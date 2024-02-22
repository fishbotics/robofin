from collections import namedtuple

import numpy as np
import torch
from geometrout import SE3, Sphere
from geometrout.maths import transform_in_place

import robofin.kinematics.numba as nfk
import robofin.kinematics.torch as tfk
from robofin.robot_constants import FrankaConstants
from robofin.torch_urdf import TorchURDF

SphereInfo = namedtuple("SphereInfo", "radii centers")

# Sphere model comes from STORM:
# https://github.com/NVlabs/storm/blob/e53556b64ca532e836f6bfd50893967f8224980e/content/configs/robot/franka_real_robot.yml


class FrankaCollisionSpheres:
    def __init__(
        self,
        margin=0.0,
    ):
        self._init_collision_spheres(margin)
        self._init_self_collision_spheres()

    def _init_collision_spheres(self, margin):
        spheres = {}
        for r, centers in FrankaConstants.SPHERES:
            for k, c in centers.items():
                spheres[k] = spheres.get(k, [])
                spheres[k].append(
                    SphereInfo(radii=r * np.ones((c.shape[0])) + margin, centers=c)
                )
        self.cspheres = {}
        for k, v in spheres.items():
            radii = np.concatenate([ix.radii for ix in v])
            radii.setflags(write=False)
            centers = np.concatenate([ix.centers for ix in v])
            centers.setflags(write=False)
            self.cspheres[k] = SphereInfo(
                radii=radii,
                centers=centers,
            )

    def _init_self_collision_spheres(self):
        link_names = []
        centers = {}
        for s in FrankaConstants.SELF_COLLISION_SPHERES:
            if s[0] not in centers:
                link_names.append(s[0])
                centers[s[0]] = [s[1]]
            else:
                centers[s[0]].append(s[1])
        self.points = [(name, np.asarray(centers[name])) for name in link_names]

        self.collision_matrix = -np.inf * np.ones(
            (
                len(FrankaConstants.SELF_COLLISION_SPHERES),
                len(FrankaConstants.SELF_COLLISION_SPHERES),
            )
        )

        link_ids = {link_name: idx for idx, link_name in enumerate(link_names)}

        # Set up the self collision distance matrix
        for idx1, (link_name1, center1, radius1) in enumerate(
            FrankaConstants.SELF_COLLISION_SPHERES
        ):
            for idx2, (link_name2, center2, radius2) in enumerate(
                FrankaConstants.SELF_COLLISION_SPHERES
            ):
                # Ignore all sphere pairs on the same link or adjacent links
                if abs(link_ids[link_name1] - link_ids[link_name2]) < 2:
                    continue
                self.collision_matrix[idx1, idx2] = radius1 + radius2

    def has_self_collision(self, config, prismatic_joint, buffer=0.0):
        fk = nfk.franka_arm_link_fk(config, prismatic_joint, np.eye(4))
        fk_points = []
        for link_name, centers in self.points:
            fk_points.append(
                transform_in_place(
                    np.copy(centers), fk[FrankaConstants.ARM_LINKS[link_name]]
                )
            )
        transformed_centers = np.concatenate(fk_points, axis=0)
        points_matrix = np.tile(
            transformed_centers, (transformed_centers.shape[0], 1, 1)
        )
        distances = np.linalg.norm(
            points_matrix - points_matrix.transpose((1, 0, 2)), axis=2
        )
        return np.any(distances < self.collision_matrix + buffer)

    def self_collision_spheres(self, config, prismatic_joint):
        fk = nfk.franka_arm_link_fk(config, prismatic_joint, np.eye(4))
        spheres = []
        for link_name, center, radius in FrankaConstants.SELF_COLLISION_SPHERES:
            spheres.append(
                Sphere(
                    (fk[FrankaConstants.ARM_LINKS[link_name]] @ np.array([*center, 1]))[
                        :3
                    ],
                    radius,
                )
            )
        return spheres

    def csphere_info(
        self, config, prismatic_joint, base_pose=np.eye(4), with_base_link=False
    ):
        fk = nfk.franka_arm_link_fk(config, prismatic_joint, base_pose)
        radii = []
        centers = []
        for link_name, info in self.cspheres.items():
            if not with_base_link and link_name == "panda_link0":
                continue
            centers.append(
                transform_in_place(
                    np.copy(info.centers), fk[FrankaConstants.ARM_LINKS[link_name]]
                )
            )
            radii.append(info.radii)
        return SphereInfo(radii=np.concatenate(radii), centers=np.concatenate(centers))

    def collision_spheres(
        self, config, prismatic_joint, base_pose=np.eye(4), with_base_link=False
    ):
        info = self.csphere_info(config, prismatic_joint, base_pose, with_base_link)
        return [Sphere(c, r) for c, r in zip(info.centers, info.radii)]

    def eef_csphere_info(self, pose, prismatic_joint, frame):
        if isinstance(pose, SE3):
            pose = pose.matrix
        pose = nfk.eef_pose_to_link8(pose, frame)
        fk = nfk.franka_eef_link_fk(
            prismatic_joint,
            pose,
        )
        radii = []
        centers = []
        for link_name in [
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ]:
            info = self.cspheres[link_name]
            centers.append(
                transform_in_place(
                    np.copy(info.centers), fk[FrankaConstants.EEF_LINKS[link_name]]
                )
            )
            radii.append(info.radii)
        return SphereInfo(radii=np.concatenate(radii), centers=np.concatenate(centers))

    def eef_collision_spheres(self, pose, prismatic_joint, frame):
        info = self.eef_csphere_info(pose, prismatic_joint, frame)
        return [Sphere(c, r) for c, r in zip(info.centers, info.radii)]

    def franka_arm_collides(
        self,
        q,
        prismatic_joint,
        primitives,
        *,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        check_self=True,
        with_base_link=False,
    ):
        if check_self and self.has_self_collision(
            q, prismatic_joint, self_collision_buffer
        ):
            return True
        cspheres = self.csphere_info(q, prismatic_joint, with_base_link=with_base_link)
        for p in primitives:
            if np.any(p.sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
        return False

    def franka_arm_collides_fast(
        self,
        q,
        prismatic_joint,
        primitive_arrays,
        *,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        check_self=True,
        with_base_link=False,
    ):
        if check_self and self.has_self_collision(
            q, prismatic_joint, self_collision_buffer
        ):
            return True
        cspheres = self.csphere_info(q, prismatic_joint, with_base_link=with_base_link)
        for arr in primitive_arrays:
            if np.any(arr.scene_sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
        return False

    def franka_eef_collides(
        self, pose, prismatic_joint, primitives, frame, scene_buffer=0.0
    ):
        cspheres = self.eef_csphere_info(pose, prismatic_joint, frame)
        for p in primitives:
            if np.any(p.sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
        return False

    def franka_eef_collides_fast(
        self, pose, prismatic_joint, primitive_arrays, frame, scene_buffer=0.0
    ):
        cspheres = self.eef_csphere_info(pose, prismatic_joint, frame)
        for arr in primitive_arrays:
            if np.any(arr.scene_sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
        return False


class TorchFrankaCollisionSpheres:
    def __init__(
        self,
        margin=0.0,
        device="cpu",
    ):
        self._init_collision_spheres(margin, device)
        self._init_self_collision_spheres(device)

    def transform_in_place(self, point_cloud, transformation_matrix):
        point_cloud_T = torch.transpose(point_cloud, -2, -1)
        ones_shape = list(
            point_cloud_T.shape
        )  # Convert shape to a list to manipulate it
        ones_shape[
            -2
        ] = 1  # Set the second last dimension to 1 (for the second dimension in (B, 3, N) or (3, N))
        ones = torch.ones(ones_shape).type_as(point_cloud)
        homogeneous_xyz = torch.cat((point_cloud_T, ones), dim=-2)
        transformed_xyz = torch.matmul(transformation_matrix, homogeneous_xyz)
        point_cloud[..., :3] = torch.transpose(transformed_xyz[..., :3, :], -2, -1)
        return point_cloud

    def _init_collision_spheres(self, margin, device):
        spheres = {}
        for r, centers in FrankaConstants.SPHERES:
            for k, c in centers.items():
                spheres[k] = spheres.get(k, [])
                spheres[k].append(
                    SphereInfo(
                        radii=r * torch.ones((c.shape[0]), device=device) + margin,
                        centers=torch.as_tensor(c).to(device),
                    )
                )
        self.cspheres = {}
        for k, v in spheres.items():
            radii = torch.cat([ix.radii for ix in v])
            centers = torch.cat([ix.centers for ix in v])
            self.cspheres[k] = SphereInfo(
                radii=radii,
                centers=centers,
            )

    def _init_self_collision_spheres(self, device):
        link_names = []
        centers = {}
        for link_name, center, _ in FrankaConstants.SELF_COLLISION_SPHERES:
            if link_name not in centers:
                link_names.append(link_name)
                centers[link_name] = [torch.as_tensor(center).to(device)]
            else:
                centers[link_name].append(torch.as_tensor(center).to(device))
        self.points = [(name, torch.stack(centers[name])) for name in link_names]

        self.collision_matrix = -np.inf * torch.ones(
            (
                len(FrankaConstants.SELF_COLLISION_SPHERES),
                len(FrankaConstants.SELF_COLLISION_SPHERES),
            ),
            device=device,
        )

        link_ids = {link_name: idx for idx, link_name in enumerate(link_names)}

        # Set up the self collision distance matrix
        for idx1, (link_name1, center1, radius1) in enumerate(
            FrankaConstants.SELF_COLLISION_SPHERES
        ):
            for idx2, (link_name2, center2, radius2) in enumerate(
                FrankaConstants.SELF_COLLISION_SPHERES
            ):
                # Ignore all sphere pairs on the same link or adjacent links
                if abs(link_ids[link_name1] - link_ids[link_name2]) < 2:
                    continue
                self.collision_matrix[idx1, idx2] = radius1 + radius2

    def has_self_collision(self, config, prismatic_joint, buffer=0.0):
        squeeze = False
        if config.ndim == 1:
            config = config[None, :]
            squeeze = True
        B = config.size(0)
        fk = tfk.franka_arm_link_fk(
            config, prismatic_joint, torch.eye(4).type_as(config)
        )
        fk_points = []
        for link_name, centers in self.points:
            fk_points.append(
                self.transform_in_place(
                    torch.clone(centers[None, ...].expand(B, -1, -1)).type_as(fk),
                    fk[:, FrankaConstants.ARM_LINKS[link_name]],
                )
            )
        transformed_centers = torch.cat(fk_points, dim=1).unsqueeze(1)
        points_matrix = torch.tile(
            transformed_centers, (1, transformed_centers.shape[2], 1, 1)
        )
        distances = torch.linalg.norm(
            points_matrix - points_matrix.permute((0, 2, 1, 3)), dim=3
        )
        self_collisions = torch.any(
            (distances < self.collision_matrix[None, ...]).reshape(
                distances.size(0), -1
            )
            + buffer,
            dim=1,
        )
        if squeeze:
            self_collisions = self_collisions.squeeze(0)
        return self_collisions

    def self_collision_spheres(self, config, prismatic_joint):
        raise NotImplementedError("Not implemented in Pytorch")

    def csphere_info(
        self, config, prismatic_joint, base_pose=None, with_base_link=False
    ):
        squeeze = False
        if config.ndim == 1:
            config = config[None, :]
            squeeze = True
        if base_pose is None:
            base_pose = torch.eye(4)[None, ...].type_as(config)
        elif base_pose.ndim == 2:
            base_pose = base_pose.unsqueeze(0)
        B = config.size(0)
        fk = tfk.franka_arm_link_fk(config, prismatic_joint, base_pose)
        radii = []
        centers = []
        for link_name, info in self.cspheres.items():
            if not with_base_link and link_name == "panda_link0":
                continue
            centers.append(
                self.transform_in_place(
                    torch.clone(info.centers[None, ...].expand(B, -1, -1)).type_as(fk),
                    fk[:, FrankaConstants.ARM_LINKS[link_name]],
                )
            )
            radii.append(info.radii[None, ...])
        radii = torch.cat(radii, dim=1)
        centers = torch.cat(centers, dim=1)
        if squeeze:
            radii = radii.squeeze(0)
            centers = centers.squeeze(0)
        return SphereInfo(radii=radii, centers=centers)

    def collision_spheres(
        self, config, prismatic_joint, base_pose=np.eye(4), with_base_link=False
    ):
        raise NotImplementedError("Not implemented in Pytorch")

    def eef_csphere_info(self, pose, prismatic_joint, frame):
        squeeze = False
        if pose.ndim == 2:
            pose = pose[None, :, :]
            squeeze = True
        pose = tfk.eef_pose_to_link8(pose, frame)
        fk = tfk.franka_eef_link_fk(
            prismatic_joint,
            pose,
        )
        radii = []
        centers = []
        for link_name in [
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ]:
            info = self.cspheres[link_name]
            centers.append(
                self.transform_in_place(
                    torch.clone(info.centers), fk[FrankaConstants.EEF_LINKS[link_name]]
                )
            )
            radii.append(info.radii)
        radii = torch.cat(radii, dim=1)
        centers = torch.cat(centers, dim=1)
        if squeeze:
            radii = radii.squeeze(0)
            centers = centers.squeeze(0)
        return SphereInfo(radii=radii, centers=centers)

    def eef_collision_spheres(self, pose, prismatic_joint, frame):
        raise NotImplementedError("Not implemented in Pytorch")

    def franka_arm_collides(
        self,
        q,
        prismatic_joint,
        primitives,
        *,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        check_self=True,
        with_base_link=False,
    ):
        if not isinstance(primitives, list):
            primitives = [primitives]
        squeeze = False
        if q.ndim == 1:
            q = q.unsqueeze(0)
            squeeze = True
        collisions = torch.zeros((q.size(0),), dtype=bool, device=q.device)
        if check_self:
            self_collisions = self.has_self_collision(
                q, prismatic_joint, self_collision_buffer
            )
            collisions = torch.logical_or(self_collisions, collisions)
        cspheres = self.csphere_info(q, prismatic_joint, with_base_link=with_base_link)
        for p in primitives:
            p_collisions = torch.any(
                p.sdf(cspheres.centers) < cspheres.radii + scene_buffer, dim=1
            )
            collisions = torch.logical_or(p_collisions, collisions)
        if squeeze:
            collisions = collisions.squeeze(0)
        return collisions

    def franka_arm_collides_sequence(
        self,
        q,
        prismatic_joint,
        primitives,
        *,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        check_self=True,
        with_base_link=False,
    ):
        """Specialized method works with sequences of configations [B, T, 7]."""
        if not isinstance(primitives, list):
            primitives = [primitives]
        assert q.ndim == 3
        B, T, _ = q.shape
        collisions = torch.zeros(
            (
                B,
                T,
            ),
            dtype=bool,
            device=q.device,
        )
        flat_q = q.reshape(B * T, -1)
        if check_self:
            self_collisions = self.has_self_collision(
                flat_q, prismatic_joint, self_collision_buffer
            ).reshape(B, T)
            collisions = torch.logical_or(self_collisions, collisions)
        cspheres = self.csphere_info(
            flat_q, prismatic_joint, with_base_link=with_base_link
        )
        centers = cspheres.centers.reshape(
            B, T, cspheres.centers.size(1), cspheres.centers.size(2)
        )
        radii = cspheres.radii.unsqueeze(0)

        for p in primitives:
            p_collisions = torch.any(
                p.sdf_sequence(centers) < radii + scene_buffer,
                dim=2,
            )
            collisions = torch.logical_or(p_collisions, collisions)
        return collisions

    def franka_eef_collides(
        self, pose, prismatic_joint, primitives, frame, scene_buffer=0.0
    ):
        if not isinstance(primitives, list):
            primitives = [primitives]
        squeeze = False
        if pose.ndim == 2:
            pose = pose.unsqueeze(0)
            squeeze = True
        collisions = torch.zeros((pose.size(0),), dtype=bool, device=pose.device)
        cspheres = self.eef_csphere_info(pose, prismatic_joint, frame)
        for p in primitives:
            p_collisions = torch.any(
                p.sdf(cspheres.centers) < cspheres.radii + scene_buffer, dim=1
            )
            collisions = torch.logical_or(p_collisions, collisions)
        if squeeze:
            collisions = collisions.squeeze(0)
        return collisions
