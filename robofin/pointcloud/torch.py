import logging
from pathlib import Path

import numpy as np
import torch
import trimesh
from geometrout.primitive import Sphere

from robofin.collision import FrankaSelfCollisionSampler as NumpySelfCollisionSampler
from robofin.robots import FrankaRobot
from robofin.torch_urdf import TorchURDF


def transform_pointcloud(pc, transformation_matrix, vector=False, in_place=True):
    """

    Parameters
    ----------
    pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography
    vector: Whether or not to apply the translation

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography

    """
    assert isinstance(pc, torch.Tensor)
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    if vector:
        homogeneous_xyz = torch.cat(
            (xyz, torch.zeros(ones_dim, device=xyz.device)), dim=M
        )
    else:
        homogeneous_xyz = torch.cat(
            (xyz, torch.ones(ones_dim, device=xyz.device)), dim=M
        )
    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)


class FrankaSampler:
    """
    This class allows for fast pointcloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running FK on a subsample
    of the per-link pointclouds that are established at initialization.

    """

    def __init__(
        self,
        device,
        num_fixed_points=None,
        use_cache=False,
        with_base_link=True,
        max_points=4096,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.num_fixed_points = num_fixed_points
        self.with_base_link = with_base_link
        self._init_internal_(device, use_cache, max_points)

    def _init_internal_(self, device, use_cache, max_points):
        self.max_points = max_points
        self.robot = TorchURDF.load(
            FrankaRobot.urdf, lazy_load_meshes=True, device=device
        )
        self.links = [l for l in self.robot.links if len(l.visuals)]
        if use_cache and self._init_from_cache_(device):
            return

        meshes = [
            trimesh.load(
                Path(FrankaRobot.urdf).parent / l.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for l in self.links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        if self.num_fixed_points is not None:
            num_points = np.round(
                self.num_fixed_points * np.array(areas) / np.sum(areas)
            )
            num_points[0] += self.num_fixed_points - np.sum(num_points)
            assert np.sum(num_points) == self.num_fixed_points
        else:
            num_points = np.round(max_points * np.array(areas) / np.sum(areas))
        self.points = {}
        self.normals = {}
        for ii in range(len(meshes)):
            # Will have to be updated with Trimesh 4.0.0 (has different return value)
            pc, face_indices = trimesh.sample.sample_surface(
                meshes[ii], int(num_points[ii])
            )
            self.points[self.links[ii].name] = torch.as_tensor(
                pc, device=device
            ).unsqueeze(0)
            self.normals[self.links[ii].name] = torch.as_tensor(
                self._init_normals(meshes[ii], pc, face_indices),
                device=device,
            ).unsqueeze(0)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        if use_cache:
            points_to_save = {}
            for key in self.points:
                assert key in self.normals
                pc = self.points[key].squeeze(0).cpu().numpy()
                normals = self.normals[key].squeeze(0).cpu().numpy()
                points_to_save[key] = {"pc": pc, "normals": normals}

            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)

    def _init_normals(self, mesh, pc, face_indices):
        bary = trimesh.triangles.points_to_barycentric(
            triangles=mesh.triangles[face_indices], points=pc
        )
        # interpolate vertex normals from barycentric coordinates
        normals = trimesh.unitize(
            (
                mesh.vertex_normals[mesh.faces[face_indices]]
                * trimesh.unitize(bary).reshape((-1, 3, 1))
            ).sum(axis=1)
        )
        return normals

    def _get_cache_file_name_(self):
        if self.num_fixed_points is not None:
            return (
                FrankaRobot.pointcloud_cache
                / f"fixed_point_cloud_{self.num_fixed_points}_{self.max_points}.npy"
            )
        else:
            return (
                FrankaRobot.pointcloud_cache / f"full_point_cloud_{self.max_points}.npy"
            )

    def _init_from_cache_(self, device):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = {
            key: torch.as_tensor(point_info["pc"], device=device).unsqueeze(0)
            for key, point_info in points.item().items()
        }
        self.normals = {
            key: torch.as_tensor(point_info["normals"], device=device).unsqueeze(0)
            for key, point_info in points.item().items()
        }
        return True

    def end_effector_pose(self, config, prismatic_joint, frame="right_gripper"):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                prismatic_joint
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        return fk[frame]

    def sample_end_effector(
        self,
        poses,
        prismatic_joint,
        num_points=None,
        all_points=False,
        frame="right_gripper",
    ):
        """
        An internal method--separated so that the public facing method can
        choose whether or not to have gradients
        """
        if self.num_fixed_points is not None:
            all_points = True
        assert bool(all_points is False) ^ bool(num_points is None)
        assert poses.ndim in [2, 3]
        assert frame in [
            "right_gripper",
            "panda_link8",
            "panda_hand",
        ], "Other frames not yet suppported"
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        default_cfg = torch.zeros((1, 9), device=poses.device)
        default_cfg[0, 7:] = prismatic_joint
        fk = self.robot.visual_geometry_fk_batch(default_cfg)
        eff_link_names = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]

        # This logic could break--really need a way to make sure that the
        # ordering is correct
        values = [
            list(fk.values())[idx]
            for idx, l in enumerate(self.links)
            if l.name in eff_link_names
        ]
        end_effector_links = [l for l in self.links if l.name in eff_link_names]
        assert len(end_effector_links) == len(values)
        fk_transforms = {}
        fk_points = []
        fk_normals = []
        if frame == "right_gripper":
            gripper_T_hand = torch.as_tensor(
                FrankaRobot.EFF_T_LIST[("panda_hand", "right_gripper")].inverse.matrix
            ).type_as(poses)
        elif frame == "panda_link8":
            gripper_T_hand = torch.as_tensor(
                FrankaRobot.EFF_T_LIST[("panda_link8", "panda_hand")].matrix
            ).type_as(poses)
        elif frame == "panda_hand":
            gripper_T_hand = torch.eye(4)

        # Could just invert the matrix, but matrix inversion is not implemented for half-types
        inverse_hand_transform = torch.zeros_like(values[0])
        inverse_hand_transform[:, -1, -1] = 1
        inverse_hand_transform[:, :3, :3] = values[0][:, :3, :3].transpose(1, 2)
        inverse_hand_transform[:, :3, -1] = -torch.matmul(
            inverse_hand_transform[:, :3, :3], values[0][:, :3, -1].unsqueeze(-1)
        ).squeeze(-1)
        right_gripper_transform = gripper_T_hand.unsqueeze(0) @ inverse_hand_transform
        for idx, link in enumerate(end_effector_links):
            fk_transforms[link.name] = values[idx]
            pc = transform_pointcloud(
                self.points[link.name].type_as(poses),
                (right_gripper_transform @ fk_transforms[link.name]),
                in_place=False,
            )
            normals = transform_pointcloud(
                self.normals[link.name].type_as(poses),
                (right_gripper_transform @ fk_transforms[link.name]),
                vector=True,
                in_place=False,
            )

            fk_points.append(
                torch.cat(
                    (pc, idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc)),
                    dim=-1,
                )
            )
            fk_normals.append(
                torch.cat(
                    (
                        normals,
                        idx
                        * torch.ones((normals.size(0), normals.size(1), 1)).type_as(
                            normals
                        ),
                    ),
                    dim=-1,
                )
            )
        pc = torch.cat(fk_points, dim=1)
        normals = torch.cat(fk_normals, dim=1)
        pc = transform_pointcloud(pc.repeat(poses.size(0), 1, 1), poses, in_place=True)
        normals = transform_pointcloud(
            normals.repeat(poses.size(0), 1, 1), poses, vector=True, in_place=True
        )
        if num_points is None:
            return pc, normals
        sample_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        return (pc[:, sample_idxs, :], normals[:, sample_idxs, :])

    def sample(
        self, config, prismatic_joint, num_points=None, all_points=False, only_eff=False
    ):
        """
        Samples points from the surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
            actuated joints.
            For example, if using the Franka, M is 9
        num_points : Number of points desired
        all_points : Simply return all points
        only_eff : Whether to only return points on the end effector

        Returns
        -------
        N x num points x 3 pointcloud of robot points

        """
        if self.num_fixed_points is not None:
            all_points = True
        assert bool(all_points is False) ^ bool(num_points is None)
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                prismatic_joint
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        fk_normals = []
        for idx, link in enumerate(self.links):
            if only_eff and link.name not in [
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
            ]:
                continue
            if not self.with_base_link and link.name == "panda_link0":
                continue
            fk_transforms[link.name] = values[idx]
            pc = transform_pointcloud(
                self.points[link.name]
                .float()
                .repeat((fk_transforms[link.name].shape[0], 1, 1)),
                fk_transforms[link.name],
                in_place=True,
            )
            normals = transform_pointcloud(
                self.normals[link.name]
                .float()
                .repeat((fk_transforms[link.name].shape[0], 1, 1)),
                fk_transforms[link.name],
                vector=True,
                in_place=True,
            )

            fk_points.append(
                torch.cat(
                    (pc, idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc)),
                    dim=-1,
                )
            )
            fk_normals.append(
                torch.cat(
                    (
                        normals,
                        idx
                        * torch.ones((normals.size(0), normals.size(1), 1)).type_as(
                            normals
                        ),
                    ),
                    dim=-1,
                )
            )
        pc = torch.cat(fk_points, dim=1)
        normals = torch.cat(fk_normals, dim=1)
        if num_points is None:
            return pc, normals
        random_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        return pc[:, random_idxs, :], normals[:, random_idxs, :]


class FrankaCollisionSampler:
    def __init__(
        self,
        device,
        with_base_link=True,
        margin=0.0,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.robot = TorchURDF.load(
            FrankaRobot.urdf, lazy_load_meshes=True, device=device
        )
        self.spheres = []
        for radius, point_set in FrankaRobot.SPHERES:
            sphere_centers = {
                k: torch.as_tensor(v).to(device) for k, v in point_set.items()
            }
            if not with_base_link:
                sphere_centers = {
                    k: v for k, v in sphere_centers.items() if k != "panda_link0"
                }
            if not len(sphere_centers):
                continue
            self.spheres.append(
                (
                    radius + margin,
                    sphere_centers,
                )
            )

        all_spheres = {}
        for radius, point_set in FrankaRobot.SPHERES:
            for link_name, centers in point_set.items():
                if not with_base_link and link_name == "panda_link0":
                    continue
                for c in centers:
                    all_spheres[link_name] = all_spheres.get(link_name, []) + [
                        Sphere(c, radius + margin)
                    ]

        total_points = 10000
        surface_scalar_sum = sum(
            [sum([s.radius**2 for s in v]) for v in all_spheres.values()]
        )
        surface_scalar = total_points / surface_scalar_sum
        self.link_points = {}
        for link_name, spheres in all_spheres.items():
            self.link_points[link_name] = torch.as_tensor(
                np.concatenate(
                    [
                        s.sample_surface(int(surface_scalar * s.radius**2))
                        for s in spheres
                    ],
                    axis=0,
                ),
                device=device,
            )

    def sample(self, config, prismatic_joint, n):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                prismatic_joint
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        pointcloud = []
        for link_name, points in self.link_points.items():
            pc = transform_pointcloud(
                points.float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            pointcloud.append(pc)
        pc = torch.cat(pointcloud, dim=1)
        return pc[:, np.random.choice(pc.shape[1], n, replace=False), :]

    def compute_spheres(self, config, prismatic_joint):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                prismatic_joint
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        points = []
        for radius, spheres in self.spheres:
            fk_points = []
            for link_name in spheres:
                pc = transform_pointcloud(
                    spheres[link_name]
                    .type_as(cfg)
                    .repeat((fk[link_name].shape[0], 1, 1)),
                    fk[link_name].type_as(cfg),
                    in_place=True,
                )
                fk_points.append(pc)
            points.append((radius, torch.cat(fk_points, dim=1)))
        return points

    def compute_eef_spheres(self, poses, prismatic_joint, frame):
        # pose should be in panda_link8 frame
        assert frame in [
            "right_gripper",
            "panda_link8",
            "panda_hand",
        ], "Other frames not yet suppported"
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        default_cfg = torch.zeros((1, 9), device=poses.device)
        default_cfg[0, 7:] = prismatic_joint
        fk = self.robot.link_fk_batch(default_cfg, use_names=True)
        eff_link_names = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        values = [fk[name] for name in eff_link_names]

        end_effector_links = [l for l in self.links if l.name in eff_link_names]
        assert len(end_effector_links) == len(values)
        fk_points = []
        if frame == "right_gripper":
            task_T_hand = torch.as_tensor(
                FrankaRobot.EFF_T_LIST[("panda_hand", "right_gripper")].inverse.matrix
            ).type_as(poses)
        elif frame == "panda_link8":
            task_T_hand = torch.as_tensor(
                FrankaRobot.EFF_T_LIST[("panda_link8", "panda_hand")].matrix
            ).type_as(poses)
        elif frame == "panda_hand":
            task_T_hand = torch.eye(4)
        # Could just invert the matrix, but matrix inversion is not implemented for half-types
        inverse_hand_transform = torch.zeros_like(poses.size(0), 4, 4)
        inverse_hand_transform[:, -1, -1] = 1
        inverse_hand_transform[:, :3, :3] = values[0][:, :3, :3].transpose(1, 2)
        inverse_hand_transform[:, :3, -1] = -torch.matmul(
            inverse_hand_transform[:, :3, :3], values[0][:, :3, -1].unsqueeze(-1)
        ).squeeze(-1)
        transform = task_T_hand.unsqueeze(0) @ inverse_hand_transform

        points = []
        for radius, spheres in self.spheres:
            fk_points = []
            for link_name in spheres:
                if link_name not in eff_link_names:
                    continue
                pc = transform_pointcloud(
                    spheres[link_name]
                    .type_as(poses)
                    .repeat((fk[link_name].shape[0], 1, 1)),
                    transform @ fk[link_name].type_as(poses),
                    in_place=True,
                )
                fk_points.append(pc)
            if fk_points:
                points.append((radius, torch.cat(fk_points, dim=1)))
        return points


class FrankaSelfCollisionSampler(NumpySelfCollisionSampler):
    def __init__(self, device):
        super().__init__()
        self.robot = TorchURDF.load(
            FrankaRobot.urdf, lazy_load_meshes=True, device=device
        )
        for k, v in self.link_points.items():
            self.link_points[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    def sample(self, config, prismatic_joint, n):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                prismatic_joint,
                *torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        pointcloud = []
        for link_name, points in self.link_points.items():
            pc = transform_pointcloud(
                points.float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            pointcloud.append(pc)
        pc = torch.cat(pointcloud, dim=1)
        return pc[:, np.random.choice(pc.shape[1], n, replace=False), :]
