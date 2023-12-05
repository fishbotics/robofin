import logging
from pathlib import Path

import numpy as np
import torch
import trimesh
import urchin
from geometrout.primitive import Sphere

from robofin.kinematics.numba import get_points_on_franka_arm, get_points_on_franka_eef
from robofin.point_cloud_tools import transform_point_cloud
from robofin.robot_constants import FrankaConstants
from robofin.torch_urdf import TorchURDF


class SamplerBase:
    def __init__(
        self,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.with_base_link = with_base_link
        self.num_robot_points = num_robot_points
        self.num_eef_points = num_eef_points

        if use_cache and self._init_from_cache_():
            return

        robot = urchin.URDF.load(FrankaConstants.urdf, lazy_load_meshes=True)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        link_points, link_normals = self._initialize_robot_points(
            robot, num_robot_points
        )
        eef_points, eef_normals = self._initialize_eef_points_and_normals(
            robot, num_eef_points
        )
        self.points = {
            **link_points,
            **eef_points,
        }
        self.normals = {
            **link_normals,
            **eef_normals,
        }

        if use_cache:
            points_to_save = {}
            for key, pc in self.points.items():
                assert key in self.normals
                normals = self.normals[key]
                points_to_save[key] = {"pc": pc, "normals": normals}
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)

    def _initialize_eef_points_and_normals(self, robot, N):
        links = [
            link
            for link in robot.links
            if link.name
            in set(
                [
                    "panda_hand",
                    "panda_rightfinger",
                    "panda_leftfinger",
                ]
            )
        ]
        meshes = [
            trimesh.load(
                Path(FrankaConstants.urdf).parent
                / link.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for link in links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        num_points = np.round(N * np.array(areas) / np.sum(areas))

        points = {}
        normals = {}
        for ii, mesh in enumerate(meshes):
            link_pc, face_indices = trimesh.sample.sample_surface(
                mesh, int(num_points[ii])
            )
            points[f"eef_{links[ii].name}"] = link_pc
            normals[f"eef_{links[ii].name}"] = self._init_normals(
                mesh, link_pc, face_indices
            )
        return points, normals

    def _initialize_robot_points(self, robot, N):
        links = [
            link
            for link in robot.links
            if link.name
            in set(
                [
                    "panda_link0",
                    "panda_link1",
                    "panda_link2",
                    "panda_link3",
                    "panda_link4",
                    "panda_link5",
                    "panda_link6",
                    "panda_link7",
                    "panda_hand",
                    "panda_rightfinger",
                    "panda_leftfinger",
                ]
            )
        ]

        meshes = [
            trimesh.load(
                Path(FrankaConstants.urdf).parent
                / link.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for link in links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        num_points = np.round(N * np.array(areas) / np.sum(areas)).astype(int)
        rounding_error = N - np.sum(num_points)
        if rounding_error > 0:
            while rounding_error > 0:
                jj = np.random.choice(np.arange(len(num_points)))
                num_points[jj] += 1
                rounding_error = N - np.sum(num_points)
        elif rounding_error < 0:
            while rounding_error < 0:
                jj = np.random.choice(np.arange(len(num_points)))
                num_points[jj] -= 1
                rounding_error = N - np.sum(num_points)

        points = {}
        normals = {}
        for ii, mesh in enumerate(meshes):
            link_pc, face_indices = trimesh.sample.sample_surface(mesh, num_points[ii])
            points[links[ii].name] = link_pc
            normals[f"{links[ii].name}"] = self._init_normals(
                mesh, link_pc, face_indices
            )
        return points, normals

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
        return (
            FrankaConstants.point_cloud_cache
            / f"franka_point_cloud_{self.num_robot_points}_{self.num_eef_points}.npy"
        )

    def _init_from_cache_(self):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = {key: v["pc"] for key, v in points.item().items()}
        self.normals = {key: v["normals"] for key, v in points.item().items()}
        return True


class NumpyFrankaSampler(SamplerBase):
    def sample(self, cfg, prismatic_joint, num_points=None):
        """num_points = 0 implies use all points."""
        assert num_points is None or 0 < num_points <= self.num_eef_points
        return get_points_on_franka_arm(
            cfg,
            prismatic_joint,
            num_points or 0,
            self.points["panda_link0"],
            self.points["panda_link1"],
            self.points["panda_link2"],
            self.points["panda_link3"],
            self.points["panda_link4"],
            self.points["panda_link5"],
            self.points["panda_link6"],
            self.points["panda_link7"],
            self.points["panda_hand"],
            self.points["panda_leftfinger"],
            self.points["panda_rightfinger"],
        )

    def sample_end_effector(
        self, pose, prismatic_joint, num_points=None, frame="right_gripper"
    ):
        assert num_points is None or 0 < num_points <= self.num_eef_points
        return get_points_on_franka_eef(
            pose,
            prismatic_joint,
            num_points or 0,
            self.points["eef_panda_hand"],
            self.points["eef_panda_leftfinger"],
            self.points["eef_panda_rightfinger"],
            frame,
        )


class TorchFrankaSampler(SamplerBase):
    """
    This class allows for fast point cloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces point clouds for each configuration by running FK on a subsample
    of the per-link point clouds that are established at initialization.

    """

    def __init__(
        self,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
        device="cpu",
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        super().__init__(num_robot_points, num_eef_points, use_cache, with_base_link)
        self.robot = TorchURDF.load(
            FrankaConstants.urdf, lazy_load_meshes=True, device=device
        )
        self.links = [l for l in self.robot.links if len(l.visuals)]
        self.points = {
            key: torch.as_tensor(val).unsqueeze(0).to(device)
            for key, val in self.points.items()
        }
        self.normals = {
            key: torch.as_tensor(val).unsqueeze(0).to(device)
            for key, val in self.normals.items()
        }

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

    def _sample_end_effector(
        self,
        with_normals,
        poses,
        prismatic_joint,
        num_points=None,
        frame="right_gripper",
    ):
        assert num_points is None or 0 < num_points <= self.num_eef_points
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
        link_fk = self.robot.link_fk_batch(default_cfg, use_names=True)
        visual_fk = self.robot.visual_geometry_fk_batch(default_cfg, use_names=True)

        fk_points = []
        if with_normals:
            fk_normals = []
        if frame == "right_gripper":
            eef_T_link8 = torch.as_tensor(
                FrankaConstants.EEF_T_LIST[
                    ("panda_link8", "right_gripper")
                ].inverse.matrix
            ).type_as(poses)
        elif frame == "panda_link8":
            eef_T_link8 = np.eye(4)
        elif frame == "panda_hand":
            eef_T_link8 = torch.as_tensor(
                FrankaConstants.EEF_T_LIST[("panda_link8", "panda_hand")].inverse.matrix
            ).type_as(poses)
        else:
            raise NotImplementedError("Other frames not supported")

        # Could just invert the matrix, but matrix inversion is not implemented for half-types
        link8_T_world = torch.zeros_like(link_fk["panda_link8"])
        link8_T_world[:, -1, -1] = 1
        link8_T_world[:, :3, :3] = link_fk["panda_link8"][:, :3, :3].transpose(1, 2)
        link8_T_world[:, :3, -1] = -torch.matmul(
            link8_T_world[:, :3, :3], link_fk["panda_link8"][:, :3, -1].unsqueeze(-1)
        ).squeeze(-1)
        eef_transform = poses @ eef_T_link8.unsqueeze(0) @ link8_T_world
        for link_name, link_idx in FrankaConstants.EEF_VISUAL_LINKS.__members__.items():
            pc = transform_point_cloud(
                self.points[f"eef_{link_name}"].float().repeat((poses.shape[0], 1, 1)),
                eef_transform @ visual_fk[link_name],
                in_place=True,
            )
            fk_points.append(
                torch.cat(
                    (
                        pc,
                        link_idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc),
                    ),
                    dim=-1,
                )
            )
            if with_normals:
                normals = transform_point_cloud(
                    self.normals[f"eef_{link_name}"]
                    .float()
                    .repeat((poses.shape[0], 1, 1)),
                    eef_transform @ fk[link_name],
                    vector=True,
                    in_place=True,
                )

                fk_normals.append(
                    torch.cat(
                        (
                            normals,
                            link_idx
                            * torch.ones((normals.size(0), normals.size(1), 1)).type_as(
                                normals
                            ),
                        ),
                        dim=-1,
                    )
                )
        pc = torch.cat(fk_points, dim=1)
        if with_normals:
            normals = torch.cat(fk_normals, dim=1)
        if num_points is None:
            if with_normals:
                return pc, normals
            else:
                return pc
        sample_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        if with_normals:
            return (pc[:, sample_idxs, :], normals[:, sample_idxs, :])
        return pc[:, sample_idxs, :]

    def sample_end_effector_with_normals(
        self,
        poses,
        prismatic_joint,
        num_points=None,
        frame="right_gripper",
    ):
        return self._sample_end_effector(
            with_normals=True,
            poses=poses,
            prismatic_joint=prismatic_joint,
            num_points=num_points,
            frame=frame,
        )

    def sample_end_effector(
        self,
        poses,
        prismatic_joint,
        num_points=None,
        frame="right_gripper",
    ):
        return self._sample_end_effector(
            with_normals=False,
            poses=poses,
            prismatic_joint=prismatic_joint,
            num_points=num_points,
            frame=frame,
        )

    def _sample(
        self, with_normals, config, prismatic_joint, num_points=None, only_eff=False
    ):
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
        fk = self.robot.visual_geometry_fk_batch(cfg, use_names=True)
        fk_points = []
        if with_normals:
            fk_normals = []
        for link_name, link_idx in FrankaConstants.ARM_VISUAL_LINKS.__members__.items():
            if only_eff and link_name not in [
                "panda_hand",
                "panda_leftfinger",
                "panda_rightfinger",
            ]:
                continue
            if not self.with_base_link and link_name == "panda_link0":
                continue
            pc = transform_point_cloud(
                self.points[link_name].float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            fk_points.append(
                torch.cat(
                    (
                        pc,
                        link_idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc),
                    ),
                    dim=-1,
                )
            )
            if with_normals:
                normals = transform_point_cloud(
                    self.normals[link_name]
                    .float()
                    .repeat((fk[link_name].shape[0], 1, 1)),
                    fk[link_name],
                    vector=True,
                    in_place=True,
                )
                fk_normals.append(
                    torch.cat(
                        (
                            normals,
                            link_idx
                            * torch.ones((normals.size(0), normals.size(1), 1)).type_as(
                                normals
                            ),
                        ),
                        dim=-1,
                    )
                )
        pc = torch.cat(fk_points, dim=1)
        if with_normals:
            normals = torch.cat(fk_normals, dim=1)
        if num_points is None:
            if with_normals:
                return pc, normals
            return pc
        random_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        if with_normals:
            return pc[:, random_idxs, :], normals[:, random_idxs, :]
        return pc[:, random_idxs, :]

    def sample(self, config, prismatic_joint, num_points=None, only_eff=False):
        return self._sample(
            with_normals=False,
            config=config,
            prismatic_joint=prismatic_joint,
            num_points=num_points,
            only_eff=only_eff,
        )

    def sample_with_normals(
        self, config, prismatic_joint, num_points=None, only_eff=False
    ):
        return self._sample(
            with_normals=True,
            config=config,
            prismatic_joint=prismatic_joint,
            num_points=num_points,
            only_eff=only_eff,
        )


class TorchFrankaCollisionSampler:
    def __init__(
        self,
        device,
        with_base_link=True,
        margin=0.0,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.robot = TorchURDF.load(
            FrankaConstants.urdf, lazy_load_meshes=True, device=device
        )
        self.spheres = []
        for radius, point_set in FrankaConstants.SPHERES:
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
        for radius, point_set in FrankaConstants.SPHERES:
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
        point_cloud = []
        for link_name, points in self.link_points.items():
            pc = transform_point_cloud(
                points.float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            point_cloud.append(pc)
        pc = torch.cat(point_cloud, dim=1)
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
                pc = transform_point_cloud(
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

        fk_points = []
        if frame == "right_gripper":
            task_T_hand = torch.as_tensor(
                FrankaConstants.EEF_T_LIST[
                    ("panda_hand", "right_gripper")
                ].inverse.matrix
            ).type_as(poses)
        elif frame == "panda_link8":
            task_T_hand = torch.as_tensor(
                FrankaConstants.EEF_T_LIST[("panda_link8", "panda_hand")].matrix
            ).type_as(poses)
        elif frame == "panda_hand":
            task_T_hand = torch.eye(4)
        # Could just invert the matrix, but matrix inversion is not implemented for half-types
        inverse_hand_transform = torch.zeros((poses.size(0), 4, 4)).type_as(poses)
        inverse_hand_transform[:, -1, -1] = 1
        inverse_hand_transform[:, :3, :3] = values[0][:, :3, :3].transpose(1, 2)
        inverse_hand_transform[:, :3, -1] = -torch.matmul(
            inverse_hand_transform[:, :3, :3],
            values[0][:, :3, -1].unsqueeze(-1).type_as(poses),
        ).squeeze(-1)
        transform = task_T_hand.unsqueeze(0) @ inverse_hand_transform

        points = []
        for radius, spheres in self.spheres:
            fk_points = []
            for link_name in spheres:
                if link_name not in eff_link_names:
                    continue
                pc = transform_point_cloud(
                    spheres[link_name].type_as(poses).repeat((poses.size(0), 1, 1)),
                    poses @ transform @ fk[link_name].type_as(poses),
                    in_place=True,
                )
                fk_points.append(pc)
            if fk_points:
                points.append((radius, torch.cat(fk_points, dim=1)))
        return points


class NumpyFrankaSelfCollisionSampler:
    def __init__(self):
        logging.getLogger("trimesh").setLevel("ERROR")

        # Set up the center points for calculating the FK position
        self._init_robot()
        self._init_points()

    def _init_points(self):
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
        self.link_points = {}
        total_points = 10000
        surface_scalar_sum = sum(
            [radius**2 for (_, _, radius) in FrankaConstants.SELF_COLLISION_SPHERES]
        )
        surface_scalar = total_points / surface_scalar_sum

        for idx1, (link_name, center, radius) in enumerate(
            FrankaConstants.SELF_COLLISION_SPHERES
        ):
            sphere = Sphere(center, radius)
            if link_name in self.link_points:
                self.link_points[link_name] = np.concatenate(
                    (
                        self.link_points[link_name],
                        sphere.sample_surface(int(surface_scalar * radius**2)),
                    ),
                    axis=0,
                )
            else:
                self.link_points[link_name] = sphere.sample_surface(
                    int(surface_scalar * radius**2)
                )

    def _init_robot(self):
        self.robot = urchin.URDF.load(FrankaConstants.urdf, lazy_load_meshes=True)

    def sample(self, config, prismatic_joint, n):
        cfg = np.ones(8)
        cfg[:7] = config
        cfg[-1] = prismatic_joint
        fk = self.robot.link_fk(cfg, use_names=True)
        pointcloud = []
        for link_name, centers in self.points:
            pc = transform_point_cloud(
                self.link_points[link_name], fk[link_name], in_place=False
            )
            pointcloud.append(pc)
        pointcloud = np.concatenate(pointcloud, axis=0)
        mask = np.random.choice(np.arange(len(pointcloud)), n, replace=False)
        return pointcloud[mask]


class TorchFrankaSelfCollisionSampler(NumpyFrankaSelfCollisionSampler):
    def __init__(self, device):
        self._init_robot(device)
        self._init_points()

        for k, v in self.link_points.items():
            self.link_points[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    def _init_robot(self, device):
        self.robot = TorchURDF.load(
            FrankaConstants.urdf, lazy_load_meshes=True, device=device
        )

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
        point_cloud = []
        for link_name, points in self.link_points.items():
            pc = transform_point_cloud(
                points.float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            point_cloud.append(pc)
        pc = torch.cat(point_cloud, dim=1)
        return pc[:, np.random.choice(pc.shape[1], n, replace=False), :]
