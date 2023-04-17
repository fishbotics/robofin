import numpy as np
import logging
import urchin
from robofin.robots import FrankaRobot

import trimesh
from pathlib import Path
from robofin.kinematics.numba import get_points_on_franka_eef, get_points_on_franka_arm


class FrankaPoints:
    def __init__(
        self,
        num_robot_points,
        num_eef_points,
        use_cache=True,
        with_base_link=True,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.with_base_link = with_base_link
        self.num_robot_points = num_robot_points
        self.num_eef_points = num_eef_points

        if use_cache and self._init_from_cache_():
            return

        robot = urchin.URDF.load(FrankaRobot.urdf, lazy_load_meshes=True)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        self.points = {
            **self._initialize_robot_points(robot, num_robot_points),
            **self._initialize_eef_points(robot, num_eef_points),
        }

        if use_cache:
            points_to_save = {k: arr for k, arr in self.points.items()}
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)

    def _initialize_eef_points(self, robot, N):
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
                Path(FrankaRobot.urdf).parent / link.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for link in links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        num_points = np.round(N * np.array(areas) / np.sum(areas))

        points = {}
        for ii, mesh in enumerate(meshes):
            points[f"eef_{links[ii].name}"] = trimesh.sample.sample_surface(
                mesh, int(num_points[ii])
            )[0]
        return points

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
                Path(FrankaRobot.urdf).parent / link.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for link in links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        num_points = np.round(N * np.array(areas) / np.sum(areas))

        points = {}
        for ii, mesh in enumerate(meshes):
            points[links[ii].name] = trimesh.sample.sample_surface(
                mesh, int(num_points[ii])
            )[0]
        return points

    def _get_cache_file_name_(self):
        return (
            FrankaRobot.pointcloud_cache
            / f"deterministic_point_cloud_{self.num_robot_points}_{self.num_eef_points}.npy"
        )

    def _init_from_cache_(self):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = points.item()
        return True

    def arm(self, cfg, prismatic_joint):
        return get_points_on_franka_arm(
            cfg,
            prismatic_joint,
            0,
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

    def end_effector(self, pose, prismatic_joint, frame="right_gripper"):
        """
        An internal method--separated so that the public facing method can
        choose whether or not to have gradients
        """
        return get_points_on_franka_eef(
            pose,
            prismatic_joint,
            0,
            self.points["eef_panda_hand"],
            self.points["eef_panda_leftfinger"],
            self.points["eef_panda_rightfinger"],
            frame,
        )


class FrankaCPUSampler:
    def __init__(
        self,
        use_cache=True,
        with_base_link=True,
        max_points=4096,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.with_base_link = with_base_link
        self.max_points = max_points

        robot = urchin.URDF.load(FrankaRobot.urdf, lazy_load_meshes=True)
        self.links = [
            l
            for l in robot.links
            if l.name
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
        if use_cache and self._init_from_cache_():
            return

        meshes = [
            trimesh.load(
                Path(FrankaRobot.urdf).parent / l.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for l in self.links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        num_points = np.round(max_points * np.array(areas) / np.sum(areas))

        self.points = {}
        for ii, mesh in enumerate(meshes):
            self.points[self.links[ii].name] = trimesh.sample.sample_surface(
                mesh, int(num_points[ii])
            )[0]

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally

        if use_cache:
            points_to_save = {
                k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.points.items()
            }
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)

    def _get_cache_file_name_(self):
        return FrankaRobot.pointcloud_cache / f"full_point_cloud_{self.max_points}.npy"

    def _init_from_cache_(self):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = points.item()
        return True

    def sample_arm(self, cfg, prismatic_joint, num_points):
        return get_points_on_franka_arm(
            cfg,
            prismatic_joint,
            num_points,
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
        self, pose, prismatic_joint, num_points, frame="right_gripper"
    ):
        """
        An internal method--separated so that the public facing method can
        choose whether or not to have gradients
        """
        return get_points_on_franka_eef(
            pose,
            prismatic_joint,
            num_points,
            self.points["panda_hand"],
            self.points["panda_leftfinger"],
            self.points["panda_rightfinger"],
            frame,
        )
