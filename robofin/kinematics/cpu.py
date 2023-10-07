import logging
from pathlib import Path

import numpy as np
import trimesh
import urchin

from robofin.kinematics.numba import get_points_on_franka_arm, get_points_on_franka_eef
from robofin.robots import FrankaRobot


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
            for key in self.points.items():
                assert key in self.normals
                pc = self.points[key]
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
                Path(FrankaRobot.urdf).parent / link.visuals[0].geometry.mesh.filename,
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
                Path(FrankaRobot.urdf).parent / link.visuals[0].geometry.mesh.filename,
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
            )[0]
            points[links[ii].name] = link_pc
            normals[f"eef_{links[ii].name}"] = self._init_normals(
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
        self.points = {key: v["pc"] for key, v in points.item().items()}
        self.normals = {key: v["normals"] for key, v in points.item().items()}
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
        if use_cache and self._init_from_cache_():
            return

        meshes = [
            trimesh.load(
                Path(FrankaRobot.urdf).parent / link.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for link in self.links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        num_points = np.round(max_points * np.array(areas) / np.sum(areas))

        self.points = {}
        for ii, mesh in enumerate(meshes):
            pc, face_indices = trimesh.sample.sample_surface(mesh, int(num_points[ii]))
            self.points[self.links[ii].name] = pc
            self.normals[self.links[ii].name] = self._init_normals(
                mesh, pc, face_indices
            )
        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally

        if use_cache:
            points_to_save = {}
            for key in self.points.items():
                assert key in self.normals
                pc = self.points[key]
                normals = self.normals[key]
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
        return FrankaRobot.pointcloud_cache / f"full_point_cloud_{self.max_points}.npy"

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
