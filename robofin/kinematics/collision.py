from collections import namedtuple

import numpy as np
from geometrout import SE3, Sphere
from geometrout.utils import transform_in_place

from robofin.kinematics.numba import (
    FrankaArmLinks,
    FrankaEefLinks,
    eef_pose_to_link8,
    franka_arm_link_fk,
    franka_eef_link_fk,
)
from robofin.robots import FrankaRobot

SphereInfo = namedtuple("SphereInfo", "radii centers")

# Sphere model comes from STORM:
# https://github.com/NVlabs/storm/blob/e53556b64ca532e836f6bfd50893967f8224980e/content/configs/robot/franka_real_robot.yml
SELF_COLLISION_SPHERES = [
    ("panda_link0", [-0.08, 0.0, 0.05], 0.06),
    ("panda_link0", [-0.0, 0.0, 0.05], 0.08),
    ("panda_link1", [0.0, -0.08, 0.0], 0.1),
    ("panda_link1", [0.0, -0.03, 0.0], 0.1),
    ("panda_link1", [0.0, 0.0, -0.12], 0.06),
    ("panda_link1", [0.0, 0.0, -0.17], 0.06),
    ("panda_link2", [0.0, 0.0, 0.03], 0.1),
    ("panda_link2", [0.0, 0.0, 0.08], 0.1),
    ("panda_link2", [0.0, -0.12, 0.0], 0.06),
    ("panda_link2", [0.0, -0.17, 0.0], 0.06),
    ("panda_link3", [0.0, 0.0, -0.06], 0.05),
    ("panda_link3", [0.0, 0.0, -0.1], 0.06),
    ("panda_link3", [0.08, 0.06, 0.0], 0.055),
    ("panda_link3", [0.08, 0.02, 0.0], 0.055),
    ("panda_link4", [0.0, 0.0, 0.02], 0.055),
    ("panda_link4", [0.0, 0.0, 0.06], 0.055),
    ("panda_link4", [-0.08, 0.095, 0.0], 0.06),
    ("panda_link4", [-0.08, 0.06, 0.0], 0.055),
    ("panda_link5", [0.0, 0.055, 0.0], 0.05),
    ("panda_link5", [0.0, 0.085, 0.0], 0.055),
    ("panda_link5", [0.0, 0.0, -0.22], 0.05),
    ("panda_link5", [0.0, 0.05, -0.18], 0.045),
    ("panda_link5", [0.015, 0.08, -0.14], 0.03),
    ("panda_link5", [0.015, 0.085, -0.11], 0.03),
    ("panda_link5", [0.015, 0.09, -0.08], 0.03),
    ("panda_link5", [0.015, 0.095, -0.05], 0.03),
    ("panda_link5", [-0.015, 0.08, -0.14], 0.03),
    ("panda_link5", [-0.015, 0.085, -0.11], 0.03),
    ("panda_link5", [-0.015, 0.09, -0.08], 0.03),
    ("panda_link5", [-0.015, 0.095, -0.05], 0.03),
    ("panda_link6", [0.0, 0.0, 0.0], 0.05),
    ("panda_link6", [0.08, 0.035, 0.0], 0.052),
    ("panda_link6", [0.08, -0.01, 0.0], 0.05),
    ("panda_link7", [0.0, 0.0, 0.07], 0.05),
    ("panda_link7", [0.02, 0.04, 0.08], 0.025),
    ("panda_link7", [0.04, 0.02, 0.08], 0.025),
    ("panda_link7", [0.04, 0.06, 0.085], 0.02),
    ("panda_link7", [0.06, 0.04, 0.085], 0.02),
    ("panda_hand", [0.0, -0.08, 0.01], 0.03),
    ("panda_hand", [0.0, -0.045, 0.01], 0.03),
    ("panda_hand", [0.0, -0.015, 0.01], 0.03),
    ("panda_hand", [0.0, 0.015, 0.01], 0.03),
    ("panda_hand", [0.0, 0.045, 0.01], 0.03),
    ("panda_hand", [0.0, 0.08, 0.01], 0.03),
    ("panda_hand", [0.0, 0.065, -0.02], 0.05),
    ("panda_hand", [0.0, -0.08, 0.05], 0.05),
    ("panda_hand", [0.0, -0.045, 0.05], 0.05),
    ("panda_hand", [0.0, -0.015, 0.05], 0.05),
    ("panda_hand", [0.0, 0.015, 0.05], 0.05),
    ("panda_hand", [0.0, 0.045, 0.05], 0.05),
    ("panda_hand", [0.0, 0.08, 0.05], 0.05),
    ("panda_hand", [0.0, 0.08, 0.08], 0.05),
    ("panda_hand", [0.0, -0.08, 0.08], 0.05),
    ("panda_hand", [0.0, 0.05, 0.08], 0.05),
    ("panda_hand", [0.0, -0.05, 0.08], 0.05),
    ("panda_hand", [0.0, 0.0, 0.08], 0.05),
    # ("panda_leftfinger", [0.0, 0.01, 0.034], 0.02),
    # ("panda_rightfinger", [0.0, -0.01, 0.034], 0.02),
]


class FrankaCollisionSpheres:
    def __init__(
        self,
        margin=0.0,
    ):
        self._init_collision_spheres(margin)
        self._init_self_collision_spheres()

    def _init_collision_spheres(self, margin):
        spheres = {}
        for r, centers in FrankaRobot.SPHERES:
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
        for s in SELF_COLLISION_SPHERES:
            if s[0] not in centers:
                link_names.append(s[0])
                centers[s[0]] = [s[1]]
            else:
                centers[s[0]].append(s[1])
        self.points = [(name, np.asarray(centers[name])) for name in link_names]

        self.collision_matrix = -np.inf * np.ones(
            (len(SELF_COLLISION_SPHERES), len(SELF_COLLISION_SPHERES))
        )

        link_ids = {link_name: idx for idx, link_name in enumerate(link_names)}

        # Set up the self collision distance matrix
        for idx1, (link_name1, center1, radius1) in enumerate(SELF_COLLISION_SPHERES):
            for idx2, (link_name2, center2, radius2) in enumerate(
                SELF_COLLISION_SPHERES
            ):
                # Ignore all sphere pairs on the same link or adjacent links
                if abs(link_ids[link_name1] - link_ids[link_name2]) < 2:
                    continue
                self.collision_matrix[idx1, idx2] = radius1 + radius2

    def has_self_collision(self, config, prismatic_joint):
        fk = franka_arm_link_fk(config, prismatic_joint, np.eye(4))
        fk_points = []
        for link_name, centers in self.points:
            fk_points.append(
                transform_in_place(np.copy(centers), fk[FrankaArmLinks[link_name]])
            )
        transformed_centers = np.concatenate(fk_points, axis=0)
        points_matrix = np.tile(
            transformed_centers, (transformed_centers.shape[0], 1, 1)
        )
        distances = np.linalg.norm(
            points_matrix - points_matrix.transpose((1, 0, 2)), axis=2
        )
        return np.any(distances < self.collision_matrix)

    def self_collision_spheres(self, config, prismatic_joint):
        fk = franka_arm_link_fk(config, prismatic_joint, np.eye(4))
        spheres = []
        for link_name, center, radius in SELF_COLLISION_SPHERES:
            spheres.append(
                Sphere(
                    (fk[FrankaArmLinks[link_name]] @ np.array([*center, 1]))[:3], radius
                )
            )
        return spheres

    def csphere_info(
        self, config, prismatic_joint, base_pose=np.eye(4), with_base_link=False
    ):
        fk = franka_arm_link_fk(config, prismatic_joint, base_pose)
        radii = []
        centers = []
        for link_name, info in self.cspheres.items():
            if not with_base_link and link_name == "panda_link0":
                continue
            centers.append(
                transform_in_place(np.copy(info.centers), fk[FrankaArmLinks[link_name]])
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
        pose = eef_pose_to_link8(pose, frame)
        fk = franka_eef_link_fk(
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
                transform_in_place(np.copy(info.centers), fk[FrankaEefLinks[link_name]])
            )
            radii.append(info.radii)
        return SphereInfo(radii=np.concatenate(radii), centers=np.concatenate(centers))

    def eef_collision_spheres(self, pose, prismatic_joint, frame):
        info = self.eef_csphere_info(pose, prismatic_joint, frame)
        return [Sphere(c, r) for c, r in zip(info.centers, info.radii)]


def franka_arm_collides(
    q, prismatic_joint, cooo, primitives, buffer=0.0, check_self=True
):
    if check_self and cooo.has_self_collision(q, prismatic_joint):
        return True
    cspheres = cooo.csphere_info(q, prismatic_joint)
    for p in primitives:
        if np.any(p.sdf(cspheres.centers) < cspheres.radii + buffer):
            return True
    return False


def franka_eef_collides(pose, prismatic_joint, cooo, primitives, frame, buffer=0.0):
    cspheres = cooo.eef_csphere_info(pose, prismatic_joint, frame)
    for p in primitives:
        if np.any(p.sdf(cspheres.centers) < cspheres.radii + buffer):
            return True
    return False
