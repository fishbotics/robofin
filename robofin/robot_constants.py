from enum import IntEnum
from pathlib import Path

import numpy as np
from geometrout import SE3, SO3


class _FrankaConstants:
    @property
    def urdf(cls):
        return cls._URDF

    @property
    def hd_urdf(cls):
        return cls._HD_URDF

    @property
    def point_cloud_cache(cls):
        return cls._POINT_CLOUD_CACHE

    @property
    def JOINT_LIMITS(cls):
        return cls._JOINT_LIMITS

    @property
    def VELOCITY_LIMIT(cls):
        return cls._VELOCITY_LIMIT

    @property
    def ACCELERATION_LIMIT(cls):
        return cls._ACCELERATION_LIMIT

    @property
    def EEF_T_LIST(cls):
        return cls._EEF_T_LIST

    @property
    def DOF(cls):
        return cls._DOF

    @property
    def NEUTRAL(cls):
        return cls._NEUTRAL

    @property
    def SPHERES(cls):
        return cls._SPHERES

    @property
    def SELF_COLLISION_SPHERES(cls):
        return cls._SELF_COLLISION_SPHERES

    @property
    def EEF_LINKS(cls):
        return cls._EEF_LINKS

    @property
    def EEF_VISUAL_LINKS(cls):
        return cls._EEF_VISUAL_LINKS

    @property
    def ARM_LINKS(cls):
        return cls._ARM_LINKS

    @property
    def ARM_VISUAL_LINKS(cls):
        return cls._ARM_VISUAL_LINKS

    _URDF = str(Path(__file__).parent / "urdf" / "franka_panda" / "panda.urdf")

    _HD_URDF = str(Path(__file__).parent / "urdf" / "franka_panda" / "hd_panda.urdf")

    # This can be a Path because it's only ever used from Python
    _POINT_CLOUD_CACHE = Path(__file__).parent / "cache" / "point_cloud" / "franka"

    _JOINT_LIMITS = np.array(
        [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ]
    )

    _VELOCITY_LIMIT = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])

    _ACCELERATION_LIMIT = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])

    _DOF = 7

    _EEF_T_LIST = {
        ("panda_link8", "panda_hand"): SE3(
            np.zeros(3),
            np.array([0.9238795325113726, 0.0, 0.0, -0.3826834323648827]),
        ),
        ("panda_link8", "right_gripper"): SE3(
            np.array([0, 0, 0.1]), SO3.from_rpy(0, 0, 2.35619449019).q
        ),
        ("panda_link8", "panda_grasptarget"): SE3(
            np.array([0, 0, 0.105]), SO3.from_rpy(0, 0, -0.785398163397).q
        ),
        ("panda_hand", "right_gripper"): SE3(
            np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 0.0, 1.0])
        ),
    }

    _NEUTRAL = np.array(
        [
            -0.017792060227770554,
            -0.7601235411041661,
            0.019782607023391807,
            -2.342050140544315,
            0.029840531355804868,
            1.5411935298621688,
            0.7534486589746342,
        ]
    )

    # Tuples of radius in meters and the corresponding links, values are centers on that link
    _SPHERES = [
        (0.08, {"panda_link0": np.array([[0.0, 0.0, 0.05]])}),
        (
            0.06,
            {
                "panda_link1": np.array(
                    [
                        [0.0, -0.08, 0.0],
                        [0.0, -0.03, 0.0],
                        [0.0, 0.0, -0.12],
                        [0.0, 0.0, -0.17],
                    ]
                ),
                "panda_link2": np.array(
                    [
                        [0.0, 0.0, 0.03],
                        [0.0, 0.0, 0.08],
                        [0.0, -0.12, 0.0],
                        [0.0, -0.17, 0.0],
                    ]
                ),
                "panda_link3": np.array([[0.0, 0.0, -0.1]]),
                "panda_link4": np.array([[-0.08, 0.095, 0.0]]),
                "panda_link5": np.array(
                    [
                        [0.0, 0.055, 0.0],
                        [0.0, 0.075, 0.0],
                        [0.0, 0.0, -0.22],
                    ]
                ),
            },
        ),
        (
            0.05,
            {
                "panda_link3": np.array([[0.0, 0.0, -0.06]]),
                "panda_link5": np.array([[0.0, 0.05, -0.18]]),
                "panda_link6": np.array([[0.0, 0.0, 0.0], [0.08, -0.01, 0.0]]),
                "panda_link7": np.array([[0.0, 0.0, 0.07]]),
            },
        ),
        (
            0.055,
            {
                "panda_link3": np.array([[0.08, 0.06, 0.0], [0.08, 0.02, 0.0]]),
                "panda_link4": np.array(
                    [
                        [0.0, 0.0, 0.02],
                        [0.0, 0.0, 0.06],
                        [-0.08, 0.06, 0.0],
                    ]
                ),
            },
        ),
        (
            0.025,
            {
                "panda_link5": np.array(
                    [
                        [0.01, 0.08, -0.14],
                        [0.01, 0.085, -0.11],
                        [0.01, 0.09, -0.08],
                        [0.01, 0.095, -0.05],
                        [-0.01, 0.08, -0.14],
                        [-0.01, 0.085, -0.11],
                        [-0.01, 0.09, -0.08],
                        [-0.01, 0.095, -0.05],
                    ]
                ),
                "panda_link7": np.array([[0.02, 0.04, 0.08], [0.04, 0.02, 0.08]]),
            },
        ),
        (0.052, {"panda_link6": np.array([[0.08, 0.035, 0.0]])}),
        (0.02, {"panda_link7": np.array([[0.04, 0.06, 0.085], [0.06, 0.04, 0.085]])}),
        (
            0.028,
            {
                "panda_hand": np.array(
                    [
                        [0.0, -0.075, 0.01],
                        [0.0, -0.045, 0.01],
                        [0.0, -0.015, 0.01],
                        [0.0, 0.015, 0.01],
                        [0.0, 0.045, 0.01],
                        [0.0, 0.075, 0.01],
                    ]
                )
            },
        ),
        (
            0.026,
            {
                "panda_hand": np.array(
                    [
                        [0.0, -0.075, 0.03],
                        [0.0, -0.045, 0.03],
                        [0.0, -0.015, 0.03],
                        [0.0, 0.015, 0.03],
                        [0.0, 0.045, 0.03],
                        [0.0, 0.075, 0.03],
                    ]
                )
            },
        ),
        (
            0.024,
            {
                "panda_hand": np.array(
                    [
                        [0.0, -0.075, 0.05],
                        [0.0, -0.045, 0.05],
                        [0.0, -0.015, 0.05],
                        [0.0, 0.015, 0.05],
                        [0.0, 0.045, 0.05],
                        [0.0, 0.075, 0.05],
                    ]
                )
            },
        ),
        (
            0.012,
            {
                "panda_leftfinger": np.array(
                    [
                        [0, 0.015, 0.022],
                        [0, 0.008, 0.044],
                    ]
                ),
                "panda_rightfinger": np.array(
                    [
                        [0, -0.015, 0.022],
                        [0, -0.008, 0.044],
                    ]
                ),
            },
        ),
    ]

    _SELF_COLLISION_SPHERES = [
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

    _EEF_LINKS = IntEnum(
        "FrankaEefLinks",
        [
            "panda_link8",
            "panda_hand",
            "panda_grasptarget",
            "right_gripper",
            "panda_leftfinger",
            "panda_rightfinger",
        ],
        start=0,
    )

    _EEF_VISUAL_LINKS = IntEnum(
        "FrankaEefVisuals",
        [
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
        ],
        start=0,
    )

    _ARM_VISUAL_LINKS = IntEnum(
        "FrankaArmVisuals",
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
            "panda_leftfinger",
            "panda_rightfinger",
        ],
        start=0,
    )

    _ARM_LINKS = IntEnum(
        "FrankaArmLinks",
        [
            "panda_link0",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_link8",
            "panda_hand",
            "panda_grasptarget",
            "right_gripper",
            "panda_leftfinger",
            "panda_rightfinger",
        ],
        start=0,
    )


class _RealFrankaConstants(_FrankaConstants):
    _JOINT_LIMITS = np.array(
        [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (0.5, 3.75),
            (-2.8973, 2.8973),
        ]
    )


class _FrankaGripperConstants:
    @property
    def JOINT_LIMITS(cls):
        return cls._JOINT_LIMITS

    @property
    def DOF(cls):
        return cls._DOF

    @property
    def urdf(cls):
        return cls._urdf

    @property
    def fully_open_mesh(cls):
        return cls._fully_open_mesh

    @property
    def half_open_mesh(cls):
        return cls._half_open_mesh

    _JOINT_LIMITS = None
    _DOF = 6
    _urdf = str(Path(__file__).parent / "urdf" / "panda_hand" / "panda.urdf")
    _fully_open_mesh = str(
        Path(__file__).parent / "standalone_meshes" / "open_gripper.obj"
    )
    _half_open_mesh = str(
        Path(__file__).parent / "standalone_meshes" / "half_open_gripper.obj"
    )


FrankaConstants = _FrankaConstants()
RealFrankaConstants = _RealFrankaConstants()
FrankaGripperConstants = _FrankaGripperConstants()
