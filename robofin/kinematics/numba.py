from enum import IntEnum

import numba
import numpy as np
from geometrout.maths import transform_in_place

FrankaEefLinks = IntEnum(
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

FrankaEefVisuals = IntEnum(
    "FrankaEefVisuals",
    [
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
    ],
    start=0,
)

FrankaArmVisuals = IntEnum(
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

FrankaArmLinks = IntEnum(
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


@numba.jit(nopython=True, cache=True)
def axis_angle(axis, angle):
    sina = np.sin(angle)
    cosa = np.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # rotation matrix around unit vector
    M = np.diag(np.array([cosa, cosa, cosa, 1.0]))
    M[:3, :3] += np.outer(axis, axis) * (1.0 - cosa)

    axis = axis * sina
    M[:3, :3] += np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )

    return M


@numba.jit(nopython=True, cache=True)
def franka_eef_link_fk(prismatic_joint: float, base_pose: np.ndarray) -> np.ndarray:
    """
    A fast Numba-based FK method for the Franka Panda

    :param pose: [TODO:description]
    :param prismatic_joint: [TODO:description]
    :return: Poses in the following order:
        [
             "panda_hand",
             "panda_grasptarget",
             "right_gripper",
             "panda_leftfinger",
             "panda_rightfinger",
         ]
    """
    joint_axes = np.array(
        [
            [1.0, 0.0, 0.0],  # panda_hand_joint
            [1.0, 0.0, 0.0],  # panda_grasptarget_hand
            [0.0, 0.0, 1.0],  # right_gripper
            [0.0, 1.0, 0.0],  # panda_finger_joint1
            [0.0, -1.0, 0.0],  # panda_finger_joint2
        ]
    )

    joint_origins = np.array(
        [
            [  # panda_hand_joint
                [0.7071067812, 0.7071067812, 0.0, 0.0],
                [-0.7071067812, 0.7071067812, -0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_grasptarget_hand
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.105],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # right_gripper
                [-0.7071067812, -0.7071067812, 0.0, 0.0],
                [0.7071067812, -0.7071067812, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_finger_joint1
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0584],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_finger_joint2
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0584],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    poses = np.zeros((6, 4, 4))
    # panda_link8 is origin
    poses[0, :, :] = base_pose
    # panda_hand is attached via fixed joint to panda_link8
    poses[1, :, :] = np.dot(poses[0], joint_origins[0])
    # panda_grasptarget is attached via fixed joint to panda_hand
    poses[2, :, :] = np.dot(poses[1], joint_origins[1])
    # right_gripper is attached via fixed joint to panda_link8
    poses[3, :, :] = np.dot(poses[0], joint_origins[0])

    # panda_leftfinger is a prismatic joint connected to panda_hand
    t_leftfinger = np.eye(4)
    t_leftfinger[:3, 3] = joint_axes[3] * prismatic_joint
    poses[4, :, :] = np.dot(poses[1], np.dot(joint_origins[3], t_leftfinger))

    # panda_rightfinger is a prismatic joint connected to panda_hand
    t_rightfinger = np.eye(4)
    t_rightfinger[:3, 3] = joint_axes[4] * prismatic_joint
    poses[5, :, :] = np.dot(poses[1], np.dot(joint_origins[4], t_rightfinger))

    return poses


@numba.jit(nopython=True, cache=True)
def franka_eef_visual_fk(prismatic_joint: float, base_pose: np.ndarray) -> np.ndarray:
    """
    base_pose must be specified in terms of panda_link8

    Returns in the following order

    panda_hand
    panda_leftfinger
    panda_rightfinger
    """
    poses = np.zeros((3, 4, 4))
    link_fk = franka_eef_link_fk(prismatic_joint, base_pose)
    poses[0, :, :] = link_fk[1, :, :]
    poses[1, :, :] = link_fk[4, :, :]
    poses[2, :, :] = np.dot(
        np.copy(link_fk[5, :, :]),
        np.array(
            [
                [-1.0, 0.0, -0.0, 0.0],
                [-0.0, -1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    return poses


@numba.jit(nopython=True, cache=True)
def franka_arm_link_fk(
    cfg: np.ndarray, prismatic_joint: float, base_pose: np.ndarray
) -> np.ndarray:
    """
    A fast Numba-based FK method for the Franka Panda

    :param cfg: [TODO:description]
    :param prismatic_joint: [TODO:description]
    :return: Poses in the following order:
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
         ]
    """
    joint_axes = np.array(
        [
            [0.0, 0.0, 1.0],  # panda_joint1
            [0.0, 0.0, 1.0],  # panda_joint2
            [0.0, 0.0, 1.0],  # panda_joint3
            [0.0, 0.0, 1.0],  # panda_joint4
            [0.0, 0.0, 1.0],  # panda_joint5
            [0.0, 0.0, 1.0],  # panda_joint6
            [0.0, 0.0, 1.0],  # panda_joint7
            [1.0, 0.0, 0.0],  # panda_joint8
            [1.0, 0.0, 0.0],  # panda_hand_joint
            [1.0, 0.0, 0.0],  # panda_grasptarget_hand
            [0.0, 0.0, 1.0],  # right_gripper
            [0.0, 1.0, 0.0],  # panda_finger_joint1
            [0.0, -1.0, 0.0],  # panda_finger_joint2
        ]
    )

    joint_origins = np.array(
        [
            [  # panda_joint1
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.333],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_joint2
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_joint3
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -0.316],
                [-0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_joint4
                [1.0, 0.0, 0.0, 0.0825],
                [0.0, 0.0, -1.0, 0.0],
                [-0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_joint5
                [1.0, -0.0, 0.0, -0.0825],
                [0.0, 0.0, 1.0, 0.384],
                [-0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_joint6
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [-0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_joint7
                [1.0, 0.0, 0.0, 0.088],
                [0.0, 0.0, -1.0, 0.0],
                [-0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_joint8
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.107],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_hand_joint
                [0.7071067812, 0.7071067812, 0.0, 0.0],
                [-0.7071067812, 0.7071067812, -0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_grasptarget_hand
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.105],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # right_gripper
                [-0.7071067812, -0.7071067812, 0.0, 0.0],
                [0.7071067812, -0.7071067812, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_finger_joint1
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0584],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [  # panda_finger_joint2
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0584],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    poses = np.zeros((14, 4, 4))
    # Base link is origin
    poses[0, :, :] = base_pose
    # panda_link0 - panda_link7 have revolute joints in simple chain
    for i in range(7):
        poses[i + 1, :, :] = np.dot(
            poses[i], np.dot(joint_origins[i], axis_angle(joint_axes[i], cfg[i]))
        )
    # panda_link8 is attached via fixed joint to panda_link7
    poses[8, :, :] = np.dot(poses[7], joint_origins[7])
    # panda_hand is attached via fixed joint to panda_link8
    poses[9, :, :] = np.dot(poses[8], joint_origins[8])
    # panda_grasptarget is attached via fixed joint to panda_hand
    poses[10, :, :] = np.dot(poses[9], joint_origins[9])
    # right_gripper is attached via fixed joint to panda_link8
    poses[11, :, :] = np.dot(poses[8], joint_origins[10])

    # panda_leftfinger is a prismatic joint connected to panda_hand
    t_leftfinger = np.eye(4)
    t_leftfinger[:3, 3] = joint_axes[11] * prismatic_joint
    poses[12, :, :] = np.dot(poses[9], np.dot(joint_origins[11], t_leftfinger))

    # panda_rightfinger is a prismatic joint connected to panda_hand
    t_rightfinger = np.eye(4)
    t_rightfinger[:3, 3] = joint_axes[12] * prismatic_joint
    poses[13, :, :] = np.dot(poses[9], np.dot(joint_origins[12], t_rightfinger))

    return poses


@numba.jit(nopython=True, cache=True)
def franka_arm_visual_fk(
    cfg: np.ndarray, prismatic_joint: float, base_pose: np.ndarray
) -> np.ndarray:
    """
    Returns in the following order

    panda_link0
    panda_link1
    panda_link2
    panda_link3
    panda_link4
    panda_link5
    panda_link6
    panda_link7
    panda_hand
    panda_leftfinger
    panda_rightfinger
    """
    poses = np.zeros((11, 4, 4))
    link_fk = franka_arm_link_fk(cfg, prismatic_joint, base_pose)
    poses[:8, :, :] = link_fk[:8, :, :]
    poses[8, :, :] = link_fk[9, :, :]
    poses[9, :, :] = link_fk[12, :, :]
    poses[10, :, :] = np.dot(
        np.copy(link_fk[13, :, :]),
        np.array(
            [
                [-1.0, 0.0, -0.0, 0.0],
                [-0.0, -1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    return poses


@numba.jit(nopython=True, cache=True)
def label(array, lbl):
    return np.concatenate((array, lbl * np.ones((array.shape[0], 1))), axis=1)


@numba.jit(nopython=True, cache=True)
def get_points_on_franka_arm(
    cfg,
    prismatic_joint,
    sample,
    panda_link0_points,
    panda_link1_points,
    panda_link2_points,
    panda_link3_points,
    panda_link4_points,
    panda_link5_points,
    panda_link6_points,
    panda_link7_points,
    panda_hand_points,
    panda_leftfinger_points,
    panda_rightfinger_points,
):
    fk = franka_arm_visual_fk(cfg, prismatic_joint, base_pose=np.eye(4))
    all_points = np.concatenate(
        (
            label(transform_in_place(np.copy(panda_link0_points), fk[0]), 0.0),
            label(transform_in_place(np.copy(panda_link1_points), fk[1]), 1.0),
            label(transform_in_place(np.copy(panda_link2_points), fk[2]), 2.0),
            label(transform_in_place(np.copy(panda_link3_points), fk[3]), 3.0),
            label(transform_in_place(np.copy(panda_link4_points), fk[4]), 4.0),
            label(transform_in_place(np.copy(panda_link5_points), fk[5]), 5.0),
            label(transform_in_place(np.copy(panda_link6_points), fk[6]), 6.0),
            label(transform_in_place(np.copy(panda_link7_points), fk[7]), 7.0),
            label(transform_in_place(np.copy(panda_hand_points), fk[8]), 8.0),
            label(transform_in_place(np.copy(panda_leftfinger_points), fk[9]), 9.0),
            label(transform_in_place(np.copy(panda_rightfinger_points), fk[10]), 10.0),
        ),
        axis=0,
    )
    if sample > 0:
        return all_points[
            np.random.choice(all_points.shape[0], sample, replace=False), :
        ]
    return all_points


@numba.jit(nopython=True, cache=True)
def eef_pose_to_link8(pose, frame):
    if frame == "right_gripper":
        pose = np.dot(
            pose,
            np.array(
                [
                    [-0.7071067812, 0.7071067812, 0.0, 0.0],
                    [-0.7071067812, -0.7071067812, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -0.1],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    elif frame == "panda_hand":
        pose = np.dot(
            pose,
            np.array(
                [
                    [0.7071067812, -0.7071067812, 0.0, 0.0],
                    [0.7071067812, 0.7071067812, -0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )
    elif frame == "panda_link8":
        pass
    else:
        raise NotImplementedError(
            "Only right_gripper, panda_hand, and panda_link8 are implemented for eef sampling"
        )
    return pose


@numba.jit(nopython=True, cache=True)
def get_points_on_franka_eef(
    pose,
    prismatic_joint,
    sample,
    panda_hand_points,
    panda_leftfinger_points,
    panda_rightfinger_points,
    frame,
):
    pose = eef_pose_to_link8(pose, frame)
    fk = franka_eef_visual_fk(prismatic_joint, pose)
    all_points = np.concatenate(
        (
            label(transform_in_place(np.copy(panda_hand_points), fk[0]), 0.0),
            label(transform_in_place(np.copy(panda_leftfinger_points), fk[1]), 1.0),
            label(transform_in_place(np.copy(panda_rightfinger_points), fk[2]), 2.0),
        ),
        axis=0,
    )
    if sample > 0:
        return all_points[
            np.random.choice(all_points.shape[0], sample, replace=False), :
        ]
    return all_points
