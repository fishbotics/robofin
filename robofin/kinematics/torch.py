import torch


def axis_angle(axis, angle):
    sina = torch.sin(angle)
    cosa = torch.cos(angle)
    axis = axis / torch.linalg.norm(axis)

    # rotation matrix around unit vector
    B = angle.size(0)
    M = torch.eye(4).unsqueeze(0).repeat((B, 1, 1)).type_as(angle)
    M[:, 0, 0] *= cosa
    M[:, 1, 1] *= cosa
    M[:, 2, 2] *= cosa
    M[:, :3, :3] += (
        torch.outer(axis, axis)[None, :, :].repeat((B, 1, 1))
        * (1.0 - cosa)[:, None, None]
    )

    M[:, 0, 1] += -axis[2] * sina
    M[:, 0, 2] += axis[1] * sina
    M[:, 1, 0] += axis[2] * sina
    M[:, 1, 2] += -axis[0] * sina
    M[:, 2, 0] += -axis[1] * sina
    M[:, 2, 1] += axis[0] * sina

    return M


@torch.compile
def franka_eef_link_fk(prismatic_joint: float, base_pose: torch.Tensor) -> torch.Tensor:
    """
    A fast Torch-based FK method for the Franka Panda

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
    joint_axes = torch.Tensor(
        [
            [1.0, 0.0, 0.0],  # panda_hand_joint
            [1.0, 0.0, 0.0],  # panda_grasptarget_hand
            [0.0, 0.0, 1.0],  # right_gripper
            [0.0, 1.0, 0.0],  # panda_finger_joint1
            [0.0, -1.0, 0.0],  # panda_finger_joint2
        ]
    ).type_as(base_pose)

    joint_origins = torch.Tensor(
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
    ).type_as(base_pose)

    squeeze = False
    if base_pose.ndim == 2:
        base_pose = base_pose.unsqueeze(0)
        squeeze = True
    B = base_pose.size(0)
    poses = torch.zeros((B, 6, 4, 4)).type_as(base_pose)
    # panda_link8 is origin
    poses[:, 0, :, :] = base_pose
    # panda_hand is attached via fixed joint to panda_link8
    poses[:, 1, :, :] = torch.matmul(poses[:, 0], joint_origins[0])
    # panda_grasptarget is attached via fixed joint to panda_hand
    poses[:, 2, :, :] = torch.matmul(poses[:, 1], joint_origins[1])
    # right_gripper is attached via fixed joint to panda_link8
    poses[:, 3, :, :] = torch.matmul(poses[:, 0], joint_origins[0])

    # panda_leftfinger is a prismatic joint connected to panda_hand
    t_leftfinger = torch.eye(4).type_as(base_pose)
    t_leftfinger[:3, 3] = joint_axes[3] * prismatic_joint
    poses[:, 4, :, :] = torch.matmul(
        poses[:, 1], torch.matmul(joint_origins[3], t_leftfinger)
    )

    # panda_rightfinger is a prismatic joint connected to panda_hand
    t_rightfinger = torch.eye(4).type_as(base_pose)
    t_rightfinger[:3, 3] = joint_axes[4] * prismatic_joint
    poses[:, 5, :, :] = torch.matmul(
        poses[:, 1], torch.matmul(joint_origins[4], t_rightfinger)
    )
    if squeeze:
        return poses.squeeze(0)

    return poses


@torch.compile
def franka_eef_visual_fk(
    prismatic_joint: float, base_pose: torch.Tensor
) -> torch.Tensor:
    """
    base_pose must be specified in terms of panda_link8

    Returns in the following order

    panda_hand
    panda_leftfinger
    panda_rightfinger
    """
    squeeze = False
    if base_pose.ndim == 2:
        base_pose = base_pose.unsqueeze(0)
        squeeze = True
    B = base_pose.size(0)
    poses = torch.zeros((B, 3, 4, 4)).type_as(base_pose)
    link_fk = franka_eef_link_fk(prismatic_joint, base_pose)
    poses[:, 0, :, :] = link_fk[:, 1, :, :]
    poses[:, 1, :, :] = link_fk[:, 4, :, :]
    poses[:, 2, :, :] = torch.matmul(
        torch.clone(link_fk[:, 5, :, :]),
        torch.Tensor(
            [
                [-1.0, 0.0, -0.0, 0.0],
                [-0.0, -1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).type_as(base_pose),
    )
    if squeeze:
        return poses.squeeze(0)
    return poses


@torch.compile
def franka_arm_link_fk(
    cfg: torch.Tensor, prismatic_joint: float, base_pose: torch.Tensor
) -> torch.Tensor:
    """
    A fast torch-based FK method for the Franka Panda

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
    joint_axes = torch.Tensor(
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
    ).type_as(cfg)

    joint_origins = torch.Tensor(
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
    ).type_as(cfg)
    squeeze = False
    if cfg.ndim == 1:
        cfg = cfg.unsqueeze(0)
        squeeze = True
    B = cfg.size(0)
    poses = [base_pose.expand(B, -1, -1)]
    for i in range(7):
        pose = torch.matmul(
            poses[-1],
            torch.matmul(joint_origins[i], axis_angle(joint_axes[i], cfg[:, i])),
        )
        poses.append(pose)
    # panda_link8 is attached via fixed joint to panda_link7
    poses.append(torch.matmul(poses[-1], joint_origins[7]))
    # panda_hand is attached via fixed joint to panda_link8
    poses.append(torch.matmul(poses[-1], joint_origins[8]))
    # panda_grasptarget is attached via fixed joint to panda_hand
    poses.append(torch.matmul(poses[-1], joint_origins[9]))
    # right_gripper is attached via fixed joint to panda_link8
    poses.append(torch.matmul(poses[8], joint_origins[10]))

    # panda_leftfinger is a prismatic joint connected to panda_hand
    t_leftfinger = torch.eye(4).type_as(cfg)
    t_leftfinger[:3, 3] = joint_axes[11] * prismatic_joint
    poses.append(torch.matmul(poses[9], torch.matmul(joint_origins[11], t_leftfinger)))

    # panda_rightfinger is a prismatic joint connected to panda_hand
    t_rightfinger = torch.eye(4).type_as(cfg)
    t_rightfinger[:3, 3] = joint_axes[12] * prismatic_joint
    poses.append(torch.matmul(poses[9], torch.matmul(joint_origins[12], t_rightfinger)))
    poses = torch.stack(poses, dim=1)

    if squeeze:
        return poses.squeeze(0)

    return poses


@torch.compile
def franka_arm_visual_fk(
    cfg: torch.Tensor, prismatic_joint: float, base_pose: torch.Tensor
) -> torch.Tensor:
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
    squeeze = False
    if cfg.ndim == 1:
        cfg = cfg.unsqueeze(0)
        squeeze = True
    B = cfg.size(0)
    poses = torch.zeros((B, 11, 4, 4)).type_as(cfg)
    link_fk = franka_arm_link_fk(cfg, prismatic_joint, base_pose)
    poses[:, :8, :, :] = link_fk[:, :8, :, :]
    poses[:, 8, :, :] = link_fk[:, 9, :, :]
    poses[:, 9, :, :] = link_fk[:, 12, :, :]
    poses[:, 10, :, :] = torch.matmul(
        torch.clone(link_fk[:, 13, :, :]),
        torch.Tensor(
            [
                [-1.0, 0.0, -0.0, 0.0],
                [-0.0, -1.0, 0.0, 0.0],
                [-0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).type_as(cfg),
    )
    if squeeze:
        return poses.squeeze(0)

    return poses


@torch.compile
def eef_pose_to_link8(pose, frame):
    if frame == "right_gripper":
        pose = torch.matmul(
            pose,
            torch.Tensor(
                [
                    [-0.7071067812, 0.7071067812, 0.0, 0.0],
                    [-0.7071067812, -0.7071067812, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -0.1],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ).type_as(pose),
        )

    elif frame == "panda_hand":
        pose = torch.matmul(
            pose,
            torch.Tensor(
                [
                    [0.7071067812, -0.7071067812, 0.0, 0.0],
                    [0.7071067812, 0.7071067812, -0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ).type_as(pose),
        )
    elif frame == "panda_link8":
        pass
    else:
        raise NotImplementedError(
            "Only right_gripper, panda_hand, and panda_link8 are implemented for eef sampling"
        )
    return pose
