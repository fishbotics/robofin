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
def franka_arm_link_fk(
    cfg: torch.Tensor, prismatic_joint: float, base_pose: torch.Tensor
) -> torch.Tensor:
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
    poses = torch.zeros((B, 14, 4, 4)).type_as(cfg)
    # Base link is origin
    poses[:, 0, :, :] = base_pose
    # panda_link0 - panda_link7 have revolute joints in simple chain
    for i in range(7):
        poses[:, i + 1, :, :] = torch.matmul(
            poses[:, i],
            torch.matmul(joint_origins[i], axis_angle(joint_axes[i], cfg[:, i])),
        )
    # panda_link8 is attached via fixed joint to panda_link7
    poses[:, 8, :, :] = torch.matmul(poses[:, 7], joint_origins[7])
    # panda_hand is attached via fixed joint to panda_link8
    poses[:, 9, :, :] = torch.matmul(poses[:, 8], joint_origins[8])
    # panda_grasptarget is attached via fixed joint to panda_hand
    poses[:, 10, :, :] = torch.matmul(poses[:, 9], joint_origins[9])
    # right_gripper is attached via fixed joint to panda_link8
    poses[:, 11, :, :] = torch.matmul(poses[:, 8], joint_origins[10])

    # panda_leftfinger is a prismatic joint connected to panda_hand
    t_leftfinger = torch.eye(4).type_as(cfg)
    t_leftfinger[:3, 3] = joint_axes[11] * prismatic_joint
    poses[:, 12, :, :] = torch.matmul(
        poses[:, 9], torch.matmul(joint_origins[11], t_leftfinger)
    )

    # panda_rightfinger is a prismatic joint connected to panda_hand
    t_rightfinger = torch.eye(4).type_as(cfg)
    t_rightfinger[:3, 3] = joint_axes[12] * prismatic_joint
    poses[:, 13, :, :] = torch.matmul(
        poses[:, 9], torch.matmul(joint_origins[12], t_rightfinger)
    )

    if squeeze:
        return poses.squeeze(0)

    return poses
