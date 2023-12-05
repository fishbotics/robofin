import numpy as np
import torch

import robofin.kinematics.numba as rkn
from robofin import samplers
from robofin.robot_constants import FrankaConstants
from robofin.robots import FrankaRobot
from robofin.torch_urdf import TorchURDF


def compare_point_clouds(pc1, pc2):
    return np.allclose(pc1, pc2)


def test_fk():
    robot = TorchURDF.load(FrankaConstants.urdf, lazy_load_meshes=True)

    rfk = rkn.franka_arm_link_fk(FrankaConstants.NEUTRAL, 0.04, np.eye(4))
    fk = robot.link_fk_batch(
        torch.as_tensor([*FrankaConstants.NEUTRAL, 0.04]).unsqueeze(0), use_names=True
    )

    for link_name, link_idx in FrankaConstants.ARM_LINKS.__members__.items():
        assert np.allclose(fk[link_name].squeeze(0).numpy(), rfk[link_idx]), link_name

    rfk = rkn.franka_arm_visual_fk(FrankaConstants.NEUTRAL, 0.04, np.eye(4))
    fk = robot.visual_geometry_fk_batch(
        torch.as_tensor([*FrankaConstants.NEUTRAL, 0.04]).unsqueeze(0), use_names=True
    )
    for link_name, link_idx in FrankaConstants.ARM_VISUAL_LINKS.__members__.items():
        assert np.allclose(fk[link_name].squeeze(0).numpy(), rfk[link_idx]), link_name


def test_eef_fk():
    robot = TorchURDF.load(FrankaConstants.urdf, lazy_load_meshes=True)

    rfk = rkn.franka_arm_link_fk(FrankaConstants.NEUTRAL, 0.04, np.eye(4))
    fk = robot.link_fk_batch(
        torch.as_tensor([*FrankaConstants.NEUTRAL, 0.04]).unsqueeze(0), use_names=True
    )

    for link_name, link_idx in FrankaConstants.ARM_LINKS.__members__.items():
        assert np.allclose(fk[link_name].squeeze(0).numpy(), rfk[link_idx]), link_name

    rfk = rkn.franka_arm_visual_fk(FrankaConstants.NEUTRAL, 0.04, np.eye(4))
    fk = robot.visual_geometry_fk_batch(
        torch.as_tensor([*FrankaConstants.NEUTRAL, 0.04]).unsqueeze(0), use_names=True
    )
    for link_name, link_idx in FrankaConstants.ARM_VISUAL_LINKS.__members__.items():
        assert np.allclose(fk[link_name].squeeze(0).numpy(), rfk[link_idx]), link_name


def test_deterministic_numpy_sampling():
    sampler1 = samplers.NumpyFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.NumpyFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    samples2 = sampler2.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    assert compare_point_clouds(samples1, samples2)
    eef_samples1 = sampler1.sample_end_effector(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        0.04,
    )
    eef_samples2 = sampler2.sample_end_effector(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        0.04,
    )
    assert compare_point_clouds(eef_samples1, eef_samples2)


def test_deterministic_torch_sampling():
    sampler1 = samplers.TorchFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.TorchFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        torch.as_tensor(FrankaConstants.NEUTRAL),
        0.04,
    )
    samples2 = sampler2.sample(
        torch.as_tensor(FrankaConstants.NEUTRAL),
        0.04,
    )
    assert compare_point_clouds(samples1.squeeze().numpy(), samples2.squeeze().numpy())
    eef_samples1 = sampler1.sample_end_effector(
        torch.as_tensor(FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix).float(),
        0.04,
    )
    eef_samples2 = sampler2.sample_end_effector(
        torch.as_tensor(FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix).float(),
        0.04,
    )
    assert compare_point_clouds(
        eef_samples1.squeeze().numpy(), eef_samples2.squeeze().numpy()
    )


def test_deterministic_gen_cache():
    sampler1 = samplers.NumpyFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.TorchFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    samples2 = sampler2.sample(
        torch.as_tensor(FrankaConstants.NEUTRAL),
        0.04,
    )
    assert compare_point_clouds(samples1, samples2.squeeze().numpy())
    eef_samples1 = sampler1.sample_end_effector(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        0.04,
    )
    eef_samples2 = sampler2.sample_end_effector(
        torch.as_tensor(FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix).float(),
        0.04,
    )
    assert compare_point_clouds(eef_samples1, eef_samples2.squeeze().numpy())


def test_deterministic_compare():
    sampler1 = samplers.NumpyFrankaSampler(
        num_robot_points=2048, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.NumpyFrankaSampler(
        num_robot_points=1024, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler3 = samplers.TorchFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
