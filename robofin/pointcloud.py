from pathlib import Path
import logging

import torch
import numpy as np
import trimesh

from robofin.torch_urdf import TorchURDF
from robofin.robots import FrankaRobot


def project(transformation_matrix, point, rotate_only=False):
    if rotate_only:
        return (transformation_matrix @ np.append(point, [0]))[:3]
    return (transformation_matrix @ np.append(point, [1]))[:3]


def transform_pointcloud(pc, transformation_matrix, in_place=True):
    """

    Parameters
    ----------
    pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography

    """
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
    homogeneous_xyz = torch.cat((xyz, torch.ones(ones_dim, device=xyz.device)), dim=M)
    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)


class SamplerBase:
    def _end_effector(self, config):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (config, torch.zeros((config.shape[0], 2), device=config.device)), dim=1
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        return fk["right_gripper"]

    def end_effector(self, config, force_no_grad=False):
        if self.no_grad or force_no_grad:
            with torch.no_grad():
                return self._end_effector(config)
        return self._end_effector(config)


class FrankaFK(SamplerBase):
    """
    This is just a very simple class that only gives the end-effector pose.
    """

    def __init__(self, device, no_grad=False):
        self.robot = TorchURDF.load(FrankaRobot.urdf, device)
        self.no_grad = no_grad


class FrankaSampler(SamplerBase):
    """
    This class allows for fast pointcloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running FK on a subsample
    of the per-link pointclouds that are established at initialization.

    """

    def __init__(self, device, no_grad=False, num_fixed_points=None):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.no_grad = no_grad
        self.num_fixed_points = num_fixed_points
        if self.no_grad:
            with torch.no_grad():
                self._init_internal_(device)
        else:
            self._init_internal_(device)

    def _init_internal_(self, device):
        self.robot = TorchURDF.load(FrankaRobot.urdf, device)
        self.links = [l for l in self.robot.links if len(l.visuals)]
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
            num_points = np.round(4096 * np.array(areas) / np.sum(areas))
        self.points = {}
        for ii in range(len(meshes)):
            pc = trimesh.sample.sample_surface(meshes[ii], int(num_points[ii]))[0]
            self.points[self.links[ii].name] = torch.as_tensor(
                pc, device=device
            ).unsqueeze(0)

    def _sample_end_effector(self, config, num_points):
        """
        An internal method--separated so that the public facing method can
        choose whether or not to have gradients
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (config, torch.zeros((config.shape[0], 2), device=config.device)), dim=1
        )
        fk = self.robot.visual_geometry_fk_batch(cfg)
        eff_link_names = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        values = list(fk.values())
        values = [
            values[idx] for idx, l in enumerate(self.links) if l.name in eff_link_names
        ]
        end_effector_links = [l for l in self.links if l.name in eff_link_names]
        assert len(end_effector_links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(end_effector_links):
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud(
                self.points[l.name]
                .float()
                .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                fk_transforms[l.name],
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def sample_end_effector(self, config, num_points=None):
        """
        Samples points from the surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of actuated joints
            For example, if using the Franka, M is 9
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot points

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None)
        if self.no_grad:
            with torch.no_grad():
                return self._sample_end_effector(config, num_points)
        return self._sample_end_effector(config, num_points)

    def _sample(self, config, num_points):
        """
        An internal method--separated so that the public facing method can
        choose whether or not to have gradients
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (config, torch.zeros((config.shape[0], 2), device=config.device)), dim=1
        )
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud(
                self.points[l.name]
                .float()
                .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                fk_transforms[l.name],
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def sample(self, config, num_points=None):
        """
        Samples points from the surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
            actuated joints.
            For example, if using the Franka, M is 9
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot points

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None)
        if self.no_grad:
            with torch.no_grad():
                return self._sample(config, num_points)
        return self._sample(config, num_points)
