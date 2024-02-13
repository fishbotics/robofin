# MIT License
#
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, University of Washington. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import random
from typing import Sequence, Union

import numpy as np
import torch
from geometrout.primitive import Cuboid, Cylinder, Sphere


def rotation_matrix_to_quaternion(rot_mat):
    """
    Convert a batch of rotation matrices to quaternions. From ChatGPT--not 100% sure its right.

    Args:
    rot_mat (Tensor): A tensor of shape (N, 3, 3) representing rotation matrices.

    Returns:
    Tensor: A tensor of shape (N, 4) representing quaternions.
    """
    batch_size = rot_mat.shape[0]

    # Preallocate a tensor for the quaternions
    quaternions = torch.zeros((batch_size, 4), device=rot_mat.device)

    # The elements of the quaternion can be directly found from the elements of the rotation matrix
    trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
    for i in range(batch_size):
        if trace[i] > 0:
            s = torch.sqrt(trace[i] + 1.0) * 2  # S=4*qw
            qw = 0.25 * s
            qx = (rot_mat[i, 2, 1] - rot_mat[i, 1, 2]) / s
            qy = (rot_mat[i, 0, 2] - rot_mat[i, 2, 0]) / s
            qz = (rot_mat[i, 1, 0] - rot_mat[i, 0, 1]) / s
        elif (rot_mat[i, 0, 0] > rot_mat[i, 1, 1]) and (
            rot_mat[i, 0, 0] > rot_mat[i, 2, 2]
        ):
            s = (
                torch.sqrt(1.0 + rot_mat[i, 0, 0] - rot_mat[i, 1, 1] - rot_mat[i, 2, 2])
                * 2
            )
            qw = (rot_mat[i, 2, 1] - rot_mat[i, 1, 2]) / s
            qx = 0.25 * s
            qy = (rot_mat[i, 0, 1] + rot_mat[i, 1, 0]) / s
            qz = (rot_mat[i, 0, 2] + rot_mat[i, 2, 0]) / s
        elif rot_mat[i, 1, 1] > rot_mat[i, 2, 2]:
            s = (
                torch.sqrt(1.0 + rot_mat[i, 1, 1] - rot_mat[i, 0, 0] - rot_mat[i, 2, 2])
                * 2
            )
            qw = (rot_mat[i, 0, 2] - rot_mat[i, 2, 0]) / s
            qx = (rot_mat[i, 0, 1] + rot_mat[i, 1, 0]) / s
            qy = 0.25 * s
            qz = (rot_mat[i, 1, 2] + rot_mat[i, 2, 1]) / s
        else:
            s = (
                torch.sqrt(1.0 + rot_mat[i, 2, 2] - rot_mat[i, 0, 0] - rot_mat[i, 1, 1])
                * 2
            )
            qw = (rot_mat[i, 1, 0] - rot_mat[i, 0, 1]) / s
            qx = (rot_mat[i, 0, 2] + rot_mat[i, 2, 0]) / s
            qy = (rot_mat[i, 1, 2] + rot_mat[i, 2, 1]) / s
            qz = 0.25 * s

        quaternions[i] = torch.tensor([qw, qx, qy, qz])

    return quaternions


class TorchSpheres:
    """
    A Pytorch representation of a batch of M spheres (i.e. B elements in the batch,
    M spheres per element). Any of these spheres can have zero volume (these
    will be masked out during calculation of the various functions in this
    class, such as sdf).
    """

    def __init__(self, centers: torch.Tensor, radii: torch.Tensor, mask=None):
        """
        :param centers torch.Tensor: a set of centers, has dim [B, M, 3]
        :param radii torch.Tensor: a set of radii, has dim [B, M, 1]
        """
        assert centers.ndim == 3
        assert radii.ndim == 3

        # TODO It would be more memory efficient to rely more heavily on broadcasting
        # in some cases where multiple spheres have the same size
        assert centers.ndim == radii.ndim

        # This logic is to determine the batch size. Either batch sizes need to
        # match or if only one variable has a batch size, that one is assumed to
        # be the batch size

        self.centers = centers
        self.radii = radii
        if mask is None:
            mask = ~torch.isclose(self.radii, torch.zeros(1).type_as(centers)).squeeze(
                -1
            )
        self.mask = mask

    def __getitem__(self, *args):
        return TorchSpheres(
            self.centers.__getitem__(*args),
            self.radii.__getitem__(*args),
            self.mask.__getitem__(*args),
        )

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the spheres

        :rtype torch.Tensor: A tensor of the surface areas of the spheres
        """
        area = 4 * np.pi * torch.pow(self.radii, 3)
        return area

    def sample_surface(self, num_points: int) -> torch.Tensor:
        """
        Samples points from all spheres, including ones with zero volume

        :param num_points int: The number of points to sample per sphere
        :rtype torch.Tensor: The points, has dim [B, M, N]
        """
        B, M, _ = self.centers.shape
        unnormalized_points = torch.rand((B, M, num_points, 3))
        normalized = (
            unnormalized_points
            / torch.linalg.norm(unnormalized_points, dim=-1)[:, :, :, None]
        )
        random_points = (
            normalized * self.radii[:, :, None, :] + self.centers[:, :, None, :]
        )
        return random_points

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the
                                    SDF, has dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                             value for each of the M spheres), has dim [B, N]
        """
        assert points.ndim == 3
        B, M, _ = self.radii.shape
        _, N, _ = points.shape
        all_sdfs = float("inf") * torch.ones(B, M, N).type_as(points)
        distances = points[:, None, :, :] - self.centers[:, :, None, :]
        all_sdfs[self.mask] = (
            torch.linalg.norm(distances[self.mask], dim=-1) - self.radii[self.mask]
        )
        return torch.min(all_sdfs, dim=1)[0]

    def sdf_sequence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates SDF values for a time sequence of point clouds
        :param points torch.Tensor: The batched sequence of point clouds with
                                    dimension [B, T, N, 3] (T in sequence length,
                                    N is number of points)
        :rtype torch.Tensor: The scene SDF for each point at each timestep (i.e. the minimum
                             SDF value across the M spheres at each timestep),
                             has dim [B, T, N]
        """
        assert points.ndim == 4
        B, M, _ = self.radii.shape
        _, T, N, _ = points.shape
        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        distances = points[:, None, :, :] - self.centers[:, :, None, None, :]
        all_sdfs[self.mask] = (
            torch.linalg.norm(distances[self.mask], dim=-1)
            - self.radii[:, :, None, :][self.mask]
        )
        return torch.min(all_sdfs, dim=1)[0]


class TorchCuboids:
    """
    A Pytorch representation of a batch of M cuboids (i.e. B elements in the batch,
    M cuboids per element). Any of these cuboids can have zero volume (these
    will be masked out during calculation of the various functions in this
    class, such as sdf).
    """

    def __init__(
        self,
        centers: torch.Tensor,
        dims: torch.Tensor,
        quaternions: torch.Tensor,
        inv_frames=None,
        mask=None,
    ):
        """
        :param centers torch.Tensor: Has dim [B, M, 3]
        :param dims torch.Tensor: Has dim [B, M, 3]
        :param quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                   w, x, y, z
        """
        assert centers.ndim == 3
        assert dims.ndim == 3
        assert quaternions.ndim == 3

        self.dims = dims
        self.centers = centers

        # It's helpful to ensure the quaternions are normalized
        self.quats = quaternions / torch.linalg.norm(quaternions, dim=2)[:, :, None]

        if inv_frames is None:
            inv_frames = self._init_frames()
        self.inv_frames = inv_frames
        # Mask for nonzero volumes
        if mask is None:
            mask = ~torch.any(
                torch.isclose(self.dims, torch.zeros(1).type_as(centers)), dim=-1
            )
        self.mask = mask

    def __getitem__(self, *args):
        return TorchCuboids(
            self.centers.__getitem__(*args),
            self.dims.__getitem__(*args),
            self.quats.__getitem__(*args),
            self.inv_frames.__getitem__(*args),
            self.mask.__getitem__(*args),
        )

    def geometrout(self):
        """
        Helper method to convert this into geometrout primitives
        """
        B, M, _ = self.centers.shape
        return [
            [
                Cuboid(
                    center=self.centers[bidx, midx, :].detach().cpu().numpy(),
                    dims=self.dims[bidx, midx, :].detach().cpu().numpy(),
                    quaternion=self.quats[bidx, midx, :].detach().cpu().numpy(),
                )
                for midx in range(M)
                if self.mask[bidx, midx]
            ]
            for bidx in range(B)
        ]

    def poses(self):
        w = self.quats[:, :, 0]
        x = self.quats[:, :, 1]
        y = self.quats[:, :, 2]
        z = self.quats[:, :, 3]

        # Naming is a little disingenuous here because I'm multiplying everything by two,
        # but can't put numbers at the beginning of variable names.
        xx = 2 * torch.pow(x, 2)
        yy = 2 * torch.pow(y, 2)
        zz = 2 * torch.pow(z, 2)

        wx = 2 * w * x
        wy = 2 * w * y
        wz = 2 * w * z
        xy = 2 * x * y
        xz = 2 * x * z
        yz = 2 * y * z

        B, M, _ = self.centers.shape
        B = self.centers.size(0)
        M = self.centers.size(1)
        poses = (
            torch.eye(4, device=self.centers.device)
            .type_as(self.centers)[None, None, :, :]
            .expand(B, M, -1, -1)
        )
        poses[:, :, :3, :3] = torch.stack(
            [
                torch.stack([1 - yy - zz, xy - wz, xz + wy], dim=2),
                torch.stack([xy + wz, 1 - xx - zz, yz - wx], dim=2),
                torch.stack([xz - wy, yz - wx, 1 - xx - yy], dim=2),
            ],
            dim=2,
        )
        poses[:, :, :3, 3] = torch.clone(self.centers)
        return poses

    def _init_frames(self):
        """
        In order to calculate the SDF, we need to calculate the inverse
        transformation of the cuboid. This is because we are transforming points
        in the world frame into the cuboid frame.
        """

        # Initialize the inverse rotation
        w = self.quats[:, :, 0]
        x = -self.quats[:, :, 1]
        y = -self.quats[:, :, 2]
        z = -self.quats[:, :, 3]

        # Naming is a little disingenuous here because I'm multiplying everything by two,
        # but can't put numbers at the beginning of variable names.
        xx = 2 * torch.pow(x, 2)
        yy = 2 * torch.pow(y, 2)
        zz = 2 * torch.pow(z, 2)

        wx = 2 * w * x
        wy = 2 * w * y
        wz = 2 * w * z
        xy = 2 * x * y
        xz = 2 * x * z
        yz = 2 * y * z

        B, M, _ = self.centers.shape
        B = self.centers.size(0)
        M = self.centers.size(1)
        inv_frames = torch.zeros((B, M, 4, 4)).type_as(self.centers)
        inv_frames[:, :, 3, 3] = 1

        R = torch.stack(
            [
                torch.stack([1 - yy - zz, xy - wz, xz + wy], dim=2),
                torch.stack([xy + wz, 1 - xx - zz, yz - wx], dim=2),
                torch.stack([xz - wy, yz - wx, 1 - xx - yy], dim=2),
            ],
            dim=2,
        )
        Rt = torch.matmul(R, -1 * self.centers.unsqueeze(3)).squeeze(3)

        # Fill in the rotation matrices
        inv_frames[:, :, :3, :3] = R

        # Invert the transform by multiplying the inverse translation by the inverse rotation
        inv_frames[:, :, :3, 3] = Rt
        return inv_frames

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the cuboids

        :rtype torch.Tensor: A tensor of the surface areas of the cuboids
        """
        area = 2 * (
            self.dims[:, :, 0] * self.dims[:, :, 1]
            + self.dims[:, :, 0] * self.dims[:, :, 2]
            + self.dims[:, :, 1] * self.dims[:, :, 2]
        )
        return area

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the SDF, has
                                    dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                             value for each of the M cuboids), has dim [B, N]
        """
        assert points.ndim == 3
        # We are going to need to map points in the global frame to the cuboid frame
        # First take the points and make them homogeneous by adding a one to the end

        # points_from_volumes = points[self.nonzero_volumes, :, :]
        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.size(0), points.size(1)).type_as(
                points
            )

        homog_points = torch.cat(
            (
                points,
                torch.ones((points.size(0), points.size(1), 1)).type_as(points),
            ),
            dim=2,
        )
        # Next, project these points into their respective cuboid frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, :, :], homog_points[:, None, :, :, None]
        ).squeeze(-1)[:, :, :, :3]
        B, M, N, _ = points_proj.shape
        masked_points = points_proj[self.mask]

        # The follow computations are adapted from here
        # https://github.com/fogleman/sdf/blob/main/sdf/d3.py
        # Move points to top corner

        distances = torch.abs(masked_points) - (self.dims[self.mask] / 2)[:, None, :]
        # This is distance only for points outside the box, all points inside return zero
        # This probably needs to be fixed or else there will be a nan gradient

        outside = torch.linalg.norm(
            torch.maximum(distances, torch.zeros_like(distances)), dim=-1
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance = torch.max(distances, dim=-1).values
        inside = torch.minimum(inner_max_distance, torch.zeros_like(inner_max_distance))
        all_sdfs = float("inf") * torch.ones(B, M, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]

    def sdf_sequence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates SDF values for a time sequence of point clouds
        :param points torch.Tensor: The batched sequence of point clouds with
                                    dimension [B, T, N, 3] (T in sequence length,
                                    N is number of points)
        :rtype torch.Tensor: The scene SDF for each point at each timestep
                             (i.e. the minimum SDF value across the M cuboids
                             at each timestep), has dim [B, T, N]
        """
        assert points.ndim == 4

        # We are going to need to map points in the global frame to the cuboid frame
        # First take the points and make them homogeneous by adding a one to the end

        # points_from_volumes = points[self.nonzero_volumes, :, :]
        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.shape[:-1]).type_as(points)

        homog_points = torch.cat(
            (
                points,
                torch.ones((*points.shape[:-1], 1)).type_as(points),
            ),
            dim=3,
        )
        # Next, project these points into their respective cuboid frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, None, :, :],
            homog_points[:, None, :, :, :, None],
        ).squeeze(-1)[:, :, :, :, :3]
        B, M, T, N, _ = points_proj.shape
        assert T == points.size(1)
        assert N == points.size(2)
        masked_points = points_proj[self.mask]

        # The follow computations are adapted from here
        # https://github.com/fogleman/sdf/blob/main/sdf/d3.py
        # Move points to top corner

        distances = (
            torch.abs(masked_points) - (self.dims[self.mask] / 2)[:, None, None, :]
        )
        # This is distance only for points outside the box, all points inside return zero
        # This probably needs to be fixed or else there will be a nan gradient

        outside = torch.linalg.norm(
            torch.maximum(distances, torch.zeros_like(distances)), dim=-1
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance = torch.max(distances, dim=-1).values
        inside = torch.minimum(inner_max_distance, torch.zeros_like(inner_max_distance))
        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]


class TorchCylinders:
    """
    A Pytorch representation of a batch of M cylinders (i.e. B elements in the batch,
    M cylinders per element). Any of these cylinders can have zero volume (these
    will be masked out during calculation of the various functions in this
    class, such as sdf).
    """

    def __init__(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        heights: torch.Tensor,
        quaternions: torch.Tensor,
        inv_frames=None,
        mask=None,
    ):
        """
        :param centers torch.Tensor: Has dim [B, M, 3]
        :param radii torch.Tensor: Has dim [B, M, 1]
        :param heights torch.Tensor: Has dim [B, M, 1]
        :param quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as
                                   (w, x, y, z)
        """
        assert centers.ndim == 3
        assert radii.ndim == 3
        assert heights.ndim == 3
        assert quaternions.ndim == 3

        self.radii = radii
        self.heights = heights
        self.centers = centers

        # It's helpful to ensure the quaternions are normalized
        self.quats = quaternions / torch.linalg.norm(quaternions, dim=2)[:, :, None]
        if inv_frames is None:
            inv_frames = self._init_frames()
        self.inv_frames = inv_frames
        # Mask for nonzero volumes
        if mask is None:
            mask = ~torch.logical_or(
                torch.isclose(self.radii, torch.zeros(1).type_as(centers)).squeeze(-1),
                torch.isclose(self.heights, torch.zeros(1).type_as(centers)).squeeze(
                    -1
                ),
            )
        self.mask = mask

    def __getitem__(self, *args):
        return TorchCylinders(
            self.centers.__getitem__(*args),
            self.radii.__getitem__(*args),
            self.heights.__getitem__(*args),
            self.quats.__getitem__(*args),
            self.inv_frames.__getitem__(*args),
            self.mask.__getitem__(*args),
        )

    def geometrout(self):
        """
        Helper method to convert this into geometrout primitives
        """
        B, M, _ = self.centers.shape
        return [
            [
                Cylinder(
                    center=self.centers[bidx, midx, :].detach().cpu().numpy(),
                    radius=self.radii[bidx, midx, 0].detach().cpu().numpy(),
                    height=self.heights[bidx, midx, 0].detach().cpu().numpy(),
                    quaternion=self.quats[bidx, midx, :].detach().cpu().numpy(),
                )
                for midx in range(M)
                if self.mask[bidx, midx]
            ]
            for bidx in range(B)
        ]

    def _init_frames(self):
        """
        In order to calculate the SDF, we need to calculate the inverse
        transformation of the cylinder. This is because we are transforming
        points in the world frame into the cylinder frame.
        """
        # Initialize the inverse rotation
        w = self.quats[:, :, 0]
        x = -self.quats[:, :, 1]
        y = -self.quats[:, :, 2]
        z = -self.quats[:, :, 3]

        # Naming is a little disingenuous here because I'm multiplying everything by two,
        # but can't put numbers at the beginning of variable names.
        xx = 2 * torch.pow(x, 2)
        yy = 2 * torch.pow(y, 2)
        zz = 2 * torch.pow(z, 2)

        wx = 2 * w * x
        wy = 2 * w * y
        wz = 2 * w * z
        xy = 2 * x * y
        xz = 2 * x * z
        yz = 2 * y * z

        B, M, _ = self.centers.shape
        B = self.centers.size(0)
        M = self.centers.size(1)
        inv_frames = torch.zeros((B, M, 4, 4)).type_as(self.centers)
        inv_frames[:, :, 3, 3] = 1

        R = torch.stack(
            [
                torch.stack([1 - yy - zz, xy - wz, xz + wy], dim=2),
                torch.stack([xy + wz, 1 - xx - zz, yz - wx], dim=2),
                torch.stack([xz - wy, yz - wx, 1 - xx - yy], dim=2),
            ],
            dim=2,
        )
        Rt = torch.matmul(R, -1 * self.centers.unsqueeze(3)).squeeze(3)

        # Fill in the rotation matrices
        inv_frames[:, :, :3, :3] = R

        # Invert the transform by multiplying the inverse translation by the inverse rotation
        inv_frames[:, :, :3, 3] = Rt
        return inv_frames

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the cuboids

        :rtype torch.Tensor: A tensor of the surface areas of the cuboids
        """
        area = self.heights * np.pi * torch.pow(self.radii, 2)
        return area

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the SDF, has
                                    dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                             value for each of the M cylinders), has dim [B, N]
        """
        assert points.ndim == 3
        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.size(0), points.size(1)).type_as(
                points
            )

        homog_points = torch.cat(
            (
                points,
                torch.ones((points.size(0), points.size(1), 1)).type_as(points),
            ),
            dim=2,
        )
        # Next, project these points into their respective cylinder frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, :, :], homog_points[:, None, :, :, None]
        ).squeeze(-1)[:, :, :, :3]
        B, M, N, _ = points_proj.shape
        masked_points = points_proj[self.mask]

        surface_distance_xy = torch.linalg.norm(masked_points[:, :, :2], dim=2)
        z_distance = masked_points[:, :, 2]

        half_extents_2d = torch.stack(
            (self.radii[self.mask], self.heights[self.mask] / 2), dim=2
        )
        points_2d = torch.stack((surface_distance_xy, z_distance), dim=2)
        distances_2d = torch.abs(points_2d) - half_extents_2d

        # This is distance only for points outside the box, all points inside return zero
        outside = torch.linalg.norm(
            torch.maximum(distances_2d, torch.zeros_like(distances_2d)), dim=2
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance_2d = torch.max(distances_2d, dim=2).values
        inside = torch.minimum(
            inner_max_distance_2d, torch.zeros_like(inner_max_distance_2d)
        )

        all_sdfs = float("inf") * torch.ones(B, M, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]

    def sdf_sequence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates SDF values for a time sequence of point clouds
        :param points torch.Tensor: The batched sequence of point clouds with
                                    dimension [B, T, N, 3] (T in sequence length,
                                    N is number of points)
        :rtype torch.Tensor: The scene SDF for each point at each timestep
                             (i.e. the minimum SDF value across the M cylinders at
                             each timestep), has dim [B, T, N]
        """
        assert points.ndim == 4

        # We are going to need to map points in the global frame to the cylinder frame
        # First take the points and make them homogeneous by adding a one to the end

        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.shape[:-1]).type_as(points)

        homog_points = torch.cat(
            (
                points,
                torch.ones((*points.shape[:-1], 1)).type_as(points),
            ),
            dim=3,
        )
        # Next, project these points into their respective cylinder frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, None, :, :],
            homog_points[:, None, :, :, :, None],
        ).squeeze(-1)[:, :, :, :, :3]
        B, M, T, N, _ = points_proj.shape
        assert T == points.size(1)
        assert N == points.size(2)
        masked_points = points_proj[self.mask]

        surface_distance_xy = torch.linalg.norm(masked_points[:, :, :, :2], dim=-1)
        z_distance = masked_points[:, :, :, 2]

        half_extents_2d = torch.stack(
            (self.radii[self.mask], self.heights[self.mask] / 2), dim=2
        )[:, :, None, :]
        points_2d = torch.stack((surface_distance_xy, z_distance), dim=3)
        distances_2d = torch.abs(points_2d) - half_extents_2d

        # This is distance only for points outside the box, all points inside return zero
        outside = torch.linalg.norm(
            torch.maximum(distances_2d, torch.zeros_like(distances_2d)), dim=3
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance_2d = torch.max(distances_2d, dim=3).values
        inside = torch.minimum(
            inner_max_distance_2d, torch.zeros_like(inner_max_distance_2d)
        )

        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]


def construct_mixed_point_cloud(
    objects: Sequence[Union[Sphere, Cuboid, Cylinder]], num_points: int
) -> np.ndarray:
    """
    Creates a random point cloud from a collection of objects. The points in
    the point cloud should be fairly(-ish) distributed amongst the objects based
    on their surface area.

    :param objects Sequence[Union[Sphere, Cuboid, Cylinder]]: The objects in the scene
    :param num_points int: The total number of points in the samples scene (not
                           the number of points per objects)
    :rtype np.ndarray: Has dim [N, 3] where N is num_points
    """
    assert len(objects) > 0, "There must be objects to create a point cloud"
    # Allocate points based on objects surface area for even sampling
    surface_areas = np.array([o.surface_area for o in objects])
    total_area = np.sum(surface_areas)
    assert total_area > 0, "Some objects must have surface area to create point cloud"
    proportions = surface_areas / total_area
    points_per_object = np.round(proportions * num_points).astype(int)
    num_points_rounded = np.sum(points_per_object)
    if num_points_rounded < num_points:
        points_per_object[np.argmin(points_per_object)] += (
            num_points - num_points_rounded
        )
    else:
        points_per_object[np.argmax(points_per_object)] += (
            num_points - num_points_rounded
        )
    points = np.ones((num_points, 4))
    indices = list(range(len(objects)))
    random.shuffle(indices)
    num_added = 0
    for i in indices:
        o, npoints = objects[i], points_per_object[i]
        points[num_added : num_added + npoints, :3] = o.sample_surface(npoints)
        points[num_added : num_added + npoints, 3] *= i + 1
        num_added += npoints
    return points
