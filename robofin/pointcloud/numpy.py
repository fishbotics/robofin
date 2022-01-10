import numpy as np


def project(transformation_matrix, point, rotate_only=False):
    if rotate_only:
        return (transformation_matrix @ np.append(point, [0]))[:3]
    return (transformation_matrix @ np.append(point, [1]))[:3]


def transform_pointcloud(pc, transformation_matrix, in_place=True):
    """

    Parameters
    ----------
    pc: A np.ndarray pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography

    """
    assert type(pc) == type(transformation_matrix)
    assert (
        pc.ndim == 2
    ), "Numpy transform pointcloud function only works on single pointcloud"
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = np.concatenate((xyz, np.ones(ones_dim)), axis=1).T
    transformed_xyz = transformation_matrix @ homogeneous_xyz
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].T
        return pc
    return np.concatenate((transformed_xyz[..., :3, :].T, pc[..., 3:]), axis=1)
