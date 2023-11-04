from pathlib import Path

import numpy as np
from geometrout import SE3
from ikfast_franka_panda import get_fk, get_ik

from robofin.collision import franka_arm_collides_fast
from robofin.robot_constants import FrankaConstants


class FrankaRobot:
    # TODO remove this after making this more general
    # These are strings because that's needed for Bullet
    @staticmethod
    def within_limits(config):
        # We have to add a small buffer because of float math
        return np.all(config >= FrankaConstants.JOINT_LIMITS[:, 0] - 1e-5) and np.all(
            config <= FrankaConstants.JOINT_LIMITS[:, 1] + 1e-5
        )

    @staticmethod
    def random_neutral(method="normal"):
        if method == "normal":
            return np.clip(
                FrankaConstants.NEUTRAL + np.random.normal(0, 0.25, 7),
                FrankaConstants.JOINT_LIMITS[:, 0],
                FrankaConstants.JOINT_LIMITS[:, 1],
            )
        if method == "uniform":
            # No need to clip this because it's always within range
            return FrankaConstants.NEUTRAL + np.random.uniform(0, 0.25, 7)
        assert False, "method must be either normal or uniform"

    @staticmethod
    def fk(config, eff_frame="right_gripper"):
        """
        Returns the SE3 frame of the end effector
        """
        assert (
            eff_frame in FrankaConstants.EFF_LIST
        ), "Default FK only calculated for a valid end effector frame"
        pos, rot = get_fk(config)
        mat = np.eye(4)
        mat[:3, :3] = np.asarray(rot)
        mat[:3, 3] = np.asarray(pos)
        if eff_frame == "panda_link8":
            return SE3.from_matrix(mat)
        elif eff_frame == "panda_hand":
            return (
                SE3.from_matrix(mat)
                * FrankaConstants.EFF_T_LIST[("panda_link8", "panda_hand")]
            )
        elif eff_frame == "right_gripper":
            return (
                SE3.from_matrix(mat)
                * FrankaConstants.EFF_T_LIST[("panda_link8", "right_gripper")]
            )
        else:
            return (
                SE3.from_matrix(mat)
                * FrankaConstants.EFF_T_LIST[("panda_link8", "panda_grasptarget")]
            )

    @staticmethod
    def ik(pose, panda_link7, eff_frame="right_gripper"):
        """
        :param pose: SE3 pose expressed in specified end effector frame
        :param panda_link7: Value for the joint panda_link7, other IK can be calculated with this joint value set.
            Must be within joint range
        :param eff_frame: Desired end effector frame, must be among [panda_link8, right_gripper, panda_grasptarget]
        :return: Typically 4 solutions to IK
        """
        assert (
            eff_frame in FrankaConstants.EFF_LIST
        ), "IK only calculated for a valid end effector frame"
        if eff_frame == "right_gripper":
            pose = (
                pose
                * FrankaConstants.EFF_T_LIST[("panda_link8", "right_gripper")].inverse
            )
        elif eff_frame == "panda_grasptarget":
            pose = (
                pose
                * FrankaConstants.EFF_T_LIST[
                    ("panda_link8", "panda_grasptarget")
                ].inverse
            )
        rot = pose.so3.matrix.tolist()
        pos = pose.xyz
        assert (
            panda_link7 >= FrankaConstants.JOINT_LIMITS[-1, 0]
            and panda_link7 <= FrankaConstants.JOINT_LIMITS[-1, 1]
        ), f"Value for floating joint must be within range {FrankaConstants.JOINT_LIMITS[-1, :].tolist()}"
        solutions = [np.asarray(s) for s in get_ik(pos, rot, [panda_link7])]
        return [
            s
            for s in solutions
            if (
                np.all(s >= FrankaConstants.JOINT_LIMITS[:, 0])
                and np.all(s <= FrankaConstants.JOINT_LIMITS[:, 1])
            )
        ]

    @staticmethod
    def random_configuration():
        limits = FrankaConstants.JOINT_LIMITS
        return (limits[:, 1] - limits[:, 0]) * (np.random.rand(7)) + limits[:, 0]

    @staticmethod
    def random_ik(pose, eff_frame="right_gripper"):
        config = FrankaConstants.random_configuration()
        try:
            return FrankaConstants.ik(pose, config[-1], eff_frame)
        except:
            raise Exception(f"IK failed with {pose}")

    @staticmethod
    def collision_free_ik(
        pose,
        prismatic_joint,
        cooo,
        primitive_arrays,
        buffer=0.0,
        frame="right_gripper",
        retries=1000,
        bad_state_callback=lambda x: False,
    ):
        for i in range(retries + 1):
            samples = FrankaConstants.random_ik(pose, frame)
            for sample in samples:
                if not (
                    franka_arm_collides_fast(
                        sample, prismatic_joint, cooo, primitive_arrays, buffer
                    )
                    or bad_state_callback(sample)
                ):
                    return sample
        return None


class FrankaRealRobot(FrankaRobot):
    @staticmethod
    def within_limits(config):
        # We have to add a small buffer because of float math
        return np.all(
            config >= RealFrankaConstants.JOINT_LIMITS[:, 0] - 1e-5
        ) and np.all(config <= RealFrankaConstants.JOINT_LIMITS[:, 1] + 1e-5)

    @staticmethod
    def random_neutral(method="normal"):
        if method == "normal":
            return np.clip(
                RealFrankaConstants.NEUTRAL + np.random.normal(0, 0.25, 7),
                RealFrankaConstants.JOINT_LIMITS[:, 0],
                RealFrankaConstants.JOINT_LIMITS[:, 1],
            )
        if method == "uniform":
            # No need to clip this because it's always within range
            return RealFrankaConstants.NEUTRAL + np.random.uniform(0, 0.25, 7)
        assert False, "method must be either normal or uniform"

    @staticmethod
    def fk(config, eff_frame="right_gripper"):
        """
        Returns the SE3 frame of the end effector
        """
        assert (
            eff_frame in RealFrankaConstants.EFF_LIST
        ), "Default FK only calculated for a valid end effector frame"
        pos, rot = get_fk(config)
        mat = np.eye(4)
        mat[:3, :3] = np.asarray(rot)
        mat[:3, 3] = np.asarray(pos)
        if eff_frame == "panda_link8":
            return SE3.from_matrix(mat)
        elif eff_frame == "right_gripper":
            return (
                SE3.from_matrix(mat)
                * RealFrankaConstants.EFF_T_LIST[("panda_link8", "right_gripper")]
            )
        else:
            return (
                SE3.from_matrix(mat)
                * RealFrankaConstants.EFF_T_LIST[("panda_link8", "panda_grasptarget")]
            )

    @staticmethod
    def ik(pose, panda_link7, eff_frame="right_gripper"):
        """
        :param pose: SE3 pose expressed in specified end effector frame
        :param panda_link7: Value for the joint panda_link7, other IK can be calculated with this joint value set.
            Must be within joint range
        :param eff_frame: Desired end effector frame, must be among [panda_link8, right_gripper, panda_grasptarget]
        :return: Typically 4 solutions to IK
        """
        assert (
            eff_frame in RealFrankaConstants.EFF_LIST
        ), "IK only calculated for a valid end effector frame"
        if eff_frame == "right_gripper":
            pose = (
                pose
                * RealFrankaConstants.EFF_T_LIST[
                    ("panda_link8", "right_gripper")
                ].inverse
            )
        elif eff_frame == "panda_grasptarget":
            pose = (
                pose
                * RealFrankaConstants.EFF_T_LIST[
                    ("panda_link8", "panda_grasptarget")
                ].inverse
            )
        rot = pose.so3.matrix.tolist()
        pos = pose.xyz
        assert (
            panda_link7 >= RealFrankaConstants.JOINT_LIMITS[-1, 0]
            and panda_link7 <= RealFrankaConstants.JOINT_LIMITS[-1, 1]
        ), f"Value for floating joint must be within range {RealFrankaConstants.JOINT_LIMITS[-1, :].tolist()}"
        solutions = [np.asarray(s) for s in get_ik(pos, rot, [panda_link7])]
        return [
            s
            for s in solutions
            if (
                np.all(s >= RealFrankaConstants.JOINT_LIMITS[:, 0])
                and np.all(s <= RealFrankaConstants.JOINT_LIMITS[:, 1])
            )
        ]

    @staticmethod
    def random_configuration():
        limits = RealFrankaConstants.JOINT_LIMITS
        return (limits[:, 1] - limits[:, 0]) * (np.random.rand(7)) + limits[:, 0]

    @staticmethod
    def random_ik(pose, eff_frame="right_gripper"):
        config = RealFrankaConstants.random_configuration()
        try:
            return RealFrankaConstants.ik(pose, config[-1], eff_frame)
        except:
            raise Exception(f"IK failed with {pose}")

    @staticmethod
    def collision_free_ik(
        pose,
        prismatic_joint,
        cooo,
        primitive_arrays,
        buffer=0.0,
        frame="right_gripper",
        retries=1000,
        bad_state_callback=lambda x: False,
    ):
        for i in range(retries + 1):
            samples = RealFrankaConstants.random_ik(pose, frame)
            for sample in samples:
                if not (
                    franka_arm_collides_fast(
                        sample, prismatic_joint, cooo, primitive_arrays, buffer
                    )
                    or bad_state_callback(sample)
                ):
                    return sample
        return None


class FrankaGripper:
    @staticmethod
    def random_configuration():
        raise NotImplementedError(
            "Random configuration not implemented for Franka Hand"
        )
