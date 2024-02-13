import numpy as np
from geometrout import SE3
from ikfast_franka_panda import get_fk, get_ik

from robofin.robot_constants import (
    FrankaConstants,
    FrankaGripperConstants,
    RealFrankaConstants,
)


class FrankaRobot:
    constants = FrankaConstants

    # TODO remove this after making this more general
    # These are strings because that's needed for Bullet
    @classmethod
    def within_limits(cls, config):
        # We have to add a small buffer because of float math
        return np.all(config >= cls.constants.JOINT_LIMITS[:, 0] - 1e-5) and np.all(
            config <= cls.constants.JOINT_LIMITS[:, 1] + 1e-5
        )

    @classmethod
    def random_neutral(cls, method="normal"):
        if method == "normal":
            return np.clip(
                cls.constants.NEUTRAL + np.random.normal(0, 0.25, 7),
                cls.constants.JOINT_LIMITS[:, 0],
                cls.constants.JOINT_LIMITS[:, 1],
            )
        if method == "uniform":
            # No need to clip this because it's always within range
            return cls.constants.NEUTRAL + np.random.uniform(0, 0.25, 7)
        assert False, "method must be either normal or uniform"

    @classmethod
    def fk(cls, config, eff_frame="right_gripper"):
        """
        Returns the SE3 frame of the end effector
        """
        assert (
            eff_frame in cls.constants.EEF_LINKS.__members__
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
                * cls.constants.EEF_T_LIST[("panda_link8", "panda_hand")]
            )
        elif eff_frame == "right_gripper":
            return (
                SE3.from_matrix(mat)
                * cls.constants.EEF_T_LIST[("panda_link8", "right_gripper")]
            )
        else:
            return (
                SE3.from_matrix(mat)
                * cls.constants.EEF_T_LIST[("panda_link8", "panda_grasptarget")]
            )

    @classmethod
    def ik(cls, pose, panda_link7, eff_frame="right_gripper"):
        """
        :param pose: SE3 pose expressed in specified end effector frame
        :param panda_link7: Value for the joint panda_link7, other IK can be calculated with this joint value set.
            Must be within joint range
        :param eff_frame: Desired end effector frame, must be among [panda_link8, right_gripper, panda_grasptarget]
        :return: Typically 4 solutions to IK
        """
        assert (
            eff_frame in cls.constants.EEF_LINKS.__members__
        ), "IK only calculated for a valid end effector frame"
        if eff_frame == "right_gripper":
            pose = (
                pose
                * cls.constants.EEF_T_LIST[("panda_link8", "right_gripper")].inverse
            )
        elif eff_frame == "panda_grasptarget":
            pose = (
                pose
                * cls.constants.EEF_T_LIST[("panda_link8", "panda_grasptarget")].inverse
            )
        elif eff_frame == "panda_hand":
            pose = (
                pose * cls.constants.EEF_T_LIST[("panda_link8", "panda_hand")].inverse
            )
        else:
            raise NotImplementedError(f"IK not implemented for eff frame {eff_frame}")
        rot = pose.so3.matrix.tolist()
        pos = pose.xyz
        assert (
            panda_link7 >= cls.constants.JOINT_LIMITS[-1, 0]
            and panda_link7 <= cls.constants.JOINT_LIMITS[-1, 1]
        ), f"Value for floating joint must be within range {cls.constants.JOINT_LIMITS[-1, :].tolist()}"
        solutions = [np.asarray(s) for s in get_ik(pos, rot, [panda_link7])]
        return [
            s
            for s in solutions
            if (
                np.all(s >= cls.constants.JOINT_LIMITS[:, 0])
                and np.all(s <= cls.constants.JOINT_LIMITS[:, 1])
            )
        ]

    @classmethod
    def random_configuration(cls):
        limits = cls.constants.JOINT_LIMITS
        return (limits[:, 1] - limits[:, 0]) * (np.random.rand(7)) + limits[:, 0]

    @classmethod
    def random_ik(cls, pose, eff_frame="right_gripper"):
        config = cls.random_configuration()
        try:
            return cls.ik(pose, config[-1], eff_frame)
        except Exception as e:
            raise Exception(f"IK failed with {pose}")

    @classmethod
    def collision_free_ik(
        cls,
        pose,
        prismatic_joint,
        cooo,
        primitive_arrays,
        buffer=0.0,
        eff_frame="right_gripper",
        retries=1000,
        bad_state_callback=lambda x: False,
        choose_close_to=None,
    ):
        options = []
        for i in range(retries + 1):
            samples = cls.random_ik(pose, eff_frame)
            for sample in samples:
                if not (
                    cooo.franka_arm_collides_fast(
                        sample, prismatic_joint, primitive_arrays, buffer
                    )
                    or bad_state_callback(sample)
                ):
                    if choose_close_to is None:
                        return sample
                    options.append(sample)
        if len(options):
            return options[
                np.argmin([np.linalg.norm(o - choose_close_to) for o in options])
            ]
        return None


class FrankaRealRobot(FrankaRobot):
    constants = RealFrankaConstants


class FrankaGripper:
    constants = FrankaGripperConstants

    @classmethod
    def random_configuration(cls):
        raise NotImplementedError(
            "Random configuration not implemented for Franka Hand"
        )
