import numpy as np
import pybullet as p
from geometrout.primitive import Cuboid, Sphere
from geometrout.transform import SE3

from robofin.robots import FrankaRobot, FrankaGripper


class BulletRobot:
    def __init__(self, clid):
        self.clid = clid
        self.id = self.load(clid)
        self._setup_robot()

    def load(self, clid, urdf_path=None):
        return p.loadURDF(
            self.robot_type.urdf,
            useFixedBase=True,
            physicsClientId=clid,
            flags=p.URDF_USE_SELF_COLLISION,
        )

    @property
    def links(self):
        """
        :return: The names and bullet ids of all links for the loaded robot
        """
        return [(k, v) for k, v in self._link_name_to_index.items()]

    def link_id(self, name):
        """
        :return: The bullet id corresponding to a specific link name
        """
        return self._link_name_to_index[name]

    def link_name(self, id):
        """
        :return: The name corresponding to a particular bullet id
        """
        return self._index_to_link_name[id]

    @property
    def link_frames(self):
        """
        :return: A dictionary where the link names are the keys
            and the values are the correponding poses as reflected
            by the current state of the environment
        """
        ret = p.getLinkStates(
            self.id,
            list(range(len(self.links) - 1)),
            computeForwardKinematics=True,
            physicsClientId=self.clid,
        )
        frames = {}
        for ii, r in enumerate(ret):
            frames[self.link_name(ii)] = SE3(
                xyz=np.array(r[4]),
                quat=Quaternion([r[5][3], r[5][0], r[5][1], r[5][2]]),
            )
        return frames

    def in_collision(self, obstacles, check_self=False):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        if check_self:
            contacts = p.getContactPoints(self.id, self.id, physicsClientId=self.clid)
            if len(contacts) > 0:
                return True

        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            if len(contacts) > 0:
                return True
        return False

    def _setup_robot(self):
        """
        Internal function for setting up the correspondence
        between link names and ids.
        """
        # Code snippet borrowed from https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12728
        self._link_name_to_index = {
            p.getBodyInfo(self.id, physicsClientId=self.clid)[0].decode("UTF-8"): -1
        }
        for _id in range(p.getNumJoints(self.id, physicsClientId=self.clid)):
            _name = p.getJointInfo(self.id, _id, physicsClientId=self.clid)[12].decode(
                "UTF-8"
            )
            self._link_name_to_index[_name] = _id
        self._index_to_link_name = {}

        for k, v in self._link_name_to_index.items():
            self._index_to_link_name[v] = k


class BulletFranka(BulletRobot):
    robot_type = FrankaRobot

    def marionette(self, state):
        for i in range(0, 7):
            p.resetJointState(self.id, i, state[i], physicsClientId=self.clid)

        if len(state) == 9:
            # Spread the fingers if they aren't included--prevents self collision
            p.resetJointState(self.id, 9, state[7], physicsClientId=self.clid)
            p.resetJointState(self.id, 10, state[8], physicsClientId=self.clid)
        elif len(state) == 7:
            p.resetJointState(self.id, 9, 0.04, physicsClientId=self.clid)
            p.resetJointState(self.id, 10, 0.04, physicsClientId=self.clid)
        else:
            raise Exception("Length of input state should be either 7 or 9")


class BulletFrankaGripper(BulletRobot):
    robot_type = FrankaGripper

    def marionette(self, state, frame="right_gripper"):
        assert isinstance(state, SE3)
        assert frame in ["base_frame", "right_gripper", "panda_grasptarget"]
        # Pose is expressed as a transformation from the desired frame to the world
        # But we need to transform it into the base frame

        # TODO maybe there is some way to cache these transforms from the urdf
        # instead of hardcoding them
        if frame == "right_gripper":
            transform = SE3(
                matrix=np.array(
                    [
                        [-0.7071067811865475, 0.7071067811865475, 0, 0],
                        [-0.7071067811865475, -0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.1],
                        [0, 0, 0, 1],
                    ]
                )
            )
            pose = pose @ transform
        elif frame == "panda_grasptarget":
            transform = SE3(
                matrix=np.array(
                    [
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.105],
                        [0, 0, 0, 1],
                    ]
                )
            )
            pose = pose @ transform

        x, y, z = pose.xyz
        p.resetJointState(self.id, 0, x, physicsClientId=self.clid)
        p.resetJointState(self.id, 1, y, physicsClientId=self.clid)
        p.resetJointState(self.id, 2, z, physicsClientId=self.clid)
        p.resetJointStateMultiDof(self.id, 3, pose.so3.xyzw, physicsClientId=self.clid)
        p.resetJointState(self.id, 4, 0.02, physicsClientId=self.clid)
        p.resetJointState(self.id, 5, 0.02, physicsClientId=self.clid)


class Bullet:
    def __init__(self, gui=False):
        """
        :param gui: Whether to use a gui to visualize the environment.
            Only one gui instance allowed
        """
        self.use_gui = gui
        if self.use_gui:
            self.clid = p.connect(p.GUI)
        else:
            self.clid = p.connect(p.DIRECT)
        self.robots = {}
        self.obstacle_ids = []

    def __del__(self):
        """
        Disconnects the client on destruction
        """
        p.disconnect(self.clid)

    def setCameraPosition(self, yaw, pitch, distance, target):
        p.resetDebugVisualizerCamera(
            distance, yaw, pitch, target, physicsClientId=self.clid
        )

    def getCameraPosition(self):
        params = p.getDebugVisualizerCamera(physicsClientId=self.clid)
        return {
            "yaw": params[8],
            "pitch": params[9],
            "distance": params[10],
            "target": params[11],
        }

    def load_robot(self, robot_type):
        """
        Generic function to load a robot.
        """
        if robot_type == FrankaRobot:
            robot = BulletFranka(self.clid)
        elif robot_type == FrankaGripper:
            robot = BulletFrankaGripper(self.clid)
        self.robots[robot.id] = robot
        return robot

    def in_collision(self, robot, check_self=False):
        return robot.in_collision(self.obstacle_ids, check_self)

    def load_cuboid(self, cuboid, color=None):
        assert isinstance(cuboid, Cuboid)
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        assert not cuboid.is_zero_volume(), "Cannot load zero volume cuboid"
        kwargs = {}
        if self.use_gui:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=cuboid.half_extents,
                rgbaColor=color,
                physicsClientId=self.clid,
            )
            kwargs["baseVisualShapeIndex"] = obstacle_visual_id
        obstacle_collision_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=cuboid.half_extents,
            physicsClientId=self.clid,
        )
        obstacle_id = p.createMultiBody(
            basePosition=cuboid.center,
            baseOrientation=cuboid.pose.so3.xyzw,
            baseCollisionShapeIndex=obstacle_collision_id,
            physicsClientId=self.clid,
            **kwargs,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_sphere(self, sphere, color=None):
        assert isinstance(sphere, Sphere)
        if color is None:
            color = [0.0, 0.0, 0.0, 1.0]
        if self.use_gui:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                rgbaColor=color,
                physicsClientId=self.clid,
            )
            kwargs["baseVisualShapeIndex"] = obstacle_visual_id
        obstacle_collision_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=sphere.radius,
            physicsClientId=self.clid,
        )
        obstacle_id = p.createMultiBody(
            basePosition=sphere.center,
            baseCollisionShapeIndex=obstacle_collision_id,
            physicsClientId=self.clid,
            **kwargs,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_primitives(self, primitives, color=None):
        for prim in primitives:
            if prim.is_zero_volume():
                continue
            elif isinstance(prim, Cuboid):
                self.load_cuboid(prim, color)
            elif isinstance(prim, Sphere):
                self.load_sphere(prim, color)
            else:
                raise Exception("Only cuboids and spheres supported as primitives")

    def clear_obstacle(self, id):
        """
        Removes a specific obstacle from the environment

        :param id: Bullet id of obstacle to remove
        """
        if id is not None:
            p.removeBody(id, physicsClientId=self.clid)
            self.obstacle_ids = [x for x in self.obstacle_ids if x != id]

    def clear_all_obstacles(self):
        """
        Removes all obstacles from bullet environment
        """
        for id in self.obstacle_ids:
            if id is not None:
                p.removeBody(id, physicsClientId=self.clid)
        self.obstacle_ids = []
