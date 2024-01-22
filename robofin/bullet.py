import math
from pathlib import Path

import numpy as np
import pybullet as p
import trimesh
from geometrout import SE3, Cuboid, Cylinder, Sphere

from robofin.point_cloud_tools import transform_point_cloud
from robofin.robots import FrankaGripper, FrankaRobot


class BulletRobot:
    def __init__(self, clid, hd=False, **kwargs):
        self.clid = clid
        self.hd = hd
        # TODO implement collision free robot
        self.id = self.load(clid)
        self._setup_robot()
        self._set_robot_specifics(**kwargs)

    def load(self, clid, urdf_path=None):
        if self.hd:
            urdf = self.robot_type.constants.hd_urdf
        else:
            urdf = self.robot_type.constants.urdf
        return p.loadURDF(
            urdf,
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
                np.array(r[4]),
                np.array([r[5][3], r[5][0], r[5][1], r[5][2]]),
            )
        return frames

    def closest_distance_to_self(self, max_radius):
        contacts = p.getClosestPoints(
            self.id, self.id, max_radius, physicsClientId=self.clid
        )
        # Manually filter out fixed connections that shouldn't be considered
        # TODO fix this somehow
        filtered = []
        for c in contacts:
            # A link is always in collision with itself and its neighbors
            if abs(c[3] - c[4]) <= 1:
                continue
            # panda_link8 just transforms the origin
            if c[3] == 6 and c[4] == 8:
                continue
            if c[3] == 8 and c[4] == 6:
                continue
            if c[3] > 8 or c[4] > 8:
                continue
            filtered.append(c)
        if len(filtered):
            return min([x[8] for x in filtered])
        return None

    def closest_distance(self, obstacles, max_radius, ignore_base=True):
        distances = []
        for id in obstacles:
            closest_points = p.getClosestPoints(
                self.id, id, max_radius, physicsClientId=self.clid
            )
            if ignore_base:
                closest_points = [c for c in closest_points if c[3] > -1]
            distances.extend([x[8] for x in closest_points])
        if len(distances):
            return min(distances)
        return None

    def in_collision(self, obstacles, radius=0.0, check_self=False, ignore_base=True):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
        raise NotImplementedError

    def get_collision_points(self, obstacles, check_self=False):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
        points = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        if check_self:
            contacts = p.getContactPoints(self.id, self.id, physicsClientId=self.clid)
            # Manually filter out fixed connections that shouldn't be considered
            # TODO fix this somehow
            filtered = []
            for c in contacts:
                # panda_link8 just transforms the origin
                if c[3] == 6 and c[4] == 8:
                    continue
                if c[3] == 8 and c[4] == 6:
                    continue
                if c[3] > 8 or c[4] > 8:
                    continue
                filtered.append(c)
            points.extend([p[5] for p in filtered])

        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            points.extend([p[5] for p in contacts])
        return points

    def get_deepest_collision(self, obstacles):
        distances = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            distances.extend([p[8] for p in contacts])

        if len(distances) > 0:
            # Distance will be negative if it's a true penetration
            deepest_collision = min(distances)
            if deepest_collision < 0:
                return abs(deepest_collision)
        return 0

    def get_collision_depths(self, obstacles):
        distances = []
        # Step the simulator (only enough for collision detection)
        p.performCollisionDetection(physicsClientId=self.clid)
        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getContactPoints(self.id, id, physicsClientId=self.clid)
            distances.extend([p[8] for p in contacts])

        return [abs(d) for d in distances if d < 0]

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

    def _set_robot_specifics(self, **kwargs):
        raise NotImplemented("Must be set in the robot specific class")


class BulletFranka(BulletRobot):
    robot_type = FrankaRobot

    def _set_robot_specifics(self, default_prismatic_value=0.025):
        self.default_prismatic_value = default_prismatic_value

    def marionette(self, state, velocities=None):
        if velocities is None:
            velocities = [0.0 for _ in state]
        assert len(state) == len(velocities)
        for i in range(0, 7):
            p.resetJointState(
                self.id,
                i,
                state[i],
                targetVelocity=velocities[i],
                physicsClientId=self.clid,
            )

        if len(state) == 9:
            p.resetJointState(
                self.id,
                9,
                state[7],
                targetVelocity=velocities[7],
                physicsClientId=self.clid,
            )
            p.resetJointState(
                self.id,
                10,
                state[8],
                targetVelocity=velocities[8],
                physicsClientId=self.clid,
            )
        elif len(state) == 7:
            # Spread the fingers if they aren't included--prevents self collision
            p.resetJointState(
                self.id,
                9,
                self.default_prismatic_value,
                targetVelocity=0.0,
                physicsClientId=self.clid,
            )
            p.resetJointState(
                self.id,
                10,
                self.default_prismatic_value,
                targetVelocity=0.0,
                physicsClientId=self.clid,
            )
        else:
            raise Exception("Length of input state should be either 7 or 9")

    def get_joint_states(self):
        """
        :return: (joint positions, joint velocities)
        """
        states = p.getJointStates(
            self.id, [0, 1, 2, 3, 4, 5, 6, 9, 10], physicsClientId=self.clid
        )
        return [s[0] for s in states], [s[1] for s in states]

    def control_position(self, state):
        assert len(state) in [7, 9]
        p.setJointMotorControlArray(
            self.id,
            jointIndices=list(range(len(state))),
            controlMode=p.POSITION_CONTROL,
            targetPositions=state,
            targetVelocities=[0] * len(state),
            forces=[250] * len(state),
            positionGains=[0.01] * len(state),
            velocityGains=[1.0] * len(state),
            physicsClientId=self.clid,
        )

    def in_collision(self, obstacles, radius=0.0, check_self=False, ignore_base=True):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
        # Step the simulator (only enough for collision detection)
        if check_self:
            contacts = p.getClosestPoints(
                self.id, self.id, radius, physicsClientId=self.clid
            )
            # Manually filter out fixed connections that shouldn't be considered
            # TODO fix this somehow
            filtered = []
            for c in contacts:
                # A link is always in collision with itself and its neighbors
                if abs(c[3] - c[4]) <= 1:
                    continue
                # panda_link8 just transforms the origin
                if c[3] == 6 and c[4] == 8:
                    continue
                if c[3] == 8 and c[4] == 6:
                    continue
                if c[3] > 8 or c[4] > 8:
                    continue
                filtered.append(c)
            if len(filtered) > 0:
                return True

        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getClosestPoints(
                self.id, id, radius, physicsClientId=self.clid
            )
            if ignore_base:
                contacts = [c for c in contacts if c[3] > -1]
            if len(contacts) > 0:
                return True
        return False


class BulletFrankaGripper(BulletRobot):
    robot_type = FrankaGripper

    def _set_robot_specifics(self, default_prismatic_value=0.025):
        self.default_prismatic_value = default_prismatic_value

    def marionette(self, state, frame="right_gripper"):
        assert isinstance(state, SE3)
        assert frame in ["base_frame", "right_gripper", "panda_grasptarget"]
        # Pose is expressed as a transformation from the desired frame to the world
        # But we need to transform it into the base frame

        # TODO maybe there is some way to cache these transforms from the urdf
        # instead of hardcoding them
        if frame == "right_gripper":
            transform = SE3.from_matrix(
                np.array(
                    [
                        [-0.7071067811865475, 0.7071067811865475, 0, 0],
                        [-0.7071067811865475, -0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.1],
                        [0, 0, 0, 1],
                    ]
                )
            )
            state = state * transform
        elif frame == "panda_grasptarget":
            transform = SE3.from_matrix(
                np.array(
                    [
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0.7071067811865475, 0.7071067811865475, 0, 0],
                        [0, 0, 1, -0.105],
                        [0, 0, 0, 1],
                    ]
                )
            )
            state = state * transform

        x, y, z = state.xyz
        p.resetJointState(self.id, 0, x, physicsClientId=self.clid)
        p.resetJointState(self.id, 1, y, physicsClientId=self.clid)
        p.resetJointState(self.id, 2, z, physicsClientId=self.clid)
        p.resetJointStateMultiDof(self.id, 3, state.so3.xyzw, physicsClientId=self.clid)
        p.resetJointState(
            self.id, 5, self.default_prismatic_value, physicsClientId=self.clid
        )
        p.resetJointState(
            self.id, 6, self.default_prismatic_value, physicsClientId=self.clid
        )

    def in_collision(self, obstacles, radius=0.0, check_self=False):
        """
        Checks whether the robot is in collision with the environment

        :return: Boolean
        """
        # Step the simulator (only enough for collision detection)
        if check_self:
            contacts = p.getClosestPoints(
                self.id, self.id, radius, physicsClientId=self.clid
            )
            # Manually filter out fixed connections that shouldn't be considered
            # TODO fix this somehow
            filtered = []
            for c in contacts:
                # A link is always in collision with itself and its neighbors
                if abs(c[3] - c[4]) <= 1:
                    continue
                # panda_link8 just transforms the origin
                if c[3] == 6 and c[4] == 4:
                    continue
                if c[3] == 4 and c[4] == 6:
                    continue
                filtered.append(c)
            if len(filtered) > 0:
                return True

        # Iterate through all obstacles to check for collisions
        for id in obstacles:
            contacts = p.getClosestPoints(
                self.id, id, radius, physicsClientId=self.clid
            )
            if len(contacts) > 0:
                return True
        return False


class VisualGripper:
    def __init__(self, clid, **kwargs):
        self.id = self.load(clid, **kwargs)
        self.clid = clid

    def load(self, clid, prismatic=0.025):
        if math.isclose(prismatic, 0.04):
            path = FrankaGripper.fully_open_mesh
        elif math.isclose(prismatic, 0.025):
            path = FrankaGripper.half_open_mesh
        else:
            raise NotImplementedError(
                "Only prismatic values [0.04, 0.025] currently supported for VisualGripper"
            )

        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=path,
            # rgbaColor=[0, 1.0, 0, 1],
            physicsClientId=clid,
        )
        return p.createMultiBody(
            basePosition=[0, 0, 0],  # t.center,
            baseOrientation=[0, 0, 0, 1],  # t.pose.so3.xyzw,
            baseVisualShapeIndex=obstacle_visual_id,
            physicsClientId=clid,
        )

    def marionette(self, pose):
        p.resetBasePositionAndOrientation(
            self.id, pose.xyz, pose.so3.xyzw, physicsClientId=self.clid
        )


class Bullet:
    def __init__(self, gui=False, headless=False):
        """
        :param gui: Whether to use a gui to visualize the environment.
            Only one gui instance allowed
        """
        assert not (gui and headless), "GUI cannot be turned on for headless mode"
        self.use_gui = gui
        self.headless = headless
        if self.use_gui:
            self.clid = p.connect(p.GUI)
        else:
            self.clid = p.connect(p.DIRECT)
        self.robots = {}
        self.obstacle_ids = []
        self.poses = []

    def __del__(self):
        """
        Disconnects the client on destruction
        """
        p.disconnect(self.clid)

    def visualize_pose(self, pose):
        root_path = Path("/tmp")

        target_ids = []
        x_mesh_path = root_path / "target_x.obj"
        if not x_mesh_path.exists():
            x_transform = np.array(
                [[0, 0, 1, 0.05], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
            )
            x_axis_mesh = trimesh.creation.cylinder(
                radius=0.005,
                height=0.1,
                transform=x_transform,
            )
            with open(x_mesh_path, "w") as f:
                f.write(
                    trimesh.exchange.obj.export_obj(x_axis_mesh, include_color=True)
                )
        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(x_mesh_path),
            physicsClientId=self.clid,
            rgbaColor=[1, 0, 0, 1],
        )
        target_ids.append(
            p.createMultiBody(
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                baseVisualShapeIndex=obstacle_visual_id,
                physicsClientId=self.clid,
            )
        )

        y_mesh_path = root_path / "target_y.obj"
        if not y_mesh_path.exists():
            y_transform = np.array(
                [[0, 1, 0, 0], [0, 0, 1, 0.05], [1, 0, 0, 0], [0, 0, 0, 1]]
            )
            y_axis_mesh = trimesh.creation.cylinder(
                radius=0.005,
                height=0.1,
                transform=y_transform,
            )
            with open(y_mesh_path, "w") as f:
                f.write(
                    trimesh.exchange.obj.export_obj(y_axis_mesh, include_color=True)
                )
        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(y_mesh_path),
            physicsClientId=self.clid,
            rgbaColor=[0, 1, 0, 1],
        )
        target_ids.append(
            p.createMultiBody(
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                baseVisualShapeIndex=obstacle_visual_id,
                physicsClientId=self.clid,
            )
        )
        z_mesh_path = root_path / "target_z.obj"
        if not z_mesh_path.exists():
            z_transform = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.05], [0, 0, 0, 1]]
            )
            z_axis_mesh = trimesh.creation.cylinder(
                radius=0.005, height=0.1, transform=z_transform
            )
            with open(z_mesh_path, "w") as f:
                f.write(
                    trimesh.exchange.obj.export_obj(z_axis_mesh, include_color=True)
                )
        obstacle_visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=str(z_mesh_path),
            physicsClientId=self.clid,
            rgbaColor=[0, 0, 1, 1],
        )
        target_ids.append(
            p.createMultiBody(
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                baseVisualShapeIndex=obstacle_visual_id,
                physicsClientId=self.clid,
            )
        )
        self.poses.append(target_ids)
        return target_ids

    def set_camera_position(self, yaw, pitch, distance, target):
        p.resetDebugVisualizerCamera(
            distance, yaw, pitch, target, physicsClientId=self.clid
        )

    def set_camera_position_from_matrix(self, pose):
        distance = pose.matrix[2, 3] / pose.matrix[2, 2]
        # distance = 5
        target = tuple((pose.matrix @ np.array([0, 0, -distance, 1]))[:3])

        # Calculations for signed angles come from here https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
        # Have to get the angle between the camera's x axis and the world x axis
        v1 = (pose.matrix @ np.array([1, 0, 0, 1]))[:3] - pose.pos
        v2 = np.array([1, 0, 0])
        vn = np.cross(v1, v2)
        vn = vn / np.linalg.norm(vn)
        yaw = np.degrees(np.arctan2(np.dot(np.cross(v1, v2), vn), np.dot(v1, v2)))

        # Have to get the
        v1 = (pose.matrix @ np.array([0, 1, 0, 1]))[:3] - pose.pos
        v2 = np.array([0, 0, 1])
        vn = np.cross(v1, v2)
        vn = vn / np.linalg.norm(vn)
        pitch = np.degrees(np.arctan2(np.dot(np.cross(v2, v1), vn), np.dot(v1, v2)))
        p.resetDebugVisualizerCamera(
            distance, yaw, pitch, target, physicsClientId=self.clid
        )

    def get_camera_position(self):
        params = p.getDebugVisualizerCamera(physicsClientId=self.clid)
        return {
            "yaw": params[8],
            "pitch": params[9],
            "distance": params[10],
            "target": params[11],
        }

    def get_projection_matrix(self, width, height, fx, fy, cx, cy, near, far):
        projection_matrix = (
            2.0 * fx / width,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 * fy / height,
            0.0,
            0.0,
            1.0 - 2.0 * cx / width,
            2.0 * cy / height - 1.0,
            (far + near) / (near - far),
            -1.0,
            0.0,
            0.0,
            2.0 * far * near / (near - far),
            0.0,
        )
        return projection_matrix

    def get_camera_images(
        self,
        camera_T_world,
        width=640,
        height=480,
        fx=616.36529541,
        fy=616.20294189,
        cx=310.25881958,
        cy=310.25881958,
        near=0.01,
        far=10,
        scale=True,
    ):
        projection_matrix = self.get_projection_matrix(
            width, height, fx, fy, cx, cy, near, far
        )
        view_matrix = camera_T_world.matrix.T.reshape(16)
        _, _, rgb, depth, seg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.clid,
        )
        if scale:
            depth = far * near / (far - (far - near) * depth)
        return rgb, depth, seg

    def get_point_cloud_from_camera(
        self,
        camera_T_world,
        width=640,
        height=480,
        fx=616.36529541,
        fy=616.20294189,
        cx=310.25881958,
        cy=310.25881958,
        near=0.01,
        far=10,
        remove_robot=None,
        keep_robot=None,
        finite_depth=True,
    ):
        assert not (keep_robot is not None and remove_robot is not None)
        _, depth_image, segmentation = self.get_camera_images(
            camera_T_world,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near,
            far * 2,
        )
        # Remove all points that are too far away
        depth_image[depth_image > far] = 0.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        if remove_robot is not None:
            depth_image[segmentation == remove_robot.id] = 0.0
        elif keep_robot is not None:
            depth_image[segmentation != keep_robot.id] = 0.0
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width))
        image_points = np.stack((x, y, ones), axis=2).reshape(width * height, 3).T
        backprojected = np.linalg.inv(K) @ image_points
        pc = np.multiply(
            np.tile(depth_image.reshape(1, width * height), (3, 1)), backprojected
        ).T
        if finite_depth:
            pc = pc[np.isfinite(pc[:, 0]), :]
        capture_camera = camera_T_world.inverse * SE3(
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0])
        )
        pc = pc[~np.all(pc == 0, axis=1)]
        transform_point_cloud(pc, capture_camera.matrix, in_place=True)
        return pc

    def load_robot(self, robot_type, hd=False, collision_free=False, **kwargs):
        """
        Generic function to load a robot.
        """
        if issubclass(robot_type, FrankaRobot):
            robot = BulletFranka(self.clid, hd, **kwargs)
        elif issubclass(robot_type, FrankaGripper):
            if collision_free:
                robot = VisualGripper(self.clid, **kwargs)
            else:
                robot = BulletFrankaGripper(self.clid, **kwargs)
        else:
            raise NotImplementedError(f"Cannot load robot of type {robot_type}")
        self.robots[robot.id] = robot
        return robot

    def in_collision(self, robot, radius=0.0, check_self=False, **kwargs):
        return robot.in_collision(self.obstacle_ids, radius, check_self, **kwargs)

    def load_mesh(self, visual_mesh_path, collision_mesh_path=None, color=None):
        if collision_mesh_path is None:
            collision_mesh_path = visual_mesh_path
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        visual_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=visual_mesh_path,
            rgbaColor=color,
            physicsClientId=self.clid,
        )
        collision_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=collision_mesh_path,
            physicsClientId=self.clid,
        )
        obstacle_id = p.createMultiBody(
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1],
            baseVisualShapeIndex=visual_id,
            baseCollisionShapeIndex=collision_id,
            physicsClientId=self.clid,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_cuboid(self, cuboid, color=None, visual_only=False):
        assert isinstance(cuboid, Cuboid)
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        assert not cuboid.is_zero_volume(), "Cannot load zero volume cuboid"
        kwargs = {}
        if self.use_gui or self.headless:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=cuboid.half_extents.tolist(),
                rgbaColor=color,
                physicsClientId=self.clid,
            )
            kwargs["baseVisualShapeIndex"] = obstacle_visual_id
        if not visual_only:
            obstacle_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=cuboid.half_extents.tolist(),
                physicsClientId=self.clid,
            )
            kwargs["baseCollisionShapeIndex"] = obstacle_collision_id
        obstacle_id = p.createMultiBody(
            basePosition=cuboid.center.tolist(),
            baseOrientation=cuboid.pose.so3.xyzw,
            physicsClientId=self.clid,
            **kwargs,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_cylinder(self, cylinder, color=None, visual_only=False):
        assert isinstance(cylinder, Cylinder)
        if color is None:
            color = [0.85882353, 0.14117647, 0.60392157, 1]
        assert not cylinder.is_zero_volume(), "Cannot load zero volume cylinder"
        kwargs = {}
        if self.use_gui or self.headless:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=cylinder.radius,
                length=cylinder.height,
                rgbaColor=color,
                physicsClientId=self.clid,
            )
            kwargs["baseVisualShapeIndex"] = obstacle_visual_id
        if not visual_only:
            obstacle_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=cylinder.radius,
                height=cylinder.height,
                physicsClientId=self.clid,
            )
            kwargs["baseCollisionShapeIndex"] = obstacle_collision_id
        obstacle_id = p.createMultiBody(
            basePosition=cylinder.center.tolist(),
            baseOrientation=cylinder.pose.so3.xyzw,
            physicsClientId=self.clid,
            **kwargs,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_sphere(self, sphere, color=None, visual_only=False):
        assert isinstance(sphere, Sphere)
        if color is None:
            color = [0.0, 0.0, 0.0, 1.0]
        kwargs = {}
        if self.use_gui or self.headless:
            obstacle_visual_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                rgbaColor=color,
                physicsClientId=self.clid,
            )
            kwargs["baseVisualShapeIndex"] = obstacle_visual_id
        if not visual_only:
            obstacle_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_SPHERE,
                radius=sphere.radius,
                physicsClientId=self.clid,
            )
            kwargs["baseCollisionShapeIndex"] = obstacle_collision_id
        obstacle_id = p.createMultiBody(
            basePosition=sphere.center,
            physicsClientId=self.clid,
            **kwargs,
        )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

    def load_primitives(self, primitives, color=None, visual_only=False):
        ids = []
        for prim in primitives:
            if prim.is_zero_volume():
                continue
            elif isinstance(prim, Cuboid):
                ids.append(self.load_cuboid(prim, color, visual_only))
            elif isinstance(prim, Cylinder):
                ids.append(self.load_cylinder(prim, color, visual_only))
            elif isinstance(prim, Sphere):
                ids.append(self.load_sphere(prim, color, visual_only))
            else:
                raise Exception(
                    "Only cuboids, cylinders, and spheres supported as primitives"
                )
        return ids

    def load_urdf_obstacle(self, path, pose=None):
        if pose is not None:
            obstacle_id = p.loadURDF(
                str(path),
                basePosition=pose.xyz,
                baseOrientation=pose.so3.xyzw,
                useFixedBase=True,
                physicsClientId=self.clid,
            )
        else:
            obstacle_id = p.loadURDF(
                str(path),
                useFixedBase=True,
                physicsClientId=self.clid,
            )
        self.obstacle_ids.append(obstacle_id)
        return obstacle_id

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

    def clear_all_poses(self):
        for pose in self.poses:
            for id in pose:
                p.removeBody(id, physicsClientId=self.clid)
        self.poses = []


class BulletController(Bullet):
    def __init__(self, gui=False, hz=12, substeps=20):
        """
        :param gui: Whether to use a gui to visualize the environment.
            Only one gui instance allowed
        """
        super().__init__(gui)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1 / hz,
            numSubSteps=substeps,
            deterministicOverlappingPairs=1,
            physicsClientId=self.clid,
        )

    def step(self):
        p.stepSimulation(physicsClientId=self.clid)
