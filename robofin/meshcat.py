import meshcat
import numpy as np
import urchin
from geometrout.primitive import Cuboid, Cylinder, Sphere

from robofin.collision import FrankaCollisionSpheres
from robofin.robot_constants import FrankaConstants


def rgb_to_hex(rgb):
    # Ensure that the values are in the valid range (0-255)
    if isinstance(rgb[0], float):
        for x in rgb:
            assert 0 <= x <= 1
        r, g, b = [min(max(int(x * 255), 0), 255) for x in rgb]
    else:
        r, g, b = [min(max(x, 0), 255) for x in rgb]

    # Convert the values to hexadecimal and format them as a string
    hex_color = "0x{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))

    return hex_color


def generate_color_gradient(rgb_color, num_colors, destination_color):
    # Extract the RGB components of the source color
    src_r, src_g, src_b = rgb_color

    # Extract the RGB components of the destination color
    dest_r, dest_g, dest_b = destination_color

    # Calculate the step size for each color channel
    r_step = (dest_r - src_r) / (num_colors - 1)
    g_step = (dest_g - src_g) / (num_colors - 1)
    b_step = (dest_b - src_b) / (num_colors - 1)

    # Generate the gradient colors
    gradient = []
    for i in range(num_colors):
        r = int(src_r + i * r_step)
        g = int(src_g + i * g_step)
        b = int(src_b + i * b_step)
        gradient.append((r, g, b))

    return gradient


class MeshcatFrankaGripper:
    def __init__(self, gripper_idx, vis):
        self.vis = vis
        self.gripper_idx = gripper_idx
        self.key = f"franka_gripper/{self.gripper_idx}"
        self.cooo = None
        self.spheres_key = f"{self.key}/spheres"
        self.franka_urdf = urchin.URDF.load(
            FrankaConstants.urdf,
        )
        trimeshes = self.franka_urdf.visual_trimesh_fk(
            np.array([*FrankaConstants.NEUTRAL, 0.04]),
            links=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
        )
        panda_hand_transform = self.franka_urdf.link_fk(
            np.array([*FrankaConstants.NEUTRAL, 0.04]), link="panda_hand"
        )

        self.inv_panda_hand_transform = np.linalg.inv(panda_hand_transform)
        transforms = {
            key: self.inv_panda_hand_transform @ val for key, val in trimeshes.items()
        }
        for idx, (mesh, transform) in enumerate(transforms.items()):
            self.vis[f"{self.key}/{idx}"].set_object(
                meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces)
            )
            self.vis[f"{self.key}/{idx}"].set_transform(
                panda_hand_transform @ transform
            )

    def __del__(self):
        self.vis[self.key].delete()

    def convert_pose(self, pose, frame):
        if frame == "panda_link8":
            pose = pose * FrankaConstants.EFF_T_LIST[("panda_link8", "panda_hand")]
        elif frame == "right_gripper":
            pose = (
                pose
                * FrankaConstants.EFF_T_LIST[("panda_hand", "right_gripper")].inverse
            )
            pass
        elif frame == "panda_grasptarget":
            pose = (
                pose
                * FrankaConstants.EFF_T_LIST[
                    ("panda_link8", "panda_grasptarget")
                ].inverse
                * FrankaConstants.EFF_T_LIST[("panda_link8", "panda_hand")]
            )
        else:
            assert frame == "panda_hand"
        return pose

    def marionette(self, pose, prismatic_joint, frame):
        pose = self.convert_pose(pose, frame)
        trimeshes = self.franka_urdf.visual_trimesh_fk(
            np.array([*FrankaConstants.NEUTRAL, prismatic_joint]),
            links=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
        )

        transforms = {
            key: self.inv_panda_hand_transform @ val for key, val in trimeshes.items()
        }
        for idx, (mesh, transform) in enumerate(transforms.items()):
            self.vis[f"{self.key}/{idx}"].set_transform(pose.matrix @ transform)

    def marionette_and_check(self, pose, prismatic_joint, frame, obstacles):
        """Slower because it has to reload meshes with colors."""
        self.marionette(pose, prismatic_joint, frame)
        if self.cooo is None:
            self.cooo = FrankaCollisionSpheres()

        spheres = self.cooo.eef_collision_spheres(pose, prismatic_joint, frame)
        has_collision = False
        for idx, sphere in enumerate(spheres):
            collides = False
            for o in obstacles:
                if o.sdf(sphere.center) <= sphere.radius:
                    self.vis[f"{self.key}/cspheres/{idx}"].set_object(
                        meshcat.geometry.Sphere(sphere.radius),
                        meshcat.geometry.MeshLambertMaterial(
                            color=rgb_to_hex([255, 0, 0]), reflectivity=0.8
                        ),
                    )
                    collides = True
                    has_collision = True
                    break
            if not collides:
                self.vis[f"{self.key}/cspheres/{idx}"].set_object(
                    meshcat.geometry.Sphere(sphere.radius),
                )
            transform = np.eye(4)
            transform[:3, -1] = sphere.center
            self.vis[f"{self.key}/cspheres/{idx}"].set_transform(transform)
        return has_collision


class MeshcatFranka:
    def __init__(self, franka_idx, vis):
        self.vis = vis
        self.franka_idx = franka_idx
        self.key = f"franka/{self.franka_idx}"
        self.franka_urdf = urchin.URDF.load(FrankaConstants.urdf)
        self.cooo = None
        self.spheres_key = f"franka/{self.franka_idx}/spheres"
        self.with_base_link = False
        trimeshes = self.franka_urdf.visual_trimesh_fk(
            np.array([*FrankaConstants.NEUTRAL, 0.04])
        )
        for idx, (mesh, transform) in enumerate(trimeshes.items()):
            self.vis[f"{self.key}/{idx}"].set_object(
                meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces)
            )
            self.vis[f"{self.key}/{idx}"].set_transform(transform)

    def __del__(self):
        self.vis[self.key].delete()

    def load_cspheres(self, with_base_link=False):
        self.cooo = FrankaCollisionSpheres()
        self.with_base_link = with_base_link
        spheres = self.cooo.collision_spheres(
            FrankaConstants.NEUTRAL, 0.04, with_base_link=self.with_base_link
        )
        for idx, sphere in enumerate(spheres):
            self.vis[f"{self.key}/cspheres/{idx}"].set_object(
                meshcat.geometry.Sphere(sphere.radius)
            )
            transform = np.eye(4)
            transform[:3, -1] = sphere.center
            self.vis[f"{self.key}/{idx}"].set_transform(transform)

    def marionette(self, config, prismatic):
        trimeshes = self.franka_urdf.visual_trimesh_fk(np.array([*config, prismatic]))
        for idx, (mesh, transform) in enumerate(trimeshes.items()):
            self.vis[f"{self.key}/{idx}"].set_transform(transform)

        # True when spheres are loaded
        if self.cooo is not None:
            spheres = self.cooo.collision_spheres(
                config, prismatic, with_base_link=self.with_base_link
            )
            for idx, sphere in enumerate(spheres):
                transform = np.eye(4)
                transform[:3, -1] = sphere.center
                self.vis[f"{self.key}/{idx}"].set_transform(transform)

    def marionette_and_check(self, config, prismatic, obstacles):
        """Slower because it has to reload meshes with colors."""
        trimeshes = self.franka_urdf.visual_trimesh_fk(np.array([*config, prismatic]))
        for idx, (mesh, transform) in enumerate(trimeshes.items()):
            self.vis[f"{self.key}/{idx}"].set_transform(transform)

        if self.cooo is None:
            self.cooo = FrankaCollisionSpheres()
        spheres = self.cooo.collision_spheres(
            config, prismatic, with_base_link=self.with_base_link
        )
        has_collision = False
        for idx, sphere in enumerate(spheres):
            collides = False
            for o in obstacles:
                if o.sdf(sphere.center) <= sphere.radius:
                    self.vis[f"{self.key}/cspheres/{idx}"].set_object(
                        meshcat.geometry.Sphere(sphere.radius),
                        meshcat.geometry.MeshLambertMaterial(
                            color=rgb_to_hex([255, 0, 0]), reflectivity=0.8
                        ),
                    )
                    collides = True
                    has_collision = True
                    break
            if not collides:
                self.vis[f"{self.key}/cspheres/{idx}"].set_object(
                    meshcat.geometry.Sphere(sphere.radius),
                )
            transform = np.eye(4)
            transform[:3, -1] = sphere.center
            self.vis[f"{self.key}/cspheres/{idx}"].set_transform(transform)
        return has_collision


class Meshcat:
    def __init__(self):
        self.vis = meshcat.Visualizer()
        self.frankas = []
        self.grippers = []
        self.spheres = []
        self.cuboids = []
        self.cylinders = []
        self.poses = []

    def _increment_type(self, item_type):
        if len(item_type) == 0:
            item_type.append(0)
        else:
            item_type.append(item_type[-1] + 1)
        return item_type[-1]

    def load_franka(self):
        if not self.frankas:
            franka_idx = 0
        else:
            franka_idx = self.frankas[-1].franka_idx + 1

        franka = MeshcatFranka(franka_idx, self.vis)
        self.frankas.append(franka)
        return franka

    def load_gripper(self):
        if not self.grippers:
            gripper_idx = 0
        else:
            gripper_idx = self.grippers[-1].gripper_idx + 1

        gripper = MeshcatFrankaGripper(gripper_idx, self.vis)
        self.grippers.append(gripper)
        return gripper

    def load_spheres(self, spheres, color_gradient=None):
        keys = []
        if color_gradient is None:
            color_gradient = [255, 0, 0], [255, 0, 0]
        elif len(color_gradient) == 1:
            color_gradient = color_gradient[0], color_gradient[0]
        elif len(color_gradient) == 3:
            color_gradient = [color_gradient, color_gradient]
        rgb_colors = generate_color_gradient(
            color_gradient[0], len(spheres), color_gradient[1]
        )
        for sphere, rgb_color in zip(spheres, rgb_colors):
            idx = self._increment_type(self.spheres)
            self.vis[f"spheres/{idx}"].set_object(
                meshcat.geometry.Sphere(sphere.radius),
                meshcat.geometry.MeshLambertMaterial(
                    color=rgb_to_hex(rgb_color), reflectivity=0.8
                ),
            )
            transform = np.eye(4)
            transform[:3, -1] = sphere.center
            self.vis[f"spheres/{idx}"].set_transform(transform)
            keys.append(f"spheres/{idx}")
        return keys

    def clear_all_spheres(self):
        self.vis["spheres"].delete()

    def load_cuboids(self, cuboids, color_gradient=None):
        keys = []
        if color_gradient is None:
            color_gradient = [74, 152, 192], [219, 36, 154]  # , [219, 36, 73]
        elif len(color_gradient) == 1:
            color_gradient = color_gradient[0], color_gradient[0]
        elif len(color_gradient) == 3:
            color_gradient = [color_gradient, color_gradient]
        rgb_colors = generate_color_gradient(
            color_gradient[0], len(cuboids), color_gradient[1]
        )
        for cuboid, rgb_color in zip(cuboids, rgb_colors):
            idx = self._increment_type(self.cuboids)
            print(rgb_to_hex(rgb_color))
            self.vis[f"cuboids/{idx}"].set_object(
                meshcat.geometry.Box(cuboid.dims),
                meshcat.geometry.MeshLambertMaterial(
                    color=rgb_to_hex(rgb_color), reflectivity=0.8
                ),
            )
            self.vis[f"cuboids/{idx}"].set_transform(cuboid.pose.matrix)
            keys.append(f"cuboids/{idx}")
        return keys

    def clear_all_cuboids(self):
        self.vis["cuboids"].delete()

    def load_cylinders(self, cylinders, color_gradient=None):
        keys = []
        if color_gradient is None:
            color_gradient = [219, 36, 154], [219, 36, 73]
        elif len(color_gradient) == 1:
            color_gradient = color_gradient[0], color_gradient[0]
        elif len(color_gradient) == 3:
            color_gradient = [color_gradient, color_gradient]
        rgb_colors = generate_color_gradient(
            color_gradient[0], len(cylinders), color_gradient[1]
        )
        for cylinder, rgb_color in zip(cylinders, rgb_colors):
            idx = self._increment_type(self.cylinders)
            self.vis[f"cylinders/{idx}"].set_object(
                meshcat.geometry.Cylinder(cylinder.height, cylinder.radius),
                meshcat.geometry.MeshLambertMaterial(
                    color=rgb_to_hex(rgb_color), reflectivity=0.8
                ),
            )
            self.vis[f"cylinders/{idx}"].set_transform(cylinder.pose.matrix)
            keys.append(f"cylinders/{idx}")
        return keys

    def clear_all_cylinders(self):
        self.vis["cylinders"].delete()

    def load_primitives(self, primitives):
        keys = []
        cuboids = [p for p in primitives if isinstance(p, Cuboid)]
        cylinders = [p for p in primitives if isinstance(p, Cylinder)]
        spheres = [p for p in primitives if isinstance(p, Sphere)]
        keys.extend(self.load_spheres(spheres))
        keys.extend(self.load_cuboids(cuboids))
        keys.extend(self.load_cylinders(cylinders))
        return keys

    def clear_all_primitives(self):
        self.clear_all_spheres()
        self.clear_all_cuboids()
        self.clear_all_cylinders()

    def load_pose(self, pose):
        idx = self._increment_type(self.poses)
        self.vis[f"poses/{idx}/x"].set_object(
            meshcat.geometry.Cylinder(0.1, 0.005),
            meshcat.geometry.MeshLambertMaterial(
                color=rgb_to_hex([255, 0, 0]), reflectivity=0.8
            ),
        )
        self.vis[f"poses/{idx}/y"].set_object(
            meshcat.geometry.Cylinder(0.1, 0.005),
            meshcat.geometry.MeshLambertMaterial(
                color=rgb_to_hex([0, 255, 0]), reflectivity=0.8
            ),
        )
        self.vis[f"poses/{idx}/z"].set_object(
            meshcat.geometry.Cylinder(0.1, 0.005),
            meshcat.geometry.MeshLambertMaterial(
                color=rgb_to_hex([0, 0, 255]), reflectivity=0.8
            ),
        )
        self.vis[f"poses/{idx}/x"].set_transform(
            pose.matrix
            @ np.array([[0, 1, 0, 0.05], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        )
        self.vis[f"poses/{idx}/y"].set_transform(
            pose.matrix
            @ np.array([[1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 1, 0], [0, 0, 0, 1]])
        )
        self.vis[f"poses/{idx}/z"].set_transform(
            pose.matrix
            @ np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, 0, 1]])
        )
        return f"poses/{idx}"

    def clear_all_poses(self):
        self.vis["poses"].delete()
