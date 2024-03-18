import os
from collections import OrderedDict

import numpy as np
import six
import torch
import trimesh
from lxml import etree as ET
from urchin import (
    URDF,
    Box,
    Collision,
    Cylinder,
    Geometry,
    Inertial,
    Joint,
    Link,
    Material,
    Mesh,
    Sphere,
    Transmission,
    URDFTypeWithMesh,
    Visual,
)
from urchin.utils import load_meshes, parse_origin

from robofin.point_cloud_tools import transform_point_cloud


def configure_origin(value, device=None):
    """Convert a value into a 4x4 transform matrix.
    Parameters
    ----------
    value : None, (6,) float, or (4,4) float
        The value to turn into the matrix.
        If (6,), interpreted as xyzrpy coordinates.
    Returns
    -------
    matrix : (4,4) float or None
        The created matrix.
    """
    assert isinstance(
        value, torch.Tensor
    ), "Invalid type for origin, expect 4x4 torch tensor"
    assert value.shape == (4, 4)
    return value.to(device)


class TorchMesh(Mesh):
    def __init__(
        self,
        filename,
        combine,
        scale=None,
        meshes=None,
        lazy_filename=None,
        device=None,
    ):
        self.device = device
        super().__init__(filename, combine, scale, meshes, lazy_filename)
        if self.meshes is not None:
            self.vertices = [
                torch.as_tensor(m.vertices, device=self.device) for m in self.meshes
            ]

    @property
    def meshes(self):
        """list of :class:`~trimesh.base.Trimesh` : The triangular meshes
        that represent this object.
        """
        if self.lazy_filename is not None and self._meshes is None:
            self.meshes = self._load_and_combine_meshes(
                self.lazy_filename, self.combine
            )
            self.vertices = [
                torch.as_tensor(m.vertices, device=self.device) for m in self.meshes
            ]
        return self._meshes

    @meshes.setter
    def meshes(self, value):
        if self.lazy_filename is not None and value is None:
            self._meshes = None
        elif isinstance(value, six.string_types):
            value = self._load_and_combine_meshes(value, self.combine)
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            value = list(value)
            if len(value) == 0:
                raise ValueError("Mesh must have at least one trimesh.Trimesh")
            for m in value:
                if not isinstance(m, trimesh.Trimesh):
                    raise TypeError(
                        "Mesh requires a trimesh.Trimesh or a " "list of them"
                    )
        elif isinstance(value, trimesh.Trimesh):
            value = [value]
        else:
            raise TypeError("Mesh requires a trimesh.Trimesh")
        self._meshes = value
        self.vertices = [
            torch.as_tensor(m.vertices, device=self.device) for m in self._meshes
        ]

    @property
    def scale(self):
        """(3,) float : A scaling for the mesh along its local XYZ axes."""
        return self._scale

    @scale.setter
    def scale(self, value):
        if value is not None:
            value = torch.as_tensor(value, device=self.device)
        self._scale = value


class TorchGeometry(Geometry):
    _ELEMENTS = {
        "box": (Box, False, False),
        "cylinder": (Cylinder, False, False),
        "sphere": (Sphere, False, False),
        "mesh": (TorchMesh, False, False),
    }

    def __init__(self, box=None, cylinder=None, sphere=None, mesh=None, device=None):
        self.device = device
        super().__init__(box, cylinder, sphere, mesh)


class TorchVisual(Visual):
    _ELEMENTS = {
        "geometry": (TorchGeometry, True, False),
        "material": (Material, False, False),
    }

    def __init__(self, geometry, name=None, origin=None, material=None, device=None):
        self.device = device
        super().__init__(geometry, name, origin, material)

    @property
    def geometry(self):
        """:class:`.Geometry` : The geometry of this element."""
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, TorchGeometry):
            raise TypeError("Must set geometry with TorchGeometry object")
        self._geometry = value

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value, self.device)

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        kwargs = cls._parse(node, path, lazy_load_meshes)
        kwargs["origin"] = torch.tensor(parse_origin(node))
        kwargs["device"] = device
        return TorchVisual(**kwargs)


class TorchCollision(Collision):
    _ELEMENTS = {
        "geometry": (TorchGeometry, True, False),
    }

    @property
    def geometry(self):
        """:class:`.Geometry` : The geometry of this element."""
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, TorchGeometry):
            raise TypeError("Must set geometry with Geometry object")
        self._geometry = value

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        kwargs = cls._parse(node, path, lazy_load_meshes)
        kwargs["origin"] = parse_origin(node)
        return cls(**kwargs)


class TorchLink(Link):
    _ELEMENTS = {
        "inertial": (Inertial, False, False),
        "visuals": (TorchVisual, False, True),
        "collisions": (TorchCollision, False, True),
    }

    def __init__(self, name, inertial, visuals, collisions, device=None):
        self.device = device
        super().__init__(name, inertial, visuals, collisions)

    @classmethod
    def _parse_simple_elements(cls, node, path, lazy_load_meshes, device):
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]
            if not m:
                v = node.find(t._TAG)
                if r or v is not None:
                    if issubclass(t, URDFTypeWithMesh):
                        v = t._from_xml(v, path, lazy_load_meshes)
                    else:
                        v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(t.__name__, cls.__name__)
                    )
                if issubclass(t, URDFTypeWithMesh):
                    v = [t._from_xml(n, path, lazy_load_meshes, device) for n in vs]
                else:
                    v = [t._from_xml(n, path, device) for n in vs]
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse(cls, node, path, lazy_load_meshes, device):
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path, lazy_load_meshes, device))
        return kwargs

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        """Create an instance of this class from an XML node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        obj : :class:`URDFType`
            An instance of this class parsed from the node.
        """
        return cls(**cls._parse(node, path, lazy_load_meshes, device))


class TorchJoint(Joint):
    def __init__(
        self,
        name,
        joint_type,
        parent,
        child,
        axis=None,
        origin=None,
        limit=None,
        dynamics=None,
        safety_controller=None,
        calibration=None,
        mimic=None,
        device=None,
    ):
        self.device = device
        super().__init__(
            name,
            joint_type,
            parent,
            child,
            axis,
            origin,
            limit,
            dynamics,
            safety_controller,
            calibration,
            mimic,
        )

    @property
    def origin(self):
        """(4,4) float : The pose of this element relative to the link frame."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = configure_origin(value, device=self.device)

    @property
    def axis(self):
        """(3,) float : The joint axis in the joint frame."""
        return self._axis

    @axis.setter
    def axis(self, value):
        if value is None:
            value = torch.as_tensor([1.0, 0.0, 0.0], device=self.device)
        elif isinstance(value, torch.Tensor):
            assert value.shape == (3,), "Invalid shape for axis, should be (3,)"
            value = value.to(self.device)
            value = value / torch.norm(value)
        else:
            value = torch.as_tensor(value, device=self.device)
            if value.shape != (3,):
                raise ValueError("Invalid shape for axis, should be (3,)")
            value = value / torch.norm(value)
        self._axis = value

    @classmethod
    def _from_xml(cls, node, path, device):
        kwargs = cls._parse(node, path)
        kwargs["joint_type"] = str(node.attrib["type"])
        kwargs["parent"] = node.find("parent").attrib["link"]
        kwargs["child"] = node.find("child").attrib["link"]
        axis = node.find("axis")
        if axis is not None:
            axis = torch.as_tensor(
                np.fromstring(axis.attrib["xyz"], sep=" "),
            )
        kwargs["axis"] = axis
        kwargs["origin"] = torch.tensor(parse_origin(node))
        kwargs["device"] = device

        return TorchJoint(**kwargs)

    def _rotation_matrices(self, angles, axis):
        """Compute rotation matrices from angle/axis representations.
        Parameters
        ----------
        angles : (n,) float
            The angles.
        axis : (3,) float
            The axis.
        Returns
        -------
        rots : (n,4,4)
            The rotation matrices
        """
        axis = axis / torch.norm(axis)
        sina = torch.sin(angles)
        cosa = torch.cos(angles)
        M = torch.eye(4, device=self.device).repeat((len(angles), 1, 1))
        M[:, 0, 0] = cosa
        M[:, 1, 1] = cosa
        M[:, 2, 2] = cosa
        M[:, :3, :3] += (
            torch.ger(axis, axis).repeat((len(angles), 1, 1))
            * (1.0 - cosa)[:, np.newaxis, np.newaxis]
        )
        M[:, :3, :3] += (
            torch.tensor(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                device=self.device,
            ).repeat((len(angles), 1, 1))
            * sina[:, np.newaxis, np.newaxis]
        )
        return M

    def get_child_poses(self, cfg, n_cfgs):
        """Computes the child pose relative to a parent pose for a given set of
        configuration values.
        Parameters
        ----------
        cfg : (n,) float or None
            The configuration values for this joint. They are interpreted
            based on the joint type as follows:
            - ``fixed`` - not used.
            - ``prismatic`` - a translation along the axis in meters.
            - ``revolute`` - a rotation about the axis in radians.
            - ``continuous`` - a rotation about the axis in radians.
            - ``planar`` - Not implemented.
            - ``floating`` - Not implemented.
            If ``cfg`` is ``None``, then this just returns the joint pose.
        Returns
        -------
        poses : (n,4,4) float
            The poses of the child relative to the parent.
        """
        if cfg is None:
            return self.origin.repeat((n_cfgs, 1, 1))
        elif self.joint_type == "fixed":
            return self.origin.repeat((n_cfgs, 1, 1))
        elif self.joint_type in ["revolute", "continuous"]:
            if cfg is None:
                cfg = torch.zeros(n_cfgs)
            return torch.matmul(
                self.origin.type_as(cfg),
                self._rotation_matrices(cfg, self.axis).type_as(cfg),
            )
        elif self.joint_type == "prismatic":
            if cfg is None:
                cfg = torch.zeros(n_cfgs)
            translation = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))
            translation[:, :3, 3] = self.axis * cfg[:, np.newaxis]
            return torch.matmul(self.origin.type_as(cfg), translation.type_as(cfg))
        elif self.joint_type == "planar":
            raise NotImplementedError()
        elif self.joint_type == "floating":
            raise NotImplementedError()
        else:
            raise ValueError("Invalid configuration")


class TorchURDF(URDF):
    _ELEMENTS = {
        "links": (TorchLink, True, True),
        "joints": (TorchJoint, False, True),
        "transmissions": (Transmission, False, True),
        "materials": (Material, False, True),
    }

    def __init__(
        self,
        name,
        links,
        joints=None,
        transmissions=None,
        materials=None,
        other_xml=None,
        device=None,
    ):
        self.device = device
        self._faces = None
        super().__init__(name, links, joints, transmissions, materials, other_xml)

    @classmethod
    def load(cls, file_obj, lazy_load_meshes=True, device=None):
        """Load a URDF from a file.
        Parameters
        ----------
        file_obj : str or file-like object
            The file to load the URDF from. Should be the path to the
            ``.urdf`` XML file. Any paths in the URDF should be specified
            as relative paths to the ``.urdf`` file instead of as ROS
            resources.
        Returns
        -------
        urdf : :class:`.URDF`
            The parsed URDF.
        """
        if isinstance(file_obj, str):
            if os.path.isfile(file_obj):
                parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
                tree = ET.parse(file_obj, parser=parser)
                path, _ = os.path.split(file_obj)
            else:
                raise ValueError("{} is not a file".format(file_obj))
        else:
            parser = ET.XMLParser(remove_comments=True, remove_blank_text=True)
            tree = ET.parse(file_obj, parser=parser)
            path, _ = os.path.split(file_obj.name)

        node = tree.getroot()
        return cls._from_xml(node, path, lazy_load_meshes, device)

    @classmethod
    def _parse_simple_elements(cls, node, path, lazy_load_meshes, device):
        """Parse all elements in the _ELEMENTS array from the children of
        this node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse children for.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from element names to the :class:`URDFType` subclass (or list,
            if ``multiple`` was set) created for that element.
        """
        kwargs = {}
        for a in cls._ELEMENTS:
            t, r, m = cls._ELEMENTS[a]
            if not m:
                v = node.find(t._TAG)
                if r or v is not None:
                    if issubclass(t, URDFTypeWithMesh):
                        v = t._from_xml(v, path, lazy_load_meshes)
                    else:
                        v = t._from_xml(v, path)
            else:
                vs = node.findall(t._TAG)
                if len(vs) == 0 and r:
                    raise ValueError(
                        "Missing required subelement(s) of type {} when "
                        "parsing an object of type {}".format(t.__name__, cls.__name__)
                    )
                if issubclass(t, URDFTypeWithMesh):
                    v = [t._from_xml(n, path, lazy_load_meshes, device) for n in vs]
                else:
                    v = [t._from_xml(n, path, device) for n in vs]
            kwargs[a] = v
        return kwargs

    @classmethod
    def _parse(cls, node, path, lazy_load_meshes, device):
        """Parse all elements and attributes in the _ELEMENTS and _ATTRIBS
        arrays for a node.
        Parameters
        ----------
        node : :class:`lxml.etree.Element`
            The node to parse.
        path : str
            The string path where the XML file is located (used for resolving
            the location of mesh or image files).
        Returns
        -------
        kwargs : dict
            Map from names to Python classes created from the attributes
            and elements in the class arrays.
        """
        kwargs = cls._parse_simple_attribs(node)
        kwargs.update(cls._parse_simple_elements(node, path, lazy_load_meshes, device))
        return kwargs

    @classmethod
    def _from_xml(cls, node, path, lazy_load_meshes, device):
        valid_tags = set(["joint", "link", "transmission", "material"])
        kwargs = cls._parse(node, path, lazy_load_meshes, device)

        extra_xml_node = ET.Element("extra")
        for child in node:
            if child.tag not in valid_tags:
                extra_xml_node.append(child)

        data = ET.tostring(extra_xml_node)
        kwargs["other_xml"] = data
        kwargs["device"] = device
        return cls(**kwargs)

    @property
    def faces(self):
        if self._faces is not None:
            return self._faces
        meshes = []
        for link in self.links:
            for visual in link.visuals:
                for mesh in visual.geometry.meshes:
                    meshes.append(mesh)
        mesh = trimesh.util.concatenate(meshes)
        self._faces = torch.as_tensor(mesh.faces, device=self.device)
        return self._faces

    def _process_cfgs(self, cfgs):
        """Process a list of joint configurations into a dictionary mapping joints to
        configuration values.
        This should result in a dict mapping each joint to a list of cfg values, one
        per joint.
        """
        joint_cfg = {}
        assert isinstance(cfgs, torch.Tensor), "Incorrectly formatted config array"
        assert len(self.actuated_joints) == cfgs.size(
            1
        ), f"cfg should have {len(self.actuated_joints)} dof"
        n_cfgs = len(cfgs)
        for i, j in enumerate(self.actuated_joints):
            joint_cfg[j] = cfgs[:, i]

        return joint_cfg, n_cfgs

    def link_fk(self, cfg=None, link=None, links=None, use_names=False):
        raise NotImplementedError("Not implemented")

    def link_fk_batch(self, cfgs=None, use_names=False):
        """Computes the poses of the URDF's links via forward kinematics in a batch.
        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        use_names : bool
            If True, the returned dictionary will have keys that are string
            link names rather than the links themselves.
        Returns
        -------
        fk : dict or (n,4,4) float
            A map from links to a (n,4,4) vector of homogenous transform matrices that
            position the links relative to the base link's frame
        """
        joint_cfgs, n_cfgs = self._process_cfgs(cfgs)

        # Process link set
        link_set = self.links

        # Compute FK mapping each link to a vector of matrices, one matrix per cfg
        fk = OrderedDict()
        for lnk in self._reverse_topo:
            if lnk not in link_set:
                continue
            poses = torch.eye(4, device=self.device).repeat((n_cfgs, 1, 1))
            path = self._paths_to_base[lnk]
            for i in range(len(path) - 1):
                child = path[i]
                parent = path[i + 1]
                joint = self._G.get_edge_data(child, parent)["joint"]

                cfg_vals = None
                if joint.mimic is not None:
                    mimic_joint = self._joint_map[joint.mimic.joint]
                    if mimic_joint in joint_cfgs:
                        cfg_vals = joint_cfgs[mimic_joint]
                        cfg_vals = (
                            joint.mimic.multiplier * cfg_vals + joint.mimic.offset
                        )
                elif joint in joint_cfgs:
                    cfg_vals = joint_cfgs[joint]

                child_poses = joint.get_child_poses(cfg_vals, n_cfgs)
                poses = torch.matmul(child_poses, poses.type_as(child_poses))

                if parent in fk:
                    poses = torch.matmul(fk[parent], poses.type_as(fk[parent]))
                    break
            fk[lnk] = poses

        if use_names:
            return {ell.name: fk[ell] for ell in fk}
        return fk

    def visual_geometry_fk_batch(self, cfgs=None, use_names=False):
        """Computes the poses of the URDF's visual geometries using fk.
        Parameters
        ----------
        cfgs : dict, list of dict, or (n,m), float
            One of the following: (A) a map from joints or joint names to vectors
            of joint configuration values, (B) a list of maps from joints or joint names
            to single configuration values, or (C) a list of ``n`` configuration vectors,
            each of which has a vector with an entry for each actuated joint.
        links : list of str or list of :class:`.Link`
            The links or names of links to perform forward kinematics on.
            Only geometries from these links will be in the returned map.
            If not specified, all links are returned.
        Returns
        -------
        fk : dict
            A map from :class:`Geometry` objects that are part of the visual
            elements of the specified links to the 4x4 homogenous transform
            matrices that position them relative to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs)

        fk = OrderedDict()
        for link in lfk:
            if use_names:
                assert len(link.visuals) <= 1
            for visual in link.visuals:
                if use_names:
                    key = link.name
                else:
                    key = visual.geometry
                fk[key] = torch.matmul(lfk[link], visual.origin.type_as(lfk[link]))
        return fk

    def visual_trimesh_fk_batch(self, cfgs=None):
        """Computes the poses of the URDF's visual trimeshes using fk.

        -------
        fk : dict
            A map from :class:`~trimesh.base.Trimesh` objects that are
            part of the visual geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs)

        fk = OrderedDict()
        for link in lfk:
            for visual in link.visuals:
                for mesh in visual.geometry.meshes:
                    poses = torch.matmul(lfk[link], visual.origin.type_as(lfk[link]))
                    if visual.geometry.mesh is not None:
                        if visual.geometry.mesh.scale is not None:
                            S = torch.eye(4).type_as(lfk[link])
                            S[:3, :3] = torch.diag(visual.geometry.mesh.scale)
                            poses = torch.matmul(poses, S)
                    fk[mesh] = poses
        return fk

    def visual_trimesh_vertices_fk_batch(self, cfgs=None):
        """Computes the poses of the URDF's visual trimeshes using fk.

        -------
        fk : dict
            A map from :class:`~trimesh.base.Trimesh` objects that are
            part of the visual geometry of the specified links to the
            4x4 homogenous transform matrices that position them relative
            to the base link's frame.
        """
        lfk = self.link_fk_batch(cfgs=cfgs)

        fk = []
        for link in lfk:
            for visual in link.visuals:
                for vertices in visual.geometry.mesh.vertices:
                    poses = torch.matmul(lfk[link], visual.origin.type_as(lfk[link]))
                    if visual.geometry.mesh is not None:
                        if visual.geometry.mesh.scale is not None:
                            S = torch.eye(4).type_as(lfk[link])
                            S[:3, :3] = torch.diag(visual.geometry.mesh.scale)
                            poses = torch.matmul(poses, S)
                    fk.append(
                        transform_point_cloud(
                            vertices.type_as(poses)[None, ...].expand(
                                poses.size(0), -1, -1
                            ),
                            poses,
                            in_place=False,
                        )
                    )
        return torch.cat(fk, dim=1)
