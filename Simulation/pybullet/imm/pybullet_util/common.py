#!/usr/bin/env python3

from typing import Tuple, Optional, Iterable, List
import numpy as np
import pybullet as pb

from Simulation.pybullet.imm.pybullet_util.typing_extra import Tuple3, Tuple4, QuaternionT
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient


_JOINT_NAME_INDEX: int = 1
_LINK_NAME_INDEX: int = 12


def get_link_pose(bc: BulletClient, body_id: int, link_id: int,
                  inertial: bool = False, **kwds) -> Tuple[Tuple3[float],
                                                           Tuple4[float]]:
    if link_id < 0:
        kwds.pop('computeForwardKinematics', False)
        return bc.getBasePositionAndOrientation(body_id)
    if inertial:
        # inertial frame (=center of mass)
        return bc.getLinkState(body_id, link_id, **kwds)[0:2]
    else:
        # URDF link frame
        return bc.getLinkState(body_id, link_id, **kwds)[4:6]


def get_relative_transform(
        bc: BulletClient, body_id_a: int, link_a: int, link_b: int,
        body_id_b: Optional[int] = None, **kwds):
    """Get relative transform between two links in a multibody."""
    if body_id_b is None:
        body_id_b = body_id_a

    kwds.setdefault('computeForwardKinematics', True)
    Ta = get_link_pose(bc, body_id_a, link_a, **kwds)  # world_from_a
    Tb = get_link_pose(bc, body_id_b, link_b, **kwds)  # world_from_b
    Tia = bc.invertTransform(Ta[0], Ta[1])  # a_from_world
    return bc.multiplyTransforms(Tia[0], Tia[1], Tb[0], Tb[1])  # a_from_b


def get_name_from_index(body_id: int, sim_id: int,
                        indices: Iterable[int], link: bool = False):
    """Query joint/link names from indices."""
    _INDEX = (_LINK_NAME_INDEX if link else _JOINT_NAME_INDEX)
    names = []
    for i in indices:
        joint_info = pb.getJointInfo(
            body_id, i, physicsClientId=sim_id)
        names.append(joint_info[_INDEX].decode('utf-8'))
    return names


def get_index_from_name(body_id: int, sim_id: int,
                        names: Iterable[str], link: bool = False):
    """Query joint/link indices from names.

    each returned entry is `None` if not found.
    """
    _INDEX = (_LINK_NAME_INDEX if link else _JOINT_NAME_INDEX)
    num_joints = pb.getNumJoints(body_id, physicsClientId=sim_id)
    indices = [None for _ in names]
    for i in range(num_joints):
        joint_info = pb.getJointInfo(
            body_id, i, physicsClientId=sim_id)
        joint_name = joint_info[_INDEX].decode('utf-8')
        if joint_name in names:
            indices[names.index(joint_name)] = i
    return indices


def get_transform_matrix(pos: Tuple3[float], orn: Tuple4[float],
                         dtype=np.float32) -> np.ndarray:
    """xyz coordinates + quaternion -> homogeneous transformation matrix."""
    pos = np.asanyarray(pos)
    orn = np.asanyarray(orn)
    R = np.asanyarray(pb.getMatrixFromQuaternion(orn))
    T = np.zeros(shape=(4, 4), dtype=dtype)
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = pos
    T[3, 3] = 1
    return T


def get_attachments(
        bc: BulletClient, body_id: int) -> List[Tuple[int, int, int, int]]:
    """Get all JOINT_FIXED attachments for a given body."""
    # TODO(ycho): what about JOINT_POINT2POINT?
    # NOTE(ycho): Also, cannot handle attachments
    # physically held by friction.
    n = bc.getNumConstraints()
    uids = [bc.getConstraintUniqueId(i) for i in range(n)]
    out = []
    for uid in uids:
        constraint_info = bc.getConstraintInfo(uid)
        if constraint_info[0] != body_id:
            continue
        if constraint_info[4] != pb.JOINT_FIXED:
            continue
        # NOTE(ycho): parent body, parent link, child body, child link
        out.append(constraint_info[0:4])
    return out


def get_joint_limits(
    bc: BulletClient, body_id: int, joint_ids: Iterable[int]) -> Tuple[
        Tuple[float, ...], Tuple[float, ...]]:
    """Query joint limits as (lo, hi) tuple, each with length same as
    `joint_ids`."""
    joint_limits = []
    for joint_id in joint_ids:
        joint_info = bc.getJointInfo(body_id, joint_id)
        joint_limit = joint_info[8], joint_info[9]
        joint_limits.append(joint_limit)
    joint_limits = np.transpose(joint_limits)  # Dx2 -> 2xD
    return joint_limits


def get_joint_positions(bc: BulletClient, body_id: int,
                        joint_ids: Iterable[int]) -> Tuple[float, ...]:
    joint_states = pb.getJointStates(body_id, joint_ids)
    joint_positions = [js[0] for js in joint_states]
    return joint_positions
