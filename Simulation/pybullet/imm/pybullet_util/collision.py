#!/usr/bin/env python3

import numpy as np
from numpy import ndarray
import numpy.typing as npt
import logging

from collections import defaultdict
from typing import Tuple, Callable, Iterable, Optional, Dict, Any, List, Union
from dataclasses import dataclass
import pybullet as pb
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient

from Simulation.pybullet.imm.pybullet_util.common import (
    get_relative_transform,
    get_link_pose,
    get_name_from_index
)


@dataclass(frozen=True)
class LinkPair:
    body_id_a: int
    link_id_a: int
    body_id_b: int
    link_id_b: int


def is_allowed(allowlist: Iterable[LinkPair], contact: List):
    """Checks whether a given contact is allowed."""
    for C in allowlist:
        if contact[1] != C.body_id_a:
            continue
        if contact[2] != C.body_id_b:
            continue

        if (C.link_id_a is not None):
            if C.link_id_a != contact[3]:
                continue

        if (C.link_id_b is not None):
            if C.link_id_b != contact[4]:
                continue
        # allowed
        return True
    # not allowed
    return False



class SetRobotState:
    def __init__(self, bc: BulletClient,
                 robot_id: int,
                 attachlist: Iterable[LinkPair]):
        self.bc = bc
        self.robot_id = robot_id

        # Lookup relative transforms
        # of attachments.
        self.attach_xfms = {
            C: get_relative_transform(bc,
                                      C.body_id_a,
                                      C.link_id_a,
                                      C.link_id_b,
                                      C.body_id_b,
                                      inertial=False)
            for C in attachlist}

    def __call__(self, value: npt.ArrayLike, debug: bool=False):

        # Set kinematic states.
        self._update_robot(value, debug)
        self._update_attached_bodies()


    def _update_robot(self, value: npt.ArrayLike, debug:bool = False):
        raise NotImplementedError

            
    def _update_attached_bodies(self):
        bc = self.bc
        # Update transforms of attached bodies.
        for C, xfm in self.attach_xfms.items():
            # NOTE(ycho): as long as we're consistent
            # about `inertial` keyword, we're fine.
            pose = get_link_pose(bc, C.body_id_a, C.link_id_a,
                                 inertial=False)
            pose = bc.multiplyTransforms(pose[0], pose[1],
                                         xfm[0], xfm[1])
            bc.resetBasePositionAndOrientation(
                C.body_id_b, pose[0], pose[1])



class SetRobotJointState(SetRobotState):
    def __init__(self, bc: BulletClient,
                 robot_id: int,
                 joint_ids: Iterable[int],
                 attachlist: Iterable[LinkPair]):
        
        super().__init__(bc=bc, robot_id=robot_id, attachlist=attachlist)
        self.joint_ids = joint_ids


    def _update_robot(self, q: npt.ArrayLike, debug: bool=False):
        for i, v in zip(self.joint_ids, q):
            self.bc.resetJointState(self.robot_id, i, v)
        
            

class SetRobotPose(SetRobotState):
    def __init__(self, bc: BulletClient,
                 robot_id: int,
                 attachlist: Iterable[LinkPair]):
        super().__init__(bc=bc, robot_id=robot_id, attachlist=attachlist)


    def _update_robot(self, value: npt.ArrayLike, debug: bool=False):
        # Set robot pose.
        if debug:
            x, y, _ = tuple(value[0])
            self.bc.addUserDebugPoints([(x, y, 0.005)], [(0, 0, 1)], pointSize=3)
        self.bc.resetBasePositionAndOrientation(self.robot_id, value[0], value[1])


class Collision:
    """General collision class.
    """

    def __init__(
            self, bc: BulletClient, robot_id: int,
            allowlist: Iterable[LinkPair] = [],
            attachlist: Iterable[LinkPair] = [],
            touchlist: Iterable[LinkPair] = [],
            tol: Optional[Dict[int, float]] = None,
            touch_tol=0.002):
        self.bc = bc
        self.robot_id = robot_id
        self.attachlist = attachlist
        self.tol = tol
        self.touch_tol = touch_tol
        self.set_state = SetRobotState(bc, robot_id, attachlist)

        # Split by `body_id_a` for convenience.
        self.allowlist = defaultdict(list)
        for C in allowlist:
            self.allowlist[C.body_id_a].append(C)

        self.touchlist = defaultdict(list)
        for C in touchlist:
            self.touchlist[C.body_id_a].append(C)

        # Lookup relative transforms
        # of attachments.
        self.attach_xfms = {
            C: get_relative_transform(bc,
                                      C.body_id_a,
                                      C.link_id_a,
                                      C.link_id_b,
                                      C.body_id_b)
            for C in attachlist}
        
    def _convert(self, value: npt.ArrayLike) -> npt.ArrayLike:
        return value

    def __call__(self, value: npt.ArrayLike, object_uid: int=None, debug: bool=False) -> bool:
        bc = self.bc
        robot_id: int = self.robot_id
        value = np.asarray(value)

        # Set robot state.
        value = self._convert(value)
        self.set_state(value, debug)

        # Perform collision detection.
        bc.performCollisionDetection()

        # Check collisions.
        # We primarily check the robot and the attached bodies.
        bodies = [robot_id] + [C.body_id_b for C in self.attachlist]

        # Configure tolerances.
        if self.tol is None:
            tol = {}
        else:
            tol = self.tol
        for body_id in bodies:
            tol.setdefault(body_id, -5e-5)
        touch_tol = self.touch_tol


        for body in bodies:
            if object_uid is None:
                contacts = bc.getContactPoints(bodyA=body)
            else:
                contacts = bc.getContactPoints(bodyA=body, bodyB=object_uid)

            filtered_contacts = []
            allowlist = self.allowlist.get(body, [])
            touchlist = self.touchlist.get(body, [])
            for contact in contacts:
                if contact[8] >= tol[body]:
                    continue
                if is_allowed(allowlist, contact):
                    continue
                if is_allowed(touchlist, contact) and contact[8] >= touch_tol:
                    continue
                filtered_contacts.append(contact)
            contacts = filtered_contacts

            if len(contacts) > 0:
                msg = ''
                # In case of contact, optionally output debug messages.
                if debug:
                    for pt in contacts:
                        try:
                            names_a = get_name_from_index(
                                pt[1], bc.sim_id, [pt[3]], link=True)
                            names_b = get_name_from_index(
                                pt[2], bc.sim_id, [pt[4]], link=True)
                            msg += F'names_a = {names_a}, names_b = {names_b}\n'
                        except pb.error:
                            msg += F'{pt[1], pt[2], pt[3], pt[4]}\n'
                            continue
                    #logging.debug(msg)
                    print(msg)
                return True, contacts
        return False, contacts



class ContactBasedCollision(Collision):
    """General contact-based collision checker.

    see pb.getContactPoints()
    """

    def __init__(
            self, bc: BulletClient, robot_id: int, joint_ids: Iterable[int],
            allowlist: Iterable[LinkPair],
            attachlist: Iterable[LinkPair],
            touchlist: Iterable[LinkPair] = [],
            joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            tol: Optional[Dict[int, float]] = None,
            touch_tol=0.002):

        self.joint_ids = joint_ids
        self.joint_limits = np.asarray(joint_limits)

        super().__init__(bc=bc, robot_id=robot_id,
                         allowlist=allowlist,
                         attachlist=attachlist,
                         touchlist=touchlist,
                         tol=tol, touch_tol=touch_tol)

        self.set_state = SetRobotJointState(bc, robot_id, joint_ids, attachlist)


    def __call__(self, q: np.ndarray, object_uid: int=None, debug: bool = False) -> bool:

        joint_limits = self.joint_limits
        q = np.asarray(q)

        # Check if exceeding joint limits.
        # NOTE(ssh): joint_to_check ignores fixed joints
        joint_to_check = np.where(joint_limits[0] <= joint_limits[1])
        if (q[joint_to_check] < joint_limits[0, joint_to_check]).any():
            logging.debug(F'Hit lower jlim: {q[joint_to_check]} < {joint_limits[0, joint_to_check]}')
            return True, []
        if (q[joint_to_check] >= joint_limits[1, joint_to_check]).any():
            logging.debug(F'Hit upper jlim: {q[joint_to_check]} >= {joint_limits[1,joint_to_check]}')
            return True, []
        
        return super().__call__(q, object_uid=object_uid, debug=debug)


class NavigationCollision(Collision):
    '''
        Collsion function for Navigation
    '''
    def __init__(self, bc: BulletClient, 
                 robot_id: int, 
                 allowlist: Iterable[LinkPair] = [], 
                 attachlist: Iterable[LinkPair] = [], 
                 touchlist: Iterable[LinkPair] = [], 
                 tol: Union[Dict[int, float], None] = None, 
                 touch_tol=0.002):
        super().__init__(bc, robot_id, allowlist, attachlist, touchlist, tol, touch_tol)
        self.set_state = SetRobotPose(self.bc, self.robot_id, attachlist=attachlist)
    
    def _convert(self, value: npt.ArrayLike):

        x, y, theta = tuple(value)
        z = self.bc.getBasePositionAndOrientation(self.robot_id)[0][2]
        orn = self.bc.getQuaternionFromEuler((0, 0, theta))        
        
        pose = ((x,y,z),orn)

        return pose


# TODO (dlee): remove this?
class GraspAffordance:
    """
    check grasp affordance based on antipodal score, described in eq.(4-6) in MetaGrasp(ICRA 2019).
    """
    def __init__(
            self, bc: BulletClient, robot,
            allowlist: Iterable[LinkPair],
            thresh: float,
            ):
        self.bc = bc
        self.robot = robot
        self.thresh = thresh
        
        # Split by `body_id_a` for convenience.
        self.allowlist = defaultdict(list)
        for C in allowlist:
            self.allowlist[C.body_id_a].append(C)
        
    def __call__(self, obj_id, obstacle_id_list, debug: bool = False) -> bool:
        bc = self.bc
        robot = self.robot
        
        finger_pos_src = self.robot.last_pose[self.robot.joint_index_finger]
        finger_pos_dst = 0.0
        steps = 100 # hardcoded
        delta = (finger_pos_dst - finger_pos_src) / steps
        trajectory = [finger_pos_src + delta * i for i in range(steps)]
        
        contacts_point_left = []
        contacts_normal_left = []
        contacts_point_right = []
        contacts_normal_right = []
        contacts_left_with_obstacle = []
        contacts_right_with_obstacle = []
        for finger_pos in trajectory:
            bc.resetJointState(robot.uid, robot.joint_index_finger, finger_pos)
            target_position = -1.0 * robot.joint_gear_ratio_mimic * np.asarray(finger_pos)
            for i, idx in enumerate(robot.joint_indices_finger_mimic):
                self.bc.resetJointState(robot.uid, idx, target_position[i])

            # Perform collision detection.
            bc.performCollisionDetection()
            if not contacts_point_left:
                contacts_left = bc.getContactPoints(bodyA=robot.uid, bodyB=obj_id, linkIndexA=robot.joint_index_last-1)
                for c in contacts_left:
                    contacts_point_left.append(c[6])
                    contacts_normal_left.append(c[7])
            if not contacts_point_right:
                contacts_right = bc.getContactPoints(bodyA=robot.uid, bodyB=obj_id, linkIndexA=robot.joint_index_last-4)
                for c in contacts_right:
                    contacts_point_right.append(c[6])
                    contacts_normal_right.append(c[7])
            
            if not contacts_left_with_obstacle:
                for obstacle_id in obstacle_id_list:
                    contact = bc.getContactPoints(bodyA=robot.uid, bodyB=obstacle_id, linkIndexA=robot.joint_index_last-1)
                    if contact:
                        contacts_left_with_obstacle.append(contact)
            if not contacts_right_with_obstacle:
                for obstacle_id in obstacle_id_list:
                    contact = bc.getContactPoints(bodyA=robot.uid, bodyB=obstacle_id, linkIndexA=robot.joint_index_last-4)
                    if contact:
                        contacts_right_with_obstacle.append(contact)
            
            if (contacts_point_right and contacts_point_left) or (contacts_left_with_obstacle or contacts_right_with_obstacle):
                break
        
        if contacts_point_left and contacts_point_right:
            contacts_point_left = np.asarray(contacts_point_left).reshape(-1, 3)
            contacts_point_right = np.asarray(contacts_point_right).reshape(-1, 3)
            contacts_normal_left = np.asarray(contacts_normal_left).reshape(-1, 3)
            contacts_normal_right = np.asarray(contacts_normal_right).reshape(-1, 3)
            
            c1 = np.mean(contacts_point_left, axis=0)
            c2 = np.mean(contacts_point_right, axis=0)
            n1 = np.mean(contacts_normal_left, axis=0)
            n2 = np.mean(contacts_normal_right, axis=0)
            
            return self._get_antipodal_score(c1, c2, n1, n2, self.thresh, debug)
        else:
            return False
    
    def _get_antipodal_score(self, c1, c2, n1, n2, thres: float, debug: bool):
        contact_line = (c1 - c2)/np.linalg.norm(c1 - c2)
        
        score_1 = np.arccos(np.abs(np.dot(n1, contact_line)) / np.linalg.norm(n1))
        score_2 = np.arccos(np.abs(np.dot(n2, contact_line)) / np.linalg.norm(n2))
        
        if debug:
            print("Antipodal score:", score_1, score_2)
            self.bc.addUserDebugLine(c1, c2, [1, 0, 0], 10, 20)
            self.bc.addUserDebugLine(c1, c1 + 0.05 * n1, [0, 1, 0], 2, 30)
            self.bc.addUserDebugLine(c2, c2 + 0.05 * n2, [0, 1, 0], 2, 30)
        
        if (score_1 < thres) and (score_2 < thres):
            return True
        else:
            return False