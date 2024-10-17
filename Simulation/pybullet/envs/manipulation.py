import math
import time
import numpy as np
import numpy.typing as npt
import random
import pybullet as p 
from copy import deepcopy

from contextlib import contextmanager
from typing import List, Tuple, Union, Callable
# from Simulation.pybullet.envs.binpick_env_real_objects import BinpickEnvRealObjects
from Simulation.pybullet.envs.shop_env import ShopEnv, ShopObjectEntry
from Simulation.pybullet.envs.robot import *
from Simulation.pybullet.envs.navigation import Navigation

from Simulation.pybullet.imm.pybullet_util.typing_extra import TranslationT, QuaternionT
from Simulation.pybullet.imm.pybullet_util.common import get_joint_positions, get_relative_transform
from Simulation.pybullet.imm.pybullet_util.collision import ContactBasedCollision, LinkPair
from Simulation.pybullet.imm.motion_planners.utils import INF
from Simulation.pybullet.imm.motion_planners.rrt_connect import birrt
from Simulation.pybullet.imm.motion_planners.prm import DegreePRM

from Simulation.pybullet.utils.process_geometry import random_sample_array_from_config


@contextmanager
def imagine(bc: BulletClient) -> int:
    """Temporarily change the simulation, but restore the state afterwards.
    NOTE(ycho): Do not abuse this function! saveState() is quite expensive.
    """
    try:
        state_id = bc.saveState()
        yield state_id
    finally:
        bc.restoreState(state_id)
        bc.removeState(state_id)



def interpolate_trajectory(cur: List, 
                           goal: List, 
                           action_duration: float, 
                           control_dt: float) -> Tuple[npt.NDArray, ...]:
    '''
    This function returns linear-interpolated (dividing straight line)
    trajectory between current and goal pose.
    Acc, Jerk is not considered.
    '''
    # Interpolation steps
    steps = math.ceil(action_duration/control_dt)
    
    # Calculate difference
    delta = [ goal[i] - cur[i] for i in range(len(cur)) ]
    
    # Linear interpolation
    trajectory: Tuple[npt.NDArray, ...] = ([
        np.array([
            cur[j] + ( delta[j] * float(i)/float(steps) ) 
            for j in range(len(cur))
        ])
        for i in range(1, steps+1)
    ])

    return trajectory


def convex_combination(x: npt.ArrayLike, y: npt.ArrayLike, w=0.5):
    return (1-w)*np.array(x) + w*np.array(y)



class Manipulation():
    def __init__(self, bc         : BulletClient,
                    #    env        : Union[BinpickEnvRealObjects, ShopEnv], 
                        env         : ShopEnv,
                       robot      : Robot,
                       config     : dict,
                       planner_type: str="prm"): 
        
        self.config      = config
        self.bc          = bc
        self.env = env
        self.robot       = robot
        
        # Sim params
        sim_params = config["sim_params"]
        self.DEBUG_GET_DATA = self.config['project_params']['debug']['get_data']
        self.delay          = sim_params["delay"]
        self.control_dt     = 1. / sim_params["control_hz"]

        self._set_manipulation_params()

        # Default RRT callbacks
        distance_fn, sample_fn, extend_fn, collsion_fn = self._get_default_functions()
        self.distance_fn          = distance_fn
        self.sample_fn            = sample_fn
        self.extend_fn            = extend_fn
        self.default_collision_fn = collsion_fn

        if planner_type in ["prm", "PRM"]:
            self.planner_type = planner_type
            self.motion_planner = self.__get_prm_planner()
        elif planner_type in ["rrt", "RRT"]:
            self.planner_type = planner_type
            # self.motion_planner = "rrt"   TODO (dlee): get RRT here
        elif planner_type in ["interpolation"]:
            self.planner_type = planner_type

        self.default_prm_verticies = {}
        self.default_prm_edges = []
        self.default_prm_empty_collision_fn = None


    
    def __get_prm_planner(self, target_degree=4, connect_distance=INF):
        motion_planner = DegreePRM(self.distance_fn, self.extend_fn, self.default_collision_fn,
                                   target_degree=target_degree, connect_distance=connect_distance)

        return motion_planner


    def _restore_default_roadmap(self):
            self.motion_planner.vertices = self.default_prm_verticies
            self.motion_planner.edges = self.default_prm_edges

    
    def get_default_roadmap(self, moveable_obj_uids: List[int]):
        samples = [tuple(self.sample_fn()) for _ in range(self.num_samples)]
        self.default_prm_empty_collision_fn = self._define_empty_collision_fn(moveable_obj_uids)
        self.motion_planner.collision_fn = self.default_prm_empty_collision_fn

        self.motion_planner.grow(samples)

        self.default_prm_verticies = self.motion_planner.vertices
        self.default_prm_edges = self.motion_planner.edges

    
    def _set_manipulation_params(self):
        # Manipulation params
        manipulation_params = self.config["manipulation_params"]
        # For numerical IK solver
        self.ik_max_num_iterations = manipulation_params["inverse_kinematics"]["max_num_iterations"]
        self.ik_residual_threshold = manipulation_params["inverse_kinematics"]["residual_threshold"]
        self.rrt_trials            = manipulation_params["rrt_trials"]
        self.sample_space_center     : TranslationT        = manipulation_params["sample_space"]["center"]
        self.sample_space_half_ranges: TranslationT        = manipulation_params["sample_space"]["half_ranges"]
        self.sample_space_yaw_range  : Tuple[float, float] = manipulation_params["sample_space"]["yaw_range"]
        self.delay                 = manipulation_params["delay"]
        self.resolutions           = manipulation_params["resolutions"]
        self.num_samples           = manipulation_params["num_samples"]


    def _get_default_functions(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Init default sample, extend, distance, collision function

        Returns:
            distance_fn
            sample_fn
            extend_fn
            collision_fn
        """
        def distance_fn(q0: np.ndarray, q1: np.ndarray):
            return np.linalg.norm(np.subtract(q1, q0))
        
        # def sample_fn():
        #     return np.random.uniform(self.robot.joint_limits[0]/2.0, self.robot.joint_limits[1]/2.0)
        def sample_fn(debug=False, return_xyz=False):
            """RRT sampling function in cartesian space"""
            pos = random_sample_array_from_config(self.sample_space_center, self.sample_space_half_ranges)
            orn_e_toppick = [-3.1416, 0, -1.57]
            orn_e_yaw_rot = [0, 0, random.uniform(*self.sample_space_yaw_range)]
            _, orn_q = self.bc.multiplyTransforms(
                [0, 0, 0], self.bc.getQuaternionFromEuler(orn_e_yaw_rot),
                [0, 0, 0], self.bc.getQuaternionFromEuler(orn_e_toppick),)
            _, dst = self.solve_ik_numerical(pos, orn_q)
            if debug:
                self.bc.addUserDebugPoints([pos], [(1, 0, 0)], pointSize=3)
            if return_xyz:
                return pos, dst
            else:
                return dst
        
        def extend_fn(q0: np.ndarray, q1: np.ndarray, num_ext=2):
            dq = np.subtract(q1, q0)  # Nx6
            return q0 + np.linspace(0, 1, num=150)[:, None] * dq
        
        touch_pair_robot_a = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=None,
            body_id_b=self.env.cabinet_uid,
            link_id_b=None)
        touch_pair_robot_b = LinkPair(
            body_id_a=self.env.cabinet_uid,
            link_id_a=None,
            body_id_b=self.robot.uid,
            link_id_b=None)
        default_collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = list(range(self.robot.joint_index_last+1)),
            attachlist   = [],
            allowlist    = [],
            touchlist    = [touch_pair_robot_a, touch_pair_robot_b],
            joint_limits = self.robot.joint_limits,
            tol          = {},
            touch_tol    = 0.005)


        return distance_fn, sample_fn, extend_fn, default_collision_fn


    def move(self, traj: List[npt.NDArray]):
        self.__move(traj)

    def __move(self, traj: List[npt.NDArray]):
        """Move

        Args:
            traj (List[npt.NDArray])
        """
        # Execute the trajectory
        for joint_pos in traj:
            # Control
            self.robot.update_arm_control(joint_pos)
            self.bc.stepSimulation()
            
            if self.delay:
                time.sleep(self.delay)
            
        # Wait until control completes
        self.wait(steps=240)
        

    def open(self, pos: TranslationT, 
                   orn_q: QuaternionT):
        """_summary_

        Args:
            pos (TranslationT): _description_
            orn_q (QuaternionT): _description_
        """        
        # Wait until control completes
        self.wait(steps=500)

        self.robot.release()

        # Pull ee back        
        modified_pos_back, modified_orn_q_back, modified_pos, modified_orn_q = self.get_ee_pose_from_target_pose(pos, orn_q)
        joint_pos_src, joint_pos_dst = self.solve_ik_numerical(modified_pos_back, modified_orn_q_back)

        traj = interpolate_trajectory(list(joint_pos_src), list(joint_pos_dst), 0.2, self.control_dt)
        for joint_pos in traj:
            # Control
            self.robot.update_arm_control(joint_pos)
            self.bc.stepSimulation()
            
            if self.delay:
                time.sleep(self.delay)

    
    def close(self, obj_uid: int, 
                    pos: TranslationT, 
                    orn_q: QuaternionT) -> Union[int, None]:
        """
        Args:
            obj_uid: uid of target object
            pos: position when grasping occurs
            orn_q: orientation when grasping occurs

        Returns:
            Union[int, None]: Returns obj_uid when grasp succeeded. None when failed.
        """
        # Approach for poke
        modified_pos_back, modified_orn_q_back, \
            modified_pos, modified_orn_q \
            = self.get_ee_pose_from_target_pose(pos, orn_q)
        joint_pos_src, joint_pos_dst = self.solve_ik_numerical(modified_pos, modified_orn_q)

        traj = interpolate_trajectory(list(joint_pos_src), list(joint_pos_dst), 0.2, self.control_dt)
        for joint_pos in traj:
            # Control
            self.robot.update_arm_control(joint_pos)
            self.bc.stepSimulation()
            
            if self.delay:
                time.sleep(self.delay)
                
        # Wait until control completes
        self.wait(steps=500)
                
        # Check contact after approaching
        if self.robot.detect_contact():
            grasp_obj_uid = self.robot.activate(self.env.object_uids)
        else:
            grasp_obj_uid = None

        return grasp_obj_uid
    

    def pick(self, obj_uid: int, 
                   pos: TranslationT, 
                   orn_q: QuaternionT, 
                   traj = None) -> Union[int, None]:
        """
        Implementation of the pick action
        
        Args:
            obj_uid: Target object id
            pos: Postion when PICK occurs
            orn_q: Orientation when PICK occurs
            traj: Collision-free trajectory for pose to reach before picking (backward as much as self.robot.poke_backward)

        Returns:
            holding_obj_uid (Union(int, None)): Return the uid of grasped object.
        """
        # Perform move
        self.__move(traj)
        
        # Close the gripper
        holding_obj_uid = self.close(obj_uid, pos, orn_q)

        if holding_obj_uid is None:
            # NOTE(ssh): Release again if grasp failed.
            self.open(pos, orn_q)
            return None
        else:
            return holding_obj_uid


    def place(self, obj_uid: int, 
                    pos: TranslationT, 
                    orn_q: QuaternionT, 
                    traj: List[npt.NDArray] = None) -> Union[int, None]:
        """
        Implementation of the place action

        Args:
            obj_uid (int): Target object uid
            pos (TranslationT): End effector position
            orn_q (QuaternionT): End effector orientation
            traj (List[npt.NDArray]): Collision-free trajectory for PLACE pose
        """
        # Perform move
        self.__move(traj)
        # Open the gripper
        self.open(pos, orn_q)
        holding_obj_uid = None  # No object is being held.
        # Wait until the control completes
        self.wait(steps=500)
        
        return holding_obj_uid
    
    
    def wait(self, steps=-1):
        """General wait function for hold or stabilization.

        Args:
            steps (int): Defaults to -1.
        """
        while steps != 0: 
            self.robot.update_arm_control()
            self.bc.stepSimulation()
            if self.delay:
                time.sleep(self.delay)
            steps -= 1
    

    def solve_ik_numerical(self, pos: TranslationT, 
                                 orn_q: QuaternionT) -> Tuple[ npt.NDArray, npt.NDArray ]:
        '''
        Solve the inverse kinematics of the robot given the finger center position.
        
        Args:
        - pos (TranslationT) : R^3 position
        - orn_q (QuaternionT): Quaternion
        
        Returns
        - joint_pos_src (npt.NDArray): Source position in buffer format
        - joint_pos_dst (npt.NDArray): Destination position in buffer format

        NOTE(ssh):
            Finger joint value will be preserved in destination position.
            The indices of the value match with `rest_pose`
            Return type is numpy because it requires advanced indexing. 
            Make sure to cast for other hashing.

        '''
        
        # Get current joint state first.
        joint_pos_src = np.array(get_joint_positions( self.bc, 
                                                self.robot.uid,
                                                range(0, self.robot.joint_index_last + 1) ))

        # Reset joint to the rest pose. This will increase the ik stability
        for i, v in enumerate(self.robot.rest_pose):
            self.bc.resetJointState(self.robot.uid, i, v)

        # IK solve
        ik = self.bc.calculateInverseKinematics(
            bodyUniqueId         = self.robot.uid, 
            endEffectorLinkIndex = self.robot.link_index_endeffector_base, 
            targetPosition       = pos, 
            targetOrientation    = orn_q,
            lowerLimits          = self.robot.joint_limits[0],     # upper
            upperLimits          = self.robot.joint_limits[1],     # lower
            jointRanges          = self.robot.joint_range,
            restPoses            = self.robot.rest_pose[self.robot.joint_indices_arm],
            maxNumIterations     = self.ik_max_num_iterations,
            residualThreshold    = self.ik_residual_threshold)

        # Reset back to src pose.
        for i, v in enumerate(joint_pos_src):
            self.bc.resetJointState(self.robot.uid, i, v)

        # Set source and destination position of motion planner (entire joint)
        # NOTE(ssh): This is awkward... but necessary as ik index ignores fixed joints.
        joint_pos_dst = np.copy(joint_pos_src)
        joint_pos_dst[self.robot.joint_indices_arm] = np.array(ik)[ : len(self.robot.joint_indices_arm)+1]
        
        # Normalize wrist angle limit
        #   Use -1.5 pi ~ 0.5 pi range for this problem.
        if (not (joint_pos_dst[self.robot.joint_index_last] > -1.5*np.pi and
                 joint_pos_dst[self.robot.joint_index_last] < 0.5*np.pi )):
            joint_pos_dst[self.robot.joint_index_last] \
                = np.arctan2(
                    np.sin(joint_pos_dst[self.robot.joint_index_last]),
                    np.cos(joint_pos_dst[self.robot.joint_index_last]))
            if joint_pos_dst[self.robot.joint_index_last] > 0.5 * np.pi:
                joint_pos_dst[self.robot.joint_index_last] -= 2*np.pi

        return joint_pos_src, joint_pos_dst
    

    def _get_attach_pair(self, grasp_uid: Union[int, List[int]]) -> List[LinkPair]:
        # Attached object will be moved together when searching the path
        attachlist = []
        if isinstance(grasp_uid, int):
            attach_pair = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=self.robot.link_index_endeffector_base,
                body_id_b=grasp_uid,
                link_id_b=-1)
            
            attachlist.append(attach_pair)
        elif isinstance(grasp_uid, list):
            if isinstance(self.robot, PR2):
                for i, uid in enumerate(grasp_uid):
                    if uid is None:
                        continue
                    if i == 0:
                        arm = self.robot.main_hand
                    else:
                        arm = self.robot.get_other_arm()
                    attach_pair = LinkPair(
                    body_id_a=self.robot.uid,
                    link_id_a=self.robot.gripper_link_indices[arm][0],
                    body_id_b=uid,
                    link_id_b=-1)
                
                    attachlist.append(attach_pair)
            else:
                raise NotImplementedError("Grasping multiple object is for PR2 only")
        
        return attachlist
    

    def _get_allow_pair(self, grasp_uid: Union[int, List[int]]) -> List[LinkPair]:
        # Allow pair is not commutative
        allowlist = []
        if isinstance(grasp_uid, int):
            allow_pair_grasp_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=grasp_uid,
                link_id_b=None)
            allow_pair_grasp_b = LinkPair(
                body_id_a=grasp_uid,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
        
            allowlist.append(allow_pair_grasp_a)
            allowlist.append(allow_pair_grasp_b)
        elif isinstance(grasp_uid, list):
            if isinstance(self.robot, PR2):
                for i, uid in enumerate(grasp_uid):
                    if uid is None:
                        continue
                    allow_pair_grasp_a = LinkPair(
                        body_id_a=self.robot.uid,
                        link_id_a=None,
                        body_id_b=uid,
                        link_id_b=None)
                    allow_pair_grasp_b = LinkPair(
                        body_id_a=uid,
                        link_id_a=None,
                        body_id_b=self.robot.uid,
                        link_id_b=None)
                    
                    allowlist.append(allow_pair_grasp_a)
                    allowlist.append(allow_pair_grasp_b)
            else:
                raise NotImplementedError("Grasping multiple object is for PR2 only")
        
        return allowlist
    

    def _get_interobject_allow_pair_list(self, allow_uid_list: List[int])->List[LinkPair]:
        # Optional allow pair
        interobject_allow_pair_list = []
        for allow_uid_a in allow_uid_list:
            for allow_uid_b in allow_uid_list:
                if allow_uid_a == allow_uid_b:
                    continue
                allow_pair_ab = LinkPair(   # ba will be accessed within double loop. dont worry.
                    body_id_a=allow_uid_a,
                    link_id_a=None,
                    body_id_b=allow_uid_b,
                    link_id_b=None)
                interobject_allow_pair_list.append(allow_pair_ab)
            allow_pair_a_robot = LinkPair(
                body_id_a=allow_uid_a,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            allow_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=allow_uid_a,
                link_id_b=None)
            interobject_allow_pair_list.append(allow_pair_a_robot)
            interobject_allow_pair_list.append(allow_pair_robot_a)

        return interobject_allow_pair_list
    

    def _get_touch_pair(self, grasp_uid: int)->Tuple[LinkPair]:
        # Touch pair is not commutative. One should define bidirectional pair.
        # NOTE(ssh): allow touch between the object and cabinet, but not the penetration.
        touch_pair_objcab_a = LinkPair(
            body_id_a=self.env.cabinet_uid, 
            link_id_a=None,
            body_id_b=grasp_uid,
            link_id_b=None)
        touch_pair_objcab_b = LinkPair(
            body_id_a=grasp_uid,
            link_id_a=None,
            body_id_b=self.env.cabinet_uid,
            link_id_b=None)
        touch_pair_objgoal_a = LinkPair(
            body_id_a=self.env.goal_table,
            link_id_a=None,
            body_id_b=grasp_uid,
            link_id_b=None)
        touch_pair_objgoal_b = LinkPair(
            body_id_a=grasp_uid,
            link_id_a=None,
            body_id_b=self.env.goal_table,
            link_id_b=None)
        # NOTE(ssh): allow touch between the robot and cabinet, but not the penetration.
        touch_pair_robot_a = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=None,
            body_id_b=self.env.cabinet_uid,
            link_id_b=None)
        touch_pair_robot_b = LinkPair(
            body_id_a=self.env.cabinet_uid,
            link_id_a=None,
            body_id_b=self.robot.uid,
            link_id_b=None)
        
        return touch_pair_objcab_a, touch_pair_objcab_b, touch_pair_objgoal_a, \
            touch_pair_objgoal_b, touch_pair_robot_a, touch_pair_robot_b
    

    def _get_default_interobject_touch_pair_list(self)->List[LinkPair]:
        # NOTE(ssh): Let's also allow touch between the objects defaultly.
        default_interobject_touch_pair_list = []
        for touch_uid_a in self.env.object_uids:
            for touch_uid_b in self.env.object_uids:
                if touch_uid_a == touch_uid_b:
                    continue
                touch_pair_ab = LinkPair(   # ba will be accessed within double loop. dont worry.
                    body_id_a=touch_uid_a,
                    link_id_a=None,
                    body_id_b=touch_uid_b,
                    link_id_b=None)
                default_interobject_touch_pair_list.append(touch_pair_ab)
            touch_pair_a_robot = LinkPair(
                body_id_a=touch_uid_a,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            touch_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=touch_uid_a,
                link_id_b=None)
            default_interobject_touch_pair_list.append(touch_pair_a_robot)
            default_interobject_touch_pair_list.append(touch_pair_robot_a)

        return default_interobject_touch_pair_list


    def _get_joint_ids(self):
        return list(range(self.robot.joint_index_last+1))
    

    def _get_joint_limits(self):
        return self.robot.joint_limits


    def _define_grasp_collision_fn(self, grasp_uid: Union[int, List[int]], 
                                         allow_uid_list: List[int], 
                                         debug=False) -> ContactBasedCollision:
        """ This function returns the collision function that allows the grasped object to
        1. collide with the robot body
        2. be attached at robot finger
        3. touch with the cabinet.
        
        Args:
            grasp_uid: This object will be move together with the end effector.
            allow_uid_list: Uids of object to allow the contact.
        """
        
        attachlist: List[LinkPair] = self._get_attach_pair(grasp_uid)
        allowlist: List[LinkPair] = self._get_allow_pair(grasp_uid)
        interobject_allow_pair_list: List[LinkPair] = self._get_interobject_allow_pair_list(allow_uid_list)

        
        # Compose collision fn
        collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = self._get_joint_ids(),
            allowlist    = allowlist + interobject_allow_pair_list,
            attachlist   = attachlist,
            touchlist = [],
            joint_limits = self._get_joint_limits(),
            tol          = {},
            touch_tol    = 0.005)   #TODO: config
        # Debug
        if debug:
            contact = np.array(self.bc.getContactPoints(bodyA=grasp_uid, bodyB=self.env.cabinet_uid), dtype=object)
            for c in contact:
                print(f"Cabinet contact point: {c[8]}")
            print(f"min: {np.min(contact[:,8])}")
            
            for uid in self.env.object_uids:
                contact2 = np.array(self.bc.getContactPoints(bodyA=grasp_uid, bodyB=uid), dtype=object)
                for c in contact2:
                    print(f"Object {uid} contact: {c[8]}")

        return collision_fn
    
    
    def _define_empty_collision_fn(self, allow_uid_list: List[int]) -> ContactBasedCollision:
        """ This function returns the collision function that allows the grasped object to
        1. collide with the robot body
        2. be attached at robot finger
        3. touch with the cabinet.
        
        Args:
            allow_uid_list: Uids of object to allow the contact.
        """
        # Optional allow pair
        interobject_allow_pair_list = []
        for allow_uid_a in allow_uid_list:
            for allow_uid_b in allow_uid_list:
                if allow_uid_a == allow_uid_b:
                    continue
                allow_pair_ab = LinkPair(   # ba will be accessed within double loop. dont worry.
                    body_id_a=allow_uid_a,
                    link_id_a=None,
                    body_id_b=allow_uid_b,
                    link_id_b=None)
                interobject_allow_pair_list.append(allow_pair_ab)
            allow_pair_a_robot = LinkPair(
                body_id_a=allow_uid_a,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            allow_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=allow_uid_a,
                link_id_b=None)
            interobject_allow_pair_list.append(allow_pair_a_robot)
            interobject_allow_pair_list.append(allow_pair_robot_a)
        # NOTE(ssh): allow touch between the robot and cabinet, but not the penetration.
        touch_pair_robot_a = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=None,
            body_id_b=self.env.cabinet_uid,
            link_id_b=None)
        touch_pair_robot_b = LinkPair(
            body_id_a=self.env.cabinet_uid,
            link_id_a=None,
            body_id_b=self.robot.uid,
            link_id_b=None)
        # NOTE(ssh): Let's also allow touch between the objects defaultly.
        default_interobject_touch_pair_list = []
        for touch_uid_a in self.env.object_uids:
            for touch_uid_b in self.env.object_uids:
                if touch_uid_a == touch_uid_b:
                    continue
                touch_pair_ab = LinkPair(   # ba will be accessed within double loop. dont worry.
                    body_id_a=touch_uid_a,
                    link_id_a=None,
                    body_id_b=touch_uid_b,
                    link_id_b=None)
                default_interobject_touch_pair_list.append(touch_pair_ab)
            touch_pair_a_robot = LinkPair(
                body_id_a=touch_uid_a,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            touch_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=touch_uid_a,
                link_id_b=None)
            default_interobject_touch_pair_list.append(touch_pair_a_robot)
            default_interobject_touch_pair_list.append(touch_pair_robot_a)
        
        # Compose collision fn
        collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = list(range(self.robot.joint_index_last+1)),
            allowlist    = [*interobject_allow_pair_list],
            # allowlist = [allow_pair_grasp_a, allow_pair_grasp_b],
            attachlist   = [],
            # touchlist    = [touch_pair_objcab_a, touch_pair_objcab_b, 
            #                 touch_pair_objgoal_a, touch_pair_objgoal_b,
            #                 touch_pair_robot_a, touch_pair_robot_b,
            #                 *default_interobject_touch_pair_list],
            touchlist = [],
            joint_limits = self.robot.joint_limits,
            tol          = {},
            touch_tol    = 0.005)   #TODO: config
        

        return collision_fn


    def get_ee_pose_from_target_pose(self, pos: TranslationT, orn_q: QuaternionT, clip_height: Union[float, None] = 0.91):
        """
        Adjust the target pose to the end-effector pose.
        It reflects adjustment as much as poke backward.

        Args:
            pos (TranslationT): Target position
            orn_q (QuaternionT): Target orientation, outward orientation (surface normal)
            clip_height (float): backward pose do not go higher than this value.
        
        Returns:
            modified_pos_back (TranslationT): Position to reach before picking (backward as much as self.robot.poke_backward, only for PICK)
            modified_orn_q_back (QuaternionT): Orientation to reach before picking (only for PICK)
            modified_pos (TranslationT)
            modified_orn_q (QuaternionT): Inward orientation (ee base)
        """
        
        orn = self.bc.getEulerFromQuaternion(orn_q)
        ee_base_link_target_pos, ee_base_link_target_orn_e = self.robot.get_target_ee_pose_from_se3(pos, orn)
        ee_base_link_target_orn_q = self.bc.getQuaternionFromEuler(ee_base_link_target_orn_e)

        modified_pos = ee_base_link_target_pos
        modified_orn_q = ee_base_link_target_orn_q
        
        poke_backward = self.robot.grasp_poke_backward
        ee_base_link_backward_pos, ee_base_link_backward_orn_q \
            = self.bc.multiplyTransforms(ee_base_link_target_pos, ee_base_link_target_orn_q,
                                            [0, 0, -poke_backward], [0, 0, 0, 1])
        # Clip... domain specific function.
        if clip_height is not None and ee_base_link_backward_pos[2] > clip_height:
            ee_base_link_backward_pos = list(ee_base_link_backward_pos)
            ee_base_link_backward_pos[2] = clip_height
            ee_base_link_backward_pos = tuple(ee_base_link_backward_pos)
        
        modified_pos_back = ee_base_link_backward_pos
        modified_orn_q_back = ee_base_link_backward_orn_q
        
        return modified_pos_back, modified_orn_q_back, modified_pos, modified_orn_q
    

    def motion_plan(self, joint_pos_src: npt.NDArray, 
                          joint_pos_dst: npt.NDArray, 
                          holding_obj_uid: Union[int, None], 
                          allow_uid_list: List[int] = [],
                          planner_type: str=None) -> Union[List[npt.NDArray], None]:
        """Get a motion plan trajectory.

        Args:
            joint_pos_src (npt.NDArray): Source joint position
            joint_pos_dst (npt.NDArray): Destination joint position
            holding_obj_uid (Union[int, None]): Uid of the holding object.
            allow_uid_list (List[int]): asdf
            use_interpolation (str, Optional): Flag for force motion planning method
        Returns:
            Union[List[npt.NDArray], None]: Generated trajectory. None when no feasible trajectory is found.
        """
        if planner_type is None:
            planner_type = self.planner_type

        # Compute motion plan
        if planner_type in ["interpolation"]:
            trajectory = interpolate_trajectory(
                cur             = joint_pos_src, 
                goal            = joint_pos_dst, 
                action_duration = 1/48, 
                control_dt      = self.control_dt)
            contacts = []
        elif planner_type in ["RRT", "rrt", "birrt"]:
            # Try RRT
            with imagine(self.bc):
                # Moving with grapsed object
                if holding_obj_uid is not None:
                    collision_fn = self._define_grasp_collision_fn(holding_obj_uid, allow_uid_list, debug=False)
                    # Get RRT path using constraints
                    trajectory, contacts = birrt(
                        joint_pos_src,
                        joint_pos_dst,
                        self.distance_fn,
                        self.sample_fn,
                        self.extend_fn,
                        collision_fn,
                        max_solutions=1,
                        restarts=self.rrt_trials)
                # Moving without grasped object
                else:
                    # Get RRT path using default constraints
                    trajectory, contacts = birrt(
                        joint_pos_src,
                        joint_pos_dst,
                        self.distance_fn,
                        self.sample_fn,
                        self.extend_fn,
                        self.default_collision_fn,
                        max_solutions=1,
                        restarts=self.rrt_trials)
        elif planner_type in ["PRM", "prm"]:
            with imagine(self.bc):
                # Moving with grapsed object
                start = tuple(joint_pos_src)
                goal = tuple(joint_pos_dst)
                # samples = [start, goal] + [tuple(self.sample_fn()) for _ in range(self.num_samples)]
                samples = [start, goal]

                if holding_obj_uid is not None:
                    collision_fn = self._define_grasp_collision_fn(holding_obj_uid, allow_uid_list, debug=False)
                    self.motion_planner.collision_fn = collision_fn
                    self.motion_planner.grow(samples=samples)


                # Moving without grasped object
                else:
                    # Get RRT path using default constraints
                    self.motion_planner.collision_fn = self.default_collision_fn
                    self.motion_planner.grow(samples=samples)
                
            trajectory = self.motion_planner(start, goal)

            # NOTE (dlee): This is because of PRM which requires the input to be in tuple for hashing.
            self.motion_planner[start].clear()
            self.motion_planner[goal].clear()

            if trajectory is not None:
                trajectory = np.array(trajectory)
            contacts = []
        else:
            raise ValueError("Mode should be interpolation, rrt, or prm")
                             

        return trajectory, contacts
    


class PR2SingleArmManipulation(Manipulation):
    
    def __init__(self, bc: BulletClient, 
                 env: ShopEnv, 
                 robot: PR2,
                 nav: Navigation, 
                 config: dict,
                 arm: str):
        assert arm in ["left", "right"], "'arm' should either be left or right"
        self.arm = arm
        self.arm_joints = self._get_controlable_joints(robot)
        self.nav = nav

        super().__init__(bc, env, robot, config, planner_type="interpolation")

        self.circular = [self._is_circular(joint) for joint in self.arm_joints]

        # Define empty collisfion function
        self.empty_collision_fn = self._define_empty_collision_fn([v.uid for v in env.movable_obj.values()])

        # receptacle
        self.receptacle_status: Dict[str, AttachConstraint] = dict()
        self.receptacle_holding_info: Tuple[Tuple, Tuple] = None

    # Initialization helpers
    def _get_controlable_joints(self, robot: PR2):
        
        if self.arm == "left":
            joints = robot.torso_joints + robot.left_arm_joints
        else:
            joints = robot.torso_joints + robot.right_arm_joints

        return joints


    def _is_circular(self, joint):
        joint_info = self.robot.get_joint_info(joint)
        if joint_info.jointType == p.JOINT_FIXED:
            return False
        return joint_info.jointUpperLimit < joint_info.jointLowerLimit
    

    def _get_default_functions(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Init default sample, extend, distance, collision function

        Returns:
            distance_fn
            sample_fn
            extend_fn
            collision_fn
        """
        def difference_fn(q0: npt.ArrayLike, q1: npt.ArrayLike):            
            diff = []
            for circular, value0, value1 in zip(self.circular, q0, q1):
                if circular:
                    angle_diff = (value1 - value0) % (2*np.pi) - np.pi
                    diff.append(angle_diff)
                else:
                    d_diff = value1 - value0
                    diff.append(d_diff)

            return tuple(diff)

        def distance_fn(q0: npt.ArrayLike, q1: npt.ArrayLike, weights=None):
            weights = 1*np.ones(len(q0))
            difference = np.array(difference_fn(q0, q1))
            return np.sqrt(np.dot(weights, difference * difference))
        

        def sample_fn():
            """node sampling function in cartesian space"""

            lower_limits, upper_limits = self.robot.get_joint_intervals(self.arm_joints)
            weights = np.random.uniform(size=len(self.arm_joints))

            assert np.less_equal(lower_limits, upper_limits).all()
            if np.equal(lower_limits, upper_limits).all():
                return lower_limits

            return convex_combination(lower_limits, upper_limits, w=weights)
        
        def extend_fn(q1, q2):
            dq = difference_fn(q1, q2)
            n = int(np.max(np.abs(dq))/self.resolutions)+1

            return q1 + np.linspace(0, 1, num=n)[:, None] * dq
        

        default_collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = list(self.arm_joints),
            attachlist   = [],
            allowlist    = [],
            touchlist    = [],
            joint_limits = self.robot.get_joint_intervals(self.arm_joints),
            tol          = {},
            touch_tol    = 0.005)


        return distance_fn, sample_fn, extend_fn, default_collision_fn


    def _define_empty_collision_fn(self, allow_uid_list: List[int]) -> ContactBasedCollision:
        """ This function returns the collision function that allows the grasped object to
        1. collide with the robot body
        2. be attached at robot finger
        3. touch with the cabinet.
        
        Args:
            allow_uid_list: Uids of object to allow the contact.
        """
        # Optional allow pair
        interobject_allow_pair_list = []
        for allow_uid_a in allow_uid_list:
            for allow_uid_b in allow_uid_list:
                if allow_uid_a == allow_uid_b:
                    continue
                allow_pair_ab = LinkPair(   # ba will be accessed within double loop. dont worry.
                    body_id_a=allow_uid_a,
                    link_id_a=None,
                    body_id_b=allow_uid_b,
                    link_id_b=None)
                interobject_allow_pair_list.append(allow_pair_ab)
            allow_pair_a_robot = LinkPair(
                body_id_a=allow_uid_a,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            allow_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=allow_uid_a,
                link_id_b=None)
            interobject_allow_pair_list.append(allow_pair_a_robot)
            interobject_allow_pair_list.append(allow_pair_robot_a)
        
        # NOTE(ssh): Let's also allow touch between the objects defaultly.
        default_interobject_touch_pair_list = []
        for touch_uid_a in self.env.uid_to_name.keys():
            for touch_uid_b in self.env.uid_to_name.keys():
                if touch_uid_a == touch_uid_b:
                    continue
                touch_pair_ab = LinkPair(   # ba will be accessed within double loop. dont worry.
                    body_id_a=touch_uid_a,
                    link_id_a=None,
                    body_id_b=touch_uid_b,
                    link_id_b=None)
                default_interobject_touch_pair_list.append(touch_pair_ab)
            touch_pair_a_robot = LinkPair(
                body_id_a=touch_uid_a,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            touch_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=touch_uid_a,
                link_id_b=None)
            default_interobject_touch_pair_list.append(touch_pair_a_robot)
            default_interobject_touch_pair_list.append(touch_pair_robot_a)
        
        # Compose collision fn
        collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = list(self.arm_joints),
            allowlist    = [*interobject_allow_pair_list],
            attachlist   = [],
            touchlist = [],
            joint_limits = self.robot.get_joint_intervals(self.arm_joints),
            tol          = {},
            touch_tol    = 0.005)
        

        return collision_fn

    # Interaction helpers
    def _get_allow_pair(self, grasp_uid: Union[int, LinkPair])-> List[LinkPair]:
        self.robot: PR2
        # Attached object will be moved together when searching the path
        allowlist = super()._get_allow_pair(grasp_uid)
        if self.robot.is_holding_receptacle():
            receptacle_uid = self.robot.activated[self.robot.get_other_arm()].uid
            for name, (receptacle_obj_info, _, _) in self.robot.receptacle_status.items():
                allow_pair_grasp_a = LinkPair(
                    body_id_a=receptacle_uid,
                    link_id_a=None,
                    body_id_b=receptacle_obj_info.uid,
                    link_id_b=None)
                allow_pair_grasp_b = LinkPair(
                    body_id_a=receptacle_obj_info.uid,
                    link_id_a=None,
                    body_id_b=receptacle_uid,
                    link_id_b=None)
                
                allowlist.append(allow_pair_grasp_a)
                allowlist.append(allow_pair_grasp_b)

        return allowlist
    

    def _get_joint_ids(self):
        return list(self.arm_joints)
    

    def _get_joint_limits(self):
        return self.robot.get_joint_intervals(self.arm_joints)


    def _check_closeness(self, 
                         pose1: Tuple[Tuple[float]], 
                         pose2: Tuple[Tuple[float]],
                         pos_thresh: float=0.01,
                         orn_thresh: float=0.02) -> bool:
        # Check XYZ
        linear_distance = np.linalg.norm((np.array(pose1[0][:2])-np.array(pose2[0][:2])))

        # Check angle
        orn1_q = pose1[1]
        orn2_q = pose2[1]
        orn_diff_q = self.bc.getDifferenceQuaternion(orn1_q, orn2_q)
        orn_diff = self.bc.getEulerFromQuaternion(orn_diff_q)

        return (linear_distance < pos_thresh) and np.all(np.array(orn_diff) < orn_thresh)


    def get_object_height(self, object_info: ShopObjectEntry):
        uid = object_info.uid
        if object_info.is_movable:
            try:
                aabb = self.bc.getAABB(uid, 0)
            except Exception as e:
                aabb = self.bc.getAABB(uid)
        else:
            aabb = self.bc.getAABB(uid)

        return aabb[1][2] - aabb[0][2]


    def get_ee_pose_from_target_pose(self, target_pos: TranslationT, 
                                     target_orn_q: QuaternionT):
        """
        Adjust the target pose to the end-effector pose.
        It reflects adjustment as much as poke backward.

        Args:
            pos (TranslationT): Target position
            orn_q (QuaternionT): Target orientation, outward orientation (surface normal)
            clip_height (float): backward pose do not go higher than this value.
        
        Returns:
            modified_pos_back (TranslationT): Position to reach before picking (backward as much as self.robot.poke_backward, only for PICK)
            modified_orn_q_back (QuaternionT): Orientation to reach before picking (only for PICK)
            modified_pos (TranslationT)
            modified_orn_q (QuaternionT): Inward orientation (ee base)
        """

        # Target_pos is already in world coordinate
        # Target_orn is in robot-base frame
        # NOTE (dlee): fix the gripper to be horizontal for now
        # tool_orn = (0, 0, math.pi/6) if self.arm == "right" else (0, 0, -math.pi/6)
        tool_orn = (0, 0, 0)
        tool_orn_q = self.bc.getQuaternionFromEuler(tool_orn)

        tool_offset = (0.03, 0, 0)

        # Transform tool pose into world-frame
        _, base_orn = self.robot.get_pose()

        target_pos_world, target_orn_q_world \
            = self.bc.multiplyTransforms(target_pos, tool_orn_q, tool_offset, base_orn)

        return (target_pos_world, target_orn_q_world)
    

    def get_joint_and_ee_pose_from_target_pose(self, target_pos: TranslationT, target_orn_q: QuaternionT):
        ee_pose = self.get_ee_pose_from_target_pose((target_pos, target_orn_q))
        q = self.robot.arm_ik(self.arm, ee_pose, debug=False)

        return q, ee_pose


    def get_handle_pose(self, object_uid: int, 
                        handle_action_info: Tuple[int, npt.ArrayLike, npt.ArrayLike]) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        
        _, link_id, handle_local_pos_candidates, handle_local_orn_candidates, \
            open_direction_candidates, _, _ = handle_action_info[0]

        idx = handle_action_info[1] 
        
        handle_local_pos = handle_local_pos_candidates[idx]
        handle_local_orn = handle_local_orn_candidates[idx]

        frame_pos, frame_orn = self.bc.getLinkState(object_uid, link_id)[4:6]


        handle_pos, handle_orn = self.bc.multiplyTransforms(frame_pos, 
                                                            frame_orn,
                                                            handle_local_pos,
                                                            handle_local_orn)
        
        return handle_pos, handle_orn


    def get_receptacle_placement_pose(self, receptacle: ShopObjectEntry, obj: ShopObjectEntry, placement_config: Tuple[int] = (3,2)):
        xs = np.linspace(receptacle.taskspace[0][0], receptacle.taskspace[1][0], placement_config[0]+2)
        ys = np.linspace(receptacle.taskspace[0][1], receptacle.taskspace[1][1], placement_config[1]+2)

        # num_total = placement_config[0]*placement_config[1]
        # num_curr = len(self.robot.receptacle_status)
        # idx = num_curr if num_curr%2 == 0 else num_total-num_curr
        idx = len(self.robot.receptacle_status)

        ix = idx % placement_config[1] + 1
        iy = idx // placement_config[1] + 1
        

        x, y = xs[ix], ys[iy]
        z = self.get_object_height(obj)/2 + 0.005

        receptacle_pose = self.bc.getBasePositionAndOrientation(receptacle.uid)

        placement_pose = self.bc.multiplyTransforms(*receptacle_pose, (x, y, z), self.bc.getQuaternionFromEuler((0, 0, 0)))

        return placement_pose
    

    def stabilize(self, debug=False):
        i = 0
        while True:
            if not debug:
                i += 1
            joints = self.robot.torso_joints + self.robot.left_arm_joints + self.robot.right_arm_joints + self.robot.joints_from_names(PR2_GROUPS["left_gripper"]) + self.robot.joints_from_names(PR2_GROUPS["right_gripper"])
            curr_joint_poses = self.robot.get_joint_positions(joints)
            self.bc.stepSimulation()
            self.robot.set_joint_positions(joints, curr_joint_poses)
            if i > 80:
                break


    def fix_object_on_receptacle(self, receptacle: ShopObjectEntry, obj: ShopObjectEntry, stabilize:bool=False):
        
        # Save pick up joint configuration
        joints = self.robot.torso_joints + self.robot.right_arm_joints + self.robot.joints_from_names(PR2_GROUPS["right_gripper"])
        joint_positions = self.robot.get_joint_positions(joints)
        T_ow = self.bc.getBasePositionAndOrientation(obj.uid)

        # Transform object_pose from T_ow to T_oh (object_pose)
        T_hw = self.robot.get_endeffector_pose(self.arm)
        T_wh = self.bc.invertTransform(*T_hw)
        object_pose = self.bc.multiplyTransforms(*T_wh, *T_ow)

        # Transport the object on the receptacle
        self.robot.release(self.arm)
        placement_pose = self.get_receptacle_placement_pose(receptacle, obj)
        self.bc.resetBasePositionAndOrientation(obj.uid, *placement_pose)
        self.robot.rest_arm(self.arm)
        if stabilize:
            self.stabilize()
        
        # Transformations
        T_tw = self.bc.getBasePositionAndOrientation(receptacle.uid)
        T_wt = self.bc.invertTransform(*T_tw)
        T_ow = self.bc.getBasePositionAndOrientation(obj.uid)
        T_ot = self.bc.multiplyTransforms(*T_wt, *T_ow)
        
        # Setting up constraint
        contact_constraint = self.bc.createConstraint(
            parentBodyUniqueId = receptacle.uid,
            parentLinkIndex    = -1,
            childBodyUniqueId = obj.uid,
            childLinkIndex = -1,
            jointType = self.bc.JOINT_FIXED,
            jointAxis = (0, 0, 1),
            parentFramePosition = T_ot[0],
            childFramePosition = (0, 0, 0),
            childFrameOrientation = T_ot[1])
        
        self.robot.receptacle_status[obj.name] = (AttachConstraint(contact_constraint, obj.uid, object_pose, joints, joint_positions), T_ot, deepcopy(obj.region))
        
        # Update obj region
        receptacle_name = self.env.uid_to_name[self.robot.activated[self.robot.get_other_arm()].uid]
        obj.region = receptacle_name
        
        if stabilize:
            self.stabilize()


    def remove_object_from_receptacle(self, obj: ShopObjectEntry):
        assert obj.name in self.robot.receptacle_status, "[Placement] the object must be on the receptacle"
        
        # Set up for the collsion func
        receptacle_uid = self.robot.activated[self.robot.get_other_arm()].uid
        grasp_uid = [obj.uid, receptacle_uid]
        allow_uid_list = [receptacle_uid] + [attach_info.uid for (attach_info, _, _) in self.robot.receptacle_status.values()]

        # Restore picking up joint configuration
        receptacle_obj_info, _, orignal_region = self.robot.receptacle_status.pop(obj.name)
        self.bc.removeConstraint(receptacle_obj_info.constraint)

        valid = False
        num_trial = 0
        max_trial = 10
        while not valid and num_trial < max_trial:
            attach_xfms = self._get_attach_xfms()
            self.robot.set_joint_positions(receptacle_obj_info.joints, receptacle_obj_info.joint_positions)
            self._update_attached_bodies(attach_xfms)

            # Transform object_pose from T_oh to T_ow
            T_hw = self.robot.get_endeffector_pose(self.arm)
            object_pose = self.bc.multiplyTransforms(*T_hw, *receptacle_obj_info.object_pose)
            
            self.bc.resetBasePositionAndOrientation(obj.uid, *object_pose)
            self.robot.activate(self.arm, [obj.uid])
            self.robot.close_gripper_motion(self.arm)

            collision_fn = self._define_grasp_collision_fn(grasp_uid, allow_uid_list)
            valid = not collision_fn(self.robot.get_joint_positions(self._get_controlable_joints(self.robot)))
            num_trial += 1
            receptacle_obj_info.joint_positions = np.array(receptacle_obj_info.joint_positions)
            receptacle_obj_info.joint_positions[0] += 0.05
            receptacle_obj_info.joint_positions = tuple(receptacle_obj_info.joint_positions)

        # Update region info
        obj.region = orignal_region

        return receptacle_obj_info


    def save_receptacle_holding_info(self, receptacle: ShopObjectEntry):
        joints = self.robot.torso_joints + self.robot.right_arm_joints + self.robot.joints_from_names(PR2_GROUPS["right_gripper"])
        joint_positions = self.robot.get_joint_positions(joints)

        # Save receptacle pose in hand coord
        arm = self.robot.get_other_arm()
        T_hw = self.robot.get_endeffector_pose(arm)
        T_wh = self.bc.invertTransform(*T_hw)
        T_tw = self.bc.getBasePositionAndOrientation(receptacle.uid)
        receptacle_pose = self.bc.multiplyTransforms(*T_wh, *T_tw)

        self.robot.receptacle_holding_info = (joints, joint_positions, receptacle_pose)


    def restore_receptacle_holding_info(self, receptacle: ShopObjectEntry):
        assert self.robot.receptacle_holding_info is not None, "[Placement] receptacle holding info is None"
        other_arm = self.robot.get_other_arm()
        
        self.robot.release(other_arm)
        joints, joint_positions, receptacle_pose = self.robot.receptacle_holding_info
        attach_xfms = self._get_attach_xfms()
        self.robot.set_joint_positions(joints, joint_positions)
        self._update_attached_bodies(attach_xfms)

        # Transform receptacle pose into the world
        #  coord from hand coord
        T_hw = self.robot.get_endeffector_pose(self.arm)
        T_tw = self.bc.multiplyTransforms(*T_hw, *receptacle_pose)

        self.bc.resetBasePositionAndOrientation(receptacle.uid, *T_tw)
        self.robot.activate(self.arm, [receptacle.uid])
        self.robot.receptacle_holding_info = None

        return (joints, joint_positions, receptacle_pose)
    

    # Interaction functions
    def open_gripper(self):
        self.robot.open_gripper_motion(self.arm)

    
    def push_close_gripper(self, object_info: ShopObjectEntry, delay=None):
        tool_offset = (object_info.pick_param.forward, 0, 0)

        # Move forward
        backward_ee_q = self.robot.get_arm_state(self.arm)
        backward_ee_pose = self.robot.get_endeffector_pose(self.arm)
        forward_ee_pose = self.bc.multiplyTransforms(backward_ee_pose[0], 
                                                     backward_ee_pose[1], 
                                                     tool_offset, 
                                                     self.bc.getQuaternionFromEuler((0,0,0)))
        valid = False
        q = None
        num_trial = 0
        max_trial = 10
        while not valid and num_trial < max_trial:

            q = self.robot.arm_ik(self.arm, forward_ee_pose, debug=False)

            if q is None:
                break

            valid = not self.empty_collision_fn(q)[0]
            num_trial += 1

        if not valid: 
            q = backward_ee_q
            self.robot.close_gripper_motion(self.arm)
            return


        if delay == 0:
            self.move([q], delay=delay)
        else:
            # Interpolation
            traj, _ = self.motion_plan(self.robot.get_arm_state(self.arm), q, holding_obj_uid=None, planner_type="interpolation")
            self.move(traj, delay=delay)

        self.robot.close_gripper_motion(self.arm)


    def close_gripper(self, object_info: ShopObjectEntry, delay=None):
        self.robot.close_gripper_motion(self.arm)
    

    def pick(self, object_info: ShopObjectEntry, 
             nav_traj: npt.ArrayLike,
             manip_traj: npt.ArrayLike,
             simple: bool=True)-> bool:
        
        self.nav.move(nav_traj, simple=simple)

        self.open_gripper()
        
        self.move(manip_traj, simple=simple)
        
        # Already holding receptacle
        other_arm = "left" if self.arm == "right" else "right"

        # Picking up receptacle
        if object_info.name in self.env.receptacle_obj:
            # Save the receptacle-holding pose
            self.save_receptacle_holding_info(object_info)

            # Make waiter arm and hold the receptacle on the other hand
            self.robot.set_waiter_arm(other_arm)
            robot_orn = self.robot.get_pose()[1]
            T_hw = self.robot.get_endeffector_pose(other_arm)
            T_th = ((0.13, 0.2, 0), self.bc.getQuaternionFromEuler((math.pi/2,0,math.pi/2)))
            T_tw = self.bc.multiplyTransforms(*T_hw, *T_th)

            self.bc.resetBasePositionAndOrientation(object_info.uid, *T_tw)
            self.robot.activate(other_arm, [object_info.uid], pick_receptacle=True)
            object_info.position = T_tw[0]
            object_info.orientation = T_tw[1]

            self.robot.rest_arm(self.arm)

        # Picking up other objects
        else:
            self.close_gripper(object_info)
            self.robot.activate(self.arm, [object_info.uid])

        return True
    

    def place(self, object_info: ShopObjectEntry, 
              region_name: str,
              nav_traj: npt.ArrayLike,
              manip_traj: npt.ArrayLike,
              stabilize: bool=False,
              simple: bool=True) -> bool:
        
        # Move to the region
        self.nav.move(nav_traj, simple=simple)
        
        # Placing the receptacle
        if object_info.name in self.env.receptacle_obj:
            if len(self.receptacle_status) > 0:
                # receptacle must be emptied first
                return False
            
            # Restore the receptacle holding pose
            self.restore_receptacle_holding_info(object_info)

        # Placing on the receptacle
        elif self.robot.is_holding_receptacle():

            # Already holding receptacle, and placing Object from receptacle
            if object_info.name in self.robot.receptacle_status:
                # Restore pickup position of the object from the receptacle
                self.remove_object_from_receptacle(object_info)
            
            # Place on the receptacle
            elif region_name in self.env.receptacle_obj:
                # Transport the object and Set up constraint on the object on receptacle
                other_arm = self.robot.get_other_arm()
                receptacle: ShopObjectEntry = self.env.regions[self.env.uid_to_name[self.robot.activated[other_arm].uid]]
                self.fix_object_on_receptacle(receptacle, object_info)

        # Placing other objects
        # else:
        #     pass

        self.move(manip_traj, simple=simple)

        self.robot.open_gripper_motion(self.arm)
        self.robot.release(self.arm)

        # Stabilize
        if stabilize:
            self.stabilize()

        # Reset arms
        if self.robot.receptacle_holding_info is None:
            attach_xfms = self._get_attach_xfms()
            self.robot.reset_arms()
            self._update_attached_bodies(attach_xfms)
        else:
            # torso = self.robot.get_joint_positions(self.robot.torso_joints)
            attach_xfms = self._get_attach_xfms()
            self.robot.reset_arms(self.arm)
            self.robot.set_joint_positions(self.robot.torso_joints, [0.45])
            self._update_attached_bodies(attach_xfms)
        

        return True


    def open_door(self, object_info: ShopObjectEntry, 
             nav_traj: npt.ArrayLike,
             manip_traj: npt.ArrayLike,
             handle_pose: Tuple[Tuple[float]],
             handle_action_info: Tuple,
             angle: float = 90.0,
             resolution: int = 20,
             simple: bool=True)-> bool:
        
        ## NOTE (SJ): Double check opening the door is not possible
        if object_info.is_open:
            print("Cannot open door while it is already open")
            # logger.warning(f"Cannot open door while it is already open")
            return False

        if self.robot.activated[self.arm] is not None:
            print("Cannot open door while holding object")
            # logger.warning(f"Cannot open door while holding object")
            return False

        
        object_uid = object_info.uid
        handle_pos, handle_orn = handle_pose
        
        self.nav.move(nav_traj, simple=simple)

        self.open_gripper()
        
        self.move(manip_traj, simple=simple)
        self.close_gripper(object_info)
        self.robot.activate(self.arm, [object_uid])

        # Open the door
        direction = handle_action_info[0][4][handle_action_info[1]]     
        dtheta = angle/180*math.pi/resolution*direction
        for i in range(resolution):
            base_pose = self.robot.get_pose()
            if self.robot.is_holding_receptacle():
                attach_xfms = self._get_attach_xfms()
            T_wh = self.bc.invertTransform(handle_pos, handle_orn)
            T_bh = self.bc.multiplyTransforms(*T_wh, *base_pose)
            theta = self.bc.getJointState(object_uid, handle_action_info[0][1])[0]
            self.bc.resetJointState(object_uid, handle_action_info[0][1], theta+dtheta)
            handle_pos, handle_orn = self.get_handle_pose(object_uid, handle_action_info)
            T_bw = self.bc.multiplyTransforms(handle_pos, handle_orn, *T_bh)
            self.robot.set_pose(T_bw)
            if self.robot.is_holding_receptacle():
                self._update_attached_bodies(attach_xfms)
            time.sleep(2*self.delay)

        self.env.all_obj[object_info.name].is_open = True

        # Release the handle
        self.robot.release(self.arm)

        # Reset arm
        attach_xfms = self._get_attach_xfms()
        if self.robot.is_holding_receptacle():
            self.robot.rest_arm(self.arm)
        else:
            self.robot.reset_arms()
        self._update_attached_bodies(attach_xfms)

        # NOTE (dlee): Do NOT reset the roadmap HERE.
        # Reset roadmap
        # self.nav.reset_roadmap(use_pickle=True, door_open=True)

        return True


    def close_door(self, object_info: ShopObjectEntry, 
                   nav_traj: npt.ArrayLike,
                   manip_traj: npt.ArrayLike,
                   handle_pose: Tuple[Tuple[float]],
                   handle_action_info: Dict,
                   angle: float = 80.0,
                   resolution: int = 20)-> bool:
        
        object_uid = object_info.uid
        handle_pos, handle_orn = handle_pose
        
        self.nav.move(nav_traj)

        self.open_gripper()
        
        self.move(manip_traj)
        self.close_gripper(object_info)
        self.robot.activate(self.arm, [object_uid])

        # close the door
        direction = handle_action_info[0][4][handle_action_info[1]] 
        dtheta = -angle/180*math.pi/resolution*direction
        for i in range(resolution):
            base_pose = self.robot.get_pose()
            T_wh = self.bc.invertTransform(handle_pos, handle_orn)
            T_bh = self.bc.multiplyTransforms(T_wh[0], T_wh[1], base_pose[0], base_pose[1])
            theta = self.bc.getJointState(object_uid, handle_action_info[0][1])[0]
            self.bc.resetJointState(object_uid, handle_action_info[0][1], theta+dtheta)
            handle_pos, handle_orn = self.get_handle_pose(object_uid, handle_action_info)
            T_bw = self.bc.multiplyTransforms(handle_pos, handle_orn, T_bh[0], T_bh[1])
            self.robot.set_pose(T_bw)
            time.sleep(2*self.delay)

        self.env.all_obj[object_info.name].is_open = False

        # Release the handle
        self.robot.release(self.arm)

        # Reset arm
        attach_xfms = self._get_attach_xfms()
        self.robot.reset_arms()
        self._update_attached_bodies(attach_xfms)

        # Reset roadmap
        self.nav.reset_roadmap(use_pickle=True, door_open=False)

        return True
    

    def _get_attach_xfms(self)-> Dict[LinkPair, npt.ArrayLike]:
        attachlist: List[LinkPair] = []
        for arm, info in self.robot.activated.items():
            if info is None:
                continue
            if self.env.uid_to_name[info.uid] in self.env.handles:
                continue
            attach_pair = LinkPair(
                    body_id_a=self.robot.uid,
                    link_id_a=self.robot.gripper_link_indices[arm][0],
                    body_id_b=info.uid,
                    link_id_b=-1)
            
            attachlist.append(attach_pair)

        other_arm = self.robot.get_other_arm()
        for name, (info, _, _) in self.robot.receptacle_status.items():
            if info is None:
                continue
            attach_pair = LinkPair(
                    body_id_a=self.robot.uid,
                    link_id_a=self.robot.gripper_link_indices[other_arm][0],
                    body_id_b=info.uid,
                    link_id_b=-1)
            
            attachlist.append(attach_pair)

        attach_xfms = {
            C: get_relative_transform(self.bc,
                                      C.body_id_a,
                                      C.link_id_a,
                                      C.link_id_b,
                                      C.body_id_b,
                                      inertial=False)
            for C in attachlist}
        
        return attach_xfms


    def _update_attached_bodies(self, attach_xfms: Dict[LinkPair, npt.ArrayLike]):
        bc = self.bc
        # Update transforms of attached bodies.
        for C, xfm in attach_xfms.items():
            # NOTE(ycho): as long as we're consistent
            # about `inertial` keyword, we're fine.
            pose = get_link_pose(bc, C.body_id_a, C.link_id_a)
            pose = bc.multiplyTransforms(pose[0], pose[1],
                                         xfm[0], xfm[1])
            bc.resetBasePositionAndOrientation(
                C.body_id_b, pose[0], pose[1])


    def move(self, traj: List[npt.NDArray], physics:bool=False, delay:float=None, simple:bool=True):
        attach_xfms = self._get_attach_xfms()

        if simple:
            self.robot.set_joint_positions(self.arm_joints, traj[-1])
            self._update_attached_bodies(attach_xfms)
        
        else:

            if delay is None:
                delay = self.delay

            for bq in traj:
                self.robot.set_joint_positions(self.arm_joints, bq)
                self._update_attached_bodies(attach_xfms)
                if delay != 0:
                    time.sleep(delay)
                if physics:
                    for i in range(int(self.delay/self.control_dt)):
                        self.bc.stepSimulation()
    

class PR2Manipulation():
    def __init__(self, bc: BulletClient, 
                 env: ShopEnv, 
                 robot: PR2,
                 nav: Navigation, 
                 config: dict):
        
        self.left = PR2SingleArmManipulation(bc, env, robot, nav, config, "left")
        self.right = PR2SingleArmManipulation(bc, env, robot, nav, config, "right")

        
        