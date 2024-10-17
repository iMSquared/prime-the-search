import os
import math
from typing import Tuple, Dict, List, Set, Union
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from Simulation.pybullet.imm.pybullet_util.typing_extra import TranslationT, QuaternionT, Tuple3, EulerT
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient, suppress_stdout
from Simulation.pybullet.imm.pybullet_util.common import get_joint_limits, get_link_pose

from Simulation.pybullet.imm.pybullet_tools.pr2_utils import TOP_HOLDING_LEFT_ARM, \
    PR2_GROUPS, PR2_GRIPPER_LINKS, RETRACT_LEFT_ARM, RECEPTACLE_LEFT_ARM

from Simulation.pybullet.imm.pybullet_tools.utils import safe_zip, JointInfo, JointState

# IK
from Simulation.pybullet.imm.pybullet_tools.ikfast.pr2.ik import get_tool_pose, get_ik_generator

SLEEP = 0.05
DEBUG = False
CIRCULAR_LIMITS = (-np.pi, np.pi)


@dataclass
class AttachConstraint:
    constraint: int
    uid: int
    object_pose: Tuple[Tuple[float], Tuple[float]]
    joints: Tuple[int]
    joint_positions: Tuple[float]


class Robot(ABC):
    '''
    This is an abstract class that governs the 
    common properties between robots.
    '''

    def __init__(self, bc: BulletClient, uid: int):
        
        # Pybullet properties
        self.bc: BulletClient = bc
        self.uid: int = uid
    

    @abstractmethod
    def update_arm_control(self, *args, **kwargs):
        '''
        Update the control of the robot arm. 
        '''
        

    @abstractmethod
    def get_arm_state(self, *args, **kwargs) -> np.ndarray:
        '''
        Get the joint positions of arm. Not the `last_pose` buffer.
        Only support format in ARM_CONTROL_FORMAT_ARM_JOINT_VALUES
        '''

    @abstractmethod
    def get_endeffector_pose(self, *args, **kwargs) -> Tuple[TranslationT, QuaternionT]:
        '''
        Get the current pose of the end-effector
        '''

    # Util functions
    # Links
    def link_from_name(self, name: str):
        for joint in self.get_joints():
            if self.get_joint_info(joint).linkName.decode('UTF-8') == name:
                return joint
        raise ValueError(self.uid, name)

    # Joints
    def get_joint_info(self, joint: int):
        return JointInfo(*self.bc.getJointInfo(self.uid, joint))

    def get_joint_name(self, joint: int):
        return self.get_joint_info(joint).jointName.decode('UTF-8')

    def get_num_joints(self):
        return self.bc.getNumJoints(self.uid)

    def get_joints(self):
        return list(range(self.get_num_joints()))

    def joint_from_name(self, name: str):
        for joint in self.get_joints():
            if self.get_joint_name(joint) == name:
                return joint
        raise ValueError(self.uid, name)

    def joints_from_names(self, names: List[str]):
        return tuple(self.joint_from_name(name) for name in names)

    def is_circular(self, joint):
        joint_info = self.get_joint_info(joint)
        if joint_info.jointType == self.bc.JOINT_FIXED:
            return False
        return joint_info.jointUpperLimit < joint_info.jointLowerLimit

    def get_joint_limits(self, joint):
        # TODO: make a version for several joints?
        if self.is_circular(joint):
            return CIRCULAR_LIMITS
        joint_info = self.get_joint_info(joint)
        return joint_info.jointLowerLimit, joint_info.jointUpperLimit
    
    def get_min_limit(self, joint):
        return self.get_joint_limits(joint)[0]
    
    def get_min_limits(self, joints):
        return [self.get_min_limit(joint) for joint in joints]

    def get_max_limit(self, joint):
        return self.get_joint_limits(joint)[1]
    
    def get_max_limits(self, joints):
        return [self.get_max_limit(joint) for joint in joints]
    
    def get_joint_intervals(self, joints):
        return self.get_min_limits(joints), self.get_max_limits(joints)

    # Navigation
    def get_pose(self):
        return self.bc.getBasePositionAndOrientation(self.uid)
    
    def get_position(self):
        return self.get_pose()[0]

    def get_quat(self):
        return self.get_pose()[1] # [x,y,z,w]

    def get_euler(self):
        return self.bc.getEulerFromQuaternion(self.get_quat())
    
    def set_pose(self, pose: Tuple[float]):
        (position, quat) = pose
        self.bc.resetBasePositionAndOrientation(self.uid, position, quat)
    
    def set_position(self, position: Tuple[float]):
        self.set_pose((position, self.get_quat()))

    def set_quat(self, quat: Tuple[float]):
        self.set_pose((self.get_position(), quat))

    def set_euler(self, euler: Tuple[float]):
        self.set_quat(self.bc.getQuaternionFromEuler(euler))

    # Manipulation    
    def get_joint_state(self, joint_id: int):
        return JointState(*self.bc.getJointState(self.uid, joint_id))

    def get_joint_position(self, joint_id):
        return self.get_joint_state(joint_id).jointPosition
    
    def get_joint_positions(self, joints):
        return tuple(self.get_joint_position(joint_id) for joint_id in joints)
    
    def set_joint_position(self, joint: int, value: float):
        self.bc.resetJointState(self.uid, joint, targetValue=value, targetVelocity=0)
    
    def set_joint_positions(self, joints: List[str], values: npt.ArrayLike):
        for joint, value in safe_zip(joints, values):
            self.set_joint_position(joint, value)


    def check_robot_object_penetration(self, object_uids: List[int], 
                                             threshold: float = -0.01) -> bool:
        """Detects the penetration between the robot and objects
        
        Args:
            threshold (float): Contact distance below threshold will be regarded as collision.

        Returns:
            bool: True when penetration exists
        """
        # Check collision
        self.bc.performCollisionDetection()
        uid_a = self.uid
        for uid_b in object_uids:

            contact_points = self.bc.getContactPoints(uid_a, uid_b, linkIndexB=-1)
            if len(contact_points) == 0:
                continue
            
            contact_distances = [cp[8] for cp in contact_points]
            for cd in contact_distances:
                if cd < threshold:
                    return True
                    
        return False


class UR5(Robot):
    '''
    This is an abstract class that governs the 
    common properties between manipulation robots.

    NOTE(ssh): Pose arrays include the value for fixed joints. 
    However, the IK solver in PyBullet returns the array without fixed joints.
    Thus, you should be careful when controlling. 
    '''

    # Flags for arm control input format
    ARM_CONTROL_FORMAT_LAST_POSE = 0        # Format in Robot.last_pose
    ARM_CONTROL_FORMAT_IK_SOLUTION = 1      # Format in ik solution
    ARM_CONTROL_FORMAT_ARM_JOINT_VALUES = 2 # Format in values for joint_indices_arm

    
    def __init__(
            self, 
            bc: BulletClient, 
            uid: int,
            joint_index_last: int,
            joint_indices_arm: np.ndarray,
            link_index_endeffector_base: int,
            rest_pose: np.ndarray):

        super().__init__(bc, uid)

        # URDF properties
        self.joint_index_last: int                  = joint_index_last
        self.joint_indices_arm: np.ndarray          = joint_indices_arm
        self.link_index_endeffector_base: int       = link_index_endeffector_base
        self.joint_limits: np.ndarray               = get_joint_limits(self.bc, 
                                                                     self.uid,
                                                                     range(self.joint_index_last + 1))
        self.joint_range: np.ndarray                = self.joint_limits[1] - self.joint_limits[0]

        # Rest pose
        self.rest_pose: np.ndarray                  = rest_pose                  # include fixed joints
        # Last pose for control input
        self.last_pose: np.ndarray                  = np.copy(self.rest_pose)    # include fixed joints
        # Reset robot with init
        for i in self.joint_indices_arm:
            self.bc.resetJointState(self.uid, i, self.last_pose[i])

        # Reset all controls
        self.update_arm_control()



    def update_arm_control(self, values: npt.NDArray = None,
                                 format = ARM_CONTROL_FORMAT_LAST_POSE):
        '''
        Update the control of the robot arm. Not the finger.
        If the parameters are not given, it will reinstate 
        the positional control from the last_pose buffer.

        Params:
        - values(optional): Position values for the joints.
        The format should either match with
            - `last_pose`
            - direct return value of pybullet ik solver.
            - contiguous values for Robot.joint_indices_arm.
        This function will ignore the finger control value, 
        but the full array should be given.
        - from_ik_solution(optional): Set this as true when putting ik solution directly.
        '''
        
        # If joint value is none, the robot stays the position
        if values is not None:
            if format == self.ARM_CONTROL_FORMAT_LAST_POSE:
                self.last_pose[self.joint_indices_arm] = values[self.joint_indices_arm]
            elif format == self.ARM_CONTROL_FORMAT_IK_SOLUTION:
                self.last_pose[self.joint_indices_arm] = values[:len(self.joint_indices_arm)]
            elif format == self.ARM_CONTROL_FORMAT_ARM_JOINT_VALUES:
                self.last_pose[self.joint_indices_arm] = values
            else:
                raise ValueError("Invalid flag")

        # Arm control
        position_gains = [1.5 for _ in self.joint_indices_arm]
        self.bc.setJointMotorControlArray(
            self.uid,
            self.joint_indices_arm,
            self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.joint_indices_arm],
            positionGains = position_gains)



    def get_arm_state(self) -> np.ndarray:
        '''
        Get the joint positions of arm. Not the `last_pose` buffer.
        Only support format in ARM_CONTROL_FORMAT_ARM_JOINT_VALUES
        '''
        positions = self.get_joint_positions(self.joint_indices_arm)
        
        return np.asarray(positions)



    def get_endeffector_pose(self) -> Tuple[TranslationT, QuaternionT]:
        '''
        Get the current pose of the end-effector
        '''

        pos, orn_q = get_link_pose(self.bc, self.uid, self.link_index_endeffector_base)
        
        return pos, orn_q


class UR5Suction(UR5):
    '''
    Wrapper class for UR5 with a suction gripper in PyBullet
    '''

    def __init__(self, bc: BulletClient,
                       config: dict):
        """
        Args:
            bc (BulletClient): PyBullet Client
            config (dict): Configuration file
        """
        # Path to URDF
        project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        urdf_path = os.path.join(project_path, config["project_params"]["custom_urdf_path"])

        # Init superclass
        robot_params = config["robot_params"]["ur5_suction"]
        super().__init__(
            bc = bc, 
            uid = bc.loadURDF(
                fileName        = os.path.join(urdf_path, robot_params["path"]),
                basePosition    = robot_params["pos"],
                baseOrientation = bc.getQuaternionFromEuler(robot_params["orn"]),
                useFixedBase    = True),
            joint_index_last                    = robot_params["joint_index_last"],                     # Not used...?
            joint_indices_arm                   = np.array(robot_params["joint_indices_arm"]),          # For control
            link_index_endeffector_base         = robot_params["link_index_endeffector_base"],          # For robot uid. Not ee uid.
            rest_pose                           = np.array(robot_params["rest_pose"]))                  # asdf


        # Gripper
        gripper_params = robot_params["gripper"]
        self.link_indices_endeffector_tip = gripper_params["link_indices_endeffector_tip"]
        self.gripper_base_to_tip_stroke   = gripper_params["base_to_tip_stroke"]
        self.grasp_poke_backward          = gripper_params["grasp_poke_backward"]
        self.grasp_poke_criteria          = gripper_params["grasp_poke_criteria"]

        # Gripper flags
        self.contact_constraint = None
        self.activated = False

        # Erase the reactions at the tip of end-effector
        for idx in self.link_indices_endeffector_tip:
            self.bc.changeDynamics(self.uid, 
                                   idx,
                                   restitution = 0.0,
                                   contactStiffness = 0.0,
                                   contactDamping = 0.0)


    # From Ravens
    def activate(self, object_uids: List[int]):
        """Simulate suction using a rigid fixed constraint to contacted object.
        
        Args:
            object_uids: objects to manipulate in the environment.
        """

        # Activate when not activated yet
        if not self.activated:
            # Collision detection
            self.bc.performCollisionDetection()

            # Check contact at all links
            points = []
            for tip_idx in self.link_indices_endeffector_tip:
                pts = self.bc.getContactPoints(bodyA=self.uid, linkIndexA=tip_idx)
                points += pts
                # Grasp fail when some link has no contact.
                if len(pts)==0:
                    points = []
                    break

            # If contact exists, apply constraint.
            if len(points) != 0:
                # Check uniqueness
                is_unique = True
                target_uid_unique, target_contact_link_unique = points[0][2], points[0][4]
                for point in points:
                    if target_uid_unique != point[2] and target_contact_link_unique != point[2]:
                        is_unique = False

                # Apply constraint if unique.
                if target_uid_unique in object_uids and is_unique:
                    # Get relative transform
                    ee_body_pose = self.bc.getLinkState(self.uid, self.link_indices_endeffector_tip[0])
                    object_pose = self.bc.getBasePositionAndOrientation(target_uid_unique)

                    world_to_ee_body = self.bc.invertTransform(ee_body_pose[0], ee_body_pose[1])
                    object_to_ee_body = self.bc.multiplyTransforms(world_to_ee_body[0], world_to_ee_body[1],
                                                                    object_pose[0], object_pose[1])
                    # Apply constraint
                    self.contact_constraint = self.bc.createConstraint(
                        parentBodyUniqueId = self.uid,
                        parentLinkIndex    = self.link_indices_endeffector_tip[0],
                        childBodyUniqueId = target_uid_unique,
                        childLinkIndex = target_contact_link_unique,
                        jointType = self.bc.JOINT_FIXED,
                        jointAxis = (0, 0, 0),
                        parentFramePosition = object_to_ee_body[0],
                        parentFrameOrientation = object_to_ee_body[1],
                        childFramePosition = (0, 0, 0),
                        childFrameOrientation = (0, 0, 0))
        
        # Always mark as activated whether succeeded or not.
        self.activated = True
        
        # |FIXME(Jiyong)|: check it is possible to set (self.activated == True and self.contact_constraint is None)
        if self.contact_constraint is None:
            return None
        else:
            return self.bc.getConstraintInfo(self.contact_constraint)[2]


    # From Ravens
    def release(self):
        """Release gripper object, only applied if gripper is 'activated'.

        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.

        To handle deformables, simply remove constraints (i.e., anchors).
        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.activated:
            self.activated = False
            # Release gripped rigit object (if any)
            if self.contact_constraint is not None:
                self.bc.removeConstraint(self.contact_constraint)
                self.contact_constraint = None


    # From Ravens
    def detect_contact(self) -> bool:
        """Detects a full contact with a grasped rigid object."""
        
        self.bc.performCollisionDetection()

        # Detect contact at grasped object if constraint exist.
        if self.activated and self.contact_constraint is not None:
            info = self.bc.getConstraintInfo(self.contact_constraint)
            uid, link = info[2], info[3]
            # Get all contact points between the suction and a rigid body.
            points = self.bc.getContactPoints(bodyA=uid, linkIndexA=link)   # Detected from +0.0005
        # Detect at cup otherwise.
        else:
            points = []
            # Suction gripper tip
            uid = self.uid
            for link in self.link_indices_endeffector_tip:
                pts = self.bc.getContactPoints(bodyA=uid, linkIndexA=link)
                points += pts
                # Grasp fail when some link has no contact.
                if len(pts)==0:
                    points = []
                    break

        if self.activated:
            points = [point for point in points if point[2] != self.uid]

        # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True
        
        return False


    # From Ravens
    def check_grasp(self) -> int:
        """
        Check a grasp object in contact?" for picking success
        
        Returns:
            None (if not grasp) / object uid (if grasp)
        """
        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = self.bc.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object

    
    # For SE3 action space
    def get_target_ee_pose_from_se3(self, pos: TranslationT, orn: EulerT) -> Tuple[TranslationT, EulerT]:
        """This function returns the SE3 target ee pose(inward) given SE3 surface pose(outward).

        Args:
            pos (TranslationT): Sampled end effector tip pose (outward, surface normal)
            orn (EulerT):  Sampled end effector orn (outward, surface normal)

        Returns:
            Tuple[TranslationT, EulerT]: End effector base pose (inward)
        """
        # Get the tip-end coordinate in world frame. This flips the contact point normal inward.
        tip_end_target_pos_in_world, tip_end_target_orn_q_in_world \
            = self.bc.multiplyTransforms(
                pos, self.bc.getQuaternionFromEuler(orn),
                [0.0, 0.0, 0.0], self.bc.getQuaternionFromEuler([3.1416, 0.0, 0.0]))

        # Target in world frame -> tip-end frame -> ur5 ee base link frame
        ee_base_link_pos_in_tip_end_frame = [0.0, 0.0, -self.gripper_base_to_tip_stroke]   # NOTE(ssh): Okay... 12cm is appropriate
        ee_base_link_orn_q_in_tip_end_frame = self.bc.getQuaternionFromEuler([0.0, 0.0, 0.0])
        ee_base_link_target_pos_in_world, ee_base_link_target_orn_q_in_world \
            = self.bc.multiplyTransforms(tip_end_target_pos_in_world, tip_end_target_orn_q_in_world,
                                    ee_base_link_pos_in_tip_end_frame, ee_base_link_orn_q_in_tip_end_frame)

        return ee_base_link_target_pos_in_world, self.bc.getEulerFromQuaternion(ee_base_link_target_orn_q_in_world)


    # For primitive objects (SE2 action space)
    def get_target_ee_pose_from_se2(self, pos: TranslationT, yaw_rad: float) -> Tuple[TranslationT, EulerT]:
        """This function returns the SE3 version of target ee pose given SE2 pose.
        NOTE(ssh): This function helps preserving the roll and pitch consistency of target object!!!!!

        Args:
            pos (TranslationT): Sampled end effector pos from SE2 action space
            yaw_rad (float): Sampled end effector yaw from SE2 action space in radians (outward, surface normals)

        Returns:
            target_ee_base_link_pos (TranslationT): Target end effector position in SE3
            target_ee_base_link_orn (Tuple3): Target end effector euler orientation in SE3 
                (inward, ee_base, preserves the roll and pitch of target objects)
        """

        # NOTE(ssh): To preserve roll and pitch, I always apply constraint that gripper has x-up orientation.

        # Reference orientation in SE2
        frame_rot1 = [0.0, 1.5708, 0.0]
        frame_rot2 = [0.0, 0.0, 3.1416]
        _, se2_reference_ee_frame_orn_q = self.bc.multiplyTransforms([0, 0, 0], self.bc.getQuaternionFromEuler(frame_rot1),
                                                                     [0, 0, 0], self.bc.getQuaternionFromEuler(frame_rot2))
        yaw_rot = [0.0, 0.0, yaw_rad]

        # Target orientation is acquired by rotating the reference frame
        _, target_ee_base_link_orn_q = self.bc.multiplyTransforms([0, 0, 0], self.bc.getQuaternionFromEuler(yaw_rot),
                                                                  [0, 0, 0], se2_reference_ee_frame_orn_q)
                                                    
        
        # Target position is acquire by pushing back by the stroke of end effector in target frame.
        translation_in_target_frame = [0, 0, -self.gripper_base_to_tip_stroke]
        target_ee_base_link_pos, _ = self.bc.multiplyTransforms(pos, target_ee_base_link_orn_q,
                                                                      translation_in_target_frame, [0, 0, 0, 1])
        target_ee_base_link_orn = self.bc.getEulerFromQuaternion(target_ee_base_link_orn_q)



        return target_ee_base_link_pos, target_ee_base_link_orn


    # Reserved
    def get_overriden_pose_wrist_joint_zero_position(self, ee_base_link_target_pos: TranslationT, 
                                                           ee_base_link_target_orn_q: QuaternionT) -> Tuple[TranslationT, QuaternionT]:
        """Force reset wrist joint to 0 and recalculate the forward kinematics of UR5 EE Link
        
        Args:
            ee_base_link_target_pos (TranslationT): End effector position to override in world frame.
            ee_base_link_target_orn_q (QuaternionT): End effector orientation to override in world frame.

        Returns:
            ee_base_link_target_pos (TranslationT): Overriden end effector position in world frame.
            ee_base_link_target_orn_q (QuaternionT): Overriden end effector orientation in world frame.
        """

        # 1. Solve IK
        joint_position_list = np.array( self.bc.calculateInverseKinematics(self.uid, 
                                                                           self.link_index_endeffector_base, 
                                                                           ee_base_link_target_pos, 
                                                                           ee_base_link_target_orn_q, 
                                                                           maxNumIterations = 1000, 
                                                                           residualThreshold = 1e-6) )

        # 2. Overriding the ee base joint
        # Override
        joint_position_list[-1] = 0.0
        # Forward kinematics
        backup_joint_state_list = self.bc.getJointStates(self.uid, self.joint_indices_arm)
        for list_i, joint_idx in enumerate(self.joint_indices_arm):
            self.bc.resetJointState(self.uid, 
                                    joint_idx, 
                                    targetValue = joint_position_list[list_i])
        ur5_ee_link_info = self.bc.getLinkState(self.uid, self.joint_index_last)
        ee_base_link_target_pos = ur5_ee_link_info[4]
        ee_base_link_target_orn_q = ur5_ee_link_info[5]

        # 3. Restore
        for list_i, joint_idx in enumerate(self.joint_indices_arm):
            target_position = backup_joint_state_list[list_i][0]
            target_velocity = backup_joint_state_list[list_i][1]
            self.bc.resetJointState(self.uid,
                                    joint_idx,
                                    targetValue = target_position,
                                    targetVelocity = target_velocity)


        return ee_base_link_target_pos, ee_base_link_target_orn_q


class UR5Grip(UR5):
    '''
    Wrapper class for UR5 robot instance in PyBullet.
    '''

    def __init__(
            self, 
            bc: BulletClient, 
            config: dict):

        # Path to URDF
        project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        urdf_path = os.path.join(project_path, config["project_params"]["custom_urdf_path"])

        # Init superclass
        robot_params = config["robot_params"]["ur5"]
        super().__init__(
            bc = bc,            # NOTE(ssh): `bc` will be class attribute in super.__init__().
            uid = bc.loadURDF(
                fileName        = os.path.join(urdf_path, robot_params["path"]),
                basePosition    = robot_params["pos"],
                baseOrientation = bc.getQuaternionFromEuler(robot_params["orn"]),
                useFixedBase    = True),
            joint_index_last                    = robot_params["joint_index_last"],
            joint_indices_arm                   = np.array(robot_params["joint_indices_arm"]),
            link_index_endeffector_base         = robot_params["link_index_endeffector_base"],
            rest_pose                           = np.array(robot_params["rest_pose"]))


        # Robot dependent params
        gripper_params = robot_params["gripper"]
        self.joint_index_finger                  = gripper_params["joint_index_finger"],
        self.joint_value_finger_open             = gripper_params["joint_value_finger_open"],
        self.distance_finger_to_endeffector_base = gripper_params["distance_finger_to_endeffector_base"],
        self.joint_indices_finger_mimic          = np.array(gripper_params["joint_indices_finger_mimic"])
        self.joint_gear_ratio_mimic              = np.array(gripper_params["joint_gear_ratio_mimic"])


        # Reset all controls
        self.update_arm_control()
        self.update_finger_control()


        # Update dynamics
        self.bc.changeDynamics(
            self.uid, 
            self.joint_index_last-1, 
            lateralFriction=15.0,
            rollingFriction=0.01,
            spinningFriction=0.2,
            restitution=0.5)
        self.bc.changeDynamics(
            self.uid, 
            self.joint_index_last-4, 
            lateralFriction=15.0,
            rollingFriction=0.01,
            spinningFriction=0.2,
            restitution=0.5)



    def __create_mimic_constraints(self):
        '''
        Create mimic constraints for the grippers with closed-loop joints
        '''
        # Finger contsraint chain:
        #   right: 11->12->13 
        #   left: 11->8->9->10
        parent_joint_info = self.bc.getJointInfo(self.uid, self.joint_index_finger)
        parent_joint_id = parent_joint_info[0]
        parent_joint_frame_pos = parent_joint_info[14]

        for list_i, joint_i in enumerate(self.joint_indices_finger_mimic):

            child_joint_info = self.bc.getJointInfo(self.uid, joint_i)
            child_joint_id = child_joint_info[0]
            child_joint_axis = child_joint_info[13]
            child_joint_frame_pos = child_joint_info[14]
            constraint_id = self.bc.createConstraint(self.uid, parent_joint_id,
                                                self.uid, child_joint_id,
                                                jointType = self.bc.JOINT_GEAR,
                                                jointAxis = child_joint_axis,
                                                parentFramePosition = parent_joint_frame_pos,
                                                childFramePosition = child_joint_frame_pos)
            
            gear_ratio = self.joint_gear_ratio_mimic[list_i]
            self.bc.changeConstraint(constraint_id, gearRatio=gear_ratio, maxForce=10000, erp=1.0)



    def update_finger_control(self, finger_value=None):
        '''
        Set joint motor control value to last_pose, and refresh the mimic joints.
        '''
        # Default controlling method
        # If joint value is none, the robot stays the position
        if finger_value is not None:
            # Update last pose buffer
            self.last_pose[self.joint_index_finger] = finger_value
        
        # Finger control
        self.bc.setJointMotorControl2(
            self.uid,
            self.joint_index_finger,
            self.bc.POSITION_CONTROL,
            targetPosition = self.last_pose[self.joint_index_finger])    

        # Mimic finger controls
        finger_state = self.bc.getJointState(
            self.uid,
            self.joint_index_finger)
        target_position = -1.0 * self.joint_gear_ratio_mimic * np.asarray(finger_state[0])
        target_velocity = -1.0 * self.joint_gear_ratio_mimic * np.asarray(finger_state[1])
        self.bc.setJointMotorControlArray(
            self.uid,
            self.joint_indices_finger_mimic,
            self.bc.POSITION_CONTROL,
            targetPositions = target_position,
            targetVelocities = target_velocity,
            positionGains = np.full_like(self.joint_indices_finger_mimic, 1.2, dtype=np.float32),
            forces = np.full_like(self.joint_indices_finger_mimic, 50, dtype=np.float32))

        # Propagate the finger control value to all fingers
        self.last_pose[self.joint_indices_finger_mimic] = np.asarray(finger_state[0])



    def get_finger_state(self) -> float:
        '''
        Get the finger position. Not the `last_pose` buffer.
        '''
        finger_state = self.bc.getJointState(
            self.uid,
            self.joint_index_finger)[0]

        return finger_state
    

class PR2(Robot): 
    '''
    Wrapper class for PR2 robot with mobile base in PyBullet
    '''
    def __init__(self, bc: BulletClient,
                       config: dict,
                       suppress_output: bool=True):
        """
        Args:
            bc (BulletClient): PyBullet Client
            config (dict): Configuration file
        """
        # Path to URDF
        project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        urdf_path = os.path.join(project_path, config["project_params"]["custom_urdf_path"])
        ## setting global client


        # Init superclass
        robot_params = config["robot_params"]["pr2"]
        if suppress_output:
            with suppress_stdout():
                uid = bc.loadURDF(
                        fileName        = os.path.join(urdf_path, robot_params["path"]),
                        basePosition    = robot_params["pos"],
                        baseOrientation = bc.getQuaternionFromEuler(robot_params["orn"]),
                        useFixedBase    = True)
        else:
            uid = bc.loadURDF(
                        fileName        = os.path.join(urdf_path, robot_params["path"]),
                        basePosition    = robot_params["pos"],
                        baseOrientation = bc.getQuaternionFromEuler(robot_params["orn"]),
                        useFixedBase    = True)
            
        
        self.init_pose: Tuple[Tuple[float], Tuple[float]] = (robot_params["pos"], bc.getQuaternionFromEuler(robot_params["orn"]))

        super().__init__(bc = bc, uid = uid)


        # z = base_aligned_z(self.uid)

        # Set joints
        self.left_arm_joints = self.joints_from_names(PR2_GROUPS['left_arm'])
        self.right_arm_joints = self.joints_from_names(PR2_GROUPS['right_arm'])
        self.left_gripper_joints = self.joints_from_names(PR2_GROUPS['left_gripper'])
        self.right_gripper_joints = self.joints_from_names(PR2_GROUPS['right_gripper'])
        self.torso_joints = self.joints_from_names(PR2_GROUPS['torso'])
        self.joint_indices_arm = self.torso_joints + self.left_arm_joints + self.right_arm_joints

        # Set arm pose
        self.last_pose = self.reset_arms("both")
        self.retracted_right_gripper_pose = self.get_tool_pose_from_base("right", self.get_arm_state("right"))
        self.retracted_left_gripper_pose = self.get_tool_pose_from_base("left", self.get_arm_state("left"))
        
        # Set grippers
        self.activated: Dict[str, AttachConstraint] = {"left": None, "right": None}
        self.get_gripper_link_indices()

        # Set main hand
        self.main_hand = config["manipulation_params"]["main_hand"]

        # Set receptacle related infos
        self.receptacle_status: Dict[str, Tuple[AttachConstraint, Tuple]] = dict()
        self.receptacle_holding_info: Tuple[Tuple[float], Tuple[float], Tuple[Tuple[float]]] = None

        # For manipulation
        self.link_index_endeffector_base = self.gripper_link_indices[self.main_hand][0]

    # Arm functions
    def update_arm_control(self, arm: str, values: npt.ArrayLike):
        '''
        Update the control of the robot arm.
        No position control or what so ever since we are not using the physics.
        '''
        self.check_valid_arm(arm)
        if arm == "left":
            joints = self.left_arm_joints
            self.last_pose[len(self.torso_joints):len(self.torso_joints)+len(self.left_arm_joints)] = tuple(values)
        else:
            joints = self.right_arm_joints
            self.last_pose[-len(self.right_arm_joints):] = tuple(values)

        self.set_joint_positions(joints, values)
        

    def get_arm_state(self, arm: str) -> np.ndarray:
        '''
        Get the joint positions of arm. Not the `last_pose` buffer.
        Only support format in ARM_CONTROL_FORMAT_ARM_JOINT_VALUES
        '''
        self.check_valid_arm(arm)
        if arm == "left":
            joints = self.torso_joints + self.left_arm_joints
        else:
            joints = self.torso_joints + self.right_arm_joints

        positions = self.get_joint_positions(joints)

        return positions
    

    def get_both_arms_state(self):
        
        left_state = self.get_arm_state("left")
        right_state = self.get_arm_state("right")[1:]

        return left_state + right_state


    def get_endeffector_pose(self, arm: str) -> Tuple[TranslationT, QuaternionT]:
        '''
        Get the current pose of the end-effector
        '''
        self.check_valid_arm(arm)

        return get_tool_pose(self.uid, arm)


    # Grab functions
    def activate(self, arm: str, object_uids: List[int], object_pose: Tuple[npt.ArrayLike]=None, pick_receptacle=False):
        """Simulate grasp using a rigid fixed constraint to contacted object.
        
        Args:
            object_uids: objects to manipulate in the environment.
        """

        self.check_valid_arm(arm)

        # Activate when not activated yet
        if self.activated[arm] is None:

            # Collision detection
            # self.bc.performCollisionDetection()

            # Check contact at all links
            points = []
            for object_uid in object_uids:
                pts = self.bc.getClosestPoints(bodyA=self.uid, bodyB=object_uid, distance=0.05)
                points += pts
                # Grasp fail when some link has no contact.
                if len(pts)==0:
                    points = []

            # If contact exists, apply constraint.
            if len(points) != 0:
                # Check uniqueness
                is_unique = True
                target_uid_unique, target_contact_link_unique = points[0][2], points[0][4]
                for point in points:
                    if target_uid_unique != point[2] and target_contact_link_unique != point[2]:
                        is_unique = False

                # Apply constraint if unique.
                if target_uid_unique in object_uids and is_unique:
                    # Get relative transform
                    
                    # T_hw = self.bc.getLinkState(self.uid, self.gripper_link_indices[arm][0])[:2]
                    T_hw = self.get_endeffector_pose(arm)

                    if object_pose is None:
                        T_ow = self.bc.getBasePositionAndOrientation(target_uid_unique)
                        T_wh = self.bc.invertTransform(*T_hw)
                        object_pose = self.bc.multiplyTransforms(*T_wh, *T_ow)

                    T_oh = object_pose
                    orn_offset = (0, np.pi, 0) if pick_receptacle else (-np.pi/2, 0, 0)
                    child_frame_offset = (0, -0.5, 0) if pick_receptacle else (0, 0, 0)
                    T_joint = self.bc.multiplyTransforms((0,0,0), self.bc.getQuaternionFromEuler(orn_offset), *T_oh)
                    
                    contact_constraint = self.bc.createConstraint(
                        parentBodyUniqueId = self.uid,
                        parentLinkIndex    = self.gripper_link_indices[arm][0],
                        childBodyUniqueId = target_uid_unique,
                        childLinkIndex = target_contact_link_unique,
                        jointType = self.bc.JOINT_FIXED,
                        jointAxis = (0, 0, 1),
                        parentFramePosition = T_joint[0],
                        childFramePosition = child_frame_offset,
                        childFrameOrientation = T_joint[1])
        
                # Always mark as activated whether succeeded or not.
                joints = self.torso_joints + getattr(self, f"{arm}_arm_joints") + self.joints_from_names(PR2_GROUPS[f"{arm}_gripper"])
                joint_positions = self.get_joint_positions(joints)
                self.activated[arm] = AttachConstraint(contact_constraint, target_uid_unique, object_pose, joints, joint_positions)
                return self.bc.getConstraintInfo(contact_constraint)[2]

        else:
            return None


    def release(self, arm: str):
        """Release gripper object, only applied if gripper is 'activated'.
        """
        if self.activated[arm] is not None:
            contact_constraint = self.activated[arm].constraint
            # Release gripped rigit object (if any)
            if contact_constraint is not None:
                self.bc.removeConstraint(contact_constraint)
                contact_constraint = None
            self.activated[arm] = None


    def open_gripper_motion(self, arm: str): # These are mirrored on the pr2
        self.check_valid_arm(arm)
        name = "{}_gripper".format(arm)
        joints = self.joints_from_names(PR2_GROUPS[name])

        for joint in joints:
            self.set_joint_position(joint, self.get_max_limit(joint))


    def close_gripper_motion(self, arm: str):
        self.check_valid_arm(arm)

        name = "{}_gripper".format(arm)
        joints = self.joints_from_names(PR2_GROUPS[name])

        for joint in joints:
            self.set_joint_position(joint, self.get_min_limit(joint))


    def stabilize(self, duration: int =100, debug: bool=False):
        if debug: duration = int(math.inf)
        for _ in range(duration):
            joints = self.torso_joints + self.left_arm_joints + self.right_arm_joints + self.joints_from_names(PR2_GROUPS["left_gripper"]) + self.joints_from_names(PR2_GROUPS["right_gripper"])
            curr_joint_poses = self.get_joint_positions(joints)
            self.bc.stepSimulation()
            self.set_joint_positions(joints, curr_joint_poses)


    def check_valid_arm(self, arm: str):
        assert arm in ["left", "right"], "Argument 'arm' should be either 'left' or 'right'"


    def is_holding_receptacle(self) -> bool:
        return (self.receptacle_holding_info is not None)


    def get_other_arm(self) -> str:
        return "left" if self.main_hand == "right" else "right"


    def get_gripper_link_indices(self):
        self.gripper_link_indices = {"left": [], "right": []}
        for arm in ["left", "right"]:
            for name in PR2_GRIPPER_LINKS[arm]:
                link_idx = self.link_from_name(name)
                self.gripper_link_indices[arm].append(link_idx)


    def rightarm_from_leftarm(self, config):
        right_from_left = np.array([-1, 1, -1, 1, -1, 1, -1])
        return config * right_from_left
    

    def rest_arm(self, arm: str) -> Tuple[float]:
        arm_start = RETRACT_LEFT_ARM

        if arm == "left":
            self.set_joint_positions(self.left_arm_joints, arm_start)

        elif arm == "right":
            arm_start = list(self.rightarm_from_leftarm(arm_start))
            self.set_joint_positions(self.right_arm_joints, arm_start)
        
        else:
            raise ValueError("wrong arm")

        return arm_start
    

    def reset_arms(self, arm: str = "both") -> Tuple[float]:
        '''
            Initialize arm pose
        '''

        if arm == "both":
            arm = ["left", "right"]

        torso_start = [0.4]
        self.set_joint_positions(self.torso_joints, torso_start)

        value = torso_start
        if "left" in arm:
            arm_start_left = RETRACT_LEFT_ARM
            self.set_joint_positions(self.left_arm_joints, arm_start_left)
            value += arm_start_left

        if "right" in arm:
            arm_start_right = list(self.rightarm_from_leftarm(TOP_HOLDING_LEFT_ARM))
            self.set_joint_positions(self.right_arm_joints, arm_start_right)
            value += arm_start_right
        
        return value
    

    def set_waiter_arm(self, arm: str):

        if arm == "left":
            self.set_joint_positions(self.left_arm_joints, RECEPTACLE_LEFT_ARM)
        else:
            self.set_joint_positions(self.right_arm_joints, self.rightarm_from_leftarm(RECEPTACLE_LEFT_ARM))


    def get_tool_pose_from_base(self, arm: str, joint_pose: npt.ArrayLike) -> Tuple[float]:
        '''
            Initialize arm pose
        '''
        
        T_bw = self.get_pose()
        T_wb = self.bc.invertTransform(*T_bw)

        T_tw = self.get_endeffector_pose(arm)
        T_tb = self.bc.multiplyTransforms(*T_wb, *T_tw)

        return T_tb
    

    def arm_ik(self, arm: str, pose: npt.ArrayLike, trial: int=100, debug=DEBUG) -> List[npt.ArrayLike]:
        
        self.check_valid_arm(arm)
        generator = get_ik_generator(self.uid, arm, pose, torso_limits=False)

        for i in range(trial):
            solutions = next(generator)
            if debug:
                print(f"Trial {i}, # Solutions: {len(solutions)}")

            if len(solutions) > 0:
                return solutions[0]


def load_gripper(bc: BulletClient, target_pose: Tuple[Tuple[float]]) -> int:
    urdf_path = os.path.join("Simulation", "pybullet", "urdf")
    uid = bc.loadURDF(fileName = os.path.join(urdf_path, "pr2", "pr2_gripper.urdf"),
                      basePosition = target_pose[0],
                      baseOrientation = target_pose[1],
                      useFixedBase = True)
    
    return uid

def remove_gripper(bc: BulletClient, uid: int):
    bc.removeBody(uid)