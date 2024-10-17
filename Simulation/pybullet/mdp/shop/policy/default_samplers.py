import math
import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Dict, Iterable
import random, json

from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.envs.shop_env import ShopEnv, ShopObjectEntry
from Simulation.pybullet.envs.robot import PR2, load_gripper, remove_gripper
from Simulation.pybullet.envs.manipulation import PR2SingleArmManipulation, PR2Manipulation
from Simulation.pybullet.envs.navigation import Navigation, dream

from Simulation.pybullet.mdp.shop.shop_MDP import ShopState, ShopDiscreteAction, ShopContinuousAction, Relation, ACTION_PICK, ACTION_PLACE, ACTION_OPEN, ACTION_CLOSE

DEBUG = False

class Sampler:
    def __init__(self, NUM_FILTER_TRIALS: int):
        """Random pick action sampler"""
        self.NUM_FILTER_TRIALS = NUM_FILTER_TRIALS


    def sample_from_region(self, min: npt.ArrayLike, max: npt.ArrayLike, margin:Tuple[float]=(0.03, 0.03, 0), eps: float=0.02) -> npt.ArrayLike:
        assert len(min)==len(max), "Number of dimension must be same"

        range = np.array(max) - np.array(min) - 2*np.array(margin) + 2*np.array([eps, eps, 0])
        rand = np.random.uniform(low=0, high=1, size=len(range))

        return range*rand + np.array(min) + np.array(margin) - np.array([eps, eps, 0])


    def get_object_height(self, bc: BulletClient, object_info: ShopObjectEntry):
        uid = object_info.uid
        if object_info.is_movable:
            try:
                aabb = bc.getAABB(uid, 0)
            except Exception as e:
                aabb = bc.getAABB(uid)
        else:
            aabb = bc.getAABB(uid)

        return aabb[1][2] - aabb[0][2]


    def _check_closeness(self,
                         bc: BulletClient, 
                         pose1: Tuple[Tuple[float]], 
                         pose2: Tuple[Tuple[float]],
                         pos_thresh: float=0.01,
                         orn_thresh: float=0.02) -> bool:
        # Check XYZ
        linear_distance = np.linalg.norm((np.array(pose1[0][:2])-np.array(pose2[0][:2])))

        # Check angle
        orn1_q = pose1[1]
        orn2_q = pose2[1]
        orn_diff_q = bc.getDifferenceQuaternion(orn1_q, orn2_q)
        orn_diff = bc.getEulerFromQuaternion(orn_diff_q)

        return (linear_distance < pos_thresh) and np.all(np.array(orn_diff) < orn_thresh)


    def get_handle_pose(self, bc: BulletClient,
                        robot: PR2,
                        object_uid: int, 
                        handle_info: Tuple[int, npt.ArrayLike, npt.ArrayLike]) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        
        _, link_id, handle_local_pos_candidates, handle_local_orn_candidates, \
            open_direction_candidates, _, _ = handle_info
        
        distances = np.linalg.norm(np.array(handle_local_pos_candidates) - robot.get_position(), axis=1)
        idx = np.argmin(distances)

        handle_local_pos = handle_local_pos_candidates[idx]
        handle_local_orn = handle_local_orn_candidates[idx]
        
        frame_pos, frame_orn = bc.getLinkState(object_uid, link_id)[4:6]

        handle_pos, handle_orn = bc.multiplyTransforms(frame_pos, frame_orn, handle_local_pos, handle_local_orn)
        
        return handle_pos, handle_orn, idx


    def get_nav_allow_uid_list(self, bc: BulletClient,
                               robot: PR2,
                               env: ShopEnv):
        
        allow_uid_list = [v[0].uid for v in robot.receptacle_status.values()] + [v.uid for v in env.regions.values()]

        for v in env.openable_obj.values():
            if v.uid in allow_uid_list:
                allow_uid_list.remove(v.uid)

        if robot.is_holding_receptacle():
            receptacle_uid = robot.activated[robot.get_other_arm()].uid
            allow_uid_list.append(receptacle_uid)

        
        return allow_uid_list
        


class PickSampler(Sampler):
    def __init__(self, NUM_FILTER_TRIALS: int):
        super().__init__(NUM_FILTER_TRIALS)
        self.nav_traj_cache: Dict[str, Iterable] = dict()

    def __call__(self, bc: BulletClient,
                       env: ShopEnv,
                       robot: PR2,
                       nav: Navigation,
                       manip: PR2Manipulation,
                       state: ShopState,
                       discrete_action: ShopDiscreteAction = None,
                       ignore_movable: bool=False,
                       predicate_check: bool=False,
                       debug: bool=False) -> ShopContinuousAction:
        """Sample a random pick action

        Args:
            bc (BulletClient)
            env (ShopEnv)
            robot (PR2)
            manip (PR2Manipulation)
            state (ShopState)
            discrete_action (ShopDiscreteAction)

        Raises:
            ValueError: Pick sampler being called while holding an object

        Returns:
            ShopContinuousAction: Sampled Pick action
        """
        DEBUG = debug
        # build_distance_table(bc, nav, env, robot, True)

        # Validity check and pre-process
        assert discrete_action.type == ACTION_PICK, "Should be PICK action"
        
        object_info = env.movable_obj[discrete_action.aimed_obj.name]
        arm = discrete_action.arm
        single_arm_manip: PR2SingleArmManipulation = getattr(manip, arm)

        # Check cache
        # if f"{arm}-{object_info.name}" in state.action_cache.pick_action:
        #     return state.action_cache.pick_action[f"{arm}-{object_info.name}"]

        # Sanity check
        if robot.activated[arm] is not None:
            if DEBUG:
                print("Can't pick if you are holding something")
            return ShopContinuousAction(discrete_action, None, None, None, None, None)

        # Navigate to the region
        nav_traj, goal_pose = self.sample_navigation(bc, robot, env, nav, 
                                                     object_info, 
                                                     ignore_movable=ignore_movable,
                                                     predicate_check=predicate_check)
        if nav_traj is None:
            if DEBUG:
                print("[Pick] failed to navigate to the region")
            return ShopContinuousAction(discrete_action, None, None, None, None, None)

        for i in range(self.NUM_FILTER_TRIALS):
            # Sample pose
            target_pose = self.sample_pose(bc, robot, env, object_info)

            if len(nav_traj) > 0:
                with dream(bc, env, robot):
                    # Move to the region
                    single_arm_manip = getattr(manip, robot.main_hand)
                    attach_xfms = single_arm_manip._get_attach_xfms()
                    robot.set_pose(goal_pose)
                    single_arm_manip._update_attached_bodies(attach_xfms)

                    # Motion plan manipulation
                    manip_traj = self.sample_manipulation(bc, env, robot, single_arm_manip, target_pose, ignore_movable)

            else:
                # Motion plan manipulation
                manip_traj = self.sample_manipulation(bc, env, robot, single_arm_manip, target_pose, ignore_movable)

            if manip_traj is not None and len(manip_traj) > 0:
                break

        if manip_traj is None:
            if DEBUG:
                print("Motion planning was unsuccessful")
        
        return ShopContinuousAction(discrete_action, target_pose[0], target_pose[1], nav_traj, manip_traj)


    def sample_pose(self, bc: BulletClient,
                    robot: PR2,
                    env: ShopEnv, 
                    object_info: ShopObjectEntry) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Sample a random pick action

        Args:
            bc (BulletClient)
            env (ShopEnv)
            robot (PR2)
            manip (PR2Manipulation)
            object_pose (Tuple[npt.ArrayLike])
        Raises:
            ValueError: Pick sampler being called while holding an object

        Returns:
            ShopContinuousAction: Sampled Pick action
        """

        assert object_info.is_movable, "Only movable object can be picked"
        robot_stand_pos, robot_stand_orn = ShopObjectEntry.find_default_robot_base_pose(robot,
                                                                                        env.regions[object_info.region],
                                                                                        env.shop_config[object_info.area + "_config"],
                                                                                        env.all_obj)
        
        robot_hand_pos, _ = robot.get_endeffector_pose(robot.main_hand)

        # Target orientation
        orn_range = object_info.pick_param.angles
        sampled_orn = self.sample_from_region(*orn_range, eps=0)

        # Target position
        if len(object_info.pick_param.points) > 0:
            target_position_candidates = []
            for point in object_info.pick_param.points:
                pos, _ = bc.multiplyTransforms(object_info.position,
                                               object_info.orientation,
                                               point,
                                               bc.getQuaternionFromEuler((0,0,0)))
                target_position_candidates.append(pos)
            distances = np.linalg.norm(np.array(target_position_candidates) - robot_hand_pos, axis=1)
            idx = np.argmin(distances)
            target_pos = object_info.pick_param.points[idx]
            
        else:
            target_pos = (0, 0, 0)

        target_orn = bc.getQuaternionFromEuler(sampled_orn)

        # offset
        tool_offset = (-object_info.pick_param.backward, 0, 0)

        # p: pick pose      b: backward pose    o: object pose
        T_p_orn = ((0, 0, 0), target_orn)
        T_turn = ((0,0,0), robot_stand_orn)
        # T_ow = (object_info.position, bc.getQuaternionFromEuler((0,0,0)))
        T_offset = (tool_offset, bc.getQuaternionFromEuler((0,0,0)))

        T_turned = bc.multiplyTransforms(*T_turn, *T_p_orn)
        target_pos = target_position_candidates[idx] if len(object_info.pick_param.points) > 0 else object_info.position
        T_pw = (target_pos, T_turned[1])
        # T_pw = bc.multiplyTransforms(*T_ow, *T_trans)
        T_bw = bc.multiplyTransforms(*T_pw, *T_offset)


        return T_bw


    def sample_manipulation(self, bc: BulletClient,
                            env: ShopEnv,
                            robot: PR2,
                            single_arm_manip: PR2SingleArmManipulation,
                            target_pose: Tuple[Tuple[float]],
                            ignore_movable: bool=False) -> npt.ArrayLike:
        # Open gripper
        robot.open_gripper_motion(single_arm_manip.arm)

        q = robot.arm_ik(single_arm_manip.arm, target_pose, debug=False)
        if q is None:
            if DEBUG:
                print("No possible pick pose")
            return None

        # Motion planning
        if robot.is_holding_receptacle():
            if robot.activated[robot.get_other_arm()] is None:
                if DEBUG:
                    print("No possible pick pose. Probably failed to grasp the receptacle.")
                return None
            receptacle_uid = robot.activated[robot.get_other_arm()].uid
            obj_on_receptacle_uids = [attach_info.uid for (attach_info, _, _) in robot.receptacle_status.values()]
            attachlist = [None] + obj_on_receptacle_uids + [receptacle_uid]
            collsion_fn = single_arm_manip._define_grasp_collision_fn(attachlist, allow_uid_list=obj_on_receptacle_uids)
        else:
            collsion_fn = single_arm_manip.default_collision_fn
        if ignore_movable:
            collsion_fn = single_arm_manip.empty_collision_fn
        with dream(bc, env, robot):
            # if single_arm_manip.motion_planner.collision_fn(q)[0]:
            if collsion_fn(q)[0]:
                if DEBUG:
                    print("Collision at goal")
                return None
            # single_arm_manip.motion_planner.grow(samples)
        traj, _ = single_arm_manip.motion_plan(single_arm_manip.robot.get_arm_state(single_arm_manip.arm), q, holding_obj_uid=None)

        # Close gripper
        robot.close_gripper_motion(single_arm_manip.arm)

        return traj


    def sample_navigation(self, bc: BulletClient,
                          robot: PR2,
                          env: ShopEnv,
                          nav: Navigation,
                          object_info: ShopObjectEntry,
                          ignore_movable: bool=False,
                          predicate_check: bool=False) -> npt.ArrayLike:
        # if handle_pose is given, this is for opening the door, 
        # where navigation already dealt by open_door method
        region = object_info.region
        base_pose = robot.get_pose()
        robot_stand_pose = ShopObjectEntry.find_default_robot_base_pose(robot,
                                                                        env.regions[region], 
                                                                        env.shop_config[object_info.area + "_config"], 
                                                                        env.all_obj)

        nav_traj = []
        goal_pose = None

        if not self._check_closeness(bc, base_pose, robot_stand_pose):
            curr_pose = (*robot.get_position()[:2], robot.get_euler()[2])
            goal_pose = (*robot_stand_pose[0][:2], bc.getEulerFromQuaternion(robot_stand_pose[1])[2])
            
            # if predicate_check:
            #     return [curr_pose, goal_pose], nav._SE3fromSE2(goal_pose)
            
            # Use cached navigation traj
            is_open = env.openable_obj["kitchen_door"].is_open
            cache_key = f"{is_open}-{ignore_movable}-{curr_pose}-{goal_pose}"
            if cache_key in self.nav_traj_cache:
                return self.nav_traj_cache[cache_key]
            
            # if ignore_movable:
            #     nav.set_empty_roadmap()
            # else:
            #     nav.set_default_roadmap()

            if robot.activated[robot.main_hand] is not None:
                holding_obj_uid = robot.activated[robot.main_hand].uid
            else:
                holding_obj_uid = None

                allow_uid_list = self.get_nav_allow_uid_list(bc, robot, env)
            
            nav_traj, _ = nav.motion_plan(curr_pose, goal_pose, holding_obj_uid=holding_obj_uid, allow_uid_list=allow_uid_list, ignore_movable=ignore_movable)                

            self.nav_traj_cache[cache_key] = (nav_traj, nav._SE3fromSE2(goal_pose))
            return nav_traj, nav._SE3fromSE2(goal_pose)
        else:
            return [], None



class RandomPlaceSampler(Sampler):

    def __init__(self, NUM_FILTER_TRIALS: int,
                       NUM_FILTER_TRIALS_FORCE_FETCH: int=1,
                       random_orientation: bool=False):
        """Random pick action sampler"""
        super().__init__(NUM_FILTER_TRIALS=NUM_FILTER_TRIALS)
        self.NUM_FILTER_TRIALS_FORCE_FETCH = NUM_FILTER_TRIALS_FORCE_FETCH
        self.random_orientation = random_orientation
        self.nav_traj_cache: Dict[str, Iterable] = dict()

    def __call__(self, bc: BulletClient,
                       env: ShopEnv,
                       robot: PR2,
                       nav: Navigation,
                       manip: PR2Manipulation,
                       state: ShopState,
                       discrete_action: ShopDiscreteAction = None,
                       ignore_movable: bool=False,
                       predicate_check: bool=False,
                       debug: bool=False) -> ShopContinuousAction:
        """Sample a random pick action

        Args:
            bc (BulletClient)
            env (ShopEnv)
            robot (PR2)
            manip (PR2Manipulation)
            state (ShopState)
            discrete_action (ShopDiscreteAction)

        Raises:
            ValueError: Pick sampler being called while holding an object

        Returns:
            ShopContinuousAction: Sampled Pick action
        """
        DEBUG = debug
        # Validity check and pre-process
        assert discrete_action.type == ACTION_PLACE, "Should be PLACE action"
        # print(f"sampling pose for {discrete_action}")
        # assert robot.is_holding_receptacle() or robot.activated[discrete_action.arm] is not None, "Should picked up something in your hand"
        if not robot.is_holding_receptacle() and robot.activated[discrete_action.arm] is None:
            # print("Should picked up something in your hand")
            return ShopContinuousAction(discrete_action, None, None, None, None,)

        object_info = discrete_action.aimed_obj
        arm = discrete_action.arm
        region = discrete_action.region
        single_arm_manip: PR2SingleArmManipulation = getattr(manip, arm)
        receptacle_status_changed = False

        if not robot.is_holding_receptacle() and object_info.uid != robot.activated[robot.main_hand].uid \
        and object_info.name not in robot.receptacle_status:

            return ShopContinuousAction(discrete_action, None, None, None, None, error_message="[Placement] Can only place the objects in hand")

        # You can only place the objects in your hand
        if not robot.is_holding_receptacle() and (robot.activated[robot.main_hand] is None \
        or object_info.uid != robot.activated[robot.main_hand].uid) \
        and object_info.name not in robot.receptacle_status:

            return ShopContinuousAction(discrete_action, None, None, None, None, error_message='[Placement] Can only place the objects in hand')

        # Place on the receptacle
        if robot.is_holding_receptacle() and region in env.receptacle_obj:
            # Dummy values: we wil compute the real placement during transition.
            pos, orn = bc.getBasePositionAndOrientation(object_info.uid)
            joint_pos = robot.activated[robot.main_hand].joint_positions[:8]
            return ShopContinuousAction(discrete_action, pos, orn, [], [joint_pos])

        # Navigate to the region
        nav_traj, goal_pose, intermediate_arm_joint_pos = self.sample_navigation(bc, robot, env, nav, 
                                                                                 single_arm_manip, 
                                                                                 object_info, region, 
                                                                                 ignore_movable=ignore_movable,
                                                                                 predicate_check=predicate_check)
        if nav_traj is None:
            return ShopContinuousAction(discrete_action, None, None, None, None, error_message="[Placement] failed to navigate to the region")

        # Hold the object on hand if placing from receptacle
        other_arm_attach_info = robot.activated[robot.get_other_arm()]
        if robot.is_holding_receptacle() and object_info.name in robot.receptacle_status:
            single_arm_manip.remove_object_from_receptacle(object_info)
            receptacle_status_changed = True

        obj2: ShopObjectEntry = discrete_action.reference_obj
        rel: str = discrete_action.direction

        for i in range(self.NUM_FILTER_TRIALS):
            # Sample pose
            target_pose = self.sample_pose(bc, env, robot, arm, region, object_info.name, obj2.name, [rel], relative=True)
                
            if target_pose is None:
                if DEBUG: print("No appropriate target pose")
                continue
            
            arm_joints = robot.left_arm_joints if single_arm_manip.arm == "left" else robot.right_arm_joints

            if len(nav_traj) > 0:
                with dream(bc, env, robot):
                    # Move to the region
                    attach_xfms = single_arm_manip._get_attach_xfms()
                    robot.set_pose(goal_pose)
                    robot.set_joint_positions(robot.torso_joints + arm_joints, intermediate_arm_joint_pos)
                    single_arm_manip._update_attached_bodies(attach_xfms)

                    # Motion plan manipulation
                    manip_traj = self.sample_manipulation(bc, env, robot, single_arm_manip, object_info, target_pose, ignore_movable)
                    # ShopObjectEntry.draw_taskspace(bc, env, env.all_obj["coke"], env.all_obj["counter1"])
            else:
                # Motion plan manipulation
                manip_traj = self.sample_manipulation(bc, env, robot, single_arm_manip, object_info, target_pose, ignore_movable)

            if manip_traj is not None and len(manip_traj) > 0:
                break

        if target_pose is None:
            # if DEBUG: print("Failed to sample appropriate placement pose")
            # logger.info("Failed to sample placement pose")
            if receptacle_status_changed:
                receptacle = env.all_obj[env.uid_to_name[other_arm_attach_info.uid]]
                single_arm_manip.fix_object_on_receptacle(receptacle, object_info)
            return ShopContinuousAction(discrete_action, None, None, nav_traj, None, error_message="Failed to sample placement pose")


        if manip_traj is None:
            # if DEBUG: print("Motion planning was unsuccessful")
            # logger.info("Motion planning was unsuccessful")
            if receptacle_status_changed:
                receptacle = env.all_obj[env.uid_to_name[other_arm_attach_info.uid]]
                single_arm_manip.fix_object_on_receptacle(receptacle, object_info)
            return ShopContinuousAction(discrete_action, target_pose[0], target_pose[1], nav_traj, None, error_message="Motion planning was unsuccessful at grounding policy")


        # Success
        if receptacle_status_changed:
            receptacle = env.all_obj[env.uid_to_name[other_arm_attach_info.uid]]
            single_arm_manip.fix_object_on_receptacle(receptacle, object_info)
        
        return ShopContinuousAction(discrete_action, target_pose[0], target_pose[1], nav_traj, manip_traj)
    

    def sample_pose(self, 
                    bc: BulletClient,
                    env: ShopEnv,
                    robot: PR2,
                    arm: str,
                    region: str,
                    obj1: str,
                    obj2: str = None,
                    rels: List[str] = ["on"],
                    relative: bool=False) -> Tuple[Tuple[float]]:


        region: ShopObjectEntry = env.regions[region]
        obj1: ShopObjectEntry = env.movable_obj[obj1]
        if obj2 is not None:
            obj2: ShopObjectEntry = env.all_obj[obj2]
        else:
            obj2 = region

        # Step 1: Decided placement position
        obj_height = self.get_object_height(bc, obj1)
        valid = False
        trial = 0
        region_pos = env.all_obj[env.regions[region.name].entity_group[0]].position
        region_orn = env.all_obj[env.regions[region.name].entity_group[0]].orientation
        region_orn = region_orn if len(region_orn) == 4 else bc.getQuaternionFromEuler(region_orn)
        while not valid:
            if trial > self.NUM_FILTER_TRIALS:
                return None
            
            # Step 1-0: determine orientation
            if self.random_orientation:
                yaw = random.uniform(*env.config["manipulation_params"]["sample_space"]["yaw_range"])
                goal_orn = self.bc.getQuaternionFromEuler((0,0,yaw))

            else:
                obj_orn = obj1.orientation if len(obj1.orientation) == 4 else bc.getQuternionFromEuler(obj1.orientation)
                T_wr = bc.invertTransform(*robot.get_pose())
                T_or = bc.multiplyTransforms(*T_wr, (0,0,0), obj_orn)
                T_dw = ShopObjectEntry.find_default_robot_base_pose(robot, 
                                                                    env.regions[region.name], 
                                                                    env.shop_config[f"{env.regions[region.name].area}_config"], 
                                                                    env.all_obj)
                _, goal_orn = bc.multiplyTransforms(*T_dw, *T_or)

            # Step 1-1: Random sample from taskspace
            margin = ShopObjectEntry.get_obj_margin(bc, obj1, goal_orn)
            sampled = self.sample_from_region(*region.taskspace, margin=margin)
            goal_pos, _ = bc.multiplyTransforms(region_pos,
                                                region_orn,
                                                sampled,
                                                bc.getQuaternionFromEuler((0,0,0)))

            goal_pos = np.array(goal_pos)
            goal_pos[2] = region.taskspace[1][2] + region_pos[2] + 0.02 # + obj_height/2


            if len(rels) == 1 and "on" in rels:
                valid = True
                break

            # Step 1-2: Get coordinates
            # Transform into Robot coord if relative
            if relative:
                T_rw = ShopObjectEntry.find_default_robot_base_pose(robot, 
                                                                    region, 
                                                                    env.shop_config[region.area + "_config"], 
                                                                    env.all_obj)
                T_o2w = (obj2.position, obj2.orientation) if len(obj2.orientation) == 4 \
                    else (obj2.position, bc.getQuaternionFromEuler(obj2.orientation))
                T_o1w = (goal_pos, goal_orn)

                T_wr = bc.invertTransform(*T_rw)
                T_o2r = bc.multiplyTransforms(*T_wr, *T_o2w)
                T_o1r = bc.multiplyTransforms(*T_wr, *T_o1w)

                o2_x, o2_y, _ = T_o2r[0]
                sampled_x, sampled_y, _ = T_o1r[0]
            else:
                o2_x, o2_y, _ = obj2.position
                sampled_x, sampled_y, _ = goal_pos


            if DEBUG:
                bc.addUserDebugPoints([obj2.position, goal_pos], [[1,0,0], [0,0,1]], 4)

            # Step 1-3: Check and reject
            curr = []
            if abs(sampled_x - o2_x) > abs(sampled_y - o2_y):
                if sampled_x > o2_x:
                    curr.append(Relation.behind.value)
                else:
                    curr.append(Relation.front.value)
            else:
                if sampled_y > o2_y:
                    curr.append(Relation.left.value)
                else:
                    curr.append(Relation.right.value)

            valid = np.array([(rel in curr) for rel in rels]).all()
            trial += 1
                
        goal_pose = (tuple(goal_pos), goal_orn)

        # Step 3: Convert to gripper pose which is little bit backward
        gripper_pose = self.get_gripper_pose_from_obj_pose(bc, robot, arm, obj1, goal_pose)

        return gripper_pose


    def sample_navigation(self, bc: BulletClient,
                          robot: PR2,
                          env: ShopEnv,
                          nav: Navigation,
                          single_arm_manip: PR2SingleArmManipulation,
                          object_info: ShopObjectEntry,
                          region: str,
                          ignore_movable: bool=False,
                          predicate_check: bool=False) -> npt.ArrayLike:
        # if handle_pose is given, this is for opening the door, 
        # where navigation already dealt by open_door method
        region: ShopObjectEntry = env.regions[region]
        base_pose = robot.get_pose()
        robot_stand_pose = ShopObjectEntry.find_default_robot_base_pose(robot,
                                                                        region, 
                                                                        env.shop_config[region.area + "_config"], 
                                                                        env.all_obj)

        object_height = self.get_object_height(bc, object_info)
        # object_height = 0.2

        nav_traj = []
        goal_pose = (*robot_stand_pose[0][:2], bc.getEulerFromQuaternion(robot_stand_pose[1])[2])
        curr_pose = (*robot.get_position()[:2], robot.get_euler()[2])

        # Use cached navigation traj
        is_open = env.openable_obj["kitchen_door"].is_open
        cache_key = f"{is_open}-{ignore_movable}-{curr_pose}-{goal_pose}"
        if cache_key in self.nav_traj_cache:
            return self.nav_traj_cache[cache_key]
        # Prepare
        # if ignore_movable:
        #     nav.set_empty_roadmap()
        if robot.activated[robot.main_hand] is not None:
            # nav.set_default_roadmap()
            grasp_uid = robot.activated[robot.main_hand].uid

            if robot.is_holding_receptacle():
                receptacle_uid = robot.activated[robot.get_other_arm()].uid
                grasp_uid = [grasp_uid, receptacle_uid]
            allow_uid_list = self.get_nav_allow_uid_list(bc, robot, env)

            nav.motion_planner.collision_fn = nav._define_pr2_navigation_collision_fn(grasp_uid, allow_uid_list)
        else:
            # if not predicate_check:
            #     nav.set_default_roadmap()
            allow_uid_list = self.get_nav_allow_uid_list(bc, robot, env)
            if robot.is_holding_receptacle():
                receptacle_uid = robot.activated[robot.get_other_arm()].uid
                grasp_uid = receptacle_uid
            nav.motion_planner.collision_fn = nav._define_pr2_navigation_collision_fn(grasp_uid, allow_uid_list)

        # Check before retract
        if nav.motion_planner.collision_fn(curr_pose)[0]:
            return None, None, None

        if not self._check_closeness(bc, base_pose, robot_stand_pose):
            with dream(bc, env, robot):
                # Retract
                attach_xfms = single_arm_manip._get_attach_xfms()
                gripper_pose_from_base = robot.retracted_left_gripper_pose if single_arm_manip.arm == "left" else robot.retracted_right_gripper_pose
                gripper_pose = bc.multiplyTransforms(*robot.get_pose(), *gripper_pose_from_base)
                
                # Raise hand if holding plate
                gripper_pos = gripper_pose[0]
                if "plate" in object_info.name:
                    gripper_pos = np.array(gripper_pose[0])
                    gripper_pos[2] += 0.2
                    gripper_pos = tuple(gripper_pos)
                    
                joints = robot.left_arm_joints if single_arm_manip.arm == "left" else robot.right_arm_joints
                joints = robot.torso_joints + joints
                retracted_hand_pose = (gripper_pos, robot.get_endeffector_pose(single_arm_manip.arm)[1])
                q = robot.arm_ik(single_arm_manip.arm, retracted_hand_pose)
                if q is None:
                    if DEBUG:
                        print("[Placement] No IK")
                    return None, None, None
                robot.set_joint_positions(joints, q)
                single_arm_manip._update_attached_bodies(attach_xfms)


                region_z = env.all_obj[region.entity_group[0]].position[2]
                if region.taskspace is not None:
                    h_diff = region_z + region.taskspace[1][2] - robot.get_endeffector_pose(single_arm_manip.arm)[0][2] + object_height
                else:
                    h_diff = -robot.get_endeffector_pose(single_arm_manip.arm)[0][2] + object_height
                curr_torso = robot.get_joint_state(robot.torso_joints[0]).jointPosition
                alleviated_torso_pos = curr_torso

                if h_diff > 0:
                    attach_xfms = single_arm_manip._get_attach_xfms()
                    alleviated_torso_pos = curr_torso + h_diff + 0.02
                    robot.set_joint_positions(robot.torso_joints, [alleviated_torso_pos])
                    single_arm_manip._update_attached_bodies(attach_xfms)

                retracted_arm_joint_pos = np.array(q)
                retracted_arm_joint_pos[0] = alleviated_torso_pos

                # if predicate_check:
                #     return [curr_pose, goal_pose], nav._SE3fromSE2(goal_pose), tuple(retracted_arm_joint_pos)
            
                allow_uid_list = self.get_nav_allow_uid_list(bc, robot, env)
                if robot.activated[robot.main_hand] is not None:
                    holding_obj_uid = robot.activated[robot.main_hand].uid
                else:
                    holding_obj_uid = None

                nav_traj, _ = nav.motion_plan(curr_pose, goal_pose, holding_obj_uid, allow_uid_list=allow_uid_list, ignore_movable=ignore_movable)

            self.nav_traj_cache[cache_key] = (nav_traj, nav._SE3fromSE2(goal_pose), tuple(retracted_arm_joint_pos))
            return nav_traj, nav._SE3fromSE2(goal_pose), tuple(retracted_arm_joint_pos)
        else:
            nav_traj = []
            goal_pose = (*robot_stand_pose[0][:2], bc.getEulerFromQuaternion(robot_stand_pose[1])[2])

            return nav_traj, None, None

    ## NOTE(SJ): Sampling manipulation 
    def sample_manipulation(self, bc: BulletClient,
                            env: ShopEnv,
                            robot: PR2,
                            single_arm_manip: PR2SingleArmManipulation,
                            object_info: ShopObjectEntry,
                            target_pose: Tuple[Tuple[float]],
                            ignore_movable: bool=False) -> npt.ArrayLike:

        grasp_uid = object_info.uid
        q = robot.arm_ik(single_arm_manip.arm, target_pose, debug=False)
        if q is None:
            if DEBUG:
                print("No possible place pose")
            return None

        before_place_arm_state = robot.get_arm_state(single_arm_manip.arm)

        if ignore_movable:
            collsion_fn = single_arm_manip.empty_collision_fn
        elif robot.is_holding_receptacle():
            receptacle_uid = robot.activated[robot.get_other_arm()].uid
            obj_on_receptacle_uids = [attach_info.uid for (attach_info, _, _) in robot.receptacle_status.values()]
            attachlist = obj_on_receptacle_uids + [receptacle_uid]
            if robot.activated[single_arm_manip.arm] is not None:
                attachlist = [object_info.uid] + attachlist
            else:
                attachlist = [None] + attachlist
            collsion_fn = single_arm_manip._define_grasp_collision_fn(attachlist, allow_uid_list=[])
        elif robot.activated[robot.main_hand] is not None:
            attachlist = grasp_uid
            collsion_fn = single_arm_manip._define_grasp_collision_fn(attachlist, allow_uid_list=[])
        else:
            collsion_fn = single_arm_manip.default_collision_fn
        
        
        with dream(bc, env, robot):
            # if single_arm_manip.motion_planner.collision_fn(q)[0]:
            valid = not collsion_fn(q)[0]
            if not valid:
                if DEBUG:
                    # print(collsion_fn(q))
                    print("Collision at goal")
                return None
            # single_arm_manip.motion_planner.grow(samples)
        traj, _ = single_arm_manip.motion_plan(before_place_arm_state, q, holding_obj_uid=grasp_uid)


        return traj


    def get_gripper_pose_from_obj_pose(self, bc: BulletClient,
                                       robot: PR2,
                                       arm: str,
                                       object_info: ShopObjectEntry,
                                       target_obj_pose: Tuple[Tuple[float]]):
        T_oh = robot.activated[robot.main_hand].object_pose
        T_ho = bc.invertTransform(*T_oh)
        T_hw = bc.multiplyTransforms(*target_obj_pose, *T_ho)

        return T_hw



class OpenDoorSampler(Sampler):

    def __init__(self, NUM_FILTER_TRIALS):
        """Random pick action sampler"""
        super().__init__(NUM_FILTER_TRIALS=1)


    def __call__(self, bc: BulletClient,
                       env: ShopEnv,
                       robot: PR2,
                       nav: Navigation,
                       manip: PR2Manipulation,
                       state: ShopState,
                       discrete_action: ShopDiscreteAction = None,
                       ignore_movable: bool=False) -> ShopContinuousAction:
        """Sample an open door action

        Args:
            bc (BulletClient)
            env (ShopEnv)
            robot (PR2)
            manip (PR2Manipulation)
            state (ShopState)
            discrete_action (ShopDiscreteAction)

        Raises:
            ValueError: Pick sampler being called while holding an object

        Returns:
            ShopContinuousAction: Sampled Pick action
        """

        # Validity check and pre-process
        assert discrete_action.type in [ACTION_OPEN, ACTION_CLOSE], "Should be OPEN or CLOSE action"
        object_info = env.all_obj[discrete_action.aimed_obj.name]
        arm = discrete_action.arm
        if discrete_action.region == None:
            discrete_action.region = discrete_action.aimed_obj.name
        single_arm_manip: PR2SingleArmManipulation = getattr(manip, arm)
        handle_info = env.handles[object_info.name]
        handle_pos, handle_orn, idx = self.get_handle_pose(bc, robot, object_info.uid, handle_info)
        handle_action_info = (handle_info, idx)

        # Navigate to the region
        nav_traj, goal_pose = self.sample_navigation(bc, robot, env, nav, object_info, handle_action_info, (handle_pos, handle_orn), ignore_movable)
        if nav_traj is None:
            if DEBUG:
                print("failed to navigate to the region")
            return ShopContinuousAction(discrete_action, None, None, None, None, handle_pose=(handle_pos, handle_orn), handle_action_info=handle_action_info)

        for i in range(self.NUM_FILTER_TRIALS):
            # Sample pose
            target_pose = self.sample_pose(bc, env, object_info, (handle_pos, handle_orn))

            if len(nav_traj) > 0:
                with dream(bc, env, robot):
                    # Move to the region
                    attach_xfms = single_arm_manip._get_attach_xfms()
                    robot.set_pose(goal_pose)
                    single_arm_manip._update_attached_bodies(attach_xfms)

                    # Motion plan manipulation
                    manip_traj = self.sample_manipulation(bc, robot, single_arm_manip, object_info, target_pose)

            else:
                # Motion plan manipulation
                manip_traj = self.sample_manipulation(bc, robot, single_arm_manip, object_info, target_pose)

            if manip_traj is None:
                if DEBUG:
                    print("Motion planning was unsuccessful")

            return ShopContinuousAction(discrete_action, *target_pose, nav_traj, manip_traj, (handle_pos, handle_orn), handle_action_info=handle_action_info)


    def sample_pose(self, bc: BulletClient,
                    env: ShopEnv, 
                    object_info: ShopObjectEntry,
                    handle_pose: Tuple[Tuple[float]]) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Sample a random pick action

        Args:
            bc (BulletClient)
            env (ShopEnv)
            robot (PR2)
            manip (PR2Manipulation)
            object_pose (Tuple[npt.ArrayLike])
        Raises:
            ValueError: Pick sampler being called while holding an object

        Returns:
            ShopContinuousAction: Sampled Pick action
        """

        handle_pos, handle_orn = handle_pose
        orn_ph = orn_ph = (0, 0, math.pi)
        pick_pose = bc.multiplyTransforms(handle_pos,
                                          handle_orn,
                                          (object_info.pick_param.backward, 0, 0), 
                                          bc.getQuaternionFromEuler(orn_ph))
    
        return pick_pose


    def sample_navigation(self, bc: BulletClient,
                          robot: PR2,
                          env: ShopEnv,
                          nav: Navigation,
                          object_info: ShopObjectEntry,
                          handle_action_info: Tuple,
                          handle_pose: Tuple[Tuple[float]],
                          ignore_movable: bool=False) -> Tuple[npt.ArrayLike, Tuple[float]]:
        # if handle_pose is given, this is for opening the door, 
        # where navigation already dealt by open_door method
        handle_pos, handle_orn = handle_pose
        base_pose = robot.get_pose()
        idx = handle_action_info[1]

        # Get hand pose related to handle
        offset = object_info.robot_offset[idx]
        T_bh = ((offset[0][0], offset[0][1], base_pose[0][-1]), bc.getQuaternionFromEuler(offset[1]))
        T_bw = bc.multiplyTransforms(handle_pos, handle_orn, T_bh[0], T_bh[1])
        curr_pose = (*robot.get_position()[:2], robot.get_euler()[2])
        goal_pose = (*T_bw[0][:2], bc.getEulerFromQuaternion(T_bw[1])[2])

        nav_traj = []
        if not self._check_closeness(bc, base_pose, nav._SE3fromSE2(goal_pose)):
            if ignore_movable:
                nav.set_empty_roadmap()
            else:
                nav.set_default_roadmap()

            allow_uid_list = self.get_nav_allow_uid_list(bc, robot, env)
            if robot.activated[robot.main_hand] is not None:
                holding_obj_uid = robot.activated[robot.main_hand].uid
            else:
                holding_obj_uid = None

            nav_traj, _ = nav.motion_plan(curr_pose, goal_pose, holding_obj_uid=holding_obj_uid, allow_uid_list=allow_uid_list)
        
        return nav_traj, nav._SE3fromSE2(goal_pose)
    

    def sample_manipulation(self, bc: BulletClient,
                            robot: PR2,
                            single_arm_manip: PR2SingleArmManipulation,
                            object_info: ShopObjectEntry,
                            target_pose: Tuple[Tuple[float]]) -> npt.ArrayLike:

        grasp_uid = object_info.uid
        q = robot.arm_ik(single_arm_manip.arm, target_pose, debug=False)
        if q is None:
            if DEBUG:
                print("No possible place pose")
            return None

        before_place_arm_state = robot.get_arm_state(single_arm_manip.arm)
        traj, _ = single_arm_manip.motion_plan(before_place_arm_state, q, holding_obj_uid=None, planner_type="interpolation")


        return traj



CloseDoorSampler = OpenDoorSampler


def draw_sampling_region(bc: BulletClient, t_min: npt.ArrayLike, t_max: npt.ArrayLike, region_pose: Tuple[npt.ArrayLike]):
    c1 = (t_min[0], t_min[1], t_max[2])
    c2 = (t_max[0], t_min[1], t_max[2])
    c3 = (t_max[0], t_max[1], t_max[2])
    c4 = (t_min[0], t_max[1], t_max[2])

    cs = [c1, c2, c3, c4]
    ps = []
    for c in cs:
        p, _ = bc.multiplyTransforms(region_pose[0],
                                    bc.getQuaternionFromEuler(region_pose[1]),
                                    c,
                                    bc.getQuaternionFromEuler((0,0,0)))
        
        p = np.array(p)
        p[2] = t_max[2] + 0.3
        p = tuple(p)
        ps.append(p)
    
    bc.addUserDebugPoints(ps, [[1,0,0]]*4, pointSize=3)
    bc.addUserDebugLine(lineFromXYZ=ps[0], lineToXYZ=ps[1], lineColorRGB=[0,0,1])
    bc.addUserDebugLine(lineFromXYZ=ps[1], lineToXYZ=ps[2], lineColorRGB=[0,0,1])
    bc.addUserDebugLine(lineFromXYZ=ps[2], lineToXYZ=ps[3], lineColorRGB=[0,0,1])
    bc.addUserDebugLine(lineFromXYZ=ps[3], lineToXYZ=ps[0], lineColorRGB=[0,0,1])


def build_distance_table(bc: BulletClient,
                         nav: Navigation,
                         env: ShopEnv,
                         robot: PR2,
                         dump: bool=False) -> Dict[Tuple[str], float]:
        
        regions = [k for k in env.regions.keys()] + ["initial_position",]
        distance_table: Dict[Tuple[str, str], float] = dict()

        for r1 in regions:
            if r1 == "initial_position":
                r1_pose = robot.init_pose
            else:
                r1_pose = ShopObjectEntry.find_default_robot_base_pose(robot,
                                                                    env.regions[r1],
                                                                    env.shop_config[env.regions[r1].area + "_config"],
                                                                    env.all_obj)
            for r2 in regions:
                if r1 == r2:
                    continue
                if f"{r2}-{r1}" in distance_table:
                    distance_table[f"{r1}-{r2}"] = distance_table[f"{r2}-{r1}"]

                if r2 == "initial_position":
                    r2_pose = robot.init_pose
                else:
                    r2_pose = ShopObjectEntry.find_default_robot_base_pose(robot,
                                                                       env.regions[r2],
                                                                       env.shop_config[env.regions[r2].area + "_config"],
                                                                       env.all_obj)
                
                q1 = nav._SE2fromSE3(r1_pose)
                q2 = nav._SE2fromSE3(r2_pose)

                nav.set_empty_roadmap()
                path = nav.motion_plan(q1, q2, None, [], ignore_movable=True)[0]
                distance = nav.get_traj_distance(path)
                distance_table[f"{r1}-{r2}"] = distance

        if dump:
            with open("navigation_distance_table.json", "w") as f:

                json.dump(distance_table, f)

        return distance_table