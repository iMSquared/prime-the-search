import random
import numpy as np


from typing import List, Dict, Optional, Any, Tuple
from copy import deepcopy
from collections import defaultdict


# Simulations
from Simulation.pybullet.envs.shop_env import ShopEnv
from Simulation.pybullet.envs.navigation import Navigation, dream, set_env_from_capture
from Simulation.pybullet.envs.manipulation import Manipulation, PR2Manipulation, PR2SingleArmManipulation
from Simulation.pybullet.envs.robot import Robot, PR2, AttachConstraint
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.mdp.MDP_framework import Dict, HistoryEntry, Tuple
from Simulation.pybullet.mdp.shop.shop_MDP import ShopGoal, capture_shopenv, capture_shopenv_state, ShopState, ShopObjectEntry, ShopDiscreteAction, ShopContinuousAction, ACTION_PICK, ACTION_PLACE, Relation
from Simulation.pybullet.mdp.shop.policy.default_samplers import PickSampler, RandomPlaceSampler

# Framework
from Simulation.pybullet.predicate.predicate_framework import AgentPredicateManager, AssetPredicateManager, PredicateManager

from Simulation.pybullet.custom_logger import LLOG
logger = LLOG.get_logger()
DEBUG = False

AREA_MAP = {"hall": "hall", "kitchen": "kitchen", "initial_position": "hall"}
     

def makePairs(state: ShopState, objs: List[ShopObjectEntry]) -> List[Tuple[ShopObjectEntry]]:
    pairs = []
    for sub in objs:
        for ref in objs:
            if sub.is_openable:
                continue
            if sub.name == ref.name:
                continue
            if state.receptacle_holding_info is not None and sub.is_receptacle:
                continue
            pairs.append((sub, ref))
    return pairs


def makeDistinctPairs(objs: List[ShopObjectEntry]):
    pairs = []
    for i in range(len(objs)):
        for j in range(i+1, len(objs)):
            pairs.append((objs[i], objs[j]))
    return pairs


class ShopAgentPredicateManager(AgentPredicateManager):
    """
    Agent Embodiment predicates
    """
    def __init__(self, config: Dict):
        super().__init__(config)    

    def atRegion(self, state: ShopState):
        return state.region

    def right_hand_holding(self, state: ShopState):
        directly_holding = getattr(state.holding_status["right"], "uid", None)
        if directly_holding is not None:
            directly_holding = self.env.uid_to_name[directly_holding]
        return directly_holding

    def left_hand_holding(self, state: ShopState):
        directly_holding = getattr(state.holding_status["left"], "uid", None)
        if directly_holding is not None:
            directly_holding = self.env.uid_to_name[directly_holding]
        return directly_holding
    
    def right_hand_empty(self, state: ShopState):
        return state.holding_status["right"] is None
    
    def left_hand_empty(self, state: ShopState):
        return state.holding_status["left"] is None
    
    
    def evaluate(self, env: ShopEnv, state:ShopState):
        self.env = env
        predicates = {
            'at': self.atRegion(state),
            'right_hand_holding': self.right_hand_holding(state),
            'left_hand_holding': self.left_hand_holding(state),
            'right_hand_empty': self.right_hand_empty(state),
            'left_hand_empty': self.left_hand_empty(state),
            'is_holding': self.right_hand_holding(state) is not None or self.left_hand_holding(state) is not None, 
            'has_empty_hand': self.right_hand_empty(state) #or self.left_hand_empty(state)
        }

        return predicates


class ShopAssetPredicateManager(AssetPredicateManager):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.skip_occlusion_predicates = config["plan_params"]["skip_occlusion_predicates"]

        self.bc : BulletClient = None # type: ignore
        self.env : ShopEnv = None # type: ignore
        self.robot : PR2 = None # type: ignore
        self.manip : PR2Manipulation = None # type: ignore
        self.nav : Navigation = None # type: ignore
        self.state : ShopState = None # type: ignore
        self.config : Dict = config # type: ignore
        self.pick_sampler = PickSampler(config["pose_sampler_params"]["num_filter_trials_state_description"])
        self.place_sampler = RandomPlaceSampler(config["pose_sampler_params"]["num_filter_trials_state_description"])
        self.NUM_FILTER_TRIALS_PICK  = config["pose_sampler_params"]["num_filter_trials_pick"]
        self.NUM_FILTER_TRIALS_PLACE = config["pose_sampler_params"]["num_filter_trials_place"]
        self.NUM_FILTER_TRIALS_STATE_DESCRIPTION = config["pose_sampler_params"]["num_filter_trials_state_description"]

        self.assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.movable_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.not_movable_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.region_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.not_region_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.openable_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.not_openable_assets : Dict[str, ShopObjectEntry] = None # type: ignore



        self.diff = 1e8
        self.eps = 1e-2
        self.region_info_cache : Dict[str, Dict[str, Any]] = {}
        self.region2objs = defaultdict(list) 

        self.is_absolute: bool = None # type: ignore
        self.use_relative_occluison: bool = None
        ## TODO (SJ): Add option for relative position predicates
        # self.use_realtive_positition: bool = None
        
        
    
    def free_class_variables(self):
        self.bc = None # type: ignore
        self.env = None # type: ignore
        self.robot = None # type: ignore
        self.manip = None # type: ignore
        self.nav = None # type: ignore
        self.state = None # type: ignore
        # self.config = None # type: ignore
        # self.pick_sampler = None # type: ignore
        # self.place_sampler = None # type: ignore
        # self.NUM_FILTER_TRIALS_PICK = None # type: ignore
        # self.NUM_FILTER_TRIALS_PLACE = None # type: ignore
        # self.NUM_FILTER_TRIALS_STATE_DESCRIPTION = None # type: ignore


        self.assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.movable_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.not_movable_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.region_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.not_region_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.openable_assets : Dict[str, ShopObjectEntry] = None # type: ignore
        self.not_openable_assets : Dict[str, ShopObjectEntry] = None # type: ignore


        self.diff = 1e8
        self.region_info_cache: Dict[str, Dict[str, Any]] = {}
        self.region2objs = defaultdict(list) 
    

        self.is_absolute : bool = None # type: ignore


    def set_class_variables(self, vars: Dict,):
        self.bc = vars['bc']
        self.env = vars['env']
        self.robot = vars['robot']
        self.manip = vars['manip']
        self.nav = vars['nav']
        self.state = vars['state']


        # self.config = vars['config']
        # self.pick_sampler = PickSampler(vars['config']["pose_sampler_params"]["num_filter_trials_state_description"])
        # self.place_sampler = RandomPlaceSampler(vars['config']["pose_sampler_params"]["num_filter_trials_state_description"])
        # self.NUM_FILTER_TRIALS_PICK  = vars['config']["pose_sampler_params"]["num_filter_trials_pick"]
        # self.NUM_FILTER_TRIALS_PLACE = vars['config']["pose_sampler_params"]["num_filter_trials_place"]
        # self.NUM_FILTER_TRIALS_STATE_DESCRIPTION = vars['config']["pose_sampler_params"]["num_filter_trials_state_description"]

        self.assets = vars['assets']
        self.movable_assets = vars['movable_assets']
        self.not_movable_assets = vars['not_movable_assets']
        self.region_assets = vars['region_assets']
        self.not_region_assets = vars['not_region_assets']
        self.openable_assets = vars['openable_assets']
        self.not_openable_assets = vars['not_openable_assets']

        self.is_absolute = vars['config']['predicate_params']['is_absolute'] if 'is_absolute' in vars['config']['predicate_params'] else True

        self.region2objs = self.make_region2objs(self.movable_assets, self.region_assets)

        self.use_relative_occluison = self.config['predicate_params']['use_relative_occlusion']

    
    def is_in_region(self, env: ShopEnv, obj: ShopObjectEntry, region: ShopObjectEntry):
        return obj.region == region.name
    

    def make_region2objs(self, movable_assets: Dict[str,ShopObjectEntry], regions: Dict[str,ShopObjectEntry]):
        region2objs = {region: [] for region in regions if 'door' not in region}
        for name, obj in movable_assets.items():
            region2objs[obj.region].append(obj)

        return region2objs


    def in_region(self, env: ShopEnv, movable_assets: Dict[str,ShopObjectEntry], regions: Dict[str,ShopObjectEntry]):
        predicates = {}
        for region, objs in self.region2objs.items():
            for obj in objs:
                predicates[obj.name] = region
        predicates = {obj_name: {'in_region': region_name} for obj_name, region_name in predicates.items()}
        return predicates
    
    
    def empty_region(self, env: ShopEnv, assets: List[ShopObjectEntry], ):
    
        predicates = {region: {'is_empty_region': len(objs) == 0} for region, objs in self.region2objs.items()}

        return predicates


    def restore_holding(self, holding_status: Dict[str, AttachConstraint]):
        for arm, attach_info in holding_status.items():
            attach_xfms = self.nav._get_attach_xfms()
            self.robot.release(arm)
            if attach_info is not None:
                joints = attach_info.joints
                joint_positions = attach_info.joint_positions
                self.robot.set_joint_positions(joints, joint_positions)
                self.nav._update_attached_bodies(attach_xfms)
                self.robot.activate(arm, [attach_info.uid])
            del attach_xfms
                
                 
    def is_occluded_pre(self, state: ShopState, 
                 obj1: ShopObjectEntry, 
                 obj2: ShopObjectEntry,
                 arm: str,
                 force_compute: bool=False):
        """
        Evaluate the predicate
        obj1 is subject object
        obj2 is reference object
        """

        
        assert arm in ["left", "right"], "arm should be either right or left"

        # Step 0. check cache. Save holding status
        if "asset" in state.predicates and obj1.name in state.predicates["asset"] \
            and "is_occ_pre" in state.predicates["asset"][obj1.name]:
            value = (obj2.name in state.predicates["asset"][obj1.name]["is_occ_pre"])
            if DEBUG:
                print(f"[OccPre] Using cached predicates: : {obj1.name}-{obj2.name}")

            if not force_compute or value:
                return value

        if self.config["project_params"]["overridable"]["value"] == "hcount":
            if (obj1.name,) in state.action_cache.collisions \
                and obj2.name in state.action_cache.collisions[(obj1.name,)]:
                return True
        
        if (obj1.name,) in state.action_cache.collisions:
            if not force_compute:
                return (obj2.name in state.action_cache.collisions[(obj1.name,)])
        else:
            state.action_cache.collisions[(obj1.name,)] = set()

        real_holding_status = deepcopy(self.robot.activated)
        single_arm_manip = getattr(self.manip, arm)

        # Dummy discrete action
        holding_status = state.holding_status
        with dream(self.bc, self.env, self.robot):

            self.robot.release(arm)
            
            # Step 1. find the motion plan with all the movable objects
            pick_discrete_action = ShopDiscreteAction(ACTION_PICK, arm, obj1)
            if f"{arm}-{obj1.name}" in state.action_cache.pick_action:
                action = state.action_cache.pick_action[f"{arm}-{obj1.name}"]
            else:
                action = self.pick_sampler(self.bc, self.env, self.robot, self.nav, 
                                            self.manip, state, pick_discrete_action,
                                            ignore_movable=False,
                                            predicate_check=True,
                                            debug=False)
                
                if action.manip_traj is not None:
                    occluded = False
                    match = False
                    for name, info in self.env.openable_obj.items():
                        area = AREA_MAP[state.region] if state.region in ["hall", "kitchen", "initial_position"] else self.env.all_obj[state.region].area
                        if not info.is_open and obj1.area != area:
                            occluded = True
                            match = (name == obj1.name)
                            state.action_cache.collisions[(obj1.name,)].add(name)
                        if not occluded:
                            state.action_cache.pick_action[f"{arm}-{obj1.name}"] = action
                    
                    return occluded and match
                
            # Step 2. find motion plan ignoring movable objects
            if action.manip_traj is not None:
                if f"{arm}-{obj1.name}" not in state.action_cache.pick_joint_pos:
                    state.action_cache.pick_joint_pos[f"{arm}-{obj1.name}"] = [action.manip_traj[-1]]
                else:
                    state.action_cache.pick_joint_pos[f"{arm}-{obj1.name}"].append(action.manip_traj[-1])
                self.restore_holding(real_holding_status)
                return False

            elif f"{arm}-{obj1.name}" in state.action_cache.no_obstacle_pick_action:
                if DEBUG:
                    print(f"[OccPre] using cached pick-traj: {arm}-{obj1.name}")
                action = state.action_cache.no_obstacle_pick_action[f"{arm}-{obj1.name}"]
            else:
                action = self.pick_sampler(self.bc, self.env, self.robot, self.nav, 
                                           self.manip, state, pick_discrete_action,
                                           ignore_movable=True,
                                           predicate_check=True,
                                           debug=False)
                
                if action.manip_traj is not None:
                    state.action_cache.no_obstacle_pick_action[f"{arm}-{obj1.name}"] = action

            if action.manip_traj is None:
                # NOTE (dlee): Should we build new roadmap here? I guess not for now.
                # The object is in the unfetchable region.
                self.restore_holding(real_holding_status)
                return True

            # Step 3. Check collision at each one of the trajectory
            if f"{arm}-{obj1.name}" not in state.action_cache.pick_joint_pos:
                state.action_cache.pick_joint_pos[f"{arm}-{obj1.name}"] = [action.manip_traj[-1]]
            else:
                state.action_cache.pick_joint_pos[f"{arm}-{obj1.name}"].append(action.manip_traj[-1])
            

            if self.robot.activated[self.robot.main_hand] is not None:
                grasp_uid = self.robot.activated[self.robot.main_hand].uid
                allow_uid_list = []
                if self.robot.is_holding_receptacle():
                    receptacle_uid = self.robot.activated[self.robot.get_other_arm()].uid
                    grasp_uid = [grasp_uid, receptacle_uid]
                    allow_uid_list = [receptacle_uid] + [attach_info.uid for (attach_info, _, _) in self.robot.receptacle_status.values()]
                collision_fn = self.nav._define_pr2_navigation_collision_fn(grasp_uid, allow_uid_list)
            else:
                if self.robot.is_holding_receptacle():
                    receptacle_uid = self.robot.activated[self.robot.get_other_arm()].uid
                    grasp_uid = receptacle_uid
                    allow_uid_list = [receptacle_uid] + [attach_info.uid for (attach_info, _, _) in self.robot.receptacle_status.values()]
                    collision_fn = self.nav._define_pr2_navigation_collision_fn(grasp_uid, allow_uid_list)
                else:
                    collision_fn = self.nav.default_collision_fn

            for q in reversed(action.nav_traj):
                is_collision, contact_infos = collision_fn(q)
                if is_collision:
                    for contact_point in contact_infos:
                        uid = contact_point[2]
                        if uid not in self.env.uid_to_name:
                            continue
                        obj_name = self.env.uid_to_name[uid]
                        state.action_cache.collisions[(obj1.name,)].add(obj_name)
                    if obj2.name in state.action_cache.collisions[(obj1.name,)]:
                        self.restore_holding(real_holding_status)
                        return True
                    
            for name, info in self.env.openable_obj.items():
                area = AREA_MAP[state.region] if state.region in ["hall", "kitchen", "initial_position"] else self.env.all_obj[state.region].area
                if not info.is_open and obj1.area != area:
                    state.action_cache.collisions[(obj1.name,)].add(name)
                    
            # # Move to the region
            if len(action.nav_traj) > 0:
                attach_xfms = self.nav._get_attach_xfms()
                self.robot.set_pose(self.nav._SE3fromSE2(action.nav_traj[-1]))
                self.nav._update_attached_bodies(attach_xfms)
            
            collision_fn = single_arm_manip.default_collision_fn
            # for q in reversed(action.manip_traj):
            manip_traj = [action.manip_traj[-1], action.manip_traj[0]]
            for q in manip_traj:
                is_collision, contact_infos = collision_fn(q)
                if is_collision:
                    for contact_point in contact_infos:
                        uid = contact_point[2]
                        if uid not in self.env.uid_to_name:
                            continue
                        obj_name = self.env.uid_to_name[uid]
                        state.action_cache.collisions[(obj1.name,)].add(obj_name)
                    if obj2.name in state.action_cache.collisions[(obj1.name,)]:
                        self.restore_holding(real_holding_status)
                        return True
                    
        
        self.restore_holding(real_holding_status)
        return False


    def occludes_pre(self, state: ShopState, objs: List[ShopObjectEntry], force_compute: bool=False):
        """
        Evaluate the predicate
        """
        output = {obj.name: [] for obj in objs if not obj.is_openable}
        evalObjPairs = makePairs(state, objs)
        for sub, ref in evalObjPairs:
            if sub.region == "receptacle":
                continue
            pred = self.is_occluded_pre(state, sub, ref, arm='right', force_compute=force_compute)
            if pred:
                output[sub.name].append(ref.name)

        predicates = {sub: {'is_occ_pre': refs} for sub, refs in output.items()}
        return predicates
    

    def is_occluded_manip(self, 
                state: ShopState,
                obj1: ShopObjectEntry, 
                obj2: ShopObjectEntry, 
                r: ShopObjectEntry,
                direction: str,
                arm: str,
                force_compute: bool=False):
        """
        Evaluate the predicate
        """
        if DEBUG:
            print(obj1.name, obj1.region, obj2.name, obj2.region, r.name)

        assert arm in ["left", "right"], "arm should be either right or left"
        assert direction in ["left_of", "right_of", "front_of", "behind_of", "on"], "direction should be left, right, front, behind"

        # Step 0. check cache. Save attach constraint, Reusing cached predicates
        if "asset" in state.predicates and obj1.name in state.predicates["asset"] \
        and "is_occ_manip" in state.predicates["asset"][obj1.name] \
        and (direction, r.name) in state.predicates["asset"][obj1.name]["is_occ_manip"]:
        # and r.name in state.predicates["asset"][obj1.name]["is_occ_manip"]:
        
            occ = (obj2.name in state.predicates["asset"][obj1.name]["is_occ_manip"][(direction, r.name)])
            if DEBUG:
                print(f"[OccManip] Using cached predicates: {obj1.name}-{obj2.name}-{direction}-{r.name}")
            
            has_place = state.predicates["asset"][obj1.name]["has_placement_pose"][(direction, r.name)]
            ## NOTE (SJ): Use cached predicates for only non occlusion cases

            if self.config["project_params"]["overridable"]["value"] == "hcount":
                return occ, has_place

            if not force_compute or occ:
                return occ, has_place

        

        if (obj1.name, direction, r.name) in state.action_cache.collisions:

            occ = (obj2.name in state.action_cache.collisions[(obj1.name, direction, r.name)])
            ## NOTE (SJ): Use cached predicates for only non occlusion cases
            if f"{arm}-{obj1.name}-{direction}-{r.name}" in state.action_cache.place_action:  
                has_place = True
            else:
                has_place = f"{arm}-{obj1.name}-{direction}-{r.name}" in state.action_cache.no_obstacle_place_action

            if not force_compute or occ:
                return occ, has_place

        
        real_holding_status = deepcopy(self.robot.activated)
        self.robot.release(arm)
        single_arm_manip: PR2SingleArmManipulation = getattr(self.manip, arm)
        
        # Set collision cache
        if (obj1.name, direction, r.name) not in state.action_cache.collisions:
            state.action_cache.collisions[(obj1.name, direction, r.name)] = set()

        # Dummy discrete action
        with dream(self.bc, self.env, self.robot):
            receptacle_status_changed = False
            value = None
            if obj1.region in self.env.receptacle_obj:
                single_arm_manip.remove_object_from_receptacle(obj1)
                receptacle_status_changed = True               
            
            else:
                # Move to the region where obj1 is
                obj1_region = self.env.regions[obj1.region]
                obj1_region_robot_pose = ShopObjectEntry.find_default_robot_base_pose(self.robot,
                                                                                    obj1_region,
                                                                                    self.env.shop_config[obj1_region.area + "_config"],
                                                                                    self.env.all_obj)
                attach_xfms = self.nav._get_attach_xfms() ##NOTE (SJ) Look into functionality
                self.robot.set_pose(obj1_region_robot_pose)
                self.nav._update_attached_bodies(attach_xfms)

                # Step 1. pick Obj1
                # Check if cache-hit

                if f"{arm}-{obj1.name}" in state.action_cache.pick_joint_pos \
                    and len(state.action_cache.pick_joint_pos[f"{arm}-{obj1.name}"]) > 0:
                    q = random.choice(state.action_cache.pick_joint_pos[f"{arm}-{obj1.name}"])
                else:
                    pick_pose = self.pick_sampler.sample_pose(self.bc, self.robot, self.env, obj1)
                    q = self.robot.arm_ik(arm, pick_pose)
                    if q is not None:
                        state.action_cache.pick_joint_pos[f"{arm}-{obj1.name}"] = [q]
                
                if q is None:
                    if receptacle_status_changed:
                        receptacle_uid = self.robot.activated[self.robot.get_other_arm()].uid
                        receptacle = self.env.regions[self.env.uid_to_name[receptacle_uid]]
                        single_arm_manip.fix_object_on_receptacle(receptacle, obj1, stabilize=False)  
                    self.restore_holding(real_holding_status)
                    if DEBUG:
                        print("[Placement] NO IK solution")
                    return False, False

                # single_arm_manip.open_gripper()
                self.robot.set_joint_positions(single_arm_manip.arm_joints, q)
                single_arm_manip.close_gripper(obj1, delay=0)
                self.robot.activate(arm, [obj1.uid])

            if self.robot.is_holding_receptacle():
                other_arm = self.robot.get_other_arm()
                if self.robot.activated[other_arm] is None:
                    if DEBUG:
                        print("No possible pick pose. Probably failed to grasp the receptacle.")
                    return False, False
                receptacle = self.env.uid_to_name[self.robot.activated[other_arm].uid]
                receptacle = self.env.all_obj[receptacle]

                T_oh = self.robot.activated[other_arm].object_pose
                # T_hw = self.bc.getLinkState(self.robot.uid, self.robot.gripper_link_indices[other_arm][0])[:2]
                T_hw = self.get_endeffector_pose(arm)
                T_ow = self.bc.multiplyTransforms(*T_hw, *T_oh)
                self.bc.resetBasePositionAndOrientation(receptacle.uid, *T_ow)

                self.robot.release(other_arm)
                self.robot.activate(other_arm, [receptacle.uid], pick_receptacle=True)

            # Step 2. find the motion plan with all the movable objects
            if r.is_region:
                place_discrete_action = ShopDiscreteAction(ACTION_PLACE, arm, obj1, r.name, direction=direction, reference_obj=r)
            else:
                place_discrete_action = ShopDiscreteAction(ACTION_PLACE, arm, obj1, r.region, direction=direction, reference_obj=r)
            
            if f"{arm}-{obj1.name}-{direction}-{r.name}" in state.action_cache.place_action:
                action = state.action_cache.place_action[f"{arm}-{obj1.name}-{direction}-{r.name}"]
            else:
                curr_pose = (*self.robot.get_position()[:2], self.robot.get_euler()[2])
                allow_uid_list = [v[0].uid for v in self.robot.receptacle_status.values()]
                collision_fn = self.nav._define_pr2_navigation_collision_fn([v.uid for v in self.robot.activated.values() if v is not None],
                                                                             allow_uid_list=allow_uid_list)
                is_collision, contact_infos = collision_fn(curr_pose)
                if is_collision:
                    # for contact_point in contact_infos:
                    #     uid = contact_point[2]
                    #     if uid not in self.env.uid_to_name or obj1.uid == uid:
                    #         continue
                    #     obj_name = self.env.uid_to_name[uid]
                    #     state.action_cache.collisions[(obj1.name, direction, r.name)].add(obj_name)
                    action = ShopContinuousAction(place_discrete_action, None, None, None, None,)
                else:
                    action = self.place_sampler(self.bc, self.env, self.robot, self.nav, 
                                                self.manip, state, place_discrete_action, 
                                                ignore_movable=False, predicate_check=True)

            if action.manip_traj is not None:
                occluded = False
                match = False
                for name, info in self.env.openable_obj.items():
                    if not info.is_open and obj1.area != r.area:
                        occluded = True
                        match = (name == obj2.name)
                        state.action_cache.collisions[(obj1.name, direction, r.name)].add(name)

                if DEBUG: 
                    print("[OccManip] Found traj w/o collision")
                
                ## NOTE (SJ): Update cache for successful trajectories
                action_cache_key = f"{arm}-{obj1.name}-{direction}-{r.name}"
                if not occluded:
                    state.action_cache.place_action[action_cache_key] = action
                else:
                    state.action_cache.no_obstacle_place_action[action_cache_key] = action
                value = (occluded and match, True)

            # Step 3. find motion plan ignoring movable objects
            elif f"{arm}-{obj1.name}-{direction}-{r.name}" in state.action_cache.no_obstacle_place_action:
                if DEBUG:
                    print("using cached place-traj")
                action = state.action_cache.no_obstacle_place_action[f"{arm}-{obj1.name}-{direction}-{r.name}"]
            else:
                action = self.place_sampler(self.bc, self.env, self.robot, self.nav, 
                                            self.manip, state, place_discrete_action,
                                            ignore_movable=True,
                                            predicate_check=True,
                                            debug=False)
                
                # Cache traj w/o movable objs
                if action.manip_traj is not None:
                    state.action_cache.no_obstacle_place_action[f"{arm}-{obj1.name}-{direction}-{r.name}"] = action
            
            if value is None and action.manip_traj is None:
                # NOTE (dlee): Should we build new roadmap here? I guess not for now.
                # The object is in the unfetchable region.
                value = (False, False)


            if value is None:
                with dream(self.bc, self.env, self.robot):
                    # Step 4. Check collision at each one of the trajectory
                    allow_uid_list = [v[0].uid for v in self.robot.receptacle_status.values()]
                    collision_fn = self.nav._define_pr2_navigation_collision_fn([v.uid for v in self.robot.activated.values() if v is not None],
                                                                             allow_uid_list=allow_uid_list)
                    for q in action.nav_traj:
                        is_collision, contact_infos = collision_fn(q)
                        if is_collision:
                            for contact_point in contact_infos:
                                uid = contact_point[2]
                                if uid not in self.env.uid_to_name or obj1.uid == uid:
                                    continue
                                obj_name = self.env.uid_to_name[uid]
                                state.action_cache.collisions[(obj1.name, direction, r.name)].add(obj_name)
                                # Navigation fails without touching door
                                if "door" not in obj_name:
                                    value = (False, False)
                    
                    for name, info in self.env.openable_obj.items():
                        if not info.is_open and obj1.area != r.area:
                            state.action_cache.collisions[(obj1.name, direction, r.name)].add(name)

                    if obj2.name in state.action_cache.collisions[(obj1.name, direction, r.name)]:
                        value = (True, True)
                    

                    if value is None or not value[1]:
                        collision_fn = single_arm_manip._define_grasp_collision_fn(obj1.uid, allow_uid_list=[])
                        # for q in reversed(action.manip_traj):
                        manip_traj = [action.manip_traj[-1], action.manip_traj[0]]
                        for q in manip_traj:
                            is_collision, contact_infos = collision_fn(q)
                            if is_collision:
                                for contact_point in contact_infos:
                                    uid = contact_point[2]
                                    if uid not in self.env.uid_to_name:
                                        continue
                                    obj_name = self.env.uid_to_name[uid]
                                    state.action_cache.collisions[(obj1.name, direction, r.name)].add(obj_name)
                        if obj2.name in state.action_cache.collisions[(obj1.name, direction, r.name)]:
                            value = (True, True)
                
        if receptacle_status_changed:
            receptacle_uid = self.robot.activated[self.robot.get_other_arm()].uid
            receptacle = self.env.regions[self.env.uid_to_name[receptacle_uid]]
            single_arm_manip.fix_object_on_receptacle(receptacle, obj1, stabilize=False)
        
        self.restore_holding(real_holding_status)

        if value is None:
            value = (False, True)  
        
        return value


    def occludes_manip(self, state: ShopState, objs: List[ShopObjectEntry], regions: List[ShopObjectEntry], force_compute: bool=False):
        """
        Evaluate the predicate
        """

        directions = ["left_of", "right_of", "front_of", "behind_of"]
        
        # Output: OccManip
        output: Dict[Dict] = defaultdict(dict)
        for sub in objs:
            if sub.is_openable: continue

            # On regions
            for r in regions:
                if r.is_openable: continue
                direction = "on"
                occluders = set()
                for ref in objs:
                    if sub == ref: continue
                    if sub.name == r.name: continue
                    if "receptacle" in [r.name, sub.name]: continue
                    pred, has_place = self.is_occluded_manip(state, sub, ref, r, direction, arm='right', force_compute=force_compute)
                    if pred:
                        occluders.add(ref.name)
                output[sub.name][(direction, r.name)] = (list(occluders), has_place)
                

            # Relative to objs
            for o in objs:
                if o.is_openable: continue
                for direction in directions:
                    occluders = set()
                    for ref in objs:
                        if sub == ref: continue
                        if sub.name == o.name: continue
                        
                        if "receptacle" in [ref.name, sub.name]: continue

                        # NOTE (dlee): Test whether we need this
                        
                        if self.use_relative_occluison:
                            pred, has_place = self.is_occluded_manip(state, sub, ref, o, direction, arm='right')
                        else:
                            pred, has_place = False, True

                        if pred:
                            occluders.add(ref.name)
                    output[sub.name][(direction, o.name)] = (list(occluders), has_place)
        
            

        # predicates = {sub: {'is_occ_manip': {r: occluders},
        #                     'has_placement_pose':{r: has_place}} for r, (occluders, has_place) in ref.items() for sub, ref in output.items()}
        
        predicates = dict()
        for sub, ref in output.items():
            predicates[sub] = defaultdict(dict)
            for r, (occluders, has_place) in ref.items():
                predicates[sub]["is_occ_manip"][r] = occluders
                predicates[sub]["has_placement_pose"][r] = has_place
        
        return predicates


    def is_relative_position(self, obj1: ShopObjectEntry, obj2: ShopObjectEntry):
        """
        Evaluate the predicate based on the specified direction.
        """

        obj1_x, obj1_y, obj1_z = self.convert_cordinate(obj1)
        obj2_x, obj2_y, obj2_z = self.convert_cordinate(obj2)

        # Check the direction and evaluate accordingly
        left = obj1_y >= obj2_y
        right = obj1_y < obj2_y
        behind = obj1_x >= obj2_x
        front = obj1_x < obj2_x

        # Decide one for x shape layout
        if abs(obj1_y - obj2_y) > abs(obj1_x - obj2_x):
            front = behind = False
        else:
            right = left = False

        output = {
            'left': left,
            'right': right,
            'front': front,
            'behind': behind
        }

        return output

    def convert_cordinate(self, obj: ShopObjectEntry):
        # World frame coordinate
        pos = obj.position
        ori = obj.orientation
        

        if self.is_absolute:
            obj_x, obj_y, obj_z = pos
        else:  # Robot frame coordinate
            region = self.env.regions[obj.region]
            robot_pose = ShopObjectEntry.find_default_robot_base_pose(self.robot,
                                                                      region, 
                                                                      self.env.shop_config[region.area + "_config"], 
                                                                      self.env.all_obj)
            T_rw = self.bc.invertTransform(*robot_pose)
            ori = ori if len(ori) == 4 else self.bc.getQuaternionFromEuler(ori)
            obj_pose = (pos, ori)
            pos_o1r, _ = self.bc.multiplyTransforms(*T_rw, *obj_pose)

            obj_x, obj_y, obj_z = pos_o1r

        return (obj_x, obj_y, obj_z)


    def region_layout(self, objs: List[ShopObjectEntry]):
        """
        Evaluate the predicate
        """
        name_cord = {}
        for obj in objs:
            obj_x, obj_y, obj_z = self.convert_cordinate(obj)
            name_cord[obj.name] =  {"x": obj_x, "y": obj_y, "z": obj_z}

        left_to_right = sorted(name_cord.items(), key=lambda x: x[1]["y"])
        front_to_behind = sorted(name_cord.items(), key=lambda x: x[1]["x"])

        left_to_right = [name for name, _ in left_to_right]
        front_to_behind = [name for name, _ in front_to_behind]

        return {"left_to_right": left_to_right, "front_to_behind": front_to_behind}
    
    
    def relative_position(self, objs: List[ShopObjectEntry]):
        """
            args:
                objs: List[ShopObjectEntry] : list of movable objects that are in the region
            condition:
                if there is only one object, return empty dictionary since there is no relative position
            return:

        """
        # predicate = {}
        # predicate = {obj.name: {'on': [], 'below': [], 'left': [], 'right': [], 'front': [], 'behind': []} for obj in objs}

        predicate = {obj.name: {'relative_position':{'left': [], 'right': [], 'front': [], 'behind': []}} for obj in objs}
        counterpart = {"left": "right", "right": "left", "front": "behind", "behind": "front"}

        if len(objs) == 1:
            return predicate
        else:
            objPairs = makeDistinctPairs(objs)

        for obj1, obj2 in objPairs:
            result = self.is_relative_position(obj1, obj2)
            for direction, value in result.items():
                if value:
                    predicate[obj1.name]['relative_position'][direction].append(obj2.name)
                    predicate[obj2.name]['relative_position'][counterpart[direction]].append(obj1.name)

        return predicate
    

    ## NOTE (SJ): absolutePosition Predicate
    def get_center_aabb(self, aabb: Tuple[Tuple[float, float, float], Tuple[float, float, float]], center_proportion: float=0.33) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
            x_min, y_min, z_min = aabb[0]
            x_max, y_max, z_max = aabb[1]
            
            proportion = (1-center_proportion)/2
            x_margin = (x_max - x_min) * proportion
            y_margin = (y_max - y_min) * proportion
            z_margin = (z_max - z_min) * proportion

            center_block = ((x_min + x_margin, y_min + y_margin, z_min + z_margin), (x_max - x_margin, y_max - y_margin, z_max - z_margin))
            return center_block


    ## NOTE (SJ): absolutePosition Predicate
    def get_regionSection(self, obj_CoM: Tuple[float, float, float], center_block: Tuple[Tuple[float, float, float], Tuple[float, float, float]]):
        # If the object is exactly at the border between two regions, 
        # this function places it in the 'higher' or 'righter' region.
        # Determine the x-axis position (front, behind, or center)
        if obj_CoM[0] < center_block[0][0]:
            x_position = "behind"
        elif obj_CoM[0] > center_block[1][0]:
            x_position = "front"
        else:
            x_position = "center"

        # Determine the y-axis position (left, right, or center)
        if obj_CoM[1] < center_block[0][1]:
            y_position = "left"
        elif obj_CoM[1] > center_block[1][1]:
            y_position = "right"
        else:
            y_position = "center"

        # Combine x and y positions to determine the region
        # If both positions are 'center', return 'center', otherwise combine the positions
        return "center" if x_position == y_position == "center" else f'{x_position}-{y_position}'


    def is_absolute_position(self, obj: ShopObjectEntry, region: ShopObjectEntry):
        output = {
            "behind-left": False,
            "behind": False,
            "behind-right": False,
            "left": False,
            "center": False,
            "right": False,
            "front-left": False,
            "front": False,
            "front-right": False
        }
        ## TODO (SJ): check if is_absolute and getting orientation for aabb is possible
        obj_CoM = obj.position
        region_aabb = ShopObjectEntry.get_aabb(self.bc, region)
        center_block = self.get_center_aabb(region_aabb)
        region_section = self.get_regionSection(obj_CoM, center_block)
        output[region_section] = True
        return output
    
    
    # NOTE (SJ): Can be better implemented with reducing aabb section calculation
    def absolute_position(self, objs: List[ShopObjectEntry], region: ShopObjectEntry):
        '''
            args:
                objs: List[ShopObjectEntry] : list of movable objects that are in the region
                region: ShopObjectEntry : region the objects are in
            return:
                output: Dict[str, List[ShopObjectEntry]] : dictionary of objects in each region section
        '''
        predicates = {}

        ## EXPLANATION (SJ): USED pre-calculated region info for efficiency if exists
        if region.name in self.region_info_cache:
            region_aabb = self.region_info_cache[region.name]['aabb']
            center_block = self.region_info_cache[region.name]['center_block']
        else: 
            region_aabb = ShopObjectEntry.get_aabb(self.bc, region)
            center_block = self.get_center_aabb(region_aabb)
            self.region_info_cache[region.name] = {
                'aabb': region_aabb,
                'center_block': center_block
            }

        for _obj in objs:
            obj_CoM = _obj.position
            region_section = self.get_regionSection(obj_CoM, center_block)
            predicates[_obj.name] = {'absolute_position': region_section}
        return predicates
    

    def is_open(self, env: ShopEnv, obj: ShopObjectEntry):
        openable_objs = env.openable_obj
        return openable_objs[obj.name].is_open

    def open_closed(self, env: ShopEnv):
        openable_objs = env.openable_obj
        predicates = {}
        for name, obj in openable_objs.items():
            predicates[name] = {"is_open" : obj.is_open}
        return predicates
            

        ## NOTE (SJ): Considering to move these functions to Predicate class
    @staticmethod
    def filter_region(object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_region == condition, object_dict.items()))
    
    @staticmethod
    def filter_movable(object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_movable == condition, object_dict.items()))
    
    @staticmethod
    def filter_openable(object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_openable == condition, object_dict.items()))
    


    def class_filter_region(self, object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_region == condition, object_dict.items()))
    
    def class_filter_movable(self, object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_movable == condition, object_dict.items()))
    
    def class_filter_openable(self, object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_openable == condition, object_dict.items()))
    

    def merge_predicates(self, original: Dict, additional: List[Dict]):
        for result in additional:
            for name, preds in result.items():
                original[name].update(preds)

    def evaluate(self, config: dict, env: ShopEnv, state: ShopState, robot: PR2, manip: PR2Manipulation, nav: Navigation, 
                 include_occlusion_predicates: bool = False,
                 force_compute: bool = False):
        self.free_class_variables()
        self.set_class_variables({
            "config": config,
            "bc": env.bc,
            "env": env,
            "state": state,
            "robot": robot,
            "manip": manip,
            "nav": nav,
            "assets": env.all_obj,
            "movable_assets": self.filter_movable(env.all_obj, condition=True),
            "not_movable_assets": self.filter_movable(env.all_obj, condition=False),
            "region_assets": self.filter_region(env.all_obj, condition=True),
            "not_region_assets": self.filter_region(env.all_obj, condition=False),
            "openable_assets": self.filter_openable(env.all_obj, condition=True),
            "not_openable_assets": self.filter_openable(env.all_obj, condition=False),
        })

        predicates = {}
        

        task_set = list(self.movable_assets.keys()) + list(self.region_assets.keys()) + list(self.openable_assets.keys())
        for name, obj in self.assets.items():
            if name not in task_set: continue
            predicates[name] = {
                'is_movable': obj.is_movable,
                'is_region': obj.is_region,
                'is_openable': obj.is_openable,
                'is_held': False,
            }

        obj_in_region = self.in_region(env, self.movable_assets, self.region_assets.values())
        empty_region = self.empty_region(env, self.assets)

        open_closed = self.open_closed(env)

        absolute_positions = {}
        relative_positions = {}

        
        for region_name, objs in self.region2objs.items():
            if len(objs) == 0: continue
            absolute_position = self.absolute_position(objs, self.region_assets[region_name])
            relative_position = self.relative_position(objs)

            absolute_positions.update(absolute_position)
            relative_positions.update(relative_position)

        region_layout = {region_name: {"layout": self.region_layout(objs)} for region_name, objs in self.region2objs.items()}

        obj_list = list(env.movable_obj.values()) + [entry[-1] for entry in env.handles.values()]
        

        if not self.skip_occlusion_predicates or include_occlusion_predicates:
            ## NOTE (SJ): Only compute occlusion predicates when needed 
            if DEBUG:
                logger.info("Computing predicates with occlusion predicates")
            occPre = self.occludes_pre(state, obj_list, force_compute=force_compute)
            occManip = self.occludes_manip(state, obj_list, self.region_assets.values(), force_compute=force_compute)
            self.merge_predicates(predicates, [obj_in_region, empty_region, open_closed, absolute_positions, 
                                                relative_positions, occPre, occManip, region_layout])
        else:
            if DEBUG:
                logger.info("Computing predicates without occlusion predicates")
            self.merge_predicates(predicates, [obj_in_region, empty_region, open_closed, absolute_positions, 
                                                relative_positions, region_layout])


        return predicates


class ShopPredicateManager(PredicateManager):
    def __init__(self, config: Dict, env: ShopEnv):
        """
        Args:
            config (Settings)
            env (ShopEnv)
        """

        # Config
        self.config = config
        self.agent_manager = ShopAgentPredicateManager(config)
        self.asset_manager = ShopAssetPredicateManager(config)
        # self.action_manager = ActionManager(config, env)


        self.problem = self.make_problem_description(env)

    def make_problem_description(self, env: ShopEnv):

        problem_params = self.config['problem_params']

        problem = {
            'goal': {
                        'objects': problem_params['goal'][0]['objects'],
                        'regions': problem_params['goal'][0]['regions'],
                    },
            'is_movable': [k for k in env.movable_obj.keys()],
            'is_region': [k for k in env.regions.keys() if 'door' not in k],
            'is_openable':[k for k in env.handles.keys()],
            'target_plan_length': env.target_plan_length,
            'directions': problem_params['directions'],
            'grippers': problem_params['grippers'],
        }
    
        return problem


    def make_goal_description(self, objects, regions):
        assert len(objects) == len(regions)

        goal_description = []
        for obj, region in zip(objects, regions):
            str_buffer = " and ".join([f"{_obj}" for _obj in obj])
            if region[1] != 'on' and '_of' not in region[1]:
                region[1] = region[1]+'_of'
            goal_description.append(f'Get {str_buffer} {region[1]} {region[0]}.')

        goal_description = " ".join(goal_description)
        
        return objects, regions, goal_description

    
    def evaluate(self, config: Dict, 
                 env: ShopEnv, 
                 state: ShopState, 
                 robot: PR2, 
                 manip: Navigation, 
                 nav: Manipulation,
                 goal: ShopGoal = None, 
                 include_occlusion_predicates: bool = False,
                 force_compute: bool=False):
        
        ## Caching Logic
        # if len(state.predicates) > 0:
        #     logger.info("Using cached predicates")
        #     return state.predicates

        raw_predicates = {}
        self.agent_predicates = self.agent_manager.evaluate(env=env,
                                                            state=state)
        self.asset_predicates = self.asset_manager.evaluate(config, env, state, 
                                                            robot, manip, nav, 
                                                            include_occlusion_predicates=include_occlusion_predicates,
                                                            force_compute=force_compute)
        self.asset_predicates['kitchen_door']['is_region'] = False
    

        region2objs = {'agent_hand': []}
        for name, objs in self.asset_manager.region2objs.items():
            if 'door' in name: continue
            region2objs[name] = [obj.name for obj in objs]
        
        
        ## cases that the robot is holding a region object
        for holding in ['right_hand_holding', 'left_hand_holding']:
            objs = []
            directly_holding = self.agent_predicates.get(holding, None)
            if directly_holding is None: 
                self.agent_predicates[holding] = []
                continue
            self.asset_predicates[directly_holding]['is_held'] = True
            region2objs[self.asset_predicates[directly_holding]['in_region']].remove(directly_holding)
            self.asset_predicates[directly_holding]['in_region'] = 'agent_hand'
            region2objs[self.asset_predicates[directly_holding]['in_region']].append(directly_holding)
            if self.asset_predicates[directly_holding]['is_region']:
                objs = region2objs[directly_holding]
                self.agent_predicates[holding] = [directly_holding] + objs
                for obj in objs:
                    self.asset_predicates[obj]['is_held'] = True
                    region2objs[self.asset_predicates[obj]['in_region']].remove(obj)
                    self.asset_predicates[obj]['in_region'] = 'agent_hand'
                    region2objs[self.asset_predicates[obj]['in_region']].append(obj)
        

        # problem_params = self.config['problem_params']
        # if goal is None: 
        #     goal_objects = []
        #     goal_regions = []
        #     goal_description = ""
        # else:
        #     goal_objects, goal_regions, goal_description = self.make_goal_description(goal.obj_list, goal.condition)

        
        raw_predicates = {
                'problem': self.problem,
                'info': {
                        'region2objs': region2objs,
                        },
                'agent': self.agent_predicates,
                'asset': self.asset_predicates, 
            }


        # options = self.action_manager.compute_available_options(raw_predicates)
        # raw_predicates['info']['options'] = options


        return raw_predicates
    



