
import os
from typing import List, Dict, Tuple
import queue as Queue

import re
import math
from pathlib import Path



from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.mdp.MDP_framework import ValueModel, HistoryEntry
from Simulation.pybullet.mdp.shop.shop_MDP import ShopState, ShopGoal, ShopContinuousAction
from Simulation.pybullet.envs.shop_env import ShopEnv
from Simulation.pybullet.envs.robot import PR2
from Simulation.pybullet.envs.manipulation import PR2Manipulation
from Simulation.pybullet.envs.navigation import Navigation
from Simulation.pybullet.envs.manipulation import Manipulation
from Simulation.pybullet.predicate.reoranize_info import PredicateCentricState
from Simulation.pybullet.custom_logger import LLOG
from Simulation.pybullet.mdp.validator import ShopValidator
from Simulation.pybullet.predicate.predicate_shop import ShopPredicateManager

DEBUG = False

logger = LLOG.get_logger()



class HCountValue(ValueModel):
    """LLM generated value function"""

    def __init__(self, bc         : BulletClient, 
                       env        : ShopEnv, 
                       robot      : PR2, 
                       manip      : PR2Manipulation,
                       nav        : Navigation,
                       config     : Dict,
                       predicate_manager: ShopPredicateManager=None):
        """
        Args:
            bc (BulletClient)
            env (ShopEnv)
            robot (PR2)
            manip (Manipulation)
            gid_table (GlobalObjectIDTable)
            config (Settings)
        """
        super().__init__(is_v_model=True)

        # Initialize policy in fetching domain
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav
        self.config = config

        self.is_v_model = True ## NOTE: This is a value model
        

        # Pose sampler for predicate computation
        self.NUM_FILTER_TRIALS_PICK       : int = config["pose_sampler_params"]["num_filter_trials_pick"]
        self.NUM_FILTER_TRIALS_PLACE      : int = config["pose_sampler_params"]["num_filter_trials_place"]
        self.NUM_FILTER_TRIALS_FORCE_FETCH: int = config["pose_sampler_params"]["num_filter_trials_force_fetch"]
        self.NUM_FILTER_TRIALS_STATE_DESCRIPTION: int = config["pose_sampler_params"]["num_filter_trials_state_description"]
        self._MAX_DEPTH                   : int = config["plan_params"]["max_depth"] + 1

        self.input_state_type = 'simulated' ## NOTE: 'current' or 'simulated'

        # Predicate_manager and Prompt Generator
        self.predicate_converter = PredicateCentricState()

        self.infeasible_value = self.config["plan_params"]["reward"]["infeasible"]
        self.default_value = self.config["plan_params"]["hcount"]["default"]*len(self.env.movable_obj)
        self.multipler = self.config["plan_params"]["hcount"]["multiplier"]
        
        # Validator
        self.validator = ShopValidator(self.bc, self.env, self.robot, self.manip, self.nav, self.config, predicate_manager=predicate_manager)
    
    
    # Helpers
    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: ShopEnv, 
                                   robot: PR2,
                                   manip: Manipulation,
                                   nav: Navigation):
        """Re-initalize a random policy model with new BulletClient.
        Be sure to pass Manipulation instance together.

        Args:
            bc (BulletClient): New bullet client
            env (ShopEnv): New simulation environment
            robot (PR2): New robot instance
            manip (Manipulation): Manipulation instance of the current client.
            nav (Navigation): Navigation instance of the current client.
        """
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav
        self.validator.set_new_bulletclient(self.bc, self.env, self.robot, self.manip, self.nav)

    
    def convert_predicates(self, predicates: Dict) -> Tuple[List[Dict], Dict]:
        
        return self.predicate_converter.organize(predicates)



    def sample(self, history: Tuple[HistoryEntry],
               state: ShopState,
               goal: ShopGoal,
               action: ShopContinuousAction) -> float:
        """Infer the value of the history using NN.

        Args:
            history (Tuple[HistoryEntry])
            state (ShopState)
            goal (List[float])
        Returns:
            float: Estimated value
        """    
        converted_goal, predicates = self.convert_predicates(state.predicates)

        # value = self.compute_hcount(converted_goal, predicates)

        hcount, objects_to_move = self.compute_hcount(converted_goal, predicates)
        objects_already_in_goal_region = self.compute_objects_already_in_goal_region(converted_goal, predicates)
        action_value = self.compute_action_value(converted_goal, predicates, action, objects_to_move)

        cost = hcount - objects_already_in_goal_region + action_value
        # cost = hcount - objects_already_in_goal_region 

        value = self.default_value - self.multipler*cost

        if not action.is_feasible():
            value = self.infeasible_value

        return value
    

    def get_objects_to_move(self, goal, predicates):
        objects_to_move = set()
        potential_obj_to_move_queue = Queue.Queue()

    # objects_to_move = set()
    # potential_obj_to_move_queue = Queue.Queue()
    # goal_r = [entity for entity in state.goal_entities if 'region' in entity][0]

    
        # Putting goal objects that are not in the goal region to objects_to_move set
        for _goal in goal['position']: 
            if not self.is_object_already_in_goal_region(_goal, predicates):
                potential_obj_to_move_queue.put(_goal['subject'])

        object_names = predicates['movable_objects'] + predicates['doors']


        while not potential_obj_to_move_queue.empty():
            obj_to_move = potential_obj_to_move_queue.get()
            if obj_to_move not in objects_to_move:
                objects_to_move.add(obj_to_move)
                if obj_to_move in predicates['doors']: ## NOTE: If the object is a door
                    continue
                for o2 in object_names: ## NOTE: For each object in the environment
                    # OccludesPre
                    # is_o2_in_way_of_obj_to_move = state.binary_edges[(o2, obj_to_move)][1] ## NOTE: C
                    is_o2_in_way_of_obj_to_move = o2 in predicates['is_pick_occluded'][obj_to_move][1]
                    # OccludesManip - should this be to any region?
                    # is_o2_in_way_of_obj_to_move_to_any_region = any(
                    #     [state.ternary_edges[(obj_to_move, o2, r)][0] for r in regions])

                    ## NOTE (SJ) This is the weird part of HCount.
                    is_o2_in_way_of_obj_to_move_to_any_region = any(
                        [o2 in attr[1] for pos, attr in predicates['is_place_occluded'].items() if obj_to_move == pos[0]])

                    ## NOTE (SJ): code version occlusion check
                    # is_o2_in_way_of_obj_to_move_to_any_region = any(
                    #     [obj_to_move in attr[1] for pos, attr in predicates['is_place_occluded'].items() if o2 == pos[0]])



                    if is_o2_in_way_of_obj_to_move or is_o2_in_way_of_obj_to_move_to_any_region:
                        potential_obj_to_move_queue.put(o2)


        # print(f"n_occludes_pre {n_occludes_pre} | n_occludes_manip {n_occludes_manip}")
        return objects_to_move
    
    def is_object_already_in_goal_region(self, _goal, predicates):
        # subject, spatial_relation, reference = _goal
        subject = _goal['subject']
        spatial_relation = _goal['spatial_relation']
        reference = _goal['reference']

        if spatial_relation in ['on']:
            if subject in predicates['objects_on_region'][reference]['on']:
                return True
        elif spatial_relation in ['left_of']:
            for r, attrib in predicates['objects_on_region'].items():
                if reference in attrib['left_to_right']: 
                    sub_index = attrib['left_to_right'].index(subject) if subject in attrib['left_to_right'] else math.inf
                    ref_index = attrib['left_to_right'].index(reference)
                    if sub_index < ref_index:
                        return True
        elif spatial_relation in ['right_of']:
            for r, attrib in predicates['objects_on_region'].items():
                if reference in attrib['left_to_right']: 
                    sub_index = attrib['left_to_right'].index(subject) if subject in attrib['left_to_right'] else math.inf
                    ref_index = attrib['left_to_right'].index(reference)
                    if sub_index > ref_index:
                        return True
        elif spatial_relation in ['front_of']:
            for r, attrib in predicates['objects_on_region'].items():
                if reference in attrib['front_to_behind']: 
                    sub_index = attrib['front_to_behind'].index(subject) if subject in attrib['front_to_behind'] else math.inf
                    ref_index = attrib['front_to_behind'].index(reference)
                    if sub_index < ref_index:
                        return True
        elif spatial_relation in ['behind_of']:
            for r, attrib in predicates['objects_on_region'].items():
                if reference in attrib['front_to_behind']: 
                    sub_index = attrib['front_to_behind'].index(subject) if subject in attrib['front_to_behind'] else math.inf
                    ref_index = attrib['front_to_behind'].index(reference)
                    if sub_index > ref_index:
                        return True
        else:
            
            raise ValueError(f"Unknown spatial relation {_goal['spatial_relation']}")
        
    







    def compute_hcount(self, goal, predicates):
        objects_to_move = self.get_objects_to_move(goal, predicates)
        cost = len(objects_to_move)
        return cost, objects_to_move
    
    
    def compute_objects_already_in_goal_region(self, goal, predicates):
        objects_already_in_goal_region = set()
        for _goal in goal['position']:
            if self.is_object_already_in_goal_region(_goal, predicates):
                objects_already_in_goal_region.add(_goal['subject'])
        
        cost = len(objects_already_in_goal_region)

        return cost


    def compute_action_value(self, goal, predicates, action: ShopContinuousAction, objects_to_move: List):
        r"""
        Check if the actions aimed_object is a goal object that is already in the goal region, which intuitively means that the action would increase the cost. 
        """
        cost = 0
        subject_entity_name = action.aimed_obj.name

        already_in_the_goal = False
        touching_goal_object = False
        for _goal in goal['position']:
            if _goal['subject'] == subject_entity_name:
                if self.is_object_already_in_goal_region(_goal, predicates) and action.discrete_action.type == "PICK":
                    already_in_the_goal = True
                elif action.discrete_action.type == "PICK":
                    touching_goal_object = True

        
        if already_in_the_goal:
            cost = 1
        elif action.is_feasible() and touching_goal_object:
            cost = -1


        return cost







