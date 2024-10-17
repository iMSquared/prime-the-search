from math import exp
from typing import List, Dict
from abc import ABC, abstractmethod

from itertools import product

from Simulation.pybullet.mdp.MDP_framework import DiscreteAction, State
from Simulation.pybullet.mdp.shop.shop_MDP import ShopState, ShopGoal, ShopDiscreteAction, ACTION_PICK, ACTION_PLACE, ACTION_OPEN
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.envs.shop_env import ShopEnv
from Simulation.pybullet.envs.robot import PR2
from Simulation.pybullet.envs.manipulation import Manipulation
from Simulation.pybullet.envs.navigation import Navigation

from Simulation.pybullet.custom_logger import LLOG
from Simulation.pybullet.predicate.predicate_shop import ShopPredicateManager

logger = LLOG.get_logger()

class FailureReason(ABC):
    def __init__(self, action: str, action_args: dict, predicate: dict):
        self.action = action
        self.action_args = self.process_action_args(action_args)
        self.predicate = predicate


    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()
    
    def process_action_args(self, action_args: dict):
        output = {}
        if len(action_args) == 2: 
            output['subject'] = action_args[1]
            output['spatial_relation'] = None
            output['reference'] = None
        elif len(action_args) == 4:
            output['subject'] = action_args[1]
            output['spatial_relation'] = action_args[2]
            output['reference'] = action_args[3]
        else:
            raise ValueError("action_args must be either 2 or 4 elements.")
        
        return output
    
    @abstractmethod
    def validate(self, ) -> bool:
        pass

class WeirdArgs(FailureReason):
    def __init__(self, action: str, action_args: dict, predicate: dict):
        super().__init__(action, action_args, predicate)
        self.movable_obj = list(predicate['problem'].keys())
    
    def __str__(self):
        return "Weird ARGS"
    
    def validate(self) -> bool:
        return False

class AgentHasEmptyHand(FailureReason):

    def __str__(self):
        return "agent has no empty hand"
    
    def validate(self) -> bool: 
        return self.predicate['agent']['has_empty_hand']

class AgentIsAlreadyHoldingObject(FailureReason):
    def __str__(self):
        return f"agent is not holding {self.action_args['subject']}"

    def validate(self) -> bool:
        return self.predicate['asset'][self.action_args['subject']]['is_held']
    
class CorrectReference(FailureReason):
    def __str__(self):
        return f"Reference {self.action_args['reference']} cannot be something we are holding."
    
    def validate(self) -> bool:
        return not self.predicate['asset'][self.action_args['reference']]['is_held']

class AgentIsNotHoldingObject(FailureReason):

    def __str__(self):
        return f"agent is already holding {self.action_args['subject']}"

    def validate(self) -> bool:
        return (not self.predicate['asset'][self.action_args['subject']]['is_held'])

class ObjectIsOpen(FailureReason):

    def __str__(self):
        return f"{self.action_args['subject']} is already closed"
    
    def validate(self) -> bool:
        return self.predicate['asset'][self.action_args['subject']]['is_open']

class ObjectIsClosed(FailureReason):

    def __str__(self):
        return f"{self.action_args['subject']} is already open"
    
    def validate(self) -> bool:
        return not self.predicate['asset'][self.action_args['subject']]['is_open']

class NoObjectOnReceptacle(FailureReason):
    def __str__(self):
        return f"{self.action_args['subject']} has other object on it."
    
    def validate(self) -> bool:
        return len(self.predicate['agent']['left_hand_holding']) < 2

class ObjectIsOcclusionFree(FailureReason):
    def __init__(self, action: str, action_args: dict, predicate: dict):
        super().__init__(action, action_args, predicate)
        self.occluders = self.get_occluders()

    @abstractmethod
    def get_occluders(self):
        pass 

    def __str__(self):
        return f"{self.action_args['subject']} is occluded by {', '.join(self.occluders)}."
    
    def validate(self) -> bool:
        return len(self.occluders) == 0
    

class ObjectIsPreOcclusionFree(ObjectIsOcclusionFree):
    def get_occluders(self):
        return self.predicate['asset'][self.action_args['subject']]['is_occ_pre'] 

class ObjectIsManipOcclusionFree(ObjectIsOcclusionFree):

    def get_occluders(self):
        try:
            output = self.predicate['asset'][self.action_args['subject']]['is_occ_manip'][(self.action_args['spatial_relation'] ,self.action_args['reference'])]
        except Exception as e:
            # print(f"Error while getting occluders for {self.action_args}")
            output = ["Error","None"]
        return output
    
class HasPlacementPose(FailureReason):
    def __str__(self):
        return f"{self.action_args['subject']} has no placement pose"
    
    def validate(self) -> bool:
        try:
            output = self.predicate['asset'][self.action_args['subject']]['has_placement_pose'][(self.action_args['spatial_relation'] ,self.action_args['reference'])]
        except Exception as e:
            # print(f"Error while getting occluders for {self.action_args}")
            output = False
        return output
    

class Validator(ABC):
    def __init__(self, *args,**kwargs):
        pass

    @abstractmethod
    def get_all_discrete_actions(self, state: State) -> List[DiscreteAction]:
        pass

    @abstractmethod
    def get_available_discrete_actions(self, state: State) -> List[DiscreteAction]:
        pass


class ShopValidator(Validator):
    """Validate the LM output"""

    def __init__(self, bc, env, robot, manip, nav, config, predicate_manager=None):
        """
        Args:
            bc (BulletClient)
            env (BinpickEnvRealObjects)
            robot (Robot)
            manip (Manipulation)
            gid_table (GlobalObjectIDTable)
            config (Settings)
        """

        # Config
        self.config = config

        # Initialize policy in shop domain
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav

        self.exclude_occluded_actions = self.config['plan_params']['exclude_occluded_actions']
        self.predicate_manager = ShopPredicateManager(self.config, self.env) if predicate_manager is None else predicate_manager

        
    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: ShopEnv, 
                                   robot: PR2,
                                   manip: Manipulation,
                                   nav: Navigation):
        """Re-initalize a random policy model with new BulletClient.
        Be sure to pass Manipulation instance together.

        Args:
            bc (BulletClient): New bullet client
            env (BinpickEnvRealObjects): New simulation environment
            robot (Robot): New robot instance
            manip (Manipulation): Manipulation instance of the current client.
        """
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav

    def validate_pick(self, lang_command, action_args, predicate: Dict):
        """Validate the pick action.
        1. Whether the agent has an empty hand.
        2. Whether the agent is already holding the object.
        3. Whether the object is occluded by other objects.


        Args:
            discrete_action (ShopDiscreteAction): The discrete action to validate.
            predicate (Dict): The predicate of the current state.

        Returns:
            bool: Whether the action is valid.
            str: The reason of the action is invalid.
        """
        conditions = []

        # 1. Whether the agent has an empty hand.
        condition1 = AgentHasEmptyHand(lang_command, action_args, predicate)
        conditions.append(condition1)

        # 2. Whether the agent is already holding the object.
        condition2 = AgentIsNotHoldingObject(lang_command, action_args, predicate)
        conditions.append(condition2)

        # 3. Whether the object is occluded by other objects.
        if self.exclude_occluded_actions:
            condition3 = ObjectIsPreOcclusionFree(lang_command, action_args, predicate)
            conditions.append(condition3)
        
        
        ## filter out explanations that have conditions == True
        explanation = [str(reason) for reason in conditions if not reason.validate()]

        ## use explanation that has valid == False to explain the failure. 
        return len(explanation) == 0, explanation
    
   
    def validate_place(self, lang_command, action_args, predicate: Dict):
        """Validate the place action.
        1. Whether the agent is holding the object.
        2. Whether the object is occluded by other objects.


        Args:
            discrete_action (ShopDiscreteAction): _description_
            predicate (Dict): _description_
        """
        conditions = []
        # condition0 = len(robot.tray_status) != 0 or "tray" not in object_info.name

        # Place on the receptacle
        if not (action_args[-1] in self.env.receptacle_obj and action_args[2] == 'on'):
            condition0 = CorrectReference(lang_command, action_args, predicate)
            conditions.append(condition0)

        condition1 = AgentIsAlreadyHoldingObject(lang_command, action_args, predicate)
        conditions.append(condition1)

        ## Whether to include occlusion from door? 
        if self.exclude_occluded_actions:
            condition2 = HasPlacementPose(lang_command, action_args, predicate)
            conditions.append(condition2)


            condition3 = ObjectIsManipOcclusionFree(lang_command, action_args, predicate)
            conditions.append(condition3)

        
        if self.robot.is_holding_receptacle() and action_args[1] in self.env.receptacle_obj:
            condition4 = NoObjectOnReceptacle(lang_command, action_args, predicate)
            conditions.append(condition4)

        
        explanation = [str(reason) for reason in conditions if not reason.validate()]

        return len(explanation) == 0, explanation


    def validate_open(self, lang_command, action_args, predicate: Dict):
        """Validate the open action.
        1. Whether the agent has an empty hand.
        2. Whether the object is closed.
        Args:
            discrete_action (ShopDiscreteAction): _description_
            predicate (Dict): _description_
        Returns:
            bool: Whether the action is valid.
            str: The reason of the action is invalid.
        """
        condition1 = AgentHasEmptyHand(lang_command, action_args, predicate)
        condition2 = ObjectIsClosed(lang_command, action_args, predicate)
        
        conditions = [condition1, condition2]
        explanation = [str(reason) for reason in conditions if not reason.validate()]

        return len(explanation) == 0, explanation


    def validate_close(self, lang_command, action_args, predicate: Dict):
        """Validate the close action.
        1. Whether the agent has an empty hand.
        2. Whether the object is open.
        Args:
            discrete_action (ShopDiscreteAction): _description_
            predicate (Dict): _description_
        Returns:
            bool: Whether the action is valid.
            str: The reason of the action is invalid.
        """
        condition1 = AgentHasEmptyHand(lang_command, action_args, predicate)
        condition2 = ObjectIsOpen(lang_command, action_args, predicate)
        
        conditions = [condition1, condition2]
        explanation = [str(reason) for reason in conditions if not reason.validate()]

        return len(explanation) == 0, explanation
    
    
    def available_discrete_actions(self, predicate, **action_args) -> List[ShopDiscreteAction]:
        # holding_status = state.holding_status[self.robot.main_hand]
        # predicate = state.predicates
        # has_empty_hand = predicate['agent']['has_empty_hand']
        discrete_actions = []

        for arg in action_args['PICK']:
            if self.validate_pick(arg['lang_command'], arg['action_args'], predicate)[0]:
                discrete_actions.append(arg['ShopDiscreteAction'])
        
        for arg in action_args['PLACE']:
            if self.validate_place(arg['lang_command'], arg['action_args'], predicate)[0]:
                discrete_actions.append(arg['ShopDiscreteAction'])

        for arg in action_args['OPEN']:
            if self.validate_open(arg['lang_command'], arg['action_args'], predicate)[0]:
                discrete_actions.append(arg['ShopDiscreteAction'])

        return discrete_actions                                                        
        
    
    def set_entities(self, predicates):
        self.movable_objects = predicates['problem']['is_movable']
        self.regions = predicates['problem']['is_region']
        self.doors = predicates['problem']['is_openable']

    def all_discrete_actions(self, movable_objects, regions, doors):
        actions = {
            "PICK": [],
            "PLACE": [],
            "OPEN": []
        }

        for obj in movable_objects:
            actions["PICK"].append({'lang_command': f"PICK {obj} {self.env.all_obj[obj].region}", 'action_args': ["PICK", obj],
                                'ShopDiscreteAction': ShopDiscreteAction(ACTION_PICK, "right", self.env.all_obj[obj], region=self.env.all_obj[obj].region ,lang_command=f"('pick', '{obj}')")})
        
        for door in doors:
            actions["OPEN"].append({'lang_command': f"OPEN {door}", 'action_args': ["OPEN", door],
                                 'ShopDiscreteAction': ShopDiscreteAction(ACTION_OPEN, "right", self.env.all_obj[door], lang_command=f"('open', '{door}')")})

        
        for args in product(['PLACE'], movable_objects, ["on"], regions):
            actions["PLACE"].append({'lang_command': f"PLACE {args[1]} {args[2]} {args[3]} {self.env.all_obj[args[3]].region}", 'action_args': args,
                                  'ShopDiscreteAction': ShopDiscreteAction(ACTION_PLACE, "right", self.env.all_obj[args[1]], region=args[3], direction=args[2], reference_obj=self.env.regions[args[3]], lang_command=f"('place', '{args[1]}', '{args[2]}', '{args[3]}')")})

        for args in product(['PLACE'], movable_objects, ["left_of", "right_of", "behind_of", "front_of"], movable_objects):
            actions["PLACE"].append({'lang_command': f"PLACE {args[1]} {args[2]} {args[3]}", 'action_args': args,
                                  'ShopDiscreteAction': ShopDiscreteAction(ACTION_PLACE, "right", self.env.all_obj[args[1]], region=self.env.all_obj[args[3]].region, direction=args[2], reference_obj=self.env.all_obj[args[3]], lang_command=f"('place', '{args[1]}', '{args[2]}', '{args[3]}')")})
    
        return actions
    
    def get_all_discrete_actions(self, state: ShopState) -> List[ShopDiscreteAction]:
        self.set_entities(state.predicates)
        output = self.all_discrete_actions(self.movable_objects, self.regions, self.doors)

        all_actions = [arg['ShopDiscreteAction'] for arg in output['PICK']] + [arg['ShopDiscreteAction'] for arg in output['PLACE']] + [arg['ShopDiscreteAction'] for arg in output['OPEN']]
        return all_actions

    def get_available_discrete_actions(self, state: ShopState, goal: ShopGoal=None, exclude_occluded_actions: bool=False) -> List[ShopDiscreteAction]:
        buff = self.exclude_occluded_actions
        self.exclude_occluded_actions = exclude_occluded_actions
        if len(state.predicates) == 0:
            include_occlusion_predicates = True if self.config["project_params"]["overridable"]["value"] == "hcount" else False
            predicates = self.predicate_manager.evaluate(self.config, self.env, state, self.robot, self.manip, self.nav, goal, include_occlusion_predicates=include_occlusion_predicates)
            state.predicates = predicates
        else:
            predicates = state.predicates

        self.set_entities(predicates)
        
        output = self.all_discrete_actions(self.movable_objects, self.regions, self.doors)

        available_actions = self.available_discrete_actions(predicates, **output)
        
        self.exclude_occluded_actions = buff
        return available_actions
    
    
    



    