
"""
MDP modelling for the Shop problem
"""
import os
import numpy as np
import numpy.typing as npt
from typing import Tuple, Dict, List, TypeVar, Union, Set, Iterable
from collections import deque
from copy import deepcopy
from enum import Enum
import random
import heapq
import numpy as np
import random
from typing import List, Tuple


from Simulation.pybullet.mdp.MDP_framework import DiscreteAction, ContinuousAction, State, Goal, Observation, TransitionModel, RewardModel, Agent, TerminationT, TERMINATION_SUCCESS, TERMINATION_FAIL, TERMINATION_CONTINUE, ValueModel, PolicyModel, RolloutPolicyModel, BlackboxModel
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.imm.pybullet_util.typing_extra import TranslationT, EulerT, QuaternionT
from Simulation.pybullet.envs.shop_env import ShopEnv, ShopObjectEntry, EnvCapture, HandleState, capture_shopenv
from Simulation.pybullet.envs.robot import PR2, AttachConstraint
from Simulation.pybullet.envs.manipulation import Manipulation, PR2Manipulation, PR2SingleArmManipulation
from Simulation.pybullet.envs.navigation import Navigation

from Simulation.pybullet.custom_logger import LLOG
logger = LLOG.get_logger()

# LLM assisted
from Simulation.pybullet.mdp.LLM_assisted_MDP import LLMassistedAgent

class CircularPriorityQueue:
    def __init__(self, config: Dict):
        self.config: Dict = config 
        self.iter_list: List[Tuple[ShopDiscreteAction, float]] = []
        self.prob_list: List[float] = []
        self.index: int = 0  # Iterator index

        self.needs_update: bool = True  # Flag to track when updates are needed
        self.full_sweep_done: bool = False

        self.temp: float = self.config['plan_params']['exponential_temperature']
        self.policy_value: str = self.config['plan_params']['policy_value']
        self.available_action_sample_method: str = self.config['plan_params']['available_action_sample_method']

    def __repr__(self) -> str:
        return self.__str__() 

    def __str__(self) -> str:
        # return f"Number of items in the queue: {len(self.iter_list)}, items: {self.iter_list}"   
        output = ""
        for item, value in self.iter_list:
            output += f"Action: {item.lang_command} -- Value: {value}\n"

        return output
    
    def __len__(self):
        return len(self.iter_list)
    
    def __eq__(self, other):
        if len(self.iter_list) != len(other):
            return False
        for i, (item, _) in enumerate(self.iter_list):
            if item not in other:
                return False
        return True
    
    def examine_indistinguishable(self, threshold=0, top_n=0):
        
        
        if top_n >= len(self.iter_list):
            top_n = len(self.iter_list)-1
        elif top_n < 0:
            top_n = 1
        else:
            top_n -= 1

        # curr_value_flag = 0
        # same_value_cnt = 0

        # for i in range(0, top_n):
        #     _, value = self.iter_list[i]
        #     if value != curr_value_flag:
        #         curr_value_flag = value
        #         same_value_cnt = 1
        #     else: 
        #         same_value_cnt += 1
            
        #     if same_value_cnt > threshold:
        #         return True

        _ ,curr_value_tracker = self.iter_list[-1]
        same_value_cnt = 0

        for i in range(1, top_n):
            _, value = self.iter_list[i]
            if value == curr_value_tracker:
                same_value_cnt += 1
            else:
                return False 

            if same_value_cnt > threshold:
                return True

        return False

    
    def sort(self):
        self.iter_list = sorted(self.iter_list, key=lambda x: x[1], reverse=True)

    def push(self, item: DiscreteAction, value: Union[float, Iterable]):
        ## NOTE (SJ): Think of way to handle the reasons that are coming over from the LLM code 
        if isinstance(value, Iterable):
            self.iter_list.append((item, value[0]))
            item.q_value = value[0]
        else:
            self.iter_list.append((item, value))
            item.q_value = value
        self.needs_update = True

    def shuffle_items_with_same_value(self):
        if self.needs_update:
            value_groups = {}
            for item, value in self.iter_list:
                if value not in value_groups:
                    value_groups[value] = [item]
                else:
                    value_groups[value].append(item)

            self.iter_list = [(item, value) for value, items in value_groups.items() for item in random.sample(items, len(items))]
            self.needs_update = False

    def assign_probs(self):
        '''
            Assign probabilities to the items in the queue based on the policy value that will be used for PUCB algorithm
        '''
        self.prob_list = []
            
        if len(self.iter_list) == 0:
            raise ValueError("No items in the queue")

        if self.policy_value == "exponential":
            total = sum(np.exp(weight / self.temp) for _, weight in self.iter_list)
            for i, (item, weight) in enumerate(self.iter_list):
                prob = np.exp(weight / self.temp) / total
                item.probability = prob
                self.prob_list.append((item, prob))
        elif self.policy_value == "weighted":
            total = sum(weight for _, weight in self.iter_list)
            for i, (item, weight) in enumerate(self.iter_list):
                prob = weight / total
                item.probability = prob
                self.prob_list.append((item, prob))
        elif self.policy_value == "uniform":
            prob = 1 / len(self.iter_list)
            for i, (item, _) in enumerate(self.iter_list):
                item.probability = prob
                self.prob_list.append((item, prob))
        else:
            raise ValueError("Policy value must be either 'exponential', 'weighted', or 'uniform'")

    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.available_action_sample_method == 'queue':
            self.assign_probs()
            if self.index >= len(self.iter_list) or self.needs_update:
                self.shuffle_items_with_same_value()
                self.index = 0  # Reset index if we reach the end or after an update
                self.full_sweep_done = True

            item, prob = self.iter_list[self.index]
            self.index += 1
            return item, prob

        elif self.available_action_sample_method == 'sample':
            self.assign_probs()
            return self.sample()

        else:
            raise ValueError("Available action sample method must be either 'queue' or 'sample'")
        
    def sample(self):
        choices, probabilities = zip(*self.prob_list)  # This could be optimized by caching if the list doesn't change
        sampled_index = np.random.choice(len(choices), p=probabilities)
        return choices[sampled_index], probabilities[sampled_index]
    


class Relation(Enum):
    left = "left_of"
    right = "right_of"
    front = "front_of"
    behind = "behind_of"
    top = "on"


# Global flags
ShopActionT = TypeVar("ShopActionT", bound=str)
ACTION_PICK  : ShopActionT = "PICK"
ACTION_PLACE : ShopActionT = "PLACE"
ACTION_OPEN  : ShopActionT = "OPEN"
ACTION_CLOSE : ShopActionT = "CLOSE"

SAVE_BULLET_STATE = False


class ShopDiscreteAction(DiscreteAction):
    """
    There are two types of actions in SE(3):
        PICK(obj, g)
        PLACE(p)

    All fields are None when infeasible action.    
    """
    def __init__(self, type: ShopActionT,
                       arm: str,
                       aimed_obj: Union[ShopObjectEntry, None],
                       region: str = None,
                       direction: str = None,
                       reference_obj: Union[ShopObjectEntry, None] = None, 
                       lang_command: str = None,
                       thought: str = None,
                       llm_data: Dict = None):
        """Simple constructor

        Args:
            type (ShopActionT): ACTION_PICK or ACTION_PLACE
            target (Union[str, None],): None when infeasible action.
            region (str): Optional for LLM. PLACE only
        """
        assert arm in ["left", "right"]

        super().__init__()

        self.type = type
        self.arm = arm
        self.aimed_obj = aimed_obj
        self.region = region
        self.direction = direction
        self.reference_obj = reference_obj
        self.lang_command = lang_command
        self.thought = thought
        self.llm_data = llm_data ## current_state, options, output,  
        self.bullet_filename = None
        

    def __eq__(self, other: "ShopDiscreteAction"):

        return (self.type == other.type) \
            and (self.aimed_obj.name == other.aimed_obj.name) \
            and (self.reference_obj is None or self.reference_obj.name == other.reference_obj.name) \
            and (self.direction == other.direction)
            # and (self.lang_command is None or (other.lang_command is not None and self.lang_command.upper() == other.lang_command.upper()))


    def __hash__(self):
        if self.reference_obj is None:
            value = hash((self.type, self.aimed_obj.name, self.direction))
        else:
            value = hash((self.type, self.aimed_obj.name, self.reference_obj.name, self.direction))

        return value


    def __str__(self):
        """Simple tostring."""
        if self.type == ACTION_OPEN:
            return f"(discrete_action){self.type.lower()} {self.aimed_obj.name}"
        elif self.type == ACTION_PICK:
            return f"(discrete_action){self.type.lower()} {self.aimed_obj.name} {self.aimed_obj.region}"
        elif self.type == ACTION_PLACE:
            return f"(discrete_action){self.type.lower()} {self.aimed_obj.name} {self.direction} {self.reference_obj.name} {self.reference_obj.region}"

    def __repr__(self):
        return self.__str__()


    def is_feasible(self):
        """Check whether generated action is feasible."""

        if self.feasible is None:
            feasiblity_flags = {
                "aimed_obj": self.aimed_obj is not None,
            }
            self.feasible = all(feasiblity_flags.values())

        return self.feasible


class ShopContinuousAction(ContinuousAction):
    """
    There are two types of actions in SE(3):
        PICK(obj, g)
        PLACE(p)

    All fields are None when infeasible action.    
    NOTE(ssh): pos and orn are represented in object's SURFACE NORMAL. (Outward)
    aimed_obj does not guarantee that the object is in contact.
    """
    def __init__(self, discrete_action: ShopDiscreteAction,
                       pos: Union[TranslationT, None],
                       orn: Union[EulerT, None],
                       nav_traj: Union[npt.NDArray, None],
                       manip_traj: Union[npt.NDArray, None],
                       handle_pose: Union[Tuple[Tuple[float]], None]=None,
                       handle_action_info: Union[Tuple, None]=None,
                       error_message: str = ""):
        """Simple constructor

        Args:
            discrete_action: ShopDiscreteAction
            pos (Union[TranslationT, None]): [xyz]
            orn (Union[EulerT, None]): [roll pitch yaw]
            traj (Union[npt.NDArray, None]): Motion trajectory
            region (str): Optional for PLACE.
            lang_command (str): Optional for LLM.
        """
        super().__init__(discrete_action=discrete_action)
        self.pos = pos
        self.orn = orn
        self.nav_traj = nav_traj
        self.manip_traj = manip_traj
        self.aimed_obj = discrete_action.aimed_obj
        self.region = discrete_action.region
        self.handle_pose = handle_pose
        self.handle_action_info = handle_action_info
        self.error_message = error_message
        self.q: float = -float("inf")
        # self.bullet_filename = None

    def __eq__(self, other: "ShopContinuousAction"):
        return (self.discrete_action == other.discrete_action) \
            and (self.pos == other.pos) \
            and (self.orn == other.orn)


    def __hash__(self):
        return hash((self.discrete_action.type, 
                     self.discrete_action.aimed_obj, 
                     self.discrete_action.region, 
                     self.discrete_action.direction, 
                     self.pos, 
                     self.orn))

    def __str__(self):
        """Simple tostring."""
        command = self.discrete_action.type if self.discrete_action.lang_command is None else self.discrete_action.lang_command
        if self.is_feasible():
            pos_str = f"({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f})"
            orn_str = f"({self.orn[0]:.3f}, {self.orn[1]:.3f}, {self.orn[2]:.3f})"
            if self.discrete_action.type == ACTION_PLACE:
                return f"{command}, {self.discrete_action.aimed_obj.name}, {self.discrete_action.direction}, {self.discrete_action.reference_obj.name}, {pos_str}, {orn_str}"
            else:
                return f"{command}, {self.discrete_action.aimed_obj.name}, {pos_str}, {orn_str}"

        else:
            return f"{command}, infeasible action"



    def __repr__(self):
        return self.__str__()

    ## BOOKMARK: for traj
    def is_feasible(self):
        if self.feasible is None:
            """Check whether generated action is feasible."""
            feasiblity_flags = {
                "discrete_action": self.discrete_action.is_feasible(),
                "pos": self.pos is not None,
                "orn": self.orn is not None,
                'manip_traj': self.manip_traj is not None,
            }

            self.feasible = all(feasiblity_flags.values())

        return self.feasible


class ActionCache:
    collisions: Dict[Tuple[str], Set[str]] = dict()
    no_obstacle_pick_action: Dict[str, List[npt.ArrayLike]] = dict()
    no_obstacle_place_action: Dict[str, List[npt.ArrayLike]] = dict()
    place_action: Dict[str, Dict[str,ShopContinuousAction]] = dict()
    pick_action: Dict[str, Dict[str,ShopContinuousAction]] = dict()
    pick_joint_pos: Dict[str, List] = dict()
    place_pos: Union[Tuple[float], npt.ArrayLike] = None
    place_orn: Union[Tuple[float], npt.ArrayLike] = None

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.collisions = deepcopy(self.collisions)
        result.no_obstacle_pick_action = deepcopy(self.no_obstacle_pick_action)
        result.no_obstacle_place_action = deepcopy(self.no_obstacle_place_action)
        result.place_action = deepcopy(self.place_action)
        result.pick_action = deepcopy(self.pick_action)
        result.place_pos = deepcopy(self.place_pos)
        result.place_orn = deepcopy(self.place_orn)

        return result        


class ShopState(State):
    """
    State consist of robot state (joint) and object states (pose, shape, target)
    NOTE(ssh): Make sure to pass frozenset. It will raise unhashable error.
    """

    def __init__(self, robot_arm_state    : Tuple[float],                  # Joint values of the robot arm
                       robot_base_state: Tuple[float],
                       robot_region: str,
                       object_states  : Dict[str, ShopObjectEntry], # Dictionary of object poses
                       holding_status: Dict[str, AttachConstraint],            # Identifier of the holding object
                       handle_status  : Dict[str, HandleState],
                       receptacle_status: Dict[str, AttachConstraint],
                       receptacle_holding_info: Tuple,
                       parent         : Union['ShopState', None]):     # RGBD image
        """Constructor initializes member variables.
        
        Args:
            robot_arm_state (Tuple[float]): Joint values of the robot arm
            object_states (Dict[str, BulletObjectPose]): Object states
            holding_status (Dict[str, AttachConstraint]): GID of the holding object.
        """
        self.robot_arm_state = robot_arm_state  # (torso_joints, left_arm_joints, right_arm_joints): len=15
        self.robot_base_state = robot_base_state
        self.region = robot_region 
        self.object_states = object_states 
        self.holding_status = holding_status # {left: None, right: None}
        self.handle_status = handle_status
        
        self.failure_queue = deque(maxlen=2)
        holding_or_not = "PICK" if holding_status is not None else "PLACE"
        # self.bullet_filename = LLOG.get_filename(suffix=f"_{holding_or_not}.bullet")
        
        # Caching and predicate check
        # TODO: refactor to_goal_traj and predicate cache
        self.predicates = dict()
        self.action_cache = ActionCache()
        self.parent: ShopState = parent
        self.action_scores: list[Tuple] = None

        # receptacle related info
        self.receptacle_status = receptacle_status
        self.receptacle_holding_info = receptacle_holding_info

        if parent is not None:
            self.inherit_cache()

        # Assigned value. This is for value function refinement
        self.assigned_value = None

        # Door timer
        if parent is None:
            self.door_timer = dict()
        else:
            self.door_timer = deepcopy(parent.door_timer)


    
    @staticmethod
    def get_ShopState_from_EnvCapture(capture: EnvCapture, 
                                      region: str = None,
                                      parent: "ShopState" = None):
        return ShopState(capture.robot_arm_state,
                         capture.robot_base_state,
                         region,
                         capture.object_states, 
                         capture.holding_status,
                         capture.handle_status,
                         capture.receptacle_status,
                         capture.receptacle_holding_info,
                         parent)


    def inherit_cache(self):
        action_cache = deepcopy(self.parent.action_cache)
        action_cache.place_action = dict()
        action_cache.pick_action = dict()
        action_cache.pick_joint_pos = dict()
        action_cache.place_pos = dict()
        action_cache.place_orn = dict()
        action_cache.collisions = dict()

        # NOTE (dlee): Should we discard no_obstacle caches?
        # Yes. The robot position has changed.
        action_cache.no_obstacle_pick_action = dict()
        action_cache.no_obstacle_place_action = dict()

        # action_scores = deepcopy(self.parent.action_scores)


        self.action_cache = action_cache
        self.action_scores = None


    def inherit_predicates(self):
        self.predicates = deepcopy(self.parent.predicates)


    def _initialize_predicates(self, predicates):
        
        # MUST do runtime import
        # from Simulation.pybullet.predicate.predicate_shop import 


        self.predicates = predicates


    @staticmethod
    def equal_predicates(state1: "ShopState", state2: "ShopState") -> bool:

        predicates1 = state1.predicates
        predicates2 = state2.predicates

        value = True

        # Problem has to be same (Can this be diverged?)
        if value and predicates1["problem"] != predicates2["problem"]:
            value = False
        
        # Brief on env have to be same
        if value and predicates1["info"] != predicates2["info"]:
            value = False
        
        # Agent status have to be same
        if value and predicates1["agent"] != predicates2["agent"]:
            value = False
        
        # Asset status must be same
        if value and predicates1["asset"] != predicates2["asset"]:
            value = False

        return value

    @staticmethod
    def predicate_diff(pddl_state1: list, pddl_state2: list) -> dict:
        '''
        NOTE (SJ): not state predicate but only pddl style 
        For PDDLState, 
        input:
            state1 : state before
            state2 : state after
        output: 
            diff: 
                True:  predicates that became True 
                False: predicates that became False
        '''
        diff = {
            'True':[],
            'False:':[]
        }





        


    def __eq__(self, other: "ShopState"):
        # NOTE(ssh): This eq seems to be wrong.
        if self.robot_arm_state != other.robot_arm_state:
            return False
        else:
            for k, v in self.object_states.items():
                if k not in other.object_states:
                    return False
                else:
                    if v != other.object_states[k]:
                        return False
                    else:
                        continue
            return True


    def __hash__(self):
        # NOTE(ssh): This hash seems not reflecting enough information...
        # but just leaving as it works fine for now. 
        return hash((self.robot_arm_state, tuple(self.object_states)))


class ShopGoal(Goal):
    """
    Goal consists of goal color and goal pose
    obj_name_list in goal_region
    """

    def __init__(self, obj_name_list: Tuple[Tuple[str]], 
                       condition: Tuple[str, str]):
        
        assert len(obj_name_list) == len(condition), "list of objects and list of regions must have same length"

        self.obj_list = obj_name_list
        self.condition = condition

    
    def __eq__(self, other: "ShopGoal"):
        if self.obj_list == other.obj_list \
            and self.condition == other.condition:
            return True
        else:
            return False


    def __hash__(self):
        return hash((self.obj_list, self.goal_region))
    
    def get_goal_obj_list(self):
        goal_obj_list = []
        for objs in self.obj_list:
            goal_obj_list += objs
        
        return list(set(goal_obj_list))


class ShopTransitionModel(TransitionModel):

    def __init__(self, bc: BulletClient, 
                       env: ShopEnv, 
                       robot: PR2,
                       manip: Manipulation,
                       nav: Navigation, 
                       config: Dict,
                       goal: ShopGoal,
                       execution: bool,
                       predicate_manager: "ShopPredicateManager"=None):
        """Initalize a pybullet simulation transition model.
        Be sure to pass Manipulation instance together."""
        self.config = config
        self.DEBUG_GET_DATA = config['project_params']['debug']['get_data']

        self.bc          = bc
        self.env = env
        self.robot       = robot
        self.manip       = manip
        self.nav         = nav
        self.execution   = execution

        num_sims = 1 if self.execution else config["plan_params"]["num_sims"]

        # Dynamic import required
        from Simulation.pybullet.predicate.predicate_shop import ShopPredicateManager
        self.predicate_manager = ShopPredicateManager(config, self.env) if predicate_manager is None else predicate_manager
        if self.config["project_params"]["overridable"]["value"] == "hcount":
            self.predicate_manager.asset_manager.skip_occlusion_predicates = False

        # Goal
        self.goal = goal

        # Door duration
        self.set_door_duration()


    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: ShopEnv, 
                                   robot: PR2,
                                   manip: Manipulation,
                                   nav: Navigation):
        """Re-initalize a pybullet simulation transition model with new BulletClient.
        Be sure to pass Manipulation instance together.

        Args:
            bc (BulletClient): New bullet client
            env (ShopEnv): New simulation environment
            robot (PR2): New robot instance
            manip (Manipulation): Manipulation instance of the current client.
        """
        self.bc          = bc
        self.env = env
        self.robot       = robot
        self.manip       = manip
        self.nav         = nav


    def probability(self, next_state: ShopState, 
                          state: ShopState, 
                          action: ShopContinuousAction) -> float:
        """
        determinisitic
        """
        return 1.0
    

    def sample(self, state: ShopState, 
                     action: ShopContinuousAction,
                     running_escape: bool = False, **kwargs) -> ShopState:
        """Sample the result of the action.
        See the definition of the policy model for detail of the `action`.

        Args:
            state (ShopState): Current state.
            action (ShopContinuousAction): Action to do.
        Raises:
            ValueError: Invalid action type error.

        Returns:
            next_state (ShopState): Transition result
        """
        ## Return the current state if the action is infeasible, This will happen if all actions sampled from the model are infeasible actions and not registered to the tree
        if action is None: 
            logger.warning("Action is None | Returning the current state")
            next_state = deepcopy(state)
            return next_state
        
        # Init door timer if not set. condition check is done inside the function.
        self.init_door_timer(state)

        # Increase door timer by 1
        # for name, timer in self.door_timer[sim_id].items():
        door_timer = deepcopy(state.door_timer)
        for name, timer in door_timer.items():
            timer += 1

        # If execution, check for the closed door.
        # Closed door is conditional exogenous event which the agent cannot expect
        # action = self.apply_exogenous_event(state, action)
        single_arm_manip: PR2SingleArmManipulation = getattr(self.manip, action.discrete_action.arm)

        # Execute action
        success = False
        if action.is_feasible():
            aimed_obj = self.env.all_obj[action.aimed_obj.name]
            if action.discrete_action.type == ACTION_PICK:
                try:
                    success = single_arm_manip.pick(aimed_obj,
                                                    action.nav_traj,
                                                    action.manip_traj,
                                                    simple=(not self.execution))
                except Exception as e:
                    logger.warning(f"Failed to pick: {e}")
                    success = False
            elif action.discrete_action.type == ACTION_PLACE:
                if self.robot.is_holding_receptacle() and aimed_obj.name not in self.env.receptacle_obj:
                    if aimed_obj.name in self.robot.receptacle_status:
                        # Place from the receptacle
                        region_before = self.robot.receptacle_status[aimed_obj.name][2]
                        
                    else:
                        # Place on the receptacle
                        region_before = aimed_obj.region

                try:
                    success = single_arm_manip.place(aimed_obj,
                                                     action.region,
                                                     action.nav_traj,
                                                     action.manip_traj,
                                                     simple=(not self.execution))
                except Exception as e:
                    logger.warning(f"Failed to place: {e}")
                    success = False
            elif action.discrete_action.type == ACTION_OPEN:
                try:
                    success = single_arm_manip.open_door(aimed_obj,
                                                        action.nav_traj,
                                                        action.manip_traj,
                                                        action.handle_pose,
                                                        action.handle_action_info,
                                                        simple=(not self.execution))
                except Exception as e:
                    success = False
                    logger.warning(f"Failed to open: {e}")

                if success:
                    door_timer[action.handle_action_info[0][0]] = 0
                    self.set_door_duration()

        action.feasible = success
        
        if not action.is_feasible():
            next_state = deepcopy(state)
            return next_state
            
        # update object states
        action_type = action.discrete_action.type
        object_states: Dict[str, ShopObjectEntry] = {}
        other_arm = self.robot.get_other_arm()
        if success:
            for name, obj in self.env.movable_obj.items():
                # Update after placement
                obj_state = deepcopy(obj)
                if (action_type == ACTION_PLACE and name == action.aimed_obj.name):
                    obj_state.area = self.env.regions[action.region].area
                    if not self.robot.is_holding_receptacle():
                        region_before = obj_state.region
                    obj_state.region = action.region
                    object_pose = self.bc.getBasePositionAndOrientation(obj.uid)
                    obj_state.position, obj_state.orientation = object_pose
                # Update after pick
                elif (action_type == ACTION_PICK and name == action.aimed_obj.name):
                    ## NOTE (dlee): separate pick and place on the receptacle
                    # if name not in self.env.receptacle_obj and len(self.robot.receptacle_status) > 0:
                    #     obj_state.region = "receptacle"
                    object_pose = self.bc.getBasePositionAndOrientation(obj.uid)
                    obj_state.position, obj_state.orientation = object_pose
                else:
                    object_pose = self.bc.getBasePositionAndOrientation(obj.uid)
                    obj_state.position, obj_state.orientation = object_pose
                updated_obj_state = deepcopy(obj_state)
                self.env.all_obj[name] = updated_obj_state
                self.env.movable_obj[name] = updated_obj_state
                object_states[name] = obj_state

                # Update robot region
                if action_type == ACTION_PLACE:
                    robot_region = action.region
                elif action_type == ACTION_PICK:
                    robot_region = action.aimed_obj.region
                else:
                    robot_region = action.aimed_obj.name
        else:
            object_states = state.object_states
            robot_region = state.region

        for name in self.env.handles.keys():
            object_states[name] = deepcopy(self.env.all_obj[name])


        # Close doors
        # Doors have to be closed after certain timestep
        door_closed = []
        for name, handle_info in self.env.handles.items():
            door_default_pos: float = handle_info[5]
            door: ShopObjectEntry = handle_info[6]
            if door.is_open and door_timer[handle_info[0]] > self.door_duration:
                self.bc.resetJointState(door.uid, handle_info[1], door_default_pos)
                door.is_open = False
                door_timer[handle_info[0]] = 0
                door_closed.append(name)
                self.nav.reset_roadmap(use_pickle=True, door_open=False)
            elif door.is_open:
                door_timer[handle_info[0]] += 1

        # Get next state of objects
        next_state = capture_shopenv_state(self.bc, self.env, self.robot, region=robot_region, parent=state)
        next_state.object_states = object_states
        next_state.door_timer = door_timer

        # Update robot region
        next_state.region = robot_region

        # Update receptacle related info
        next_state.receptacle_status = deepcopy(self.robot.receptacle_status)
        next_state.receptacle_holding_info = deepcopy(self.robot.receptacle_holding_info)

        if running_escape:
            ## Return State without updating the predicates
            return next_state
    

        # Inherit Cache if previous action is PICK
        next_state.inherit_predicates()
        if action_type == ACTION_OPEN and success:
            next_state.predicates = self.remove_obj_from_predicates_after_open(next_state.predicates, action.aimed_obj)
        elif action_type == ACTION_PLACE and success:
            next_state.predicates = self.remove_obj_from_predicates_after_place(next_state.predicates, action, region_before)
            next_state.action_cache.pick_joint_pos[f"{self.robot.main_hand}-{action.aimed_obj.name}"] = [action.manip_traj[-1]]
        else:
            next_state.action_cache = self.bring_cache(state.action_cache)

        if len(door_closed) > 0:
            next_state.predicates = self.add_obj_to_predicates_after_door_closed(door_closed, next_state.predicates, state, action, success)

        if self.execution:
            buffer = self.predicate_manager.asset_manager.skip_occlusion_predicates
            self.predicate_manager.asset_manager.skip_occlusion_predicates = False
            next_state.predicates = self.predicate_manager.evaluate(self.config, self.env, next_state, 
                                                                self.robot, self.manip, self.nav, self.goal)
            self.predicate_manager.asset_manager.skip_occlusion_predicates = buffer
        else:
            next_state.predicates = self.predicate_manager.evaluate(self.config, self.env, next_state, 
                                                                self.robot, self.manip, self.nav, self.goal)

        return next_state
    

    def init_door_timer(self, state: ShopState):
        if state.parent is None:
            for name, handle_info in self.env.handles.items():
                door: ShopObjectEntry = handle_info[6]
                if door.is_open:
                    state.door_timer[handle_info[0]] = 0


    def set_door_duration(self):
        if self.execution:
            max_duration = self.config["env_params"]["shop_env"]["dynamics"]["door_duration"]
            # self.door_duration = random.randint(1, max_duration)
            self.door_duration = 10000
        else:
            self.door_duration = 10000
        # logger.warning(f"Door duration is set to {self.door_duration}")


    def apply_exogenous_event(self, state: ShopState, action: ShopContinuousAction):
        if self.execution:
            # Condition 1: action is feasible
            # Condition 2: action is PICK or PLAC#
            # Condition 3: action is going through the door
            # Condition 4: door is closed
            condition1 = action.is_feasible()
            condition2 = action.discrete_action.type in [ACTION_PICK, ACTION_PLACE]
            condition3 = action.aimed_obj.area != self.env.all_obj[state.region].area \
                if state.region in self.env.all_obj else action.aimed_obj.area != state.region
            for name, timer in state.door_timer.items():
                condition4 = not self.env.openable_obj[name].is_open
                if condition1 and condition2 and condition3 and condition4:
                    action.nav_traj = None
                    action.manip_traj = None
            
        return action


    def remove_obj_from_predicates_after_open(self, predicates: Dict[str, Dict], obj: ShopObjectEntry):
        if "asset" in predicates:
            for name, predicate in predicates["asset"].items():
                if "is_occ_pre" not in predicate or "is_occ_manip" not in predicate:
                    continue
                # Remove obj from is_occ_pre
                predicate["is_occ_pre"]: List
                predicate["is_occ_manip"]: Dict[str, List]
                if obj.name in predicate["is_occ_pre"]:
                    idx = [i for i, x in enumerate(predicate["is_occ_pre"]) if x == obj.name]
                    for i in idx:
                        predicate["is_occ_pre"].pop(i)

                # Remove obj from is_occ_manip
                for region_name, occ_set in predicate["is_occ_manip"].items():
                    # NOTE: each object has a unique name, so there is only one item in idx if matched.
                    if obj.name in occ_set:
                        occ_set.remove(obj.name)

        return predicates

    def remove_obj_from_predicates_after_place(self, predicates: Dict[str, Dict], 
                                               action: ShopContinuousAction,
                                               region_before: str):
        
        
        assert action.discrete_action.type == ACTION_PLACE
        obj = action.aimed_obj
        region_after = action.region

        # Remove predicates related to the object
        # NOTE (dlee): Actually, is_occ_manip for the moved object stays same as long as the navigation is guaranteed.
        # if obj.name in predicates["asset"]:
        #     predicates["asset"].pop(obj.name)
        if "asset" in predicates:
            for name, predicate in predicates["asset"].items():
                if "is_occ_pre" not in predicate or "is_occ_manip" not in predicate:
                    continue
                
                # Remove the moved object in the original region
                if self.env.all_obj[name].region == region_before:
                    # Update is_occ_pre
                    # NOTE: each object has a unique name, so there is only one item in idx if matched.
                    idx = [i for i, x in enumerate(predicate["is_occ_pre"]) if x == obj.name]
                    if len(idx) > 0:
                        predicate["is_occ_pre"].pop(idx[0])

                    # Update is_occ_manip
                    for region_name, occ_set in predicate["is_occ_manip"].items():
                        if obj.name in occ_set:
                            occ_set.remove(obj.name)

                # Reset predicate computation in the moved region.
                if self.env.all_obj[name].region == region_after and name != action.aimed_obj.name:
                    # Recompute all is_occ_pre in the region
                    predicate.pop("is_occ_pre")

                    # Recompute all is_occ_manip in the region
                    predicate.pop("is_occ_manip")

                    # Recompute all has_placement in the region
                    predicate.pop("has_placement_pose")

                # Reset predicate compution w.r.t moved object
                if "is_occ_manip" in predicate:
                    for direction in ["left_of", "right_of", "front_of", "behind_of"]:
                        try:
                            predicate["is_occ_manip"].pop((direction, obj.name))
                        except KeyError as e:
                            logger.warning(f"KeyError: {e}")

        return predicates

    def add_obj_to_predicates_after_door_closed(self, door_closed: List[str],
                                                predicates: Dict[str, Dict],
                                                state: ShopState,
                                                action: ShopContinuousAction,
                                                action_success: bool):
        
        if action_success:
            curr_area = action.aimed_obj.area
        else:
            curr_area = self.env.all_obj[state.region].area
            
        for name, predicate in predicates["asset"].items():
            if "is_occ_pre" not in predicate or "is_occ_manip" not in predicate:
                continue
            
            # NOTE (dlee): if we have multiple doors in the future, we may fix this
            if self.env.all_obj[name].area != curr_area:
                for door_name in door_closed:
                    predicate["is_occ_pre"].append(door_name)

            for door_name in door_closed:
                for (direction, ref), occluders in predicate["is_occ_manip"].items(): 
                    if self.env.all_obj[name].area != self.env.all_obj[ref].area:
                        occluders.append(door_name)

        return predicates
                    

    def bring_cache(self, cache: ActionCache):
        new_cache = deepcopy(cache)
        # new_cache.collisions = dict()

        return new_cache


class ShopRewardModel(RewardModel):
    def __init__(self, bc: BulletClient, 
                       env: ShopEnv, 
                       robot: PR2,
                       goal: ShopGoal, 
                       manip: Manipulation,
                       nav: Navigation,
                       config: Dict,
                       execution: bool=False,
                       predicate_manager: "ShopPredicateManager"=None):
        """Initalize a reward model."""
        from Simulation.pybullet.predicate.predicate_shop import ShopPredicateManager

        # Some configurations...
        self.config = config
        self.DEBUG_GET_DATA: bool         = self.config['project_params']['debug']['get_data']
        
        # Reward configuration
        value_or_pref = "value"
        self.REWARD_SUCCESS       : float = config["plan_params"]["reward"]["success"]
        self.REWARD_FAIL          : float = config["plan_params"]["reward"]["fail"]
        self.REWARD_INFEASIBLE    : float = config["plan_params"]["reward"]["infeasible"]
        self.REWARD_TIMESTEP_PICK : float = config["plan_params"]["reward"]["timestep_pick"]
        self.REWARD_TIMESTEP_PLACE: float = config["plan_params"]["reward"]["timestep_place"]
        self.REWARD_TIMESTEP_OPEN : float = config["plan_params"]["reward"]["timestep_open"]
        self.REWARD_TIMESTEP_CLOSE: float = config["plan_params"]["reward"]["timestep_close"]
        self.REWARD_DISTANCE_MULTIPLIER: float = config["plan_params"]["reward"]["distance_multiplier"]
        self.REWARD_PICK_GOAL_OBJ : float = config["plan_params"]["reward"]["pick_goal"]
        self.REWARD_PLACE_GOAL_OBJ : float = config["plan_params"]["reward"]["place_goal"]

        # Variables
        self.bc = bc
        self.env = env
        self.robot = robot
        self.goal = goal
        self.manip = manip
        self.nav = nav
        self.execution = execution

        # Solved
        self.solved = np.zeros((len(self.goal.condition,)))
        self.predicate_manager = ShopPredicateManager(config, self.env) if predicate_manager is None else predicate_manager

    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: ShopEnv, 
                                   robot: PR2,
                                   manip: PR2Manipulation,
                                   nav: Navigation):
        """Re-initalize a reward model with new BulletClient.

        Args:
            bc (BulletClient): New bullet client
            env (ShopEnv): New simulation environment
            robot (PR2): New robot instance
        """
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav


    def probability(self, reward: float, 
                          state: ShopState, 
                          action: ShopContinuousAction, 
                          next_state: ShopState) -> float:
        """
        determinisitic
        """
        return 1.0


    def _check_termination(self, state: ShopState) -> TerminationT:
        """
        Evaluate the termination of the given state.
        Terminate conditions:
            success condition: holding the target object in front of the robot without falling down the other objects
            fail condition: falling down any object

        Args:
            state (ShopState): A state instance to evaluate
        
        Returns:
            TerminationT: Termination condition.
        """
        if self.is_fail(state):     # NOTE(ssh): Checking the failure first is safer.
            return TERMINATION_FAIL
        elif self.is_success(state):
            return TERMINATION_SUCCESS
        else:
            return TERMINATION_CONTINUE


    def is_success(self, state: ShopState) -> bool:
        """Check only the success"""

        obj_lists = self.goal.obj_list
        regions = self.goal.condition
        values = []
        
        ## NOTE (SJ): Caching Predicates 
        temp_pred = self._get_current_predicates(state)

        for objs, (ref, relation) in zip(obj_lists, regions):
            value = True
            if relation == "on":
                for obj in objs:
                    obj_info = state.object_states[obj]
                    if obj_info.region != ref:
                        value = False
                        break
            elif relation == "holding":
                for obj in objs:
                    if obj not in temp_pred["agent"]["right_hand_holding"]:
                        value = False
                        break
            else:
                for obj in objs:
                    obj_info = state.object_states[obj]
                    objects = temp_pred["asset"][obj]["relative_position"][relation.split("_")[0]]
                    if ref not in objects:
                        value = False
                        break
            values.append(value)
        success = np.array(values).all()
        if success:
            state.predicates = temp_pred

        return success


    def is_fail(self, state: ShopState) -> bool:
        """Check only the failure"""
        for name, obj in state.object_states.items():
            if name in state.handle_status:
                continue
            # dropped
            if obj.position[2] < 0.3:
                return True
            
        return False
    

    def get_travel_cost(self, next_state: ShopState,
                        action: ShopContinuousAction,
                        state: ShopState) -> float:
        
        travel_distance = 0
        if action.nav_traj is not None and len(action.nav_traj) > 0:
            travel_distance = self.nav.get_traj_distance(action.nav_traj)

        return travel_distance * self.REWARD_DISTANCE_MULTIPLIER
    
    
    def get_action_reward(self, action: ShopContinuousAction, state: ShopState) -> float:
        
        action_type = action.discrete_action.type
        reward = 0
        if action.is_feasible():

            if action_type == ACTION_PICK:
                if action.aimed_obj.name in self.goal.get_goal_obj_list():
                    reward += self.REWARD_PICK_GOAL_OBJ

            elif action_type == ACTION_PLACE:

                for objs, (ref, relation) in zip(self.goal.obj_list, self.goal.condition):
                    obj = action.aimed_obj.name
                    if obj not in objs:
                        continue

                    if relation == "on":
                        if action.region == ref:
                            reward += self.REWARD_PLACE_GOAL_OBJ
                    elif relation == "holding":
                        continue
                    else:
                        predicates = self._get_current_predicates(state)
                        objects = predicates["asset"][obj]["relative_position"][relation.split("_")[0]]
                        if ref in objects:
                            reward += self.REWARD_PLACE_GOAL_OBJ

        return reward
                

    def _get_current_predicates(self, state: ShopState):
        if len(state.predicates) == 0:
            buffer = self.predicate_manager.asset_manager.skip_occlusion_predicates 
            self.predicate_manager.asset_manager.skip_occlusion_predicates = True
            temp_pred = self.predicate_manager.evaluate(self.config, self.env, state, self.robot, self.manip, self.nav, self.goal, 
                                                        include_occlusion_predicates=False)
            self.predicate_manager.asset_manager.skip_occlusion_predicates = buffer
        else:
            temp_pred = state.predicates

        return temp_pred

        
    def compute_reward(self, next_state: ShopState,
                       action: ShopContinuousAction,
                       state: ShopState) -> float:
        
        action_type = action.discrete_action.type

        travel_cost = self.get_travel_cost(next_state, action, state)
        
        if action_type == ACTION_PICK:
            action_reward = self.REWARD_TIMESTEP_PICK + self.get_action_reward(action, state)
        elif action_type == ACTION_PLACE:
            action_reward = self.REWARD_TIMESTEP_PLACE + self.get_action_reward(action, state)
        elif action_type == ACTION_OPEN:
            action_reward = self.REWARD_TIMESTEP_OPEN + self.get_action_reward(action, state)
        elif action_type == ACTION_CLOSE:
            action_reward == self.REWARD_TIMESTEP_CLOSE + self.get_action_reward(action, state)

        return action_reward - travel_cost


    def sample(self, next_state: ShopState, 
                     action: ShopContinuousAction, 
                     state: ShopState, **kwargs) -> Tuple[float, TerminationT]:
        """Evaluate the reward of the `next_state`.
        +100 for satisfying success condition -> terminate
        -100 for fail -> terminate
        -1 for action failed
            infeasible case:
                - not found collision free trajectory
                - fail to grasp with collision free trajectory
        -1 for each step

        Arguments should match with TransitionModel.sample() returns.

        Args:
            next_state (ShopState): A state to evaluate the reward
            action (ShopContinuousAction): Action used
            state (ShopState): Previous state

        Returns:
            reward (float): Reward value
            termination (TerminationT): Termination flag
        """
        is_rollout = kwargs.get("is_rollout", False)

        # State is set outside of this function.

        # No feasible action is found
        if action is None or not action.is_feasible():
            if self.execution:
                return self.REWARD_INFEASIBLE, TERMINATION_CONTINUE
            else:
                return self.REWARD_INFEASIBLE, TERMINATION_FAIL

        # Check termination condition
        termination = self._check_termination(next_state)

        # Success
        if termination == TERMINATION_SUCCESS:
            if not is_rollout:
                logger.success('======== Success ========')
            # print('======== Success ========')
            return self.REWARD_SUCCESS, termination
        # Fail
        elif termination == TERMINATION_FAIL:
            if not is_rollout:
                logger.warning('======== Fail ========')
            # print('======== Fail ========')
            return self.REWARD_FAIL, termination
        # Penalize all picks.
        elif termination == TERMINATION_CONTINUE:
            if action is None:
                return 0, termination
            return self.compute_reward(next_state, action, state), termination

        # Unhandled reward
        else:
            raise ValueError("Unhandled reward")


class ShopAgent(LLMassistedAgent):

    def __init__(self, bc: BulletClient, 
                       env: ShopEnv, 
                       robot: PR2, 
                       manip: Manipulation,
                       nav: Navigation,
                       config: Dict,
                       sim_blackbox_model: BlackboxModel, 
                       policy_model: PolicyModel, 
                       rollout_policy_model: RolloutPolicyModel, 
                       value_model: ValueModel = None, 
                       goal_condition: ShopGoal = None):
        """Shop agent keeps the blackbox and policy, value model.

        Args:
            bc (BulletClient): skipped
            env (ShopEnv): skipped
            robot (PR2): skipped
            config (Dict): skipped
            sim_blackbox_model (BlackboxModel): Blackbox model instance
            policy_model (PolicyModel): Policy model instance
            rollout_policy_model (RolloutPolicyModel): Policy model instance
            value_model (ValueModel, optional): Value model instance. Defaults to None.
            init_belief (WeightedParticles, optional): Initial belief. Defaults to None.
            init_observation (ShopObservation): Initial ShopObservation instance
            goal_condition (List[float]): Goal condition. [rgbxyz]
        """
        super(ShopAgent, self).__init__(sim_blackbox_model, 
                                            policy_model, 
                                            rollout_policy_model, 
                                            value_model, 
                                            goal_condition)
        self.config = config
        self.DEBUG_GET_DATA  = self.config["project_params"]["debug"]["get_data"]
        self.PLAN_NUM_SIMS   = config["plan_params"]["num_sims"]


        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav

        # Some helpful typing of inherited variables.
        self.goal_condition: ShopGoal

    
    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: ShopEnv, 
                                   robot: PR2,
                                   manip: PR2Manipulation,
                                   nav: Navigation):
        """Re-initalize a reward model with new BulletClient.

        Args:
            bc (BulletClient): New bullet client
            env (ShopEnv): New simulation environment
            robot (PR2): New robot instance
        """
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav


    def _set_simulation(self, state: ShopState):
        set_env_to_state(self.bc, self.env, self.robot, state)


    def get_llm_plan(self, state:State, history:Tuple, goal:Goal=None, **kwargs):
        return super().get_llm_plan(state, history, goal, **kwargs)


    def update(self, real_action: ShopContinuousAction,
                     real_state : ShopState,
                     real_reward: float):
        # Update history
        self._update_history(real_action, real_state, real_reward)


    def detect_exogenous(self, real_action: ShopContinuousAction, real_state: ShopState):
        imagined_state: ShopState = self.tree.parent.parent[real_action.discrete_action][real_action].state
        
        return not ShopState.equal_predicates(imagined_state, real_state)



class ShopReActAgent(ShopAgent):
    def __init__(self, bc: BulletClient, 
                       env: ShopEnv, 
                       robot: PR2, 
                       manip: Manipulation,
                       nav: Navigation,
                       config: Dict,
                       sim_blackbox_model: BlackboxModel, 
                       policy_model: PolicyModel, 
                       rollout_policy_model: RolloutPolicyModel, 
                       value_model: ValueModel = None, 
                       goal_condition: ShopGoal = None):
        """Shop agent keeps the blackbox and policy, value model.

        Args:
            bc (BulletClient): skipped
            env (ShopEnv): skipped
            robot (PR2): skipped
            config (Dict): skipped
            sim_blackbox_model (BlackboxModel): Blackbox model instance
            policy_model (PolicyModel): Policy model instance
            rollout_policy_model (RolloutPolicyModel): Policy model instance
            value_model (ValueModel, optional): Value model instance. Defaults to None.
            init_belief (WeightedParticles, optional): Initial belief. Defaults to None.
            init_observation (ShopObservation): Initial ShopObservation instance
            goal_condition (List[float]): Goal condition. [rgbxyz]
        """
        super(ShopReActAgent, self).__init__(bc, env, robot, manip, nav, config, sim_blackbox_model, policy_model, rollout_policy_model, value_model, goal_condition)
        self.config = config

        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav

        # Some helpful typing of inherited variables.
        self.goal_condition: ShopGoal


def set_env_to_state(bc: BulletClient,
                     env: ShopEnv,   
                     robot: PR2,
                     state: ShopState):
    """
    - Reset robot joint configuration.
    - Remove objects in current simulation.
    - Reload objects according to state in simulation.

    Args:
        bc (BulletClient)
        env (ShopEnv)
        robot (PR2)
        state (ShopState)
    """

    activation_status = deepcopy(state.holding_status)
    obj_in_hands = dict()
    for arm in activation_status.keys():
        obj_in_hands[arm] = env.uid_to_name[activation_status[arm].uid] if activation_status[arm] is not None else None

    for arm in ["left", "right"]:
        robot.release(arm)

    for i in range(bc.getNumConstraints()):
        bc.removeConstraint(bc.getConstraintUniqueId(0))
    
    # Reposition the objects
    for name, entry in state.object_states.items():
        uid = env.all_obj[name].uid
        pos = entry.position
        orn = bc.getQuaternionFromEuler(entry.orientation) if len(entry.orientation) == 3 else entry.orientation
        bc.resetBasePositionAndOrientation(uid, pos, orn)

        env.all_obj[name].position = pos
        env.all_obj[name].orientation = orn
        env.all_obj[name].region = entry.region
        env.all_obj[name].area = entry.area
        
    for name in env.handles.keys():
        env.all_obj[name].is_open = state.object_states[name].is_open


    # Reset robot base
    robot.set_pose(state.robot_base_state)

    # Reset robot joints
    robot.last_pose = state.robot_arm_state
    for value_i, joint_i in zip(state.robot_arm_state, robot.joint_indices_arm):
        bc.resetJointState(robot.uid, joint_i, value_i)

    # Adjust dynamics
    env.reset_dynamics()
    
    # Reactivate grasp
    for arm, name in obj_in_hands.items():
        if name is not None:
            pick_receptacle = name in env.receptacle_obj
            uid = activation_status[arm].uid

            T_oh = activation_status[arm].object_pose
            # T_hw = bc.getLinkState(robot.uid, robot.gripper_link_indices[arm][0])[:2]
            T_hw = robot.get_endeffector_pose(arm)
            object_pose = bc.multiplyTransforms(*T_hw, *T_oh)
            bc.resetBasePositionAndOrientation(uid, *object_pose)
            robot.activate(arm, [uid], object_pose=T_oh, pick_receptacle=pick_receptacle)

            env.all_obj[name].position = object_pose[0]
            env.all_obj[name].orientation = object_pose[1]



    # Restore handle status
    for name, handle_state in state.handle_status.items():
        bc.resetJointState(handle_state.uid, handle_state.link_index, handle_state.position)
        env.all_obj[name].is_open = handle_state.is_open

    # Restore receptacle related info
    if len(state.receptacle_status) > 0:
        new_receptacle_status = dict()
        other_arm = robot.get_other_arm()
        receptacle_uid = robot.activated[other_arm].uid
        for name, entry in state.receptacle_status.items():
            obj = env.all_obj[name]
            contact_constraint = bc.createConstraint(
                parentBodyUniqueId = receptacle_uid,
                parentLinkIndex    = -1,
                childBodyUniqueId = obj.uid,
                childLinkIndex = -1,
                jointType = bc.JOINT_FIXED,
                jointAxis = (0, 0, 1),
                parentFramePosition = entry[1][0],
                childFramePosition = (0, 0, 0),
                childFrameOrientation = entry[1][1])
            attach_info: AttachConstraint = entry[0]
            new_entry = (AttachConstraint(contact_constraint, 
                                        obj.uid, 
                                        attach_info.object_pose,
                                        attach_info.joints, 
                                        attach_info.joint_positions), 
                        entry[1], entry[2])
            new_receptacle_status[name] = new_entry
        state.receptacle_status = new_receptacle_status
    
    robot.receptacle_status = deepcopy(state.receptacle_status)
    robot.receptacle_holding_info = deepcopy(state.receptacle_holding_info)



def capture_shopenv_state(bc: BulletClient, 
                          env: ShopEnv, 
                          robot: PR2,
                          region: str = None,
                          parent: ShopState = None) -> ShopState:
    
    capture = capture_shopenv(bc, env, robot)
    
    state = ShopState.get_ShopState_from_EnvCapture(capture, region, parent)

    return state
    


def make_shop_goal(goal_config) -> ShopGoal:
    """Make goal object from the configuration.

    Args:
        config (Dict)

    Returns:
        ShopGoal
    """

    return ShopGoal(goal_config["objects"], goal_config["regions"])




