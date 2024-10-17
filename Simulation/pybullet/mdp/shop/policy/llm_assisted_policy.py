import random
from typing import List, Dict, Union, Tuple

from Simulation.pybullet.mdp.MDP_framework import DiscreteAction, HistoryEntry, State
from Simulation.pybullet.mdp.shop.shop_MDP import ShopContinuousAction, ShopGoal, ShopDiscreteAction, ShopState, CircularPriorityQueue

from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.envs.shop_env import ShopEnv
from Simulation.pybullet.envs.robot import PR2, Robot
from Simulation.pybullet.envs.manipulation import Manipulation, PR2Manipulation

from Simulation.pybullet.mdp.MDP_framework import HistoryEntry
from Simulation.pybullet.mdp.shop.shop_MDP import Navigation
from Simulation.pybullet.mdp.LLM_assisted_MDP import LLMassistedPolicyModel
from Simulation.pybullet.mdp.shop.policy.default_samplers import PickSampler, RandomPlaceSampler, OpenDoorSampler, CloseDoorSampler,ACTION_PICK, ACTION_PLACE, ACTION_OPEN, ACTION_CLOSE

# LLM
from Simulation.pybullet.mdp.validator import ShopValidator
from LLM.LMP.prompts_shop import ShopPDDLStylePolicyPromptGenerator
from LLM.utils.text_parsers import action_parser

# Logger
from Simulation.pybullet.custom_logger import LLOG
from Simulation.pybullet.predicate.predicate_shop import ShopPredicateManager
logger = LLOG.get_logger()

MP = "prm"

class ShopLLMassistedRandomPolicy(LLMassistedPolicyModel):
    """policy model guided by LLM"""
    def __init__(self, bc: BulletClient, 
                       env: ShopEnv, 
                       robot: PR2, 
                       manip: PR2Manipulation,
                       nav: Navigation,
                       config: Dict,
                       predicate_manager: ShopPredicateManager=None):
        """
        Args:
            bc (BulletClient)
            env (BinpickEnvRealObjects)
            robot (Robot)
            manip (Manipulation)
            gid_table (GlobalObjectIDTable)
            config (Settings)
        """

        super().__init__(config, predicate_manager=predicate_manager)
        # Config
        self.config = config
        self.NUM_FILTER_TRIALS_PICK       : int = config["pose_sampler_params"]["num_filter_trials_pick"]
        self.NUM_FILTER_TRIALS_PLACE      : int = config["pose_sampler_params"]["num_filter_trials_place"]
        self.NUM_FILTER_TRIALS_FORCE_FETCH: int = config["pose_sampler_params"]["num_filter_trials_force_fetch"]

        # Initialize policy in shop domain
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav

        # Pick sampler
        # Samplers
        self.pick_action_sampler = PickSampler(self.NUM_FILTER_TRIALS_PICK)
        self.place_action_sampler = RandomPlaceSampler(self.NUM_FILTER_TRIALS_PLACE)
        self.open_action_sampler = OpenDoorSampler(self.NUM_FILTER_TRIALS_PICK)
        self.close_action_sampler = CloseDoorSampler(self.NUM_FILTER_TRIALS_PICK)

        self.validator = ShopValidator(bc, env, robot, manip, nav, config, predicate_manager=predicate_manager)
        self.prompt_generator = ShopPDDLStylePolicyPromptGenerator(config)
        self.text_parser = action_parser

        self.num_return = config['plan_params']['llm_trigger']['llm_beam_num']

        self.use_hcount = self.config["project_params"]["overridable"]["value"] == "hcount"


    def get_available_discrete_actions(self, state: ShopState, *args, **kwargs) -> List[DiscreteAction]:
        if self.use_hcount:
            state.predicates = self.validator.predicate_manager.evaluate(self.config, self.env, state, \
                                                                         self.robot, self.manip, self.nav, \
                                                                         include_occlusion_predicates=True)
        
        available_discrete_actions = self.validator.get_available_discrete_actions(state)
        return available_discrete_actions

        
    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: ShopEnv, 
                                   robot: Robot,
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
    

    def sample_discrete_action(self, state: ShopState, 
                        history: Tuple[HistoryEntry],
                        goal: ShopGoal) -> ShopDiscreteAction:
        """Sample an discrete_action from available choices.
        For PICK, both learned and random policy uses random pick, so choose randomly from the available choices.
        For PLACE, there is only one discrete_action (i.e. PLACE <holding_obj_gid>), so random choice always results in the single discrete_action.
        Args:
            state (ShopState)
        Returns:
            ShopDiscreteAction: sampled discrete_action
        """
        discrete_actions = self.validator.get_available_discrete_actions(state)
        op = random.choice(discrete_actions)

        return op


    def sample(self, discrete_action: Union[ShopDiscreteAction, None],
                     history: Tuple[HistoryEntry], 
                     state: ShopState, 
                     goal: ShopGoal) -> ShopContinuousAction:
        """Infer next_action using neural network

        Args:
            init_observation (ShopObservation)
            history (Tuple[HistoryEntry])
            state (ShopState): Current state to start rollout
            goal (List[float]): Goal passed from the agent.
        Returns:
            ShopContinuousAction: Sampled next action
        """

        # Select primitive action.
        assert discrete_action is not None, "DiscreteAction must be provided to sample continuous parameters"

        if len(history) > 1 and history[-1].action.nav_traj is None:
            self.sync_roadmap(state, use_pickle=False)
        else:
            self.sync_roadmap(state, use_pickle=True)

        # PICK action
        if discrete_action.type == ACTION_PICK:
            next_action = self.infer_pick_action(discrete_action, history, state, goal)
        # PLACE: Use guided policy
        elif discrete_action.type == ACTION_PLACE:
            next_action = self.infer_place_action(discrete_action, history, state, goal)
        # OPEN action
        elif discrete_action.type == ACTION_OPEN:
            next_action = self.infer_open_action(discrete_action, history, state, goal)
        # CLOSE action
        elif discrete_action.type == ACTION_CLOSE:
            next_action = self.infer_close_action(discrete_action, history, state, goal)


        return next_action


    def sync_roadmap(self, state: ShopState, use_pickle: bool=True):

        # NOTE (dlee): Change the trigger if needed
        is_open = self.env.openable_obj["kitchen_door"].is_open

        if is_open:
            self.nav.reset_roadmap(use_pickle=use_pickle, door_open=True)
        else:
            self.nav.reset_roadmap(use_pickle=use_pickle, door_open=False)


    def infer_pick_action(self, discrete_action: ShopDiscreteAction, 
                          history: Tuple[HistoryEntry],
                          state: ShopState,
                          goal: ShopGoal) -> ShopContinuousAction:
        """Infer pick action

        Args:
            discrete_action (ShopDiscreteAction)
            history (Tuple[HistoryEntry])
            state (ShopState)
            goal (List[float])
            
        Returns:
            ShopContinuousAction: next action
        """

        next_action = self.pick_action_sampler(self.bc, self.env, self.robot, self.nav, self.manip, 
                                               state, discrete_action, ignore_movable=False)


        return next_action


    def infer_place_action(self, discrete_action: ShopDiscreteAction,
                           history: Tuple[HistoryEntry], 
                           state: ShopState,
                           goal: ShopGoal) -> ShopContinuousAction:
        """Infer place action

        Args:
            discrete_action (ShopDiscreteAction)
            history (Tuple[HistoryEntry])
            state (ShopState)
            goal (List[float])
            
        Returns:
            ShopContinuousAction: next action
        """

        next_action = self.place_action_sampler(self.bc, self.env, self.robot, self.nav, self.manip, 
                                                state, discrete_action, ignore_movable=False)


        return next_action
    

    def infer_open_action(self, discrete_action: ShopDiscreteAction, 
                          history: Tuple[HistoryEntry],
                          state: ShopState,
                          goal: ShopGoal) -> ShopContinuousAction:
        """Infer open action

        Args:
            discrete_action (ShopDiscreteAction)
            history (Tuple[HistoryEntry])
            state (ShopState)
            goal (List[float])
            
        Returns:
            ShopContinuousAction: next action
        """

        next_action = self.open_action_sampler(self.bc, self.env, self.robot, self.nav, self.manip, 
                                               state, discrete_action, ignore_movable=False)


        return next_action
    

    def infer_close_action(self, discrete_action: ShopDiscreteAction, 
                           history: Tuple[HistoryEntry],
                           state: ShopState,
                           goal: ShopGoal) -> ShopContinuousAction:
        """Infer close action

        Args:
            discrete_action (ShopDiscreteAction)
            history (Tuple[HistoryEntry])
            state (ShopState)
            goal (List[float])
            
        Returns:
            ShopContinuousAction: next action
        """

        raise NotImplementedError
    

    def str_to_discrete_action(self, state: ShopState, action_args: Tuple[str]):
        r"""
        Convert a tuple of string to a DiscreteAction object.

        args:
        - action_args: Tuple[str] tuple of string.
        return:
        - action: DiscreteAction, DiscreteAction object."""

        try:
            action_type = action_args[0].lower()
            aimed_obj = self.env.all_obj[action_args[1]]

            if action_type == 'open':
                action = ShopDiscreteAction(ACTION_OPEN, self.robot.main_hand, aimed_obj, aimed_obj.region, lang_command=f"open {aimed_obj.name}")
            elif action_type == 'pick':
                action = ShopDiscreteAction(ACTION_PICK, self.robot.main_hand, aimed_obj, aimed_obj.region, lang_command=f"pick {aimed_obj.name}")
            elif action_type == 'place':
                direction = action_args[2]
                reference = self.env.all_obj[action_args[3]]
                action = ShopDiscreteAction(ACTION_PLACE, self.robot.main_hand, aimed_obj, reference.region, direction, reference, lang_command=f"place {aimed_obj.name} {direction} {reference.name}")
            else:
                logger.warning(f"Invalid action type: {action_args}")
                action = None
        except KeyError as e:
            logger.warning(e)
            action = None

        return action

    def convert_prompt_list_to_string(self, prompt: List[Dict]):
        output = ""

        for p in prompt:
            output += f"# {p['role']}\n{'---'*10}\n{p['content']}\n\n{'==='*10}\n\n"
        return output

    ##  LLM assisted functionalities
    def sample_llm_plans(self, state: ShopState, **kwargs) -> List[Tuple[ShopDiscreteAction]]:
        '''
        Policy should make escape plan when it is in the deepest valley of death.
        Args:
            state (State): current state.
        Returns:
            escape_plans (List(Tuple(DiscreteAction))): A plan in tuple of discrete actions. This method must return a list of plans.
        '''
        logger.info('Querying LLM for next actions.')

        prompt = self.prompt_generator.generate_prompt(state.predicates, scenario_id=self.scenario_id, **kwargs)
        # print(prompt[1]['content'])
        
        logger.trace(f"$$Prompt$$\n {self.convert_prompt_list_to_string(prompt)}")
        raw_response = self.client.get_response(
            prompt, 
            generation_args={
                'num_return': self.num_return,
                'logprobs': False,
                'max_tokens': 500,
                'temperature': 1
            }
        )

        response = [r.text for r in raw_response]


        ## ADD TRACE 
        parsed_plans = []

        ## NOTE(SJ): Change code that it also returns discrete action from plan, we don't consider the feasibility of the plan here
        for _response in response:
            logger.trace(f"$$ Response from LLM $$\n {_response}")
            parsed_plan = self.text_parser(_response)
            discrete_action_plan = []
            for str_action in parsed_plan:
                parsed_action = self.str_to_discrete_action(state, str_action)
                if parsed_action is None:
                    break 
                else: 
                    parsed_action.llm_data = {'response':_response}
                    discrete_action_plan.append(parsed_action)
                
            if len(discrete_action_plan) == 0:
                logger.warning(f"Failed to parse the plan: {_response}")
            else:
                parsed_plans.append(discrete_action_plan)     

        return parsed_plans
    

    def discrete_action_probability(self, action: ShopDiscreteAction, available_actions: List[ShopDiscreteAction], state: ShopState) -> float:
        num_available_actions = len(available_actions)
        if num_available_actions == 0: 
            return 0
        else:
            return float(1/len(available_actions))
        
        
    def generate_priority_queue(self, state: ShopState):
        available_discrete_actions = self.validator.get_available_discrete_actions(state)

        if state.action_scores is None or state.action_scores != available_discrete_actions:
            ## Initialize the action rank
            logger.info("Ranking available actions...")
            circular_queue = CircularPriorityQueue(self.config)

            success, action_scores = self.rank_actions(circular_queue, available_discrete_actions, state)
            state.action_scores = action_scores
        else:
            logger.info("Using cached action scores...")


    def rank_actions(self, action_rank: CircularPriorityQueue, available_discrete_actions: List[ShopDiscreteAction], state: ShopState):
        for action in available_discrete_actions:
            action_value = self.discrete_action_probability(action, available_discrete_actions, state)
            action_rank.push(action, action_value)

        action_rank.sort()
        action_rank.shuffle_items_with_same_value()
        return True, action_rank
    


class ShopLLMassistedRolloutPolicy(ShopLLMassistedRandomPolicy):

    def compute_predicates(self, state: ShopState, history: Tuple[HistoryEntry], goal: ShopGoal, step=0, include_occlusion_predicates=True):
        return dict()