import os
import random
from matplotlib.style import available
import numpy as np

from typing import List, Union, Dict, Tuple

# Simulations
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.mdp.MDP_framework import DiscreteAction, HistoryEntry, State

from Simulation.pybullet.envs.shop_env import ShopEnv, save_predicates, load_predicates
from Simulation.pybullet.mdp.shop.shop_MDP import ShopDiscreteAction, ShopContinuousAction, ShopState, ShopGoal, \
                                                                  ACTION_PICK, ACTION_PLACE, ACTION_OPEN, ACTION_CLOSE, \
                                                                  CircularPriorityQueue

from Simulation.pybullet.envs.manipulation import PR2Manipulation
from Simulation.pybullet.envs.navigation import Navigation, dream
from Simulation.pybullet.envs.robot import PR2, load_gripper, remove_gripper
from Simulation.pybullet.predicate.predicate_shop import ShopPredicateManager

# Default policy
from Simulation.pybullet.mdp.shop.policy.llm_assisted_policy import ShopLLMassistedRandomPolicy

# Validator
from Simulation.pybullet.mdp.validator import ShopValidator

# LLM
from LLM.models import LMClient
from LLM.utils.text_parsers import action_parser
from LLM.utils.text_parsers import plan_parser
from LLM.LMP.prompts_shop import ShopPDDLStylePolicyPromptGenerator
from Simulation.pybullet.predicate.reoranize_info import PDDLStyleState

# Scenarios
from Simulation.pybullet.mdp.shop.policy.scenarios import Scripter

# Logger
from Simulation.pybullet.custom_logger import LLOG
logger = LLOG.get_logger()


DEBUG = False

class ShopLLMassistedLLMPolicy(ShopLLMassistedRandomPolicy):
    
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
            env (ShopEnv)
            robot (Robot)
            manip (Manipulation)
            nav (Navigation)
            config (Settings)
        """
        super().__init__(bc, env, robot, manip, nav, config, predicate_manager=predicate_manager)

        self.NUM_FILTER_TRIALS_SAMPLE = config["pose_sampler_params"]["num_filter_trials_sample"]
        self.num_trial = config['project_params']['overridable']['prompt_params']['policy']['num_trial']


        self.lm_inference = LMClient(config).get_response

        # Predicate Manager and Prompt Generator
        self.predicate_manager = ShopPredicateManager(config, self.env)

        # Validator
        self.validator = ShopValidator(self.bc, self.env, self.robot, self.manip, self.nav, self.config)

        self.random_threshold = self.config['plan_params']['policy']['random']

        # Scripter (for debug)
        if DEBUG:
            scenario: Union[int, str] = self.config["problem_params"]["scenario"]
            self.scripter = Scripter(self.env, scenario=scenario)

    

    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: ShopEnv, 
                                   robot: PR2,
                                   manip: PR2Manipulation,
                                   nav: Navigation):
        
        super().set_new_bulletclient(bc, env, robot, manip, nav)
        self.validator.set_new_bulletclient(self.bc, self.env, self.robot, self.manip, self.nav)
        
                                                                
    def compute_predicates(self, state: ShopState, history: Tuple[HistoryEntry], goal: ShopGoal, 
                           step=0, include_occlusion_predicates: bool=False,
                           force_compute: bool=False):
        """Compute the predicates

        Args:
            state (ShopState): Current state
            history (Tuple[HistoryEntry]): History
            goal (ShopGoal): Goal
        """
        # if len(state.predicates) != 0: 
        #     logger.info("Predicate already exists, using cached predicates")
        #     return

        if self.config["predicate_params"]["use_saved"] and len(history) == 1:
            logger.info("Loading the saved predicates...")
            state.predicates = load_predicates(self.config["problem_params"]["scenario"], step)
            return
        
        really_force_compute = force_compute and (history[-1].action is None or history[-1].action.discrete_action.type == ACTION_PLACE)


        # logger.info("Computing predicates...")
        state.predicates = self.predicate_manager.evaluate(self.config, self.env, state, 
                                                           self.robot, self.manip, self.nav, goal,
                                                           include_occlusion_predicates=include_occlusion_predicates,
                                                           force_compute=really_force_compute)
        return


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

        self.compute_predicates(state, history, goal)
        available_actions = self.validator.get_available_discrete_actions(state)

        # Try random action by set ratio
        if random.random() > self.random_threshold:
            valid = False
            trial = 0

            # NOTE: Add scripted behavior here to debug
            if DEBUG:
                valid = True
                discrete_action, parsed_commands, scores = self.scripter.get_next_action(history)
                if len(history) == 1 or len(history) > 1 and history[-1].action.is_feasible():
                    save_predicates(self.bc, self.env, self.robot,
                                    self.scripter.scenario,
                                    self.scripter.counter,
                                    state.predicates)

            while trial < self.num_trial and not valid:
                # Step 1. query LLM
                plan = self.sample_llm_plans(state)
                # discrete_action = self.str_to_discrete_action(state, next_action[0])
                # next_action = plan[0][0]
                valid = (discrete_action is not None) # and self.validator(state, discrete_action)
                if valid: break
                trial += 1
            
            if not valid:
                # if DEBUG: print("No feasible discrete action sampled")
                logger.info("No valid discrete action sampled, sampling random discrete action")
                available_actions = self.validator.get_available_discrete_actions(state)
                discrete_action = random.choice(available_actions)

            discrete_action.probability = self.discrete_action_probability(discrete_action, available_actions, state)    
        else:
            # Sample random action among available choices
            logger.info("Randomly Sampling Discrete Action")

            if len(available_actions) == 0:
                logger.warning(f"No available actions exist at current state:\n{state.predicates}")
                available_actions = self.validator.get_all_discrete_actions(state)
            discrete_action = random.choice(available_actions)
            discrete_action.probability = self.discrete_action_probability(discrete_action, available_actions, state)
        
        return discrete_action


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
        if len(history) > 1 and history[-1].action.nav_traj is None:
            self.sync_roadmap(state, use_pickle=False)
        else:
            self.sync_roadmap(state, use_pickle=True)

        try:
            next_action = super().sample(discrete_action, history, state, goal)
        except Exception as e:
            logger.warning(e)
            next_action = ShopContinuousAction(discrete_action, None, None, None, None, None, None)

        if next_action.manip_traj is None:
            ## TODO (SJ): Think about the failure queue
            failed_lang_command = next_action.discrete_action.lang_command
            state.failure_queue.appendleft(failed_lang_command)
        else: 
            state.failure_queue.appendleft("")

        
        next_action.bullet_filename = discrete_action.bullet_filename

        return next_action
    



class ShopLLMassistedPDDLPolicy(ShopLLMassistedLLMPolicy):
    def __init__(self, bc: BulletClient, 
                 env: ShopEnv, 
                 robot: PR2, 
                 manip: PR2Manipulation, 
                 nav: Navigation, 
                 config: Dict,
                 predicate_manager:ShopPredicateManager=None):
        ShopLLMassistedLLMPolicy.__init__(self, bc, env, robot, manip, nav, config, predicate_manager=predicate_manager)
        self.predicate_converter, self.prompt_generator = self.set_prompt_processors(config)

        ## Propmt for LLM plans
        ## TODO []: Implement the part for generating LMP function
        self.objects = None
        self.predicate_manager = ShopPredicateManager(config, self.env)
        self.validator = ShopValidator(self.bc, self.env, self.robot, self.manip, self.nav, self.config)
        
        self.text_parser = plan_parser
        self.client = LMClient(config)

        self.num_return = config['plan_params']['llm_trigger']['llm_beam_num']

        self.skip_occlusion_predicates = self.config["plan_params"]["skip_occlusion_predicates"]


    def set_prompt_processors(self, config):
        predicate_converter = PDDLStyleState(config)
        if self.config['baseline'] in ['SAYCAN','ReAct']:
            prompt_generator = ReActPromptGenerator(config)
        else:
            prompt_generator = ShopPDDLStylePolicyPromptGenerator(config)
        return predicate_converter, prompt_generator


    def generate_priority_queue(self, state: ShopState):
       
        if self.lmp_function is None:
            self.lmp_function = self.load_lmp_function(self.lmp_filename, trial=0, feedback=[], predicates=state.predicates)
        
        available_discrete_actions = self.validator.get_available_discrete_actions(state)
        if len(available_discrete_actions) == 0:
            if self.validator.exclude_occluded_actions:
                logger.info("No actions available if occluded objects are excluded.")
                self.validator.exclude_occluded_actions = False
                available_discrete_actions = self.validator.get_available_discrete_actions(state)
                self.validator.exclude_occluded_actions = True
            else:
                logger.info("No actions available.")
                available_discrete_actions = self.validator.get_all_discrete_actions(state)

        if state.action_scores != available_discrete_actions:
            ## Initialize the action rank
            logger.info("Ranking available actions...")
            circular_queue = CircularPriorityQueue(self.config)

            trial = 0
            n = 3 
            while trial < n:
                success, action_scores = self.rank_actions(circular_queue, available_discrete_actions, state)
                if success:
                    state.action_scores = action_scores
                    break
                else:
                    trial += 1
                    logger.error(f"[{trial}/{n}] Error in value function")

                    self.lmp_filename.rename(self.lmp_filename.with_suffix(f".failed{trial}.py"))
                    

                    self.lmp_function = self.load_lmp_function(self.lmp_filename ,trial=trial, predicates=state.predicates, 
                                                        feedback=[])
            if trial == n:
                logger.error(f"Failed to generate the value function. Returning infeasible value.")
                raise ValueError("Failed to generate the value function. Returning infeasible value.")
        else:
            logger.info("Using cached action scores...")


    def get_available_discrete_actions(self, state: ShopState, history=[], goal=None) -> List[ShopDiscreteAction]:
        # Validator
        available_discrete_actions = self.validator.get_available_discrete_actions(state)
        return available_discrete_actions


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
            
        self.compute_predicates(state, history, goal)
        self.generate_priority_queue(state)
        
        # # Try random action by set ratio
        # if random.random() > self.config["plan_params"]["policy"]["random"]:
        #     assert False, "Not implemented"

        # # Sample random action among available choices
        # else:
        discrete_action, action_prob = next(state.action_scores)
        
        logger.info(f"Discrete action sampled from policy: {discrete_action}")
        
        return discrete_action


    def sample(self, discrete_action: Union[ShopDiscreteAction, None],
                     history: Tuple[HistoryEntry], 
                     state: ShopState, 
                     goal: ShopGoal) -> ShopContinuousAction:
        
        return ShopLLMassistedLLMPolicy.sample(self, discrete_action, history, state, goal)
    

    def parse_objects(self):
        objects = []
        for obj in self.env.movable_obj.keys():
            objects.append((str(obj), 'movable_object'))
        for obj in self.env.regions.keys():
            if 'door' in obj: continue
            objects.append((str(obj), 'region'))
        for obj in self.env.openable_obj.keys():
            objects.append((str(obj), 'openable'))
        return objects


    def make_code_pick(self, args):
        # return ('pick', args[0], args[1])
        return ('pick', args[0])
    
    def make_code_place(self, args):
        # return ('place', args[0], args[1], args[2], args[3])
        return ('place', args[0], args[1], args[2])
    
    def make_code_open(self, args):
        return ('open', args[0])
    
    def make_code_action(self, action: ShopDiscreteAction):
        if action.type == ACTION_PICK:
            args = (action.aimed_obj.name, action.aimed_obj.region)
            return self.make_code_pick(args)
        elif action.type == ACTION_PLACE:
            args = (action.aimed_obj.name, action.direction, action.reference_obj.name, action.region)
            return self.make_code_place(args)
        elif action.type == ACTION_OPEN:
            args = (action.aimed_obj.name,)
            return self.make_code_open(args)


    ## NOTE (SJ): This function is the only difference from the upper class but the current weird inheritance structure makes it necessary to redefine the function
    def rank_actions(self, action_rank: CircularPriorityQueue, available_discrete_actions: List[ShopDiscreteAction], state: ShopState):
        if self.objects is None:
            self.objects = self.parse_objects()
        ## goal and predicates are organized for the PDDLstyle Qfunction 
        organized_goal, organized_predicates = self.convert_predicates(state.predicates)

        ## The part below remains the same as upper class
        for action in available_discrete_actions:
            code_action = self.make_code_action(action)
            success, action_value = self.lmp_wrapper(self.lmp_function, objects=self.objects, current_state=organized_predicates, goals=organized_goal, next_action=code_action)
            if success:
                action_rank.push(action, action_value)
            else:
                return False, action_rank
            
        action_rank.sort()
        action_rank.shuffle_items_with_same_value()
        return True, action_rank
    

    def sample_llm_plans(self, state: ShopState, *args, **kwargs) -> List[Tuple[ShopDiscreteAction]]:
        '''
        Policy should make escape plan when it is in the deepest valley of death.
        Args:
            state (State): current state.
        Returns:
            escape_plans (List(Tuple(DiscreteAction))): A plan in tuple of discrete actions. This method must return a list of plans.
        '''
        return ShopLLMassistedLLMPolicy.sample_llm_plans(self, state, num_return=self.num_return, **kwargs)