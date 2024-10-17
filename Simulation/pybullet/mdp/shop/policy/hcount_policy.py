import os
from typing import List, Dict, Tuple
import traceback

# Simulations
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.mdp.MDP_framework import HistoryEntry
from Simulation.pybullet.envs.shop_env import ShopEnv, save_predicates, load_predicates
from Simulation.pybullet.mdp.shop.shop_MDP import ShopDiscreteAction, ShopContinuousAction, ShopState, ShopGoal, ShopObjectEntry, Relation, \
                                                                  ACTION_PICK, ACTION_PLACE, ACTION_OPEN, ACTION_CLOSE, CircularPriorityQueue
from Simulation.pybullet.envs.manipulation import PR2Manipulation
from Simulation.pybullet.envs.navigation import Navigation
from Simulation.pybullet.envs.robot import PR2, load_gripper, remove_gripper
from Simulation.pybullet.predicate.reoranize_info import EntityCentricState, PredicateCentricState, PredicateCentricStateV2

# Default policy
from Simulation.pybullet.mdp.shop.policy.llm_assisted_llm_policy import ShopLLMassistedLLMPolicy
from Simulation.pybullet.mdp.shop.policy.llm_assisted_policy import ShopLLMassistedRandomPolicy
from Simulation.pybullet.mdp.shop.value.h_count import HCountValue
from Simulation.pybullet.predicate.predicate_shop import ShopPredicateManager

# Logger
DEBUG = False
USE_LOGGER = True
if USE_LOGGER:
    from Simulation.pybullet.custom_logger import LLOG
    logger = LLOG.get_logger()


class ShopHcountPolicy(ShopLLMassistedLLMPolicy):
    
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


        # Predicate_manager and Prompt Generator
        self.predicate_converter = PredicateCentricStateV2()
        self.error_history = []


        self.sample_mode = config['project_params']['overridable']['mode']  

        self.dump_prompt = config['dump_prompt']

        self.infeasible_value = self.config["plan_params"]["reward"]["infeasible"]

        self.hcount = HCountValue(bc, env, robot, manip, nav, config).sample
        

    
    def rank_actions(self, action_rank: CircularPriorityQueue, available_discrete_actions: List[ShopDiscreteAction], state: ShopState, goal: ShopGoal):

        for action in available_discrete_actions:
            action_value = self.hcount([], state, goal, action)
            action_rank.push(action, action_value)
        
        return True, action_rank
    
    
    @staticmethod
    def error_wrapper(generated_function, **test_input):
        status = False
        message = ""
        try:
            result = generated_function(test_input['action'], test_input['state'], test_input['goal'])

            status = True
            message = result
        except SyntaxError as e:
            status = False
            message = traceback.format_exc()
            # Add suggestions or corrective actions for syntax errors here.
        except Exception as e:
            print(e)
            status = False
            message = traceback.format_exc()
            # Add general troubleshooting steps or suggestions here.


        return status, message



    def normalize_values(self, action_scores):
        """Normalize the scores of the actions

        Args:
            action_scores (Dict[float]): Scores of the actions
        """
        # Normalize the scores
        max_score = max(action_scores.values())
        min_score = min(action_scores.values())
        for action in action_scores:
            action_scores[action] = (action_scores[action] - min_score) / (max_score - min_score)


    def action_score_probs(self, action_scores):
        """Convert scores to probabilities

        Args:
            action_scores (Dict[float]): _description_

        Returns:
            _type_: _description_
        """


        total_score = sum(action_scores.values())
        probs = {action: action_scores[action] / total_score for action in action_scores}
        return probs


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

        if len(history) == 1:
            if self.config["predicate_params"]["use_saved"]:
                state.predicates = load_predicates(self.config["problem_params"]["scenario"], 0)
            else:
                state.predicates = self.predicate_manager.evaluate(self.config, self.env, state, 
                                                                   self.robot, self.manip, self.nav, goal)
                save_predicates(self.bc, self.env, self.robot,
                                self.config["problem_params"]["scenario"],
                                len(history)-1,
                                state.predicates)

        if len(state.predicates) == 0 and len(history) > 1:
                state.predicates = self.predicate_manager.evaluate(self.config, self.env, state, 
                                                                   self.robot, self.manip, self.nav, goal)  
        

        if state.action_scores is None:
            ## Initialize the action rank
            logger.info("Ranking actions...")
            circular_queue = CircularPriorityQueue(self.config)
            available_discrete_actions = self.validator.get_available_discrete_actions(state)

            trial = 0
            n = 3 
            while trial < n:
                success, action_scores = self.rank_actions(circular_queue, available_discrete_actions, state, goal)
                if success:
                    logger.info(f"Action rank: {action_scores}")
                    action_scores.shuffle_items_with_same_value()
                    state.action_scores = action_scores
                    break
                else:
                    trial += 1
                    logger.error(f"[{trial}/{n}] Error in value function: {action_scores}")

            if trial == n:
                logger.error(f"Failed to generate the value function. Returning infeasible value.")
                raise ValueError("Failed to generate the value function. Returning infeasible value.")
            
        else:
            logger.info("Using cached action scores...")
            action_scores = state.action_scores
        
        discrete_action, probs = next(action_scores)
        
        logger.info(f"Discrete action sampled from policy: {discrete_action}")
        return discrete_action