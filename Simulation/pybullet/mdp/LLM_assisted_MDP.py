from math import log
import os 
import re
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy
from Simulation.pybullet.mdp.MDP_framework import (BlackboxModel, 
                                                                   RolloutPolicyModel,
                                                                   PolicyModel,
                                                                   ValueModel, 
                                                                   Agent,
                                                                   DiscreteAction,
                                                                   ContinuousAction, 
                                                                   HistoryEntry,
                                                                   State,
                                                                   Goal,
                                                                   PHASE_SIMULATION,
                                                                   PHASE_EXECUTION,
                                                                   TERMINATION_CONTINUE,
                                                                   TERMINATION_FAIL,
                                                                   TERMINATION_SUCCESS)
from LLM.models import LMClient


from Simulation.pybullet.custom_logger import LLOG
logger = LLOG.get_logger()
        

class LLMassistedPolicyModel(PolicyModel):
    def __init__(self, config: Dict, predicate_manager=None):
        super().__init__()
        self.config = config
        
        self.scenario_id = config["problem_params"]["scenario"]
        self.client = LMClient(config)
        self.prompt_generator = None
        self.predicate_manager = predicate_manager
        self.num_plans = 5
        
    
    @abstractmethod
    def str_to_discrete_action(self, state: State, action_args: Tuple[str]):
        r"""
        Convert a tuple of string to a DiscreteAction object.

        args:
        - action_args: Tuple[str] tuple of string.
        return:
        - action: DiscreteAction, DiscreteAction object."""

        raise NotImplementedError


    @abstractmethod
    def sample_llm_plans(self, state: State, *args, **kwargs) -> List[Tuple[DiscreteAction]]:
        '''
        Policy should make escape plan when it is in the deepest valley of death.
        Args:
            state (State): current state.
        Returns:
            escape_plans (List(Tuple(DiscreteAction))): A plan in tuple of discrete actions. This method must return a list of plans.
        '''
        
        raise NotImplementedError
    

    @abstractmethod
    def compute_predicates(self, state: State, *args ,**kwargs):
        pass

    @abstractmethod
    def generate_priority_queue(self, state: State, *args, **kwargs):
        pass

class LLMassistedAgent(Agent):
    def __init__(self, blackbox_model: BlackboxModel, 
                 policy_model: LLMassistedPolicyModel, 
                 rollout_policy_model: RolloutPolicyModel, 
                 value_model: ValueModel = None, 
                 goal_condition=None):
        super().__init__(blackbox_model, policy_model, rollout_policy_model, value_model, goal_condition)
        self._policy_model: LLMassistedPolicyModel
        self.LLM_CALL_CNT = 0
        self.ACC_LLM_CALL_TIME = 0

    def get_llm_plan(self, state:State, history:Tuple, goal:Goal=None, **kwargs):
        '''
        Get escape plan from the policy. Sample continuous parameters.
        Args:
            state (State): current state
            history (Tuple): history
            goal (Goal): goal
        Returns:
            plans (List, List of Tuple, Tuple of Action arguments)
        '''

        start_time = time.time()
        self.imagine_state(state)
        self._policy_model.compute_predicates(state, history, goal, include_occlusion_predicates=True, force_compute=True)
        self.LLM_CALL_CNT += 1
        escape_plans: List[Tuple[DiscreteAction]] = self._policy_model.sample_llm_plans(state, history=history, goal=goal, **kwargs)


        self.ACC_LLM_CALL_TIME += time.time()-start_time
        logger.info(f"Escape plan generation time: {time.time()-start_time}")
        # for idx, plans in enumerate(escape_plans):
        #     logger.info(f"Escape plan {idx}: {plans}")

        return escape_plans
        