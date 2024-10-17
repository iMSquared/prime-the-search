from pdb import run
import time, sys
import math
import random
from copy import deepcopy
from typing import Tuple, Union, List
from collections import deque
import numpy as np


from Simulation.pybullet.mdp.MDP_framework import *
from Simulation.pybullet.mdp.MDP_framework import Agent, Environment
from Simulation.pybullet.mdp.online_planner_framework import *
from Simulation.pybullet.mdp.LLM_assisted_MDP import LLMassistedAgent, LLMassistedPolicyModel

# Debugging
from Simulation.pybullet.mdp.shop.shop_MDP import CircularPriorityQueue

## LLM-MCTS code 
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils




# Collecting data
from data_generation.collect_data import collect_shop_trajectory_data
from Simulation.pybullet.custom_logger import LLOG, CustomLogger



logger: CustomLogger = LLOG.get_logger()
# tracer = LLOG.get_viztrace()

class MCTS(Planner):

    def __init__(self, agent: Agent, env: Environment, config: Dict):
        """
        Args:
            max_depth (int): Depth of the MCTS tree. Default: 5.
            planning_time (float): amount of time given to each planning step (seconds). Defaults to -1.
                if negative, then planning terminates when number of simulations `num_sims` reached.
                If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
            num_sims (int): Number of simulations for each planning step.
                If negative, then will terminate when planning_time is reached.
                If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
            rollout_policy (RolloutPolicy): rollout policy. Default: RandomRollout.
            action_prior (ActionPrior|TODO|): a prior over preferred actions given state and history.
        """
        self.config = config
        # To simplify function calls; plan only for one agent at a time
        self._agent = agent
        self._env = env

        # Configuration
        self._COLLECT_DATA    : bool    = config["project_params"]["overridable"]["collect_data"]
        self._MAX_DEPTH       : int     = config["plan_params"]["max_depth"] + 1
        self._MAX_DEPTH_REWARD: float   = config["plan_params"]["reward"]["max_depth_reward"]
        self._NUM_SIMS        : int     = config["plan_params"]["num_sims"]
        self._PLANNING_TIME   : float   = config["plan_params"]["planning_time"]
        if self._NUM_SIMS < 0 and self._PLANNING_TIME < 0:
            self._PLANNING_TIME = 1.
        self._NUM_VISITS_INIT   : int   = 0
        self._VALUE_INIT        : int   = 0.
        self._DISCOUNT_FACTOR   : float = config["plan_params"]["discount_factor"]
        self._EXPLORATION_CONST : float = config["plan_params"]["exploration_const"]["ucb1"]
        self._PUCB_EXPLORATION_CONST: float = config["plan_params"]["exploration_const"]["pucb"]
        self._K_A               : float = config["plan_params"]["k_a"]
        self._ALPHA_A           : float = config["plan_params"]["alpha_a"]
        self._SELECT_BEST_ACTION: bool  = config["plan_params"]["select_best_action"]
        self._FOLLOW_SUCCESS_SIMULATION: bool = config["plan_params"]["follow_success_simulation"]
        self._rollout_depth     : int   = config["plan_params"]["rollout_depth"]

        # For collecting data
        self.num_sim_total   = 0
        self.num_sim_success = 0
        self.llm_called_count = 0
        self.sim_trajs       = []   # Only active when in COLLECT_DATA = True

        self.problem = config["project_params"]["problem"]
        if self.problem == "shop":
            self.collect_mdp_trajectory_data = collect_shop_trajectory_data
        else:
            raise NotImplementedError
        
        self._TIME_LIMIT = config["plan_params"]["time_limit"]



    def plan(self) -> Tuple[Action, float, int, 
                            int, int,
                            List[Dict]]:
        """Plan!

        Returns:
            next_action (Action): Planned next action.
            time_taken (float): Time taken for planning one action.
            num_sim_total (int): Number of tried simulations.   
            num_sim_success (int): Number of simulation success. 
            sim_trajs (List[Dict]): Data of all trajectories. Activated when _COLLECT_DATA=True
        """

        # Initialize the root node of the search tree to the agent.
        # |TODO(ssh)|: This cause the root node to have unweighted particle.
        #   When looking into the POMCPOW paper, the root has unweighted particle....????
        if not hasattr(self._agent, "tree"):
            root = self.create_node(root=True)
            self._agent.add_attr("tree", root)

        # Logging variables...
        time_taken = 0.0
        start_time = time.time()
        self.num_sim_total = 0
        self.num_sim_success = 0

        # Data collection
        if self._COLLECT_DATA:
              self.sim_trajs = []

        # Planning loop
        state = self._env.current_state
        if self._agent.tree.parent is None or self._agent.tree.num_visits < self.config["plan_params"]["visit_threshold"]:
            for i in range(self._NUM_SIMS):
                if self._FOLLOW_SUCCESS_SIMULATION and self._agent._success_plan_found: 
                    break
                LLOG.sim_idx += 1

                # Setting up the simulation of a particle
                #   Imagine simulation with sampled state
                self._agent.imagine_state(state)
                #   Initialize simulation history as excuted history
                history = self._agent.history

                # Selection, Expansion, Simulation, Backpropagation
                total_reward = self._simulate(
                    curr_node   = self._agent.tree,
                    history = history,
                    depth   = len(history))
                
                # Update root V (V Backup)
                # self._agent.tree.value = self._agent.tree.value\
                #                        + (total_reward - self._agent.tree.value) / (self._agent.tree.num_visits)
                
                # Log cumulative time
                self.num_sim_total +=1
                time_taken = time.time() - start_time
                logger.info(f"#sim: {self.num_sim_total}, accumulated_time: {time_taken}")

        # Selecting the next action. 
        next_action = self.get_next_action()


        return next_action, \
            time_taken, self.num_sim_total, self.num_sim_success, \
            self.sim_trajs
    

    def _simulate(self, curr_node: Union[ContinuousNode, DiscreteNode],
                        history: Tuple[HistoryEntry], 
                        depth: int) -> float:
        """Implementation of `simulate(s, h, d)`
        root<-class:VNode, parent<-class:QNode
        
        Args:
            state (State): A state to simulate from
            history (Tuple[HistoryEntry]): History of the state particle
            root (RootVNodeParticles): Root node of the subtree to search.
            depth (int): Current depth

        Returns:
            float: The total discounted reward during the plannnig
        """

        
        # Select the best discrete_action child of the current state.
        op = self.choose_discrete_action(curr_node, history)
        # logger.info(f"Selected discrete action: {op}")

        # Select the best action (discrete_action + cont param) of the current state
        action = self.action_widening(curr_node[op], history)

        logger.info(f"Selected action: {action}")
        # Generative model: take one step in simulation. We don't know what new action does at this point. Success reward should be given here. 
        next_state, reward, termination = self._sample_generative_model(curr_node.state, action)
        
        # If the sampled action (sampled cont param) is new
        if curr_node[op][action].num_visits == 0:
            # Log history
            history += (HistoryEntry(action, next_state, reward, PHASE_SIMULATION),)
            curr_node[op][action].state = next_state


            # Skip rollout if already terminated
            if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):
                total_reward = self.update_and_log_at_termination(curr_node, op, action, next_state, history, termination, reward)
                # return total_reward
            # Rollout or value guidance
            else:
                total_reward = self.evaluate(curr_node, next_state, history, reward, depth, op, action, verbose=True)

        # If revisiting the existing node
        else:
            history += (HistoryEntry(action, next_state, reward, PHASE_SIMULATION),)

            if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):
                total_reward = self.update_and_log_at_termination(curr_node, op, action, next_state, history, termination, reward)
                # return total_reward
            
            else:
                total_reward = reward + self._DISCOUNT_FACTOR * self._simulate(
                    curr_node = curr_node[op][action],
                    history = history,
                    depth   = depth + 1)


        # Update visit counts and Q values.
        if curr_node.is_root:
            curr_node.num_visits += 1

        curr_node[op].num_visits += 1
        curr_node[op].value = curr_node[op].value + (total_reward - curr_node[op].value) / (curr_node[op].num_visits)

        curr_node[op][action].num_visits += 1
        curr_node[op][action].value = curr_node[op][action].value \
                                    + (total_reward - curr_node[op][action].value) / (curr_node[op][action].num_visits)

        return total_reward
    

    def evaluate(self, curr_node: DiscreteNode,
                 next_state: State,
                 history: Tuple[HistoryEntry], 
                 reward: float,
                 depth: int, 
                 discrete_action: DiscreteAction=None, 
                 action: ContinuousAction=None,
                 verbose: bool=False):
        
        if self._agent._value_model is not None:
            # V or Q
            
            if self._agent._value_model.input_state_type == 'current':
                    next_state = curr_node.state

            if self._agent._value_model.is_v_model:
                accumulated_reward = self._agent.get_value(history, next_state, goal=self._agent.goal_condition, action=action)
                logger.info(f"LMP Value: {accumulated_reward}")
                logger.info(f"Reward: {reward}")
                total_reward = reward + self._DISCOUNT_FACTOR * accumulated_reward

            else:
                accumulated_reward = self._agent.get_value(history, next_state, goal=self._agent.goal_condition, action=action)
                total_reward = accumulated_reward
            
            if verbose:
                logger.info(f"Value: {accumulated_reward}")
                logger.info(f"Total Reward: {total_reward}")

        else:
            logger.info("Proceeding with Rollout")
            value = self._rollout(
                node   = curr_node[discrete_action][action],
                history = history,
                depth   = len(history)+1)
            logger.info(f"Rollout Value: {value}")
            logger.info(f"Reward: {reward}")
            total_reward = reward + self._DISCOUNT_FACTOR * value
            
            
        return total_reward

    
    def update_and_log_at_termination(self, curr_node: DiscreteNode, discrete_action: DiscreteAction, action: Action, 
                              next_state: State, history: Tuple[HistoryEntry], 
                              termination: TerminationT, reward: float) -> float:

        total_reward = reward

        # Logging
        if termination == TERMINATION_SUCCESS:
            setattr(curr_node, "succeeded", True)
            setattr(curr_node[discrete_action], "succeeded", True)
            setattr(curr_node[discrete_action][action], "succeeded", True)
            while not curr_node.is_root:
                curr_node = curr_node.parent
                setattr(curr_node, "succeeded", True)
            self.num_sim_success += 1

        # Data collection
        if self._COLLECT_DATA:
            self.sim_trajs.append(
                self.collect_mdp_trajectory_data(agent   = self._agent,
                                                history     = history, 
                                                termination = termination))
            
        # Return with the reward when simulation terminates.
        return total_reward


    def assign_action_prob(self, action_value_pairs: List[Tuple[DiscreteAction, float]]):
        temp = self.config['plan_params']['exponential_temperature']
        method: str = self.config['plan_params']['policy_value']

        prob_list: List[Tuple[DiscreteAction, float, float]] = []

        if len(action_value_pairs) == 0:
            raise ValueError("No items in the queue")

        if method == "exponential":
            total = sum(np.exp(weight / temp) for _, weight in action_value_pairs)
            for i, (item, weight) in enumerate(action_value_pairs):
                prob = np.exp(weight / temp) / total
                prob_list.append((item, weight, prob))
        elif method == "weighted":
            total = sum(weight for _, weight in action_value_pairs)
            for i, (item, weight) in enumerate(action_value_pairs):
                prob = weight / total
                prob_list.append((item, weight, prob))
        elif method == "uniform":
            prob = 1 / len(action_value_pairs)
            for i, (item, _) in enumerate(action_value_pairs):
                prob_list.append((item, weight, prob))
        else:
            raise ValueError("Policy value must be either 'exponential', 'weighted', or 'uniform'")

        return prob_list


    def _rollout(self, node: DiscreteNode, history: Tuple[HistoryEntry], depth: int) -> float:
        """Implementation of Rollout
        This function does not manipulate any tree nodes in the planner, 
        i.e. VNode or QNode. It only uses the POMDP model.

        Args:
            state (State): The initial state to start the rollout from.
            history (Tuple[HistoryEntry]): The history of the state
            depth (int): The depth of the state.

        Returns:
            float: total_discounted_reward
        """
        discount = self._DISCOUNT_FACTOR
        total_discounted_reward = 0.0

        state = node.state
        # discrete_action = node.discrete_action

        ## NOTE(SJ): We need to change this into another type of rollout since we are not using max depth now. 
        while depth <= self._MAX_DEPTH:
        # while depth <= self._rollout_depth:
            discrete_action = random.choice(self._agent.get_available_discrete_actions(state, rollout=True))

            action = self._agent.get_action(discrete_action, history, state, goal=self._agent.goal_condition, rollout=True)
            
            next_state, reward, termination = self._sample_generative_model(state, action, running_escape=True, is_rollout=True)
            history += (HistoryEntry(action, next_state, reward, PHASE_ROLLOUT),)

            depth += 1
            total_discounted_reward += reward * discount
            discount *= self._DISCOUNT_FACTOR
            state = next_state

            # Check goal condition while rollout
            if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):
                # # Logging
                # if termination == TERMINATION_SUCCESS:
                #     self.num_sim_success += 1
                # # Data collection (rollout reached termination condition)
                # if self._COLLECT_DATA:
                #       self.sim_trajs.append(
                #         self.collect_mdp_trajectory_data(agent   = self._agent, 
                #                                         history     = history,
                #                                         termination = termination))
                # # Return reward
                return total_discounted_reward

    
        # Max depth penalty
        total_discounted_reward += self._MAX_DEPTH_REWARD * discount
        # Return reward
        return total_discounted_reward

    def register_legal_actions(self, node: DiscreteNode, history: Tuple[HistoryEntry]):
        """
        Register all the legal actions to the node.
        """
        available_discrete_actions = self._agent.get_all_discrete_actions(node.state, history, goal=self._agent.goal_condition, rollout=False)
        for op in available_discrete_actions:
            new_continuous_node = self.create_node(parent_node=node, parent_action=op)
            node[op] = new_continuous_node


    def choose_discrete_action(self, node: DiscreteNode, history: Tuple[HistoryEntry]):
        """Choose discrete_action as the operation skeleton.
            For discrete_action, there is no limit on the expand limit.
        Args:
            curr_node (DiscreteNode)
        """

        # Add all available actions
        if len(node.children) == 0:
            self.register_legal_actions(node, history)

        if self.config["plan_params"]["selection"] == "ucb1":
            return self._ucb(node)
        elif self.config["plan_params"]["selection"] == "pucb":
            return self._pucb(node)
        else:
            raise NotImplementedError("UCB1 or PUCB only")
        


    def action_widening(self, node: ContinuousNode, history: Tuple[HistoryEntry]) -> Action:
        """Implementation of action progressive widening.
        1. Appends a new action child if the number of action child 
           is below the (k_a * vnode.num_visits ** alpha_a).
        2. Returns the next action child, where a = argmax_a UCB(s, a).
           
        Args:
            vnode (VNode): Vnode to evaluate.
            history (Tuple[HistoryEntry]): The history will be passed to the policy models.
            state (State): The particle state will be passed to the policy model.

        Returns:
            Action: Selected action.
        """
        if len(node.children) <= self._K_A * node.num_visits ** self._ALPHA_A:
            _action = self._agent.get_action(node.discrete_action, history, node.state, goal=self._agent.goal_condition, rollout=False)

            if node[_action] is None:
                new_node = self.create_node(parent_node=node, parent_action=_action)
                node[_action] = new_node
            else:
                print("")

        return self._ucb(node)


    def _ucb(self, node: Union[DiscreteNode, ContinuousNode]) -> Action:
        """UCB1
        Selects the action child of the `root` with the best UCB value.

        Args:
            root (VNode): A vnode to select the action child from.

        Returns:
            Action: Selected action.
        """
        best_action, best_value = None, float('-inf')
        actions = list(node.children.keys())
        random.shuffle(actions)
        for action in actions:
            # if node[action].num_visits == 0:
            #     val = float('inf')
            # else:
            val = node[action].value + \
                self._EXPLORATION_CONST * math.sqrt(math.log(node.num_visits + 1) / (node[action].num_visits+1))
                    
            if val > best_value:
                best_action = action
                best_value = val

        return best_action
    

    def _pucb(self, node: DiscreteNode) -> Action:
        """PUCB
        Selects the action child of the `root` with the best PUCB value.

        Args:
            root (VNode): A vnode to select the action child from.

        Returns:
            Action: Selected action.
        """
        best_action, best_value = None, float('-inf')
        # Assign values
        action_value_pairs:  List[Tuple] = []
        for action in node.children.keys():
            if node[action].num_visits == 0:
                # q = 0
                if self.config['plan_params']['initialize_q'] == 'zero':
                    q = 0
                else:
                    raise NotImplementedError("Initial Q value should be zero or Qvalue")

            else:
                q = node[action].value
            
            action_value_pairs.append((action, q))

        ## NOTE(SJ): This is a bit of a hacky way to implement the PUCB.
        prob_list = self.assign_action_prob(action_value_pairs)
        
        for action, q, p in prob_list:
            # p: action.probability
            u = self._PUCB_EXPLORATION_CONST * math.sqrt(node.num_visits) / (1+node[action].num_visits)
            
            if node[action].exploration_bonus == 0:
                node[action].exploration_bonus = p*u

            val = q + p*u
                    
            if val > best_value:
                best_action = action
                best_value = val

        return best_action
    

    def _sample_generative_model(self, state: State, action: Action, **kwargs) \
                                        -> Tuple[ State, Observation, float, str ]:
        """(s', o, r) ~ G(s, a)

        Args:
            state (State): Current state
            action (Action): Action to do.

        Returns:
            next_state (State): Generated next state
            observation (Observation): Generated observation 
            reward (float): Generated reward
            termination (str): Termination condition. "success" or "fail" or None.
        """
        is_rollout = kwargs.get("is_rollout", False)
        next_state, reward, termination = self._agent.blackbox_model.sample(state, action, **kwargs)
        
        if termination == TERMINATION_SUCCESS and not is_rollout:
            self._agent._success_plan_found = True

        return next_state, reward, termination


    def create_node(self, parent_node: Union[DiscreteNode, ContinuousNode] = None, 
                    parent_action: Union[Action, DiscreteAction] = None, root: bool = False) -> TreeNode:
        """
        Returns a Node with default values;
        The function naming makes it clear that this function is about creating a VNode object.
        """
        is_ops_skeleton_node = (parent_node is None) or not isinstance(parent_node, DiscreteNode)

        # Determine state and value
        if root:
            state = self._env.current_state
            value = self._VALUE_INIT
        elif isinstance(parent_node, DiscreteNode):
            state = parent_node.state
            # value = parent_node.
            value = self._VALUE_INIT
        else:   # parent_node is DiscreteNode
            state = None
            value = self._VALUE_INIT

        if is_ops_skeleton_node:
            node = DiscreteNode(state, self._NUM_VISITS_INIT, value, parent=parent_node, is_root=root)
        else:
            node = ContinuousNode(state, self._NUM_VISITS_INIT, value, parent_action, parent=parent_node, is_root=root)

        return node


    # |FIXME(Jiyong)|: how to implement changing root node to existing node in discrete case and particle reinvigoration.
    def update(self, agent: Agent, real_action: Action, real_state: State):
        # agent.tree[real_action.discrete_action][real_action] = self.create_node(root=True)
        # Should we recylce the tree? Yes

        # agent.tree = self.create_node(root=True)
        if real_action is None or (not real_action.is_feasible()):
            return agent.tree
        else:
            agent.tree = agent.tree.children[real_action.discrete_action][real_action]
            ## NOTE (SJ): If a node that has never been visited i.e. num_visits == 0 because the nods value was initialized with 0, the node will not contain a state.
            if agent.tree.state is None: 
                agent.tree.state = real_state
            # if self.config["plan_params"]["detect_exogenous"] and agent.detect_exogenous(real_action, real_state):
            #     agent.tree.children = dict()
            agent.tree.is_root = True

            return agent.tree

    def update_num_visits(self, node: DiscreteNode):
        assert False, "NOTE (SJ): This function is currently not used."
        # This is to update the node num_visits when exogenous event happens
        # We subtract 1 because 1 is num_visit to the node. 
        # Rest of them are visits to the child nodes which need to be cleared.
        cleared_num_visit = node.num_visits - 1
        while not node.parent is None:
            node.num_visits -= cleared_num_visit
            node = node.parent.parent

    def get_next_action(self):
        '''
            Based on the designated method, 
        '''
        success_plan_action = None
        for discrete_action, continuous_node in self._agent.tree.children.items():
            for action, discrete_node in continuous_node.children.items():
                if hasattr(discrete_node, "succeeded") and discrete_node.succeeded:
                    success_plan_action = action
        
        if success_plan_action is None:
            if self._SELECT_BEST_ACTION:
                next_action = self._agent.tree.argmax()
            else:
                next_action = self._agent.tree.sample()
        else:
            next_action = success_plan_action

        return next_action
    
    def get_success_plan(self) -> List[Action]:

        success_plan = []
        if not self._agent._success_plan_found:
            return success_plan
        
        curr_node = self._agent.tree
        
        while len(curr_node.children) != 0:
            for discrete_action, continuous_node in curr_node.children.items():
                found = False
                for action, discrete_node in continuous_node.children.items():
                    if hasattr(discrete_node, "succeeded") and discrete_node.succeeded:
                        success_plan.append(action)
                        curr_node = discrete_node
                        found = True
                        break
                if found:
                    break


        return success_plan
         


class LLMMCTS(MCTS):
    def __init__(self, agent: Agent, env: Environment, config: st_utils.Dict):
        super().__init__(agent, env, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translation_lm = SentenceTransformer('stsb-roberta-large').to(self.device)
        self.config["plan_params"]["selection"] = "pucb"
        
    
    def _calculate_emperical_prob(self, state, valid_action_list):
        LAMBDA = 0.5
        valid_actions_lang = [action.lang_command for action in valid_action_list]
        # print(valid_actions_lang)
        valid_action_embedding = self.translation_lm.encode(valid_actions_lang, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        # task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\nNext plan:"
        action_dis, valid_action_lists, num_steps, is_done = self.ground_actions_softmax(state, valid_action_list, valid_action_embedding)
        emperical_prob = LAMBDA * action_dis + (1-LAMBDA) /len(valid_action_list)
        return emperical_prob 

    

    def _find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        cos_scores = cos_scores - np.mean(cos_scores) 
        return cos_scores


    def ground_actions_softmax(self, state, valid_actions, valid_action_embedding):
        logger.warning('Sampling actions from llm for PUCT')
        samples = self._agent.get_llm_plan(state, self._agent.history, self._agent.goal_condition)
        # samples = self.query_llm(task, instruction, observation)
        
        actions_dis = np.zeros(len(valid_actions))
        num_of_steps = []
        is_done = []
        for sample in samples:
            cos_sim = self._find_most_similar(sample[0].lang_command, valid_action_embedding)
            # if "done" in sample[-1] or "Done" in sample[-1]:
            if False:
                is_done.append(True)
            else:
                is_done.append(False)
            num_of_steps.append(len(sample)) 
            # use softmax to get the distribution using cos_sim
            softmax = np.exp(100 * cos_sim) / np.sum(np.exp(100 * cos_sim), axis=0)
            actions_dis += (softmax / len(samples))
        # print(valid_actions)
        return actions_dis, valid_actions, num_of_steps, is_done

    def _find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        cos_scores = cos_scores - np.mean(cos_scores) 
        return cos_scores
    
    def assign_action_prob(self, action_value_pairs, state):
        # valid_actions_list = self._agent._policy_model.validator.get_available_discrete_actions(state)
        valid_actions_list = [i[0] for i in action_value_pairs]
        probs = self._calculate_emperical_prob(state, valid_actions_list)

        action_value_prob_trip = []
        for prob, action_value in zip(probs, action_value_pairs):
            action, q = action_value 
            action_value_prob_trip.append((action, q, prob))


        return action_value_prob_trip

        
    

    def _pucb(self, node: DiscreteNode) -> Action:
        """PUCB
        Selects the action child of the `root` with the best PUCB value.

        Args:
            root (VNode): A vnode to select the action child from.

        Returns:
            Action: Selected action.
        """
        best_action, best_value = None, float('-inf')
        # Assign values
        action_value_pairs:  List[Tuple] = []
        for action in node.children.keys():
            if node[action].num_visits == 0:
                # q = 0
                if self.config['plan_params']['initialize_q'] == 'zero':
                    q = 0
                else:
                    raise NotImplementedError("Initial Q value should be zero or Qvalue")

            else:
                q = node[action].value
            
            action_value_pairs.append((action, q))

        ## NOTE(SJ): This is a bit of a hacky way to implement the PUCB.
        if node.state.action_scores is None:
            prob_list = self.assign_action_prob(action_value_pairs, node.state)
            node.state.action_scores = prob_list

        prob_list = node.state.action_scores

        for action, q, p in prob_list:
            # p: action.probability
            u = self._PUCB_EXPLORATION_CONST * math.sqrt(node.num_visits) / (1+node[action].num_visits)
            
            if node[action].exploration_bonus == 0:
                node[action].exploration_bonus = p*u

            val = q + p*u
                    
            if val > best_value:
                best_action = action
                best_value = val

        return best_action
    


class SayCan(MCTS):
    def __init__(self, agent: Agent, env: Environment, config: st_utils.Dict):
        super().__init__(agent, env, config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translation_lm = SentenceTransformer('stsb-roberta-large').to(self.device)
        self.config["plan_params"]["selection"] = "pucb"
        
    
    def _calculate_emperical_prob(self, state, valid_action_list):
        LAMBDA = 0.5
        valid_actions_lang = [action.lang_command for action in valid_action_list]
        # print(valid_actions_lang)
        valid_action_embedding = self.translation_lm.encode(valid_actions_lang, convert_to_tensor=True, show_progress_bar=False, device=self.device)
        # task = f"{instruction}\nCompleted actions: {', '.join(hist_text)}.\nNext plan:"
        action_dis, valid_action_lists, num_steps, is_done = self.ground_actions_softmax(state, valid_action_list, valid_action_embedding)
        emperical_prob = LAMBDA * action_dis + (1-LAMBDA) /len(valid_action_list)
        return emperical_prob 
    

    def ground_actions_softmax(self, state, valid_actions, valid_action_embedding):
        logger.warning('Sampling actions from llm for empirical policy distribution')
        samples = self._agent.get_llm_plan(state, self._agent.history, self._agent.goal_condition)
        # samples = self.query_llm(task, instruction, observation)
        
        actions_dis = np.zeros(len(valid_actions))
        num_of_steps = []
        is_done = []
        for sample in samples:
            cos_sim = self._find_most_similar(sample[0].lang_command, valid_action_embedding)
            # if "done" in sample[-1] or "Done" in sample[-1]:
            if False:
                is_done.append(True)
            else:
                is_done.append(False)
            num_of_steps.append(len(sample)) 
            # use softmax to get the distribution using cos_sim
            softmax = np.exp(100 * cos_sim) / np.sum(np.exp(100 * cos_sim), axis=0)
            actions_dis += (softmax / len(samples))
        # print(valid_actions)
        return actions_dis, valid_actions, num_of_steps, is_done

    def _find_most_similar(self, query_str, corpus_embedding):
        # helper function for finding similar sentence in a corpus given a query
        query_embedding = self.translation_lm.encode(query_str, convert_to_tensor=True, device=self.device, show_progress_bar=False,)
        # calculate cosine similarity against each candidate sentence in the corpus
        cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
        # retrieve high-ranked index and similarity score
        cos_scores = cos_scores - np.mean(cos_scores) 
        return cos_scores
    
    def assign_action_prob_feasibility(self, action_value_pairs, state):
        valid_actions_list = [i[0] for i in action_value_pairs]
        probs = self._calculate_emperical_prob(state, valid_actions_list)



        actions_with_no_occlusion = self._agent._policy_model.validator.get_available_discrete_actions(state, exclude_occluded_actions = True)

        action_prob_feasibilty_trip = []
        for prob, action_value in zip(probs, action_value_pairs):
            action, q = action_value 
            feasibility = 1 if action in actions_with_no_occlusion else 0
            action_prob_feasibilty_trip.append((action, prob, feasibility))


        return action_prob_feasibilty_trip
    
    def get_next_discrete_action(self, state):
        
        actions = self._agent._policy_model.validator.get_available_discrete_actions(state, exclude_occluded_actions = False)
        action_value_pairs = [(action, 0) for action in actions]

        triplet = self.assign_action_prob_feasibility(action_value_pairs, state)

        best_action = None
        best_score = -100

        for action, prob, feasibility in triplet:
            score = prob * feasibility
            logger.info(f"Action: {action.lang_command}, Score: {score}")
            if score > best_score:
                best_action = action
                best_score = score
        
        return best_action                                                                                                                                                                                                                                                                                                                                         

    
    def plan(self,): 
        '''
        Args:
        Returns:
        '''
        self._agent.tree = self.create_node(root=True)
        self.num_sim_total = 0
        self.num_sim_success = 0
        self.num_sim_failure = 0
        self.sim_trajs = []
        self.llm_called_count = 0
        start_time = time.time()
        
        
        # Sample continuous parameters
        setattr(self._agent.tree, 'reward', 0)

        curr_node: DiscreteNode = self._agent.tree
        curr_state = curr_node.state
        self._agent.imagine_state(curr_state)
        termination = TERMINATION_CONTINUE

        curr_history = deepcopy(self._agent.history)
        
        if len(curr_history) > self._MAX_DEPTH:
            termination = TERMINATION_FAIL

        discrete_action = self.get_next_discrete_action(curr_state)
        # Sample continuous parameters
        trial = 0
        # is_action_feasible = False
        action = None 
        while trial < 3:  
            action = self._agent._policy_model.sample(discrete_action=discrete_action, 
                                            history=curr_history, 
                                            state=curr_state, 
                                            goal=self._agent.goal_condition)
            trial += 1
            if not action.is_feasible():
                continue
            break ## NOTE(SJ): Stop binding continuous action if it is feasible
        if not discrete_action in curr_node:
            next_continuous_node: ContinuousNode = self.create_node(curr_node, parent_action=discrete_action)
            curr_node[discrete_action] = next_continuous_node
        else:
            next_continuous_node: ContinuousNode = curr_node[discrete_action]

        next_state, reward, termination = self._sample_generative_model(curr_state, action, running_escape=True)

        if not action in next_continuous_node:
            next_discrete_node: DiscreteNode = self.create_node(parent_node=next_continuous_node, parent_action=action)
            next_continuous_node[action] = next_discrete_node
        else:
            next_discrete_node = next_continuous_node[action]
           
        return action, time.time() - start_time, self.num_sim_total, self.num_sim_success, self.sim_trajs



class ReAct(MCTS):
    def __init__(self, agent: Agent, env: Environment, config: st_utils.Dict):
        super().__init__(agent, env, config)

        
    
   
    def get_next_discrete_action(self, state):
        samples = self._agent.get_llm_plan(state, self._agent.history, self._agent.goal_condition)
        # samples = self.ground_actions(state)
        best_action = None
        
        try:
            for sample in samples:
                best_action = sample[0]
        except:
            best_action = None

        return best_action                                                                                                                                                                                                                                                                                                                          

    
    def plan(self,): 
        '''
        Args:
        Returns:
        '''
        self._agent.tree = self.create_node(root=True)
        self.num_sim_total = 0
        self.num_sim_success = 0
        self.num_sim_failure = 0
        self.sim_trajs = []
        self.llm_called_count = 0
        start_time = time.time()
        
        
        # Sample continuous parameters
        setattr(self._agent.tree, 'reward', 0)

        curr_node: DiscreteNode = self._agent.tree
        curr_state = curr_node.state
        self._agent.imagine_state(curr_state)

        curr_history = deepcopy(self._agent.history)
        
        discrete_action = None
        
        trial = 0
        while trial < 3:  
            trial +=1
            try:
                discrete_action = self.get_next_discrete_action(curr_state)
                action = self._agent._policy_model.sample(discrete_action=discrete_action, 
                                                history=curr_history, 
                                                state=curr_state, 
                                                goal=self._agent.goal_condition)
                if not action.is_feasible():
                    continue
                break ## NOTE(SJ): Stop binding continuous action if it is feasible
            except:
                continue
        
        if action is None:
            return None, time.time() - start_time, self.num_sim_total, self.num_sim_success, self.sim_trajs
        
        if not action.is_feasible():
            action.discrete_action.llm_data['observation'] = f"{action.discrete_action.lang_command} is not feasible. Try again."

        if not discrete_action in curr_node:
            next_continuous_node: ContinuousNode = self.create_node(curr_node, parent_action=discrete_action)
            curr_node[discrete_action] = next_continuous_node
        else:
            next_continuous_node: ContinuousNode = curr_node[discrete_action]

        # next_state, reward, termination = self._sample_generative_model(curr_state, action, running_escape=True) 

        if not action in next_continuous_node:
            next_discrete_node: DiscreteNode = self.create_node(parent_node=next_continuous_node, parent_action=action)
            next_continuous_node[action] = next_discrete_node
        else:
            next_discrete_node = next_continuous_node[action]
           
        return action, time.time() - start_time, self.num_sim_total, self.num_sim_success, self.sim_trajs



class LLMAssistedMCTS(MCTS):
    def __init__(self, agent: LLMassistedAgent, env: Environment, config: Dict):
        super().__init__(agent, env, config)
        self._agent: LLMassistedAgent

        self.llm_triggers = config['plan_params']['llm_trigger']

        self.single_chunk_threshold = 2
        self.double_chunk_threshold = 2
        self.MCTS_FIRST: bool = config['plan_params']['MCTS_FIRST']


    def register_llm_plan(self, do_reflexion: bool = False):
        '''
        Call LLM to make N escape plans. Register them on the search tree.
        Args:
        Returns:
        '''
        # Sample discrete escape plan
        
        discrete_plans = self._agent.get_llm_plan(self._env.current_state, self._agent.history, self._agent.goal_condition, do_reflexion = do_reflexion)
        
        # Sample continuous parameters
        setattr(self._agent.tree, 'reward', 0)

        for idx, discrete_plan in enumerate(discrete_plans):
            curr_node: DiscreteNode = self._agent.tree
            curr_state = curr_node.state
            self._agent.imagine_state(curr_state)
            termination = TERMINATION_CONTINUE
            logger.info(f"{self.llm_called_count}th llm call | no.{idx} Discrete Plan: {discrete_plan}")

            for action_depth, discrete_action in enumerate(discrete_plan):
                logger.info(f"{self.llm_called_count}th call| no.{idx} plan | {action_depth+1}th Discrete Action: {discrete_action}")
                curr_history = deepcopy(self._agent.history)
                
                if len(curr_history) + action_depth > self._MAX_DEPTH:
                    termination = TERMINATION_FAIL
                    break

                # Sample continuous parameters
                trial = 0
                # is_action_feasible = False
                while trial < 3:  
                    action = self._agent._policy_model.sample(discrete_action=discrete_action, 
                                                    history=curr_history, 
                                                    state=curr_state, 
                                                    goal=self._agent.goal_condition)
                    trial += 1
                    if not action.is_feasible():
                        continue
                    break ## NOTE(SJ): Stop binding continuous action if it is feasible
                
                # Register new continuous node  
                if len(curr_node.children) == 0:
                    self.register_legal_actions(curr_node, curr_history)                
                              
                if not discrete_action in curr_node:
                    next_continuous_node: ContinuousNode = self.create_node(curr_node, parent_action=discrete_action)
                    curr_node[discrete_action] = next_continuous_node
                else:
                    next_continuous_node: ContinuousNode = curr_node[discrete_action]
               
                # Run transition model
                next_state, reward, termination = self._sample_generative_model(curr_state, action, running_escape=True)


                if not action in next_continuous_node:
                    next_discrete_node: DiscreteNode = self.create_node(parent_node=next_continuous_node, parent_action=action)
                    next_continuous_node[action] = next_discrete_node
                else:
                    next_discrete_node = next_continuous_node[action]

                # Update visit counts and Q values.
                # if curr_node.is_root:
                #     curr_node.num_visits += 1
                next_continuous_node.num_visits += 1
                next_discrete_node.num_visits += 1


                self.num_sim_total += 1


                # First log with reward. We will back-up after the loop
                setattr(next_discrete_node, "reward", reward)
    
                # Update nodes
                curr_history += (HistoryEntry(action, next_state, reward, PHASE_SIMULATION),)
                curr_node = next_discrete_node
                curr_node.state = next_state
                curr_state = curr_node.state

                ## Stop following LLM plan if hitting an infeasible action
                if not action.is_feasible():
                    logger.warning(f"Infeasible LLM action: {action.discrete_action}")
                    break


                if termination == TERMINATION_SUCCESS:
                    self.num_sim_success += 1
                    break

            
            # Back up
            # Curr_node is the leaf node at this point. End or Infeasible 
            logger.warning(f"LLM Plan {idx} terminated with {termination}")
            if termination == TERMINATION_CONTINUE:
                self._agent._policy_model.compute_predicates(curr_state, curr_history, self._agent.goal_condition)
                # curr_node.value = curr_node.reward + self._agent.get_value(curr_history, curr_state, action, goal=self._agent.goal_condition)
                curr_node.value = self.evaluate(curr_node.parent.parent, next_state, curr_history, reward, 0, discrete_action, action, verbose=False)
            elif termination == TERMINATION_FAIL:
                curr_node.value = curr_node.reward + 0
            elif termination == TERMINATION_SUCCESS:
                curr_node.value = curr_node.reward + 0 
                setattr(curr_node, "succeeded", True)
            else:
                raise ValueError("Invalid termination condition")

            while not curr_node.is_root:
                # Back up values to the parent Continuous Node
                curr_state = curr_node.state
                curr_node = curr_node.parent
                if termination == TERMINATION_SUCCESS:
                    setattr(curr_node, "succeeded", True)

                values = [node.value for node in curr_node.children.values() if node.num_visits != 0]
                curr_node.value = sum(values)/len(values)

                # Back up values to the parent Discrete Node
                curr_node = curr_node.parent
                if termination == TERMINATION_SUCCESS:
                    setattr(curr_node, "succeeded", True)

                if not hasattr(curr_node, "reward"):
                    curr_reward = self._agent.history[-1].reward
                else:
                    curr_reward = curr_node.reward
                 
                values = [node.value for node in curr_node.children.values() if node.num_visits != 0]
                curr_node.value = curr_reward + self._DISCOUNT_FACTOR * sum(values)/len(values)

            if self._agent._success_plan_found:
                break
            
        return


    def plan(self) -> Tuple[Action, float, int, int, int, List[Dict]]:
        start_time = time.time()
        time_taken = 0.0
        self.num_sim_total = 0
        self.num_sim_success = 0

        if not hasattr(self._agent, "tree"):
            root = self.create_node(root=True)
            self._agent.add_attr("tree", root)

        if self._FOLLOW_SUCCESS_SIMULATION and self._agent._success_plan_found:
            next_action = self.get_next_action()

            time_taken = time.time() - start_time
            return next_action, time_taken, self.num_sim_total, self.num_sim_success, \
                self.sim_trajs

        curr_node: DiscreteNode = self._agent.tree
        curr_state: State = curr_node.state
        curr_history: Tuple[HistoryEntry] = self._agent.history
        goal: Goal = self._agent.goal_condition
        budget = self._NUM_SIMS

        
        # MCTS first
        if self.MCTS_FIRST:
            next_action, _time_taken, _num_sim_total, _num_sim_success, _sim_trajs = super().plan()

        if self.llm_triggers['always_llm_plan']:
            ## NOTE (SJ): Using this path should imply that you are only going to use LLM plan
            logger.info("Shooting from the start")
            self.register_llm_plan()
            next_action = self.get_next_action()
            time_taken = time.time() - start_time

            return next_action, time_taken, self.num_sim_total, self.num_sim_success, self.sim_trajs
        

        
        if self.llm_triggers['after_every_action']:
            self.register_llm_plan()


        if not self.MCTS_FIRST:
            next_action, _time_taken, _num_sim_total, _num_sim_success, _sim_trajs = super().plan()
        time_taken = time.time() - start_time

        return next_action, time_taken, self.num_sim_total, self.num_sim_success, self.sim_trajs



class Reflexion(MCTS):
    def __init__(self, agent: LLMassistedAgent, env: Environment, config: Dict):
        super().__init__(agent, env, config)
        self._agent: LLMassistedAgent

        self.llm_triggers = config['plan_params']['llm_trigger']

        self.single_chunk_threshold = 2
        self.double_chunk_threshold = 2
        self.MCTS_FIRST: bool = config['plan_params']['MCTS_FIRST']
        self.long_memory = deque(maxlen=2)


    def compare_chunks(self, chunk1: List[DiscreteAction], chunk2: List[DiscreteAction]) -> bool:
        
        for i in range(len(chunk1)):
            if chunk1[i] != chunk2[i]:
                return 0
        return 1


    def count_action_chunk(self, chunk: List[DiscreteAction], History: List[DiscreteAction]) -> int:
        '''
        Starting from the most recent history, count the number of consecuatively redundantly appearing action chunks, 
        which are the same as the given chunk.
        '''

        cnt = 0 
        for idx in range(len(History)): 
            if History[idx] == chunk[0]:
                cnt += self.compare_chunks(chunk, History[idx:idx+len(chunk)])


        return cnt


    def detect_redunant_action(self, curr_history) -> bool:
        if len(curr_history) < 4:
            return False
        

        action_history = [entry.action.discrete_action for entry in curr_history[1:]]        
        single_chunk = [curr_history[-1].action.discrete_action]
        if self.count_action_chunk(single_chunk, action_history) >= self.single_chunk_threshold:
            self.single_chunk_threshold += 1
            return True
        
        double_chunk = [curr_history[-2].action.discrete_action, curr_history[-1].action.discrete_action]

        if self.count_action_chunk(double_chunk, action_history) >= self.double_chunk_threshold:
            self.double_chunk_threshold += 1
            return True
        

        return False

    def register_llm_plan(self, ):
        self.long_memory.clear()


        for _ in range(5):
            if self._agent._success_plan_found: 
                break
            discrete_plans = self._agent.get_llm_plan(self._env.current_state, self._agent.history, self._agent.goal_condition, do_reflexion=True, memory= self.long_memory)        
            # Sample continuous parameters
            setattr(self._agent.tree, 'reward', 0)
            
            for idx, discrete_plan in enumerate(discrete_plans):
                curr_node: DiscreteNode = self._agent.tree
                curr_state = curr_node.state
                self._agent.imagine_state(curr_state)
                termination = TERMINATION_CONTINUE
                logger.info(f"{self.llm_called_count}th llm call | no.{idx} Discrete Plan: {discrete_plan}")

                for action_depth, discrete_action in enumerate(discrete_plan):
                    logger.info(f"{self.llm_called_count}th call| no.{idx} plan | {action_depth+1}th Discrete Action: {discrete_action}")
                    curr_history = deepcopy(self._agent.history)
                    
                    if len(curr_history) + action_depth > self._MAX_DEPTH:
                        termination = TERMINATION_FAIL
                        break

                    # Sample continuous parameters
                    trial = 0
                    # is_action_feasible = False
                    while trial < 3:  
                        action = self._agent._policy_model.sample(discrete_action=discrete_action, 
                                                        history=curr_history, 
                                                        state=curr_state, 
                                                        goal=self._agent.goal_condition)
                        trial += 1
                        if not action.is_feasible():
                            continue
                        break ## NOTE(SJ): Stop binding continuous action if it is feasible
                    
                    # Register new continuous node  

                    if len(curr_node.children) == 0:
                        self.register_legal_actions(curr_node, curr_history)                
                                
                    if not discrete_action in curr_node:
                        next_continuous_node: ContinuousNode = self.create_node(curr_node, parent_action=discrete_action)
                        curr_node[discrete_action] = next_continuous_node
                    else:
                        next_continuous_node: ContinuousNode = curr_node[discrete_action]
                
                    next_state, reward, termination = self._sample_generative_model(curr_state, action, running_escape=True)


                    if not action in next_continuous_node:
                        next_discrete_node: DiscreteNode = self.create_node(parent_node=next_continuous_node, parent_action=action)
                        next_continuous_node[action] = next_discrete_node
                    else:
                        next_discrete_node = next_continuous_node[action]

                    # Update visit counts and Q values.
                    # if curr_node.is_root:
                    #     curr_node.num_visits += 1
                    next_continuous_node.num_visits += 1
                    next_discrete_node.num_visits += 1


                    self.num_sim_total += 1


                    # First log with reward. We will back-up after the loop
                    setattr(next_discrete_node, "reward", reward)
        
                    # Update nodes
                    curr_history += (HistoryEntry(action, next_state, reward, PHASE_SIMULATION),)
                    curr_node = next_discrete_node
                    curr_node.state = next_state
                    curr_state = curr_node.state

                    # ## Stop following LLM plan if hitting an infeasible action
                    # if not action.is_feasible():
                    #     logger.warning(f"Infeasible LLM action: {action.discrete_action}")
                    #     if action_depth == 0: 
                    #         action.discrete_action.llm_data['observation'] = f"{action.discrete_action.lang_command} is not feasible."
                    #         self.long_memory.appendleft(
                    #             [{'role': 'assistant', 'content':action.discrete_action.llm_data['response']},
                    #             {'role': 'user', 'content':f"{action.discrete_action.lang_command} is infeasible."}])
                    #     break
                    # else: 
                    #     if action_depth == 0:
                    #         self.long_memory.clear()

                    ## Stop following LLM plan if hitting an infeasible action
                    if not action.is_feasible():
                        logger.warning(f"Infeasible LLM action: {action.discrete_action}")
                        action.discrete_action.llm_data['observation'] = f"{action.discrete_action.lang_command} is not feasible."
                        self.long_memory.appendleft(
                            [{'role': 'assistant', 'content':action.discrete_action.llm_data['response']},
                            {'role': 'user', 'content':f"{action.discrete_action.lang_command} is infeasible."}])
                        break
                    else: 
                        self.long_memory.clear()
                    


                    if termination == TERMINATION_SUCCESS:
                        self.num_sim_success += 1
                        break

                
                # Back up
                # Curr_node is the leaf node at this point. End or Infeasible 
                logger.warning(f"LLM Plan {idx} terminated with {termination}")
                if termination == TERMINATION_CONTINUE:
                    pass
                elif termination == TERMINATION_FAIL:
                    pass

                elif termination == TERMINATION_SUCCESS:
                    setattr(curr_node, "succeeded", True)
                    pass
                else:
                    raise ValueError("Invalid termination condition")

                while not curr_node.is_root:
                    # Back up values to the parent Continuous Node
                    curr_state = curr_node.state
                    curr_node = curr_node.parent
                    if termination == TERMINATION_SUCCESS:
                        setattr(curr_node, "succeeded", True)


                    # Back up values to the parent Discrete Node
                    curr_node = curr_node.parent
                    if termination == TERMINATION_SUCCESS:
                        setattr(curr_node, "succeeded", True)

                if self._agent._success_plan_found:
                    break
            
        return
       


    def plan(self) -> Tuple[Action, float, int, int, int, List[Dict]]:
        start_time = time.time()
        time_taken = 0.0
        self.num_sim_total = 0
        self.num_sim_success = 0

        if not hasattr(self._agent, "tree"):
            root = self.create_node(root=True)
            self._agent.add_attr("tree", root)

        if self._FOLLOW_SUCCESS_SIMULATION and self._agent._success_plan_found:
            next_action = self.get_next_action()

            time_taken = time.time() - start_time
            return next_action, time_taken, self.num_sim_total, self.num_sim_success, \
                self.sim_trajs

        curr_node: DiscreteNode = self._agent.tree
        curr_state: State = curr_node.state
        curr_history: Tuple[HistoryEntry] = self._agent.history
        goal: Goal = self._agent.goal_condition
        budget = self._NUM_SIMS

            

        # MCTS first
        if self.MCTS_FIRST:
            next_action, _time_taken, _num_sim_total, _num_sim_success, _sim_trajs = super().plan()

        ## NOTE (SJ): Using this path should imply that you are only going to use LLM plan and not the LMP 

        self.register_llm_plan()

        next_action = self.get_next_action()
        
        ## NOTE (dlee): for debugging only.
        # curr_node.state = self._env.current_state
        # curr_state = self._env.current_state
        # self._agent.imagine_state(curr_state)
        # if len(self._agent.history) == 2:
        #     self._agent._policy_model.compute_predicates(curr_state, curr_history, self._agent.goal_condition)
        #     op = [i for i in self._agent._policy_model.validator.get_all_discrete_actions(curr_state) if i.aimed_obj.name == "beef_grill" and i.type == "PLACE" and i.direction == "on" and i.region == "counter1"][0]
        #     next_action = self._agent._policy_model.sample(op, history=self._agent.history, state=curr_state, goal=self._agent.goal_condition)
        #     self._agent.tree[op] = self.create_node(curr_node)
        #     self._agent.tree[op][next_action] = self.create_node(self._agent.tree[op], op)

        time_taken = time.time() - start_time

        return next_action, time_taken, self.num_sim_total, self.num_sim_success, self.sim_trajs



       