""" 
For defining POMDP problem, this API contain basic classes & functions of POMDP.

[Terminology]
S: state space
A: action space
O: observation space
T: Transition model
Z: observation model
R: reward model

Adapt from the reference: https://github.com/h2r/pomdp-py/tree/master/pomdp_py/framework
"""

from abc import ABC, abstractmethod
import random
from typing import Any, Callable, Dict, Tuple, NamedTuple, Iterator, TypeVar, List, Union
from dataclasses import dataclass
from copy import deepcopy


TerminationT = TypeVar("TerminationT", bound=str)
TERMINATION_FAIL    : TerminationT = "fail"
TERMINATION_SUCCESS : TerminationT = "success"
TERMINATION_CONTINUE: TerminationT = "continue"
PhaseT = TypeVar("PhaseT", bound=str)
PHASE_EXECUTION : PhaseT = "execution"
PHASE_SIMULATION: PhaseT = "simulation"
PHASE_ROLLOUT   : PhaseT = "rollout"


class State(ABC):
    """
    The State Class. State must be `hashable`.
    """

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError


class Goal(ABC):
    """
    The Goal class. Goal must be `hashable`
    """
    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError
    
class Action(ABC):

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError
    
    @abstractmethod
    def is_feasible(self) -> bool:
        raise NotImplementedError

class DiscreteAction(Action):
    """
    The DiscreteAction class. DiscreteAction must be `hashable`.
    """
    def __init__(self):
        self.probability: float = -1.0
        self.q_value: float = 0.0 
        self.feasible = None
        self.type = None

class ContinuousAction(Action):
    def __init__(self, discrete_action: DiscreteAction):
        self.discrete_action = discrete_action
        self.feasible = None


class Observation(ABC):
    """
    The Observation class. Observation must be `hashable`.
    """

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError



@dataclass(eq=True, frozen=True)
class HistoryEntry:
    """
    A history entry which records the history of (a, o, r) sequence.
    @dataclass(frozen=True) will define the __eq__() and __hash__().
    """
    action: ContinuousAction # Thought
    # observation: Observation
    state: State
    reward: float
    phase: PhaseT


class TransitionModel(ABC):
    """
    T(s,a,s') = Pr(s'|s,a)
    """

    @abstractmethod
    def probability(self, next_state: State, cur_state: State, action: ContinuousAction, *args, **kwargs):
        """
        Args:
            next_state(s'): State
            cur_state(s): State
            action(a): ContinuousAction
        Returns:
            Pr(s'|s,a)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: State, action: ContinuousAction, *args, **kwargs):
        """
        Returns next state randomly sampled according to the distribution of this transition model.
        
        Args:
            state(s): State
            action(a): ContinuousAction
        Returns:
            next_state(s'): State, ...
        """
        raise NotImplementedError


class ObservationModel(ABC):
    """
    Z(s',a,o) = Pr(o|s',a). This means perception model, not sensor mesurement.
    """

    @abstractmethod
    def probability(self, observation: Observation, next_state: State, action: ContinuousAction, *args, **kwargs):
        """
        Args:
            observation(o): Observation
            next_state(s'): State
            action(a): Action
        Returns:
            Pr(o|s',a)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, next_state: State, action: ContinuousAction, *args, **kwargs):
        """sample(self, next_state, action, **kwargs)
        Returns observation randomly sampled according to the distribution of this observation model.

        Args:
            next_state(s'): State
            action(a): Action
        Returns:
            observation(o): Observation, ...
        """
        raise NotImplementedError

    @abstractmethod
    def get_sensor_observation(self, next_state: State, action: ContinuousAction, *args, **kwargs):
        """
        Returns an sensor_observation
        """
        raise NotImplementedError


class RewardModel(ABC):
    """
    R(s,a,s') = Pr(r|s',a)
    """

    @abstractmethod
    def probability(self, reward: float, state: State, action: ContinuousAction, next_state: State, *args, **kwargs):
        """
        Args:
            reward(r): float
            next_state(s'): State
            action(a): Action
        Returns:
            Pr(r|s',a)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, state: State, action: ContinuousAction, next_state: State, *args, **kwargs):
        """
        Returns reward randomly sampled according to the distribution of this reward model.
        Args:
            state(s) <- basics.State
            action(a) <- basics.Action
            next_state(s') <- basics.State
        Returns:
            reword(r): float, termination
        """
        raise NotImplementedError

    @abstractmethod
    def _check_termination(self, state: State) -> bool:
        """
        Check the state satisfy terminate condition
        """
        raise NotImplementedError


class BlackboxModel:
    """
    A BlackboxModel is the generative distribution G(s,a)
    which can generate samples where each is a tuple (s',o,r) according to T, Z, and R.
    (These models can be different to models of environment.)
    """
    def __init__(self,
        transition_model: TransitionModel,
        reward_model: RewardModel):

        self._transition_model = transition_model
        self._reward_model = reward_model

    def sample(self, state: State, action: ContinuousAction, running_escape: bool = False, **kwargs):
        """
        Sample (s',o,r) ~ G(s',a)
        """
        # |NOTE(Jiyong)|: how to deal rest information of models
        next_state = self._transition_model.sample(state, action, running_escape, **kwargs)
        reward, termination = self._reward_model.sample(next_state, action, state, **kwargs)

        return next_state, reward, termination
    
    @property
    def transition_model(self):
        return self._transition_model
    
    @property
    def reward_model(self):
        return self._reward_model


class PolicyModel(ABC):
    """
    \pi(a|h)
    """
    def __init__(self, *args, **kwargs):
        pass


    @abstractmethod
    def sample_discrete_action(self, state: State, history: Tuple[HistoryEntry], goal: Goal) -> DiscreteAction:
        """
        Returns discrete_action.
        Args:
            state (State): Current state of a particle.
            history (Tuple): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
            goal (goal): Goal condition
        Returns:
            action (DiscreteAction): Discrete action
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, discrete_action: Union[DiscreteAction, None], history: Tuple[HistoryEntry], state: State, goal: object, *args, **kwargs) -> ContinuousAction:
        """
        Samples continuous paramter for the discrete action
        Args:
            discrete_action (DiscreteAction): discrete action
            history (Tuple): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
            state (State): Current state of a particle.
            goal (goal): Goal condition
        Returns:
            action (ContinuousAction): ContinuousAction
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_available_discrete_actions(self, state: State, *args, **kwargs) -> List[DiscreteAction]:
        """
        Returns currently available discrete_actions
        Args:
            history (Tuple): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
            state (State): Current state of a particle.
            goal (goal): Goal condition
        Returns:
            discrete_actions (List): DiscreteAction
        """
        raise NotImplementedError



class RolloutPolicyModel(ABC):

    @abstractmethod
    def sample(self, discrete_action: Union[DiscreteAction, None], history: Tuple[HistoryEntry], state: State, goal: object, *args, **kwargs) -> ContinuousAction:
        """
        Returns action randomly sampled according to the policy.
        Args:
            init_observation (Observation): Initial observation
            history (Tuple): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
            state (State): Current state of a particle.
            goal (goal): Goal condition
        Returns:
            action (Action): Action
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_available_discrete_actions(self, state: State, *args, **kwargs) -> List[DiscreteAction]:
        """
        Returns currently available discrete_actions
        Args:
            history (Tuple): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
            state (State): Current state of a particle.
            goal (goal): Goal condition
        Returns:
            discrete_actions (List): DiscreteAction
        """
        raise NotImplementedError


class ValueModel(ABC):

    def __init__(self, is_v_model: bool):
        """Value model can either be V model or Q model

        Args:
            is_v_model (bool): Mark this as True if the model predicts the value model.
        """
        self.is_v_model = is_v_model
        self.validator = None

    @abstractmethod
    def sample(self, *args, **kwargs):
        """
        Returns value of the node using value network or heuristic function.
        Args:
            init_observation (Observation): Initial observation
            history(h): Tuple of sequece of observations, and actions (o_0, a_1, o_1, (r_1,) a_2, o_2, (r_2,) ...)
        Returns:
            value: float
        """
        raise NotImplementedError


class Agent(ABC):
    """
    An Agent operates in an environment by giving actions, receiving observations and rewards, and updating its belief.
    """
    def __init__(self, blackbox_model: BlackboxModel,
                       policy_model: PolicyModel,
                       rollout_policy_model: RolloutPolicyModel,
                       value_model: ValueModel = None,
                       goal_condition = None):

        self._blackbox_model = blackbox_model
        self._policy_model = policy_model
        self._rollout_policy_model = rollout_policy_model
        self._value_model = value_model
        self._goal_condition = goal_condition 
        self._success_plan_found = False

        # For online planning
        self._history: Tuple[HistoryEntry] = ()

    # getter & setter
    @property
    def blackbox_model(self):
        return self._blackbox_model

    @property
    def history(self):
        """
        Current history form of ((a_1, o_1, r_1), (a_2, o_2, r_2), ...)
        """
        return self._history

    # |NOTE(Jiyong)|: This is for real history after execution, not for histories during planning. For histories during planning, use history of nodes
    def _update_history(self, real_action: ContinuousAction, real_state: State, real_reward: float):
        self._history += (HistoryEntry(real_action, real_state, real_reward, PHASE_EXECUTION),)

    @property
    def init_belief(self):
        """
        Initial belief distribution
        """
        return self._init_belief

    @property
    def goal_condition(self):
        """Goal condition"""
        return self._goal_condition

    def get_action(self, discrete_action: DiscreteAction, history: Tuple, state:State, goal = None, rollout: bool= True):
        if rollout:
            return self._rollout_policy_model.sample(discrete_action, history, state, goal)
        else:
            return self._policy_model.sample(discrete_action, history, state, goal)
        
    def get_discrete_action(self, state: State, history: Tuple, goal = None, rollout: bool = True) -> DiscreteAction:
        if rollout:
            return self._rollout_policy_model.sample_discrete_action(state, history, goal)
        else:
            return self._policy_model.sample_discrete_action(state, history, goal)
        
    def get_all_discrete_actions(self, state: State, history: Tuple, goal = None, rollout: bool = True) -> List[DiscreteAction]:
        if rollout:
            return self._rollout_policy_model.get_available_discrete_actions(state, history=history, goal=goal)
        else:
            return self._policy_model.get_available_discrete_actions(state, history=history, goal=goal)
        
    def get_available_discrete_actions(self, state:State, rollout: bool = True):
        if rollout:
            return self._rollout_policy_model.get_available_discrete_actions(state)
        else:
            return self._policy_model.get_available_discrete_actions(state)

    def get_value(self, history: Tuple, state:State, action: Action = None, goal = None):
        if action is None:
            return self._value_model.sample(history, state, goal, action=action)
        else:
            return self._value_model.sample(history, state, goal, action)

    def imagine_state(self, state: State):
        self._set_simulation(state)

    @abstractmethod
    def _set_simulation(self, state: State):
        """
        Set simulation environment with given state.
        It is used to imagine state during planning.
        User should implement according to the simulator they want to use.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, real_action: ContinuousAction, real_state: State, real_reward: float):
        """
        updates the history and performs belief update
        """
        raise NotImplementedError

    def add_attr(self, attr_name: str, attr_value: Any):
        """
        TODO(ssh): Deprecate all add_attr in this project.
        add_attr(self, attr_name, attr_value)
        A function that allows adding attributes to the agent.
        Sometimes useful for planners to store agent-specific information.
        """
        if hasattr(self, attr_name):
            raise ValueError("attributes %s already exists for agent." % attr_name)
        else:
            setattr(self, attr_name, attr_value)

    def detect_exogenous(self, real_action: Action, real_state: State) -> bool:
        """
        Detect whether simulation diverged from the real due to exogenous event.
        """
        raise NotImplementedError


class Environment(ABC):
    """
    An Environment gives an observation and an reward to the agent after receiving an action
    while maintains the true state of the world.
    """
    def __init__(self, exec_transition_model: TransitionModel,
                       exec_reward_model: RewardModel,
                       initial_state: Union[State, None] = None):
        """Simple constructor
        Init with initial_state as current_state."""
        self.exec_transition_model = exec_transition_model
        self.exec_reward_model = exec_reward_model
        self.current_state = initial_state


    def set_current_state(self, state: State):
        """Set current_state of the POMDP environment.

        Args:
            state (State): State instance.
        """
        self.current_state = state

        

    def execute(self, action: ContinuousAction, **kwargs):
        """
        Execute action, apply state transition given `action` and return sensor observation, reward, and is_goal.
        (should not return next_state, due to the agent can't know about true state)

        Args:
            action (ContinuousAction): Action that triggers the state transition.
        Returns:
            observation (Observation): Observation instance
            reward (float): Reward scalar value
            termination (bool): True when success, False when fail and None when not ended.
        """

        # Sampling next_state and reward
        next_state = self.exec_transition_model.sample(self.current_state, action, **kwargs)
        reward, termination = self.exec_reward_model.sample(next_state, action, self.current_state, **kwargs)

        # Apply state transitioin
        self.current_state = next_state

        return reward, termination


class MDP:
    """
    A MDP instance <- agent(class:'Agent'), env(class:'Enviroment')

    Because MDP modelling may be different according to the agent and the environment, 
    it does not make much sense to have the MDP class to hold this information.
    Instead, Agent should have its own belief, policy and the Environment should have its own T, Z, R.
    """

    def __init__(self, agent: Agent,
                       env: Environment):
        self.agent = agent
        self.env = env