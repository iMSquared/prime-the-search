"""
For solving POMDP problem using online planner with given policy, this API contain basic classes & functions of planner.
"""
import random

from abc import ABC
from typing import Union, Generic, Tuple, Dict

from Simulation.pybullet.mdp.MDP_framework import Action, Observation, State, DiscreteAction, HistoryEntry


class TreeNode(ABC):
    def __init__(self, parent: "TreeNode"=None, is_root: bool=False):
        self.parent: TreeNode = None
        self.children: Dict[Union[Action, Observation], TreeNode] = {}
        self.is_root: bool = is_root

    def __getitem__(self, key: Union[Action, Observation]) -> "TreeNode":
        if key not in self.children and type(key) == int:
            clist = list(self.children)
            if key >= 0 and key < len(clist):
                return self.children[clist[key]]
            else:
                return None
        return self.children.get(key, None)

    def __setitem__(self, key: Union[Action, Observation], 
                          value: "TreeNode"):
        self.children[key] = value

    def __contains__(self, key: Union[Action, Observation]):
        return key in self.children
    


class DiscreteNode(TreeNode):
    def __init__(self, state: State, num_visits: int, value: float, parent: TreeNode=None, is_root:bool=False):
        super().__init__(is_root=is_root)
        self.state = state
        self.num_visits = num_visits
        self.value = value
        self.children: Dict[DiscreteAction, ContinuousNode]
        self.parent: ContinuousNode = parent

    def __getitem__(self, key: DiscreteAction) -> "ContinuousNode":
        return super().__getitem__(key)
    
    def __str__(self):
        return "DiscreteNode(#visit: %d, val: %.3f | keys: %s)" % (self.num_visits, self.value, str(self.children.keys()))

    def __repr__(self):
        return self.__str__()
    
    def deprecated_argmax(self):
        """
        Returns the action of the child with highest value
        """
        best_value = float("-inf")
        best_action = None
        ops = list(self.children.keys())
        random.shuffle(ops)
        for op in ops:
            childrens = list(self.children[op].children.keys())
            random.shuffle(childrens)
            for action in childrens:
                if self.children[op][action].value > best_value:
                    best_action = action
                    best_value = self.children[op][action].value  
                    # best_bullet = self.children[op][action].state.bullet_filename
        # print("best_action: ", best_bullet)
        return best_action
    
    def argmax(self):
        """
        Returns the action of the child with highest value
        """
        best_value = float("-inf")
        best_op = None
        ops = list(self.children.keys())
        random.shuffle(ops)
        for op in ops:
            if self.children[op].num_visits > 0 and self.children[op].value > best_value:
                best_op = op
                best_value = self.children[op].value
        try:
            actions = list(self.children[best_op].children.keys())
            best_value = float("-inf")
            best_action = None
            random.shuffle(actions)
            for action in actions:
                if self.children[best_op][action].num_visits > 0 and self.children[op].value > best_value:
                    best_action = action
                    best_value = self.children[best_op][action].value  
        except Exception as e:
            print("Error: ", e)
            best_action = None

        return best_action
    
    def sample(self):
        """
        Returns an action w.r.t #visit
        """
        actions = list(self.children.keys())
        p_actions = []
        for action in actions:
            p_actions.append(self.children[action].num_visits / self.num_visits)
        return random.choices(actions, p_actions)
    

class ContinuousNode(TreeNode):
    def __init__(self, state: State, num_visits: int, value: float, discrete_action: DiscreteAction, parent: TreeNode=None, is_root:bool=False):
        super().__init__(is_root=is_root)
        self.state = state
        self.num_visits = num_visits
        self.value = value
        self.discrete_action = discrete_action
        self.children: Dict[Action, DiscreteNode]
        self.parent: DiscreteNode = parent
        self.exploration_bonus = 0

    def __getitem__(self, key: Action) -> "DiscreteNode":
        return super().__getitem__(key)
    
    def __str__(self):
        return "ContinuousNode(#visit: %d, val: %.3f | keys: %s)" % (self.num_visits, self.value, str(self.discrete_action))

    def __repr__(self):
        return self.__str__()
    
    def argmax(self):
        """
        Returns the action of the child with highest value
        """
        best_value = float("-inf")
        best_action = None
        for action in self.children:
            if self.children[action].value > best_value:
                best_action = action
                best_value = self.children[action].value  
        return best_action
    
    def sample(self):
        """
        Returns an action w.r.t #visit
        """
        actions = list(self.children.keys())
        p_actions = []
        for action in actions:
            p_actions.append(self.children[action].num_visits / self.num_visits)
        return random.choices(actions, p_actions)


class QNode(TreeNode):
    def __init__(self, num_visits: int, 
                       value: float):
        super().__init__()
        self.num_visits = num_visits
        self.value = value
        self.children: Dict[Observation, "VNode"]  # o -> VNode
    
    def __getitem__(self, key: Observation) -> "VNode":
        item: VNode = super().__getitem__(key)
        return item

    def __str__(self):
        return "QNode(%d, %.3f | %s)" % (self.num_visits, self.value, str(self.children.keys()))

    def __repr__(self):
        return self.__str__()


class VNode(TreeNode):
    def __init__(self, num_visits: int, 
                       value: float):
        super().__init__()
        self.num_visits = num_visits
        self.value = value
        self.children: Dict[Action, "QNode"]       # a -> QNode

    def __getitem__(self, key: Action) -> "QNode":
        item: QNode = super().__getitem__(key)
        return item

    def __str__(self):
        return "VNode(%d, %.3f | %s)" % (self.num_visits, self.value, str(self.children.keys()))

    def __repr__(self):
        return self.__str__()

    def print_children_value(self):
        for action in self.children:
            print("   action %s: %.3f" % (str(action), self.children[action].value))
    
    def argmax(self):
        """
        Returns the action of the child with highest value
        """
        best_value = float("-inf")
        best_action = None
        for action in self.children:
            if self.children[action].value > best_value:
                best_action = action
                best_value = self.children[action].value  
        return best_action
    
    def sample(self):
        """
        Returns an action w.r.t #visit
        """
        actions = list(self.children.keys())
        p_actions = []
        for action in actions:
            p_actions.append(self.children[action].num_visits / self.num_visits)
        return random.choices(actions, p_actions)

        
class RootVNode(VNode):
    def __init__(self, num_visits: int, 
                       value: float, 
                       history: Tuple[HistoryEntry]):
        super().__init__(num_visits, value)
        self.history = history
    
    @classmethod
    def from_vnode(cls, vnode: VNode, 
                        history: Tuple[HistoryEntry]):
        """from_vnode(cls, vnode, history)"""
        rootnode = RootVNode(vnode.num_visits, vnode.value, history)
        rootnode.children = vnode.children
        return rootnode


class Planner:
    """
    A Planner can `plan` the next action to take for a given POMDP problem (online planning).
    The planner can be updated (which may also update the agent belief) once an action is executed and observation received.
    """

    def plan(self, **kwargs):    
        """
        The agent carries the information: Belief, history, BlackboxModel, and policy, necessary for planning
        Return:
            next_action: Action, *rest
        """
        raise NotImplementedError

    def update(self, real_action: Action, real_observation: Observation, reward: float):
        """
        Updates the planner based on real action and observation.
        Updates the agent accordingly if necessary.
        Args:
            real_action: Action
            real_observation: Observation
        """
        raise NotImplementedError