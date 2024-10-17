import copy
import random
import numpy as np
import numpy.typing as npt
from typing import List


from Simulation.pybullet.envs.shop_env import ShopEnv, ShopObjectEntry
from Simulation.pybullet.envs.robot import PR2, PR2
from Simulation.pybullet.envs.manipulation import PR2Manipulation, PR2SingleArmManipulation
from Simulation.pybullet.envs.navigation import Navigation
from Simulation.pybullet.mdp.MDP_framework import *
from Simulation.pybullet.mdp.shop.shop_MDP import *
from Simulation.pybullet.mdp.shop.policy.default_samplers import PickSampler, RandomPlaceSampler, OpenDoorSampler, CloseDoorSampler



class ShopRolloutPolicyModel(RolloutPolicyModel):

    def __init__(self, bc: BulletClient, 
                       env: ShopEnv, 
                       robot: PR2, 
                       manip: PR2Manipulation,
                       nav: Navigation,
                       config: Dict):
        """Initialize a random rollout policy model."""

        # configs
        self.config = config
        self.NUM_FILTER_TRIALS_PICK       : int = config["pose_sampler_params"]["num_filter_trials_pick"]
        self.NUM_FILTER_TRIALS_PLACE      : int = config["pose_sampler_params"]["num_filter_trials_place"]
        self.NUM_FILTER_TRIALS_FORCE_FETCH: int = config["pose_sampler_params"]["num_filter_trials_force_fetch"]
        self.NUM_FILTER_TRIALS_STATE_DESCRIPTION: int = config["pose_sampler_params"]["num_filter_trials_state_description"]
        self.DEBUG_GET_DATA   : bool = config['project_params']['debug']['get_data']
        self.SHOW_GUI         : bool = config['project_params']['debug']['show_gui']
        self.collect_data     : bool = config["project_params"]["overridable"]["collect_data"]
        self.policy           : str = config["project_params"]["overridable"]["policy"]
        # Bullet
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav

        # Samplers
        self.pick_action_sampler = PickSampler(self.NUM_FILTER_TRIALS_PICK)
        self.place_action_sampler = RandomPlaceSampler(self.NUM_FILTER_TRIALS_PLACE,
                                                       self.NUM_FILTER_TRIALS_FORCE_FETCH)
        self.open_action_sampler = OpenDoorSampler(self.NUM_FILTER_TRIALS_PICK)
        self.close_action_sampler = CloseDoorSampler(self.NUM_FILTER_TRIALS_PICK)
        

    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: ShopEnv, 
                                   robot: PR2,
                                   manip: PR2Manipulation,
                                   nav: Navigation):
        """Re-initalize a random policy model with new BulletClient.
        Be sure to pass PR2Manipulation instance together.
        Args:
            bc (BulletClient): New bullet client
            env (ShopEnv): New simulation environment
            robot (PR2): New robot instance
            manip (PR2Manipulation): PR2Manipulation instance of the current client.
        """
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        self.nav = nav


    def get_available_discrete_actions(self, state: ShopState) -> List[ShopDiscreteAction]:
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
        discrete_actions = self.get_available_discrete_actions(state)
        op = random.choice(discrete_actions)

        return op



    def sample(self, discrete_action: Union[ShopDiscreteAction, None],
                     history: Tuple[HistoryEntry],
                     state: ShopState, 
                     goal: ShopGoal) -> ShopContinuousAction:
        """Random rollout policy model!
        Args:
            init_observation (ShopObservation)
            history (Tuple[HistoryEntry])
            state (ShopState): Current state to start rollout
            goal (ShopGoal): Goal passed from the agent.
        Returns:
            ShopContinuousAction: Sampled next action
        """

        # Select primitive action.
        if discrete_action is None:
            discrete_action = self.sample_discrete_action(state, history, goal)

        # PICK action
        if discrete_action.type == ACTION_PICK:
            next_action = self.pick_action_sampler(self.bc, self.env, self.robot, self.nav, 
                                                   self.manip, state, discrete_action)
        # PLACE action
        elif discrete_action.type == ACTION_PLACE:
            next_action = self.place_action_sampler(self.bc, self.env, self.robot, self.nav, 
                                                    self.manip, state, discrete_action)
        # OPEN action
        elif discrete_action.type == ACTION_OPEN:
            next_action = self.open_action_sampler(self.bc, self.env, self.robot, self.nav, 
                                                    self.manip, state, discrete_action)
        # CLOSE action
        elif discrete_action.type == ACTION_CLOSE:
            next_action = self.close_action_sampler(self.bc, self.env, self.robot, self.nav, 
                                                    self.manip, state, discrete_action)
        else:
            raise ValueError("[Action Sampling Error] Wrong action type")

        return next_action


