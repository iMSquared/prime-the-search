import torch
from typing import List, Union, Dict, Tuple
from enum import Enum

# Simulations
from Simulation.pybullet.envs.shop_env import ShopEnv
from Simulation.pybullet.mdp.MDP_framework import HistoryEntry
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient

from Simulation.pybullet.envs.manipulation import PR2Manipulation, PR2SingleArmManipulation, imagine
from Simulation.pybullet.mdp.MDP_framework import *
from Simulation.pybullet.mdp.shop.shop_MDP import *
from Simulation.pybullet.predicate.predicate_shop import *
from Simulation.pybullet.mdp.shop.policy.default_samplers import *


class Scripter():

    def __init__(self, env: ShopEnv,
                 plan: List[Tuple[ShopDiscreteAction, str, float]]=None, 
                 scenario: int=None):

        self.env = env
        self.scenario = scenario
        
        if plan is None:
            plan = []

        if self.scenario is not None:
            if isinstance(scenario, str) and 'v' in self.scenario:
                self.scenario = int(self.scenario[1:])
                plan = compose_variant_scenario(self.env, self.scenario)
            else:
                plan = compose_scenario(self.env, self.scenario)

        self.plan = plan
        self.counter = 0

    def has_plan(self):
        return len(self.plan) > 0
    
    def get_next_action(self, history: Tuple[HistoryEntry]):

        # First action
        if len(history) == 1:
            return self.plan[self.counter]
        
        if history[-1].action.is_feasible():
            self.counter += 1
        
        return self.plan[self.counter]



def compose_scenario(env: ShopEnv, scenario: int):
    plan = []
    # Scenario 0
        
    if scenario == 0:
        # plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle"]),
        #             "PICK bottle",
        #             {"next_action": {"0": 1}}))

        # plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle"], "table1", "on", env.all_obj["table1"]),
        #             "PLACE bottle on table1",
        #             {"next_action": {"0": 1}}))
        
        # plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter"]),
        #             "PICK salter",
        #             {"next_action": {"0": 1}}))

        # plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter"], "table1", "on", env.all_obj["table1"]),
        #             "PLACE salter on table1",
        #             {"next_action": {"0": 1}}))
        
        return plan
    
        
    # Scenario 1
    if scenario == 1:
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle"]),
                                    "PICK bottle",
                                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle"], "table2", "left_of", env.all_obj["plate"]),
                    "PLACE bottle left_of plate",
                    {"next_action": {"0": 1}}))

        
        return plan
    
    # Scenario 2
    if scenario == 2:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                                    "OPEN kitchen_door",
                                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                                    "PICK bottle2",
                                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "table1", "left_of", env.all_obj["bottle7"]),
                    "PLACE bottle2 left_of bottle7",
                    {"next_action": {"0": 1}}))
        
        return plan
    
    # Scenario 3
    if scenario == 3:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "table1", "on", env.all_obj["table1"]),
                    "PLACE salter1 on table1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "table2", "on", env.all_obj["table2"]),
                    "PLACE bottle1 on table2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE bottle2 on counter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE salter2 on counter1",
                    {"next_action": {"0": 1}}))
        
        return plan
    
    # Scenario 4
    if scenario == 4:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter"]),
                    "PICK salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter"], "counter2", "left_of", env.all_obj["bottle2"]),
                    "PLACE salter left_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter2", "right_of", env.all_obj["bottle2"]),
                    "PLACE bottle1 right_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "sink_counter_left", "on", env.all_obj["sink_counter_left"]),
                    "PLACE bottle2 on sink_counter_left",
                    {"next_action": {"0": 1}}))
        
        return plan
        
    # Scenario 5
    if scenario == 5:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter2", "left_of", env.all_obj["bottle2"]),
                    "PLACE bottle1 left_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "table1", "on", env.all_obj["table1"]),
                    "PLACE bottle2 on table1",
                    {"next_action": {"0": 1}}))
        
        return plan

    # Scenario 6    
    if scenario == 6:

        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "table1", "left_of", env.all_obj["salter1"]),
                    "PLACE bottle1 left_of salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "table2", "left_of", env.all_obj["bottle2"]),
                    "PLACE salter1 left_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "sink_counter_left", "on", env.all_obj["sink_counter_left"]),
                    "PLACE salter2 on sink_counter_left",
                    {"next_action": {"0": 1}}))
        
        return plan

    # Scenario 7
    if scenario == 7:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "table1", "left_of", env.all_obj["plate"]),
                    "PLACE bottle1 left_of plate",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["plate"]),
                    "PICK plate",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["plate"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE plate on counter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter2", "left_of", env.all_obj["salter"]),
                    "PLACE bottle2 left_of salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle3"]),
                    "PICK bottle3",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle3"], "counter2", "right_of", env.all_obj["salter"]),
                    "PLACE bottle3 right_of salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter"]),
                    "PICK salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter"], "table1", "right_of", env.all_obj["bottle1"]),
                    "PLACE salter right_of bottle1",
                    {"next_action": {"0": 1}}))
        
        return plan

    # Scenario 8
    if scenario == 8:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "counter2", "left_of", env.all_obj["bottle2"]),
                    "PLACE salter1 left_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter2", "right_of", env.all_obj["bottle2"]),
                    "PLACE bottle1 right_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "counter1", "left_of", env.all_obj["bottle4"]),
                    "PLACE salter2 left_of bottle4",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle3"]),
                    "PICK bottle3",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle3"], "counter1", "right_of", env.all_obj["bottle4"]),
                    "PLACE bottle3 right_of bottle4",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle4"]),
                    "PICK bottle4",
                    {"next_action": {"0": 1}}))
        
        
        
        return plan
    
    # Scenario 9
    if scenario == 9:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter"]),
                    "PICK salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter"], "counter1", "left_of", env.all_obj["bottle2"]),
                    "PLACE salter left_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter1", "right_of", env.all_obj["bottle2"]),
                    "PLACE bottle1 right_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter2", "on", env.all_obj["counter2"]),
                    "PLACE bottle2 on counter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["plate"]),
                    "PICK plate",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["plate"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE plate on counter1",
                    {"next_action": {"0": 1}}))
        
        return plan
    
    # Scenario 10
    if scenario == 10:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["plate1"]),
                    "PICK plate1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["plate1"], "table1", "on", env.all_obj["table1"]),
                    "PLACE plate1 on table1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "counter2", "on", env.all_obj["counter2"]),
                    "PLACE salter1 on counter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "shelf_lower", "on", env.all_obj["shelf_lower"]),
                    "PLACE salter2 on shelf_lower",
                    {"next_action": {"0": 1}}))
        
        return plan
    
    # Sceneario 11
    if scenario == 11:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter2", "left_of", env.all_obj["salter2"]),
                    "PLACE bottle2 left_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter2", "left_of", env.all_obj["salter2"]),
                    "PLACE bottle1 left_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "counter2", "left_of", env.all_obj["salter2"]),
                    "PLACE salter1 left_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "shelf_lower", "on", env.all_obj["shelf_lower"]),
                    "PLACE salter2 on shelf_lower",
                    {"next_action": {"0": 1}}))
        
        return plan

    # Sceneario 12
    if scenario == 12:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter"]),
                    "PICK salter",
                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE bottle1 on counter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE salter on counter1",
                    {"next_action": {"0": 1}}))
        
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter1", "on", env.all_obj["bottle2"]),
                    "PLACE bottle2 on counter1",
                    {"next_action": {"0": 1}}))
        
        return plan
    
    # Sceneario 13
    if scenario == 13:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter2", "left_of", env.all_obj["salter2"]),
                    "PLACE bottle2 left_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle3"]),
                    "PICK bottle3",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle3"], "counter2", "left_of", env.all_obj["salter2"]),
                    "PLACE bottle3 left_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "minifridge", "on", env.all_obj["minifridge"]),
                    "PLACE bottle1 on minifridge",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "minifridge", "on", env.all_obj["minifridge"]),
                    "PLACE salter1 on minifridge",
                    {"next_action": {"0": 1}}))
        
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "minifridge", "on", env.all_obj["minifridge"]),
                    "PLACE salter2 on minifridge",
                    {"next_action": {"0": 1}}))
        
        return plan
    
    # Sceneario 14
    if scenario == 14:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter2", "right_of", env.all_obj["salter2"]),
                    "PLACE bottle1 right_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE bottle2 on counter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "table1", "on", env.all_obj["table1"]),
                    "PLACE salter1 on table1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "shelf_lower", "on", env.all_obj["shelf_lower"]),
                    "PLACE salter2 on shelf_lower",
                    {"next_action": {"0": 1}}))
        
        return plan

    # Sceneario 15
    if scenario == 15:
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))

        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "table1", "on", env.all_obj["table1"]),
                    "PLACE bottle1 on table1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "table2", "left_of", env.all_obj["bottle3"]),
                    "PLACE bottle2 left_of bottle3",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "counter2", "right_of", env.all_obj["salter2"]),
                    "PLACE salter1 right_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((None if env is None else ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "sink_counter_left", "on", env.all_obj["sink_counter_left"]),
                    "PLACE salter2 on sink_counter_left",
                    {"next_action": {"0": 1}}))
        
        return plan


def compose_variant_scenario(env: ShopEnv, scenario: int):
    plan = []
    
    
    # Scenario 3
    if scenario == 3:
        plan.append((ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))

        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))

        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "table1", "on", env.all_obj["table1"]),
                    "PLACE bottle1 on table1",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "table2", "on", env.all_obj["table2"]),
                    "PLACE salter1 on table2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE bottle2 on counter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter2"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE salter2 on counter1",
                    {"next_action": {"0": 1}}))
        
        return plan
    
    # Scenario 4
    if scenario == 4:
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter2", "left_of", env.all_obj["salter"]),
                    "PLACE bottle2 left_of salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter2", "right_of", env.all_obj["salter"]),
                    "PLACE bottle1 right_of salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter"]),
                    "PICK salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter"], "counter1", "on", env.all_obj["counter1"]),
                    "PLACE salter on counter1",
                    {"next_action": {"0": 1}}))
        
        return plan
        

    # Scenario 7
    if scenario == 7:
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["plate"]),
                    "PICK plate",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["plate"], "table1", "right_of", env.all_obj["bottle1"]),
                    "PLACE plate right_of bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "minifridge", "on", env.all_obj["minifridge"]),
                    "PLACE bottle1 on minifridge",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter"]),
                    "PICK salter",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter"], "counter2", "left_of", env.all_obj["bottle2"]),
                    "PLACE salter left_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle3"]),
                    "PICK bottle3",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle3"], "counter2", "right_of", env.all_obj["bottle2"]),
                    "PLACE bottle3 right_of bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "table1", "left_of", env.all_obj["plate"]),
                    "PLACE bottle2 left_of plate",
                    {"next_action": {"0": 1}}))
        
        return plan

    # Scenario 8
    if scenario == 8:
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["tray"]),
                    "PICK tray",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_OPEN, "right", env.all_obj["kitchen_door"]),
                    "OPEN kitchen_door",
                    {"next_action": {"0": 1}}))

        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle1"]),
                    "PICK bottle1",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle1"], "counter2", "left_of", env.all_obj["salter2"]),
                    "PLACE bottle1 left_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter1"]),
                    "PICK salter1",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter1"], "counter2", "right_of", env.all_obj["salter2"]),
                    "PLACE salter1 right_of salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter2"]),
                    "PICK salter2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["bottle2"]),
                    "PICK bottle2",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["bottle2"], "counter1", "left_of", env.all_obj["salter4"]),
                    "PLACE bottle2 left_of salter4",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter3"]),
                    "PICK salter3",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PLACE, "right", env.all_obj["salter3"], "counter1", "right_of", env.all_obj["salter4"]),
                    "PLACE salter3 right_of salter4",
                    {"next_action": {"0": 1}}))
        
        plan.append((ShopDiscreteAction(ACTION_PICK, "right", env.all_obj["salter4"]),
                    "PICK salter4",
                    {"next_action": {"0": 1}}))
        
        
        
        return plan