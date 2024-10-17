from typing import Any
from pathlib import Path
from loguru import logger as loguru_logger
import sys
import json 
import os 
import time
import numpy as np
from pydantic import BaseModel
import wandb

'''
Action
predicate
prompt
response
value
'''



class TimeCheck: 
    '''
        tracks time and prints average 
    '''
    def __init__(self) -> None:
        self.timecheck = {1:[],
                        5:[],}
    def add(self, n, time):
        self.timecheck[n].append(time)
    def __str__(self) -> str:
        return f"Average: {np.mean(np.array(self.timecheck[1]))}s, {np.mean(np.array(self.timecheck[5]))}s"

class CustomLogger():
    def __init__(self, _logger_) -> None:

        # self.tracer = VizTracer(max_stack_depth=17, tracer_entries=20000000)
        self.tracer = None

        self.timecheck = TimeCheck()
        
        self.scenario_id = 0
        self.episode_idx = 0
        self.depth_idx = 1
        self.sim_idx = 0
        self.is_infeasible = False
        self.is_success = True
        self.data = {
            "idx": [],
            "depth": [],
            "sim": [],
            "history": [],
            "current_state": [],
            "output": [],
            "occlusion yes or not": [],
        }

        self.log_prefix = ""
        self.__logger = self.set_logger(_logger_)

    def set_log_prefix(self, prefix: str):
        self.log_prefix = prefix

    def __call__(self):
    # with open(file_name, 'w') as outfile:
    #     json.dump(self.data_buffer, outfile, indent=4)
    # self.data_buffer = {}
    
        print("Saving data to")

    def set_logger(self, __logger):

        __logger.level("LLM_INPUT", no=11, color="<yellow>")
        __logger.level("LLM_OUTPUT", no=12, color="<cyan>")
        __logger.level("LLM_PARSED", no=31, color="<magenta>")

        return __logger

        
    def get_depth_idx(self):
        return self.depth_idx
    def get_sim_idx(self):
        return self.sim_idx
    
    def get_filename(self, prefix: str="", suffix: str=""):
        return f"{prefix}episode{self.episode_idx}_depth{self.depth_idx}_sim{self.sim_idx}{suffix}"
    
    def get_viztrace(self):
        return self.tracer


    def record_simulation_llm(self, history: Any, condition: Any, output: Any, occlusion: Any):
        self.data['idx'].append(time.time())
        self.data['depth'].append(str(self.depth_idx))
        self.data['sim'].append(str(self.sim_idx))
        self.data["history"].append(history)
        self.data["current_state"].append(condition)
        self.data["output"].append(output)
        self.data["occlusion yes or not"].append(occlusion)


        return {"history": "", "condition": "", "output": ""}
    
    def save_episode_llm(self, file_name: str):
        with open(file_name, 'w') as outfile:
            json.dump(self.data, outfile, indent=4)
        self.data = {
            "idx": [],
            "depth": [],
            "sim": [],
            "history": [],
            "current_state": [],
            "output": [],
        }

    def get_logger(self):
        return self.__logger
    
    def add_time(self, n, time):
        self.timecheck.add(n, time)
        
LLOG = CustomLogger(loguru_logger)