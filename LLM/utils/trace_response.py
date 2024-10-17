import os 
import sys
import time 
import wandb 
from typing import Optional, List, Dict, Any, Union, Callable, Tuple
from wandb.sdk.data_types.trace_tree import Trace



class TraceResponse:
    def __init__(self, 
                 project_name : str = "dormamu",
                 entity_name : str = "joocjun",
                 mode : str = "disabled") -> None:
        r"""
            Args:
                project_name (str): project name for wandb
                entity_name (str): entity name for wandb
                mode (str): mode for wandb (disabled, online, offline)"""
        self.root_span : Trace = None
        wandb.init(project="dormamu", entity="joocjun", mode="disabled")
        

    def generate_trace(self, **kwargs):
        '''
            args:
                name: str,
                kind: Optional[str] = None,
                status_code: Optional[str] = None,
                status_message: Optional[str] = None,
                metadata: Optional[dict] = None,
                start_time_ms: Optional[int] = None,
                end_time_ms: Optional[int] = None,
                inputs: Optional[dict] = None,
                outputs: Optional[dict] = None,
                model_dict: Optional[dict] = None,
        '''
        trace_instance = Trace(**kwargs)

        if self.root_span is not None:
            self.root_span.add_child(trace_instance)
        else:
            self.root_span = self.set_root(trace_instance)
        

    def set_root(self, trace: Trace):
        self.root_span = trace

    def reset_root(self):
        self.root_span = None
        
    def log(self, name: str, reset_root: bool = True):
        self.root_span.log(name)
        if reset_root:
            self.reset_trace_root()


        



