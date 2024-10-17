from typing import Dict
from abc import *

# Simulations
from Simulation.pybullet.custom_logger import LLOG
logger = LLOG.get_logger()
DEBUG = False


class AgentPredicateManager(metaclass=ABCMeta):
    """
    Agent Embodiment predicates
    """
    def __init__(self, config: Dict):
        self.config = config


    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict:
        pass


class AssetPredicateManager(metaclass=ABCMeta):
    def __init__(self, config: Dict):
        self.config = config
   
    @abstractmethod
    def evaluate(self, *args, **kwargs)-> Dict:
        pass


class PredicateManager(metaclass=ABCMeta):
    def __init__(self, config: Dict):
        # Config
        self.config = config
        self.agent_manager = AgentPredicateManager(config)
        self.asset_manager = AssetPredicateManager(config)

    @abstractmethod
    def evaluate(self, *args, **kwargs)->Dict:
        pass