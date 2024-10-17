from abc import *

class PromptGenerator(metaclass=ABCMeta):
    def __init__(self, config):
        self.config = config

        # Prompts
        self.trigger = None
        self.system_prompt = None

        
    @abstractmethod
    def generate_prompt(self, *args, **kwargs):
        pass
