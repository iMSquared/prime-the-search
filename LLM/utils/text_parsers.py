'''
TODO 
 - [ ] Should We try other matching algorithms? https://medium.com/codex/best-libraries-for-fuzzy-matching-in-python-cbb3e0ef87dd 
'''
from gettext import find
import re 
from typing import List, Tuple
from pydantic import BaseModel

from Simulation.pybullet.mdp.shop.shop_MDP import (
    ACTION_OPEN,
    ACTION_PICK,
    ACTION_PLACE,
    Relation,
)
from Simulation.pybullet.custom_logger import LLOG

logger = LLOG.get_logger()


PYTHON_FUNCTION_SIGNATURE = re.compile(r"```python\n(.*?)```", re.DOTALL)

'''
Mainly 3 types of parsers 
1. Python Function Parser
2. Action Parser
3. Plan Parser 
'''

def python_function_parser(s: str, **args):
    r"""
    Parse the response from the model and return the parsed response.
    args:
    - response: str, response from the model.
    return:
    - parsed_response: str, parsed response."""

    python_function = re.findall(PYTHON_FUNCTION_SIGNATURE, s)


    return python_function[0]


def plan_parser(s: str, **kwargs):
    r"""
    Parse a plan from the input string.

    args:
    - s: str, input string.
    return:
    - plan: list, list of actions.
    """
    plan = find_plan(s, **kwargs)
    return plan


def action_parser(s: str, **args):
    r"""
    Parse an action from the input string.

    args:
    - s: str, input string.
    - args: dict, additional arguments.
        - pattern: str, pattern to match.
    return:
    - action: dict, action.
    """
    plan = plan_parser(s)
    action = plan[0]
    return action


def remove_parts(s: str, override_parts=None):
    if override_parts is not None:
        remove_dict = {ord(char): None for char in override_parts}
    else:
        remove_dict = {
            40: None,  # '('
            41: None,  # ')'
            39: None,  # Single quote
            34: None,  # Double quote
            32: None   # Space
        }

    return s.translate(remove_dict)


def parse_args(s: str):
    r"""
    Parse the arguments of a bracketed string.

    args:
    - s: str, bracketed string.
    return:
    - args: list, list of arguments."""
    args = [remove_parts(arg) for arg in s.split(',')]
    return args


## LLM assisted plan
def convert_to_discrete_action(s: str, for_camera:bool=False):
    r"""
    Convert a bracketed string to a DiscreteAction object.

    args:
    - s: str, bracketed string.
    return:
    - action: DiscreteAction, DiscreteAction object."""

    action = None
    args = parse_args(s)
    action_type = args[0].lower()
    if action_type not in ["pick", "place", "open"] and not for_camera:
        logger.warning(f"Action type {action_type} not recognized.")
        
    else:
        try:
            if action_type == 'open':
                action = (args[0], args[1])
            elif action_type == 'pick':
                action = (args[0], args[1])
            elif action_type == 'place':
                action = (args[0], args[1], args[2], args[3])
            else:
                if not for_camera:
                    logger.warning(f"Action type {action_type} not recognized.")
                action = None
        except Exception as e:
            logger.warning(f"Error in converting to discrete action: {e}")
            action = None
    return action


def find_plan(s: str, for_camera:bool=False):
    r"""
    Find every bracketed string in the input string.

    args:
    - s: str, input string.
    return:
    - bracketed_strings: list, list of bracketed strings."""

    bracketed_strings = re.findall(r'\(.*?\)', s)
    plan = []
    for bracketed_string in bracketed_strings:
        discrete_action_args = convert_to_discrete_action(bracketed_string, for_camera=for_camera)
        if discrete_action_args is not None:
            plan.append(discrete_action_args)
    return plan




    
