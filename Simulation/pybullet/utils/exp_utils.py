import argparse
import os, sys
import yaml
import json
from typing import Dict
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Collecting data
from data_generation.collect_data import time_stamp
from Simulation.pybullet.custom_logger import LLOG
logger = LLOG.get_logger()

METHOD_CONFIG = {

    'LLM_MCTS': {
        'override_policy': {
            'key': ["project_params", "overridable", "policy"],
            'value': 'llm_plan',
        },
        'override_value': {
            'key': ["project_params", "overridable", "value"],
            'value': 'rollout',
        },
        'override_rollout': {
            'key': ["project_params", "overridable", "rollout"],
            'value': 'random',
        },
        'llm_mcts': {
            'key': ["plan_params", "llm_mcts"],
            'value': True,
        },
        'always_llm_plan': {
            'key': ["plan_params", "llm_trigger", "always_llm_plan"],
            'value': True,
        },
        'llm_beam_num': {
            'key': ["plan_params", "llm_trigger", "llm_beam_num"],
            'value': 5,
        }
    },

    'NO_UCT': {
        'override_policy': {
            'key': ["project_params", "overridable", "policy"],
            'value': 'llm_plan',
        },
        'override_value': {
            'key': ["project_params", "overridable", "value"],
            'value': 'rollout',
        },
        'override_rollout': {
            'key': ["project_params", "overridable", "rollout"],
            'value': 'random',
        },
        'always_llm_plan': {
            'key': ["plan_params", "llm_trigger", "always_llm_plan"],
            'value': True,
        },
        'llm_beam_num': {
            'key': ["plan_params", "llm_trigger", "llm_beam_num"],
            'value': 5,
        }
    },

    'STaLM':{
        'override_policy': {
            'key': ["project_params", "overridable", "policy"],
            'value': 'llm_plan',
        },
        'override_value': {
            'key': ["project_params", "overridable", "value"],
            'value': 'rollout',
        },
        'override_rollout': {
            'key': ["project_params", "overridable", "rollout"],
            'value': 'random',
        },
        'after_every_action': {
            'key': ["plan_params", "llm_trigger", "after_every_action"],
            'value': True,
        },
        'llm_beam_num': {
            'key': ["plan_params", "llm_trigger", "llm_beam_num"],
            'value': 5,
        }
    },

    "MCTS_UCT": {
        'override_policy': {
            'key': ["project_params", "overridable", "policy"],
            'value': 'random',
        },
        'override_value': {
            'key': ["project_params", "overridable", "value"],
            'value': 'rollout',
        },
        'override_rollout': {
            'key': ["project_params", "overridable", "rollout"],
            'value': 'random',
        },
    },

    "Iterative_Replanning": {
        'override_policy': {
            'key': ["project_params", "overridable", "policy"],
            'value': 'llm_plan',
        },
        'override_value': {
            'key': ["project_params", "overridable", "value"],
            'value': 'rollout',
        },
        'override_rollout': {
            'key': ["project_params", "overridable", "rollout"],
            'value': 'random',
        },
        'always_llm_plan': {
            'key': ["plan_params", "llm_trigger", "always_llm_plan"],
            'value': True,
        },
        'llm_beam_num': {
            'key': ["plan_params", "llm_trigger", "llm_beam_num"],
            'value': 1,
        }
    },
    
    "SAYCAN": {
        'override_policy': {
            'key': ["project_params", "overridable", "policy"],
            'value': 'llm_plan',
        },
        'llm_beam_num': {
            'key': ["plan_params", "llm_trigger", "llm_beam_num"],
            'value': 5,
        }
    },
}


## OPENAI stuff
def load_openai_token():

    OPEN_AI_TOKEN_PATH = Path(__file__).parent.parent.parent.parent / 'LLM' / 'api_token.json'

    if not OPEN_AI_TOKEN_PATH.exists():
        if os.environ.get("OPENAI_API_KEY") is not None:
            api_token = os.environ["OPENAI_API_KEY"]
        else:
            logger.info(f"OpenAI API token is not found at {OPEN_AI_TOKEN_PATH}. Please enter your OpenAI API token.")
            api_token = input("Please enter your OpenAI API token: ")
            with open(OPEN_AI_TOKEN_PATH, 'w') as f:
                json.dump({"api_token": api_token}, f)
        os.environ["OPENAI_API_KEY"] = api_token
    else:
        with open(OPEN_AI_TOKEN_PATH, 'r') as f:
            api_token = json.load(f)['api_token']
            os.environ["OPENAI_API_KEY"] = api_token


## Configuration setters
def override_scenario_configs(params, config):

    config["problem_params"]["scenario"] = params.override_scenario_num if params.override_scenario_num is not None else config["problem_params"]["scenario"]

    SCENARIO_PATH = Path(__file__).parent.parent / "cfg" / "scenarios" / f"scenario{config['problem_params']['scenario']}.yaml"

    with open(SCENARIO_PATH, "r") as f:
        scenario_config = yaml.load(f, Loader=yaml.FullLoader)

    config["plan_params"]["max_depth"] = 20 ## CONFIG_POSSIBLE(SJ)
    config["env_params"]["shop_env"]["dynamics"]["door_duration"] = scenario_config["door_duration"]

    config['problem_params']['goal'] = scenario_config['goal']
    

    return config


def override_method_configs(method, config):
    if method not in METHOD_CONFIG:
        return config
    
    for k, v in METHOD_CONFIG[method].items():
        config = set_config(config, v['key'], v['value'])
    return config


def set_config(config, key, value):
    if len(key) == 1:
        config[key[0]] = value
    elif len(key) == 2:
        config[key[0]][key[1]] = value
    elif len(key) == 3:
        config[key[0]][key[1]][key[2]] = value
    elif len(key) == 4:
        config[key[0]][key[1]][key[2]][key[3]] = value
    return config


def override_configs(params):

    # Open yaml config file
    with open(Path(__file__).parent.parent / "cfg" / params.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # override experimental configurations
    if params.policy_lmp_filename != "":
        config['project_params']['overridable']['prompt_params']['policy']['lmp_filename'] = params.policy_lmp_filename
    if params.value_lmp_filename != "":
        config['project_params']['overridable']['prompt_params']['value']['lmp_filename'] = params.value_lmp_filename
    config['baseline'] = params.baseline
    config['plan_params']['time_limit'] = params.time_limit

    config['project_params']['overridable']['prompt_params']['policy']['add_strategy'] = params.policy_strategy
    config['project_params']['overridable']['prompt_params']['policy']['add_action_preconditions'] = params.prompt_precondition
    config['project_params']['overridable']['prompt_params']['policy']['validate_preconditions'] = params.validate_precondition
    config['project_params']['overridable']['prompt_params']['policy']['add_history'] = params.add_history

    # Override configuration file when given the arguments.
    config = override_scenario_configs(params, config)

    if params.uniform_random: ## check if the config is already set to 1
        if config["plan_params"]["policy"]["random"] == 1:
            logger.info("Uniform random search is already set to 1. Skipping...")
        else:
            logger.info(f"Changing random p from {config['plan_params']['policy']['random']} to 1 for uniform random search.")
            config["plan_params"]["policy"]["random"] = 1

    config['dump_prompt'] = params.dump_prompt

    if params.override_num_sims is not None :
        config["plan_params"]["num_sims"] = params.override_num_sims
    if params.infeasible_action_reward is not None:
        config["plan_params"]["reward"]['infeasible'] = params.infeasible_action_reward
    if params.override_num_conti_param_sample is not None:
        config['pose_sampler_params']['num_filter_trials_pick'] = params.override_num_conti_param_sample
        config['pose_sampler_params']['num_filter_trials_place'] = params.override_num_conti_param_sample

    if params.override_exp_log_dir_path is not None:
        config["project_params"]["overridable"]["default_exp_log_dir_path"] \
            = params.override_exp_log_dir_path

    if params.override_policy is not None:
        config["project_params"]["overridable"]["policy"] \
            = params.override_policy
    if params.override_value is not None:
        config["project_params"]["overridable"]["value"] \
            = params.override_value
        
    if params.override_rollout is not None:
        config["project_params"]["overridable"]["rollout"] \
            = params.override_rollout
    if params.rollout_depth is not None:
        config["plan_params"]["rollout_depth"] \
            = params.rollout_depth

    if params.override_inference_device is not None:
        config["project_params"]["overridable"]["inference_device"] \
            = params.override_inference_device

    if params.num_sims is not None:
        config["plan_params"]["num_sims"] \
            = params.num_sims

    if params.k_a is not None:
        config["plan_params"]["k_a"] \
            = params.k_a       
        
    if params.full_predicates:
        config["predicate_params"]["use_relative_occlusion"] \
            = params.full_predicates
        
    config["plan_params"]['llm_trigger']["always_llm_plan"] \
        = params.always_llm_plan
    config["plan_params"]['llm_trigger']["after_every_action"] \
        = params.after_every_action
    config["plan_params"]['llm_trigger']["start_search_with_llm_plan"] \
        = params.start_search_with_llm_plan
    config["plan_params"]["plan_with_cot"] \
        = params.plan_with_cot
    
    config['plan_params']['llm_trigger']["llm_beam_num"] = params.llm_beam_num
    config['plan_params']['MCTS_FIRST'] = params.MCTS_FIRST
    config["plan_params"]["llm_assisted"] = any([v for k, v in config["plan_params"]["llm_trigger"].items()])
    config['plan_params']['policy_value'] = params.pucb_prob_method
    config['plan_params']['llm_mcts'] = params.llm_mcts
    config["project_params"]["project_name"] = params.project_name

    config = override_method_configs(params.baseline, config)
    
    return config


def tags_for_logging(config):
    ## Problem 
    scenario = str(config["problem_params"]["scenario"])
    value_name = config["project_params"]["overridable"]["value"]

    if 'hcount' in value_name:
        config["project_params"]["overridable"]["value"] = 'hcount'
    
    name_value = {
        "scenario": scenario,
        
    } 

    tags = [v for k, v in name_value.items()]

    run_name = f"S{scenario}"

    return tags, run_name



def get_params():
    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config",                          type=str, default="config.yaml", help="Specify the config file to use.")
    parser.add_argument("--time_limit" ,                    type=int, default=300, help="Time limit for planning")
    parser.add_argument("--num_episodes",                    type=int, default=1, help="Number of episodes")
    parser.add_argument("--override_scenario_num",           type=int, default=None, help="Scenario ID")
    parser.add_argument("--override_num_sims",               type=int, default=None, help="overrides number of simulations when called")
    parser.add_argument("--override_num_conti_param_sample", type=int, default=None)
    parser.add_argument("--override_exp_log_dir_path",       type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_policy",                 type=str, default=None, help="Overrides policy")
    parser.add_argument("--override_value",                  type=str, default=None, help="Overrides value function")
    parser.add_argument("--override_rollout",                type=str, default=None, help="Overrides rollout. random or llm_plan")
    parser.add_argument("--rollout_depth",                   type=int, default=None, help="Overrides rollout depth")
    parser.add_argument("--override_inference_device",       type=str, default=None, help="Overrides the default inference device in config.")
    parser.add_argument("--num_sims",                        type=int, default=None, help="Number of simulations")
    parser.add_argument("--infeasible_action_reward",        type=float, default=None, help="Reward for infeasible action")
    parser.add_argument("--k_a",                             type=float, default=None, help="k_a for ucb")
    parser.add_argument("--override_model",                  type=str, default=None, help="Overrides model name")
    parser.add_argument("--override_ip",                     type=str, default=None, help="Overrides ip")
    parser.add_argument("--pucb_prob_method",                type=str, default='exponential', help="Overrides pucb_prob_method")
    parser.add_argument("--uniform_random", action='store_true', help="Conduct uniform random search by setting plan_params/policy/random to 1")
    parser.add_argument("--policy_strategy", action='store_true', help="Enable policy strategy")
    parser.add_argument("--value_strategy", action='store_true')
    parser.add_argument("--prompt_precondition", action='store_true')
    parser.add_argument("--validate_precondition", action='store_true')
    parser.add_argument("--add_history", action='store_true')
    parser.add_argument('--value_aggregation', type=str, default='normalize', help="Value aggregation", choices=['normalize', 'average'])

    parser.add_argument("--dump_prompt", action='store_true', help="Dump prompt")
    parser.add_argument("--full_predicates", action="store_true", help="Compute left, right, front, behind of object.")
    
    parser.add_argument("--policy_lmp_filename", type=str, default="", help="Policy LMP filename")
    parser.add_argument("--value_lmp_filename", type=str, default="", help="value LMP filename")
    parser.add_argument("--domain_pddl_filename", type=str, default="", help="Domain PDDL filename")


    ## LLM Trigger arguments 
    parser.add_argument("--always_llm_plan", action='store_true', help="always use LLM to plan and do search.")
    parser.add_argument("--llm_beam_num" , type=int, default=5, help="Beam number for LLM")
    parser.add_argument("--start_search_with_llm_plan", action='store_true', help="Start search with LLM plan.")
    parser.add_argument("--after_every_action", action='store_true', help="Trigger LLM after every action.")
    parser.add_argument("--llm_mcts", action='store_true', help="Use LLM MCTS")

    parser.add_argument("--plan_with_cot", action='store_true', help="use value function to backup escape plan.")
    parser.add_argument("--use_value_escape_plan", action='store_true', help="use value function to backup escape plan.")
    parser.add_argument("--MCTS_FIRST", action='store_true', help="use MCTS first for planning then LLM Plan.")
    parser.add_argument("--baseline", default=None, choices=["ONE_SHOT","SAYCAN","MCTS_UCT", "LLM_MCTS", "NO_UCT", "STaLM", "ReAct", "Reflexion", None], help="Baseline method")
    
    parser.add_argument("--project_name", type=str, default="commonsense_robotics", help="wandb project name")

    params = parser.parse_args()

    return params


def dump_config(config: Dict):
    folder_name = f"{str(time_stamp())}_policy={config['project_params']['overridable']['policy']}_value={config['project_params']['overridable']['value']}"
    log_base_path = Path(config["project_params"]["overridable"]["default_exp_log_dir_path"]) / f"scenario{config['problem_params']['scenario']}" / folder_name

    os.makedirs(log_base_path, exist_ok=True)
    config["project_params"]["overridable"]["default_exp_log_dir_path"] = log_base_path.as_posix()

    with open(log_base_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)

