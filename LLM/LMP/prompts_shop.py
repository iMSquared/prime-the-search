import os
import re
from regex import F
from typing import List, Dict
from pathlib import Path

from Simulation.pybullet.mdp.MDP_framework import HistoryEntry
from Simulation.pybullet.predicate.reoranize_info import PDDLStyleState
from LLM.utils.misc import read_json, read_yaml, read_txt
from LLM.LMP.prompts_framework import PromptGenerator


class ShopPDDLStylePolicyPromptGenerator(PromptGenerator):
    def __init__(self, config: Dict):
        super().__init__(config) 
        self.trigger = "Generate a plan to achieve the goals from init."
        
        self.system_prompt = self.get_system_prompt()
        self.domain_pddl_path = Path(__file__).parent.parent.parent / "Simulation/pybullet/predicate/pddl" / self.config['plan_params']['domain_pddl_filename']
        self.problem_pddl_path = Path(__file__).parent.parent.parent / "Simulation/pybullet/predicate/pddl/"

        self.state_rearrange = PDDLStyleState(config)


    def get_system_prompt(self, path = None):
        r"""
        Get the system prompt for the policy model.
        return:
        - system_prompt: str, system prompt for the policy model."""
        
        self.system_prompt_path = Path(__file__).parent.parent / "prompts/plan" / f"system_{'w_cot' if self.config['plan_params']['plan_with_cot'] else 'wo_cot'}.txt"

        
        system_prompt = read_txt(self.system_prompt_path)
        

        return system_prompt


    def replace_init_state(self, original_problem, new_init_content):
        # Find the start index of the init section
        init_start_index = original_problem.find('(:init')

        # If init section is not found, we can't proceed
        if init_start_index == -1:
            print("Init section not found.")
            return

        # Find the end of the init section by counting parentheses
        paren_count = 0
        init_end_index = init_start_index
        for i in range(init_start_index, len(original_problem)):
            if original_problem[i] == '(':
                paren_count += 1
            elif original_problem[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    init_end_index = i
                    break

        # Replace the init section
        new_content = original_problem[:init_start_index] + new_init_content + original_problem[init_end_index + 1:]


        return new_content

    def get_domain_pddl(self, ):
        return read_txt(self.domain_pddl_path)      


    def get_problem_pddl(self, scenario_id, predicates: Dict):
        problem_pddl = read_txt(self.problem_pddl_path / f"scenario{scenario_id}.pddl")
        goal, state = self.state_rearrange(predicates, config={'python_compatible':False, 'goal_related': False, 'truncate_elements': -1})
        problem_pddl = self.replace_init_state(problem_pddl, f"(:init\n\t{state}\n)")
        return problem_pddl  
    
    def get_problem_pddl_saycan(self, scenario_id, predicates: Dict, history: List[HistoryEntry]):
        problem_pddl = read_txt(self.problem_pddl_path / f"scenario{scenario_id}.pddl")
        goal, state = self.state_rearrange(predicates, config={'python_compatible':False, 'goal_related': False, 'truncate_elements': -1})
        problem_pddl = self.replace_init_state(problem_pddl, f"(:init\n\t{state}\n)")
        return problem_pddl


    def get_hint(self, **kwargs) -> str:

        hint = " Moving to other regions cost time and resources. Try to minimize the navigation to other regions. When placing an object."

        return hint
    
    def get_reflection(self, history): 
        output = f"The plans you are suggesting seems to be problematic since they are repeating redundant actions {history}"


    def generate_prompt(self, predicates: Dict, scenario_id: str, **kwargs):
        """
        Args:
        - predicates: Dict, predicates of the current state.
        - scenario_id: str, id of the current scenario.
        return:
        - prompt: str, prompt for the current state.
        """
        do_reflexion = kwargs.get('do_reflexion', False)
        memory = kwargs.get('memory', [])

        DOMAIN_PDDL = self.get_domain_pddl()
        PROBLEM_PDDL = self.get_problem_pddl(scenario_id, predicates,)

        instance = "## Instance ##\n"
        instance = "### Domain ###\n" + DOMAIN_PDDL + "\n\n"
        instance += "### Problem ###\n" + PROBLEM_PDDL + "\n\n"
        
        instance += self.trigger
        # instance += self.get_hint()
        
        prompt = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': instance}
        ]

        if do_reflexion:
            for _mem in memory:
                prompt.extend(_mem)


        return prompt
    
    
class ShopPromptGenerator(PromptGenerator):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.env_config = config["env_params"]
        self.prompt_config = config["project_params"]["overridable"]["prompt_params"]["value"]
        self.generation_instructions = {
            "chain_of_thought": '"chain_of_thought": str, //Think of remaining steps and challengs to satisfy the goal condition.',
            "plan": '"plan": List, //List of actions',
            "next_action": '"next_action": str, //Next action to take',
            "actions_left": "'actions_left': int",
            "rationale": '"rationale": str, //Justify why certain actions were chosen over others and why the overall plan will be effective and feasible.',
        }

        self.predicates = None
        self.options: List[str] = None
        self.system_prompt = None
        # self.strategies = read_json(os.path.join(os.path.dirname(__file__), 'strategy/strategy.json'))
        self.strategies = read_json(Path(__file__).parent / "strategy/strategy.json")

    
    def get_system_prompt(self):
        prompt = """You are an expert action planning agent for suggesting a optimal plan of actions to satisfy the goal in cluttered regions with occlusions.
You will be given Instructions and a Problem. Make sure to be aware of the Instructions when solving the Problem."""

        return prompt


    def make_action_types(self, prompt_keyword: str = 'Action Type', include_actions: dict =  {'PICK': True, 'PLACE': True, 'OPEN': True, 'CLOSE': False}, add_action_preconditions: bool = True):
        # Define the available actions and their details
        actions = {
            'PICK': {
                'description': 'PICK {movable object}',
                'example': 'PICK bottle',
                'preconditions': [
                    'Agent has an empty hand',
                    '{movable object} is not held',
                ]
            },
            'PLACE': {
                'description': 'PLACE {movable object} {direction} {region/movable object}',
                'example': 'PLACE bottle behind_of plate',
                'preconditions': [
                    'Agent is holding {movable object}'
                ]
            },
            'OPEN': {
                'description': 'OPEN {door}',
                'example': 'OPEN door',
                'preconditions': [
                    'Agent has an empty hand',
                    '{door} is closed'
                ]
            },
            'CLOSE': {
                'description': 'CLOSE {door}',
                'example': 'CLOSE door',
                'preconditions': [
                    'Agent has an empty hand',
                    '{door} is open'
                ]
            }
        }

        action_types = f"### {prompt_keyword} ###\nThese are the available types of actions:\n"


        for action, include in include_actions.items():
            if include and action in actions:
                action_detail = actions[action]
                action_line = f"- {action_detail['description']}\n    - example: {action_detail['example']}"
                if add_action_preconditions:
                    preconditions = '\n'.join([f"        - {precondition}" for precondition in action_detail['preconditions']])
                    action_line += f"\n    - preconditions:\n{preconditions}"
                action_types += f"{action_line}\n"

        return action_types.strip()


    def make_action_arguments(self, predicates, prompt_keyword: str='Action Arguments',):
        action_arguments = f"### {prompt_keyword} ###\n"
        action_arguments += """Arguments for actions are:
- Movable Objects=[{movable_objects}]
- Regions=[{regions}]
- Directions=[{directions}]
- Doors=[{doors}]""".format(
            movable_objects=", ".join(
                [
                    f'"{movable_object}"'
                    for movable_object in predicates["problem"]["is_movable"]
                ]
            ),
            regions=", ".join(
                [f'"{region}"' for region in predicates["problem"]["is_region"]]
            ),
            directions=", ".join(
                [f'"{direction}"' for direction in predicates["problem"]["directions"]]
            ),
            doors=", ".join(
                [f'"{door}"' for door in predicates["problem"]["is_openable"]]
            ),
        )

        return action_arguments


    def make_response_format(self, prompt_keyword: str='Response Format'):
        response_format = f"### {prompt_keyword} ###\n"
        response_format += """You must respond by filling in this JSON format\n{"""
        response_format += "\n".join([self.generation_instructions[k] for k in self.prompt_config["generation_requirements"]])
        response_format += "}"

        return response_format
    

    def make_strategy(self, scenario_id: str=None, prompt_keyword: str='Strategy', **kwargs):
        
        if scenario_id is not None:
            strategy = f"### {prompt_keyword} ###\n"
            strategy += "\n".join([f"- {strategy}" for strategy in self.strategies[scenario_id].values()])
        else:
            raise NotImplementedError


        return strategy
    

    def complete_instructions(self, predicates: Dict):
        instruction_prompt = "## Role ##\n"
        instruction_prompt += self.system_prompt


        instruction_prompt += "\n\n## Instructions ##\n"

        ACTION_TYPE = self.make_action_types(add_action_preconditions=self.prompt_config.get("add_action_preconditions", True))
        ACTION_ARGUMENTS = self.make_action_arguments(predicates)
        RESPONSE_FORMAT = self.make_response_format()



        instruction_prompt += ACTION_TYPE + "\n\n"
        if self.prompt_config.get("add_actionArguments", True):
            instruction_prompt += ACTION_ARGUMENTS + "\n\n"

        
        instruction_prompt += RESPONSE_FORMAT

        return instruction_prompt
    

    def sort_order(self, predicates, objs):
        """
        in predicates there is a key called 'relative_position' which has 'left', 'right', 'front', 'back' as keys and list of objects as values.
        i want to sort the list of objects based on the order of 'front' and 'left' keys.
        """
        # Sorting the dictionary based on the length of the list in the nested dictionary
        objsInregion = {k: v for k, v in predicates["asset"].items() if k in objs}
        front_sorted = dict(
            sorted(
                objsInregion.items(),
                key=lambda item: len(item[1]["relative_position"]["front"]),
            )
        )
        left_sorted = dict(
            sorted(
                objsInregion.items(),
                key=lambda item: len(item[1]["relative_position"]["left"]),
            )
        )

        return {"front": list(front_sorted.keys()), "left": list(left_sorted.keys())}


    def make_obj_observations(
        self, target_obj: str, predicates: Dict, include_occ_free=True
    ):
        ## Get the target object and other objects on the region target object is currently on.

        target_predicate = predicates["asset"][target_obj]
        preOccs: List[str] = target_predicate["is_occ_pre"]
        manipOccs = target_predicate["is_occ_manip"]
        manipFreeRegions: List[str] = list(set([region_name for region_name, occObjs in target_predicate["is_occ_manip"].items() if len(occObjs) == 0]))

        group_occs = {}
        for region, occs in manipOccs.items():
            if region == target_obj: continue
            if len(occs) == 0:
                continue
            sorted_occs = sorted(occs)
            occ_key = ", ".join([f'"{occ}"' for occ in sorted_occs])
            if occ_key not in group_occs.keys():
                group_occs[occ_key] = [region]
            else:
                group_occs[occ_key].append(region)


        description = f'- "{target_obj}":\n'

        if include_occ_free:
            if len(preOccs)==0:  # it is occlusion free for PICK
                description += "\t- occlusion free PICK.\n"
            else:  # occlusion even for pick
                str_buff = ", ".join([f'"{occ}"' for occ in preOccs])
                description += f"\t- occlusion by {str_buff} for PICK.\n"

        if len(manipFreeRegions) == 0:
            pass
        elif len(manipFreeRegions) < len(predicates["problem"]["is_region"]):
            str_buff = ", ".join([f'"{region}"' for region in manipFreeRegions])

            description += f"\t- occlusion free PLACE on {str_buff}.\n"
        else:  ## happy path
            description += f"\t- occlusion free PLACE everywhere.\n"


        for occs, regions in group_occs.items():
            str_buff2 = ", ".join([f'"{region}"' for region in regions])
            description += f"\t- PLACE on {str_buff2} occluded by {occs}.\n"

        return description.strip()


    def make_scene(self, predicates: Dict, region2objs: Dict):
        empty = []
        scene = ""
        hand_info = {k:v for k,v in predicates['agent'].items() if '_hand_holding' in k}


        hand_desc = "Agent is holding"
        if all(len(holding)==0 for holding in hand_info.values()):
            hand_desc += " nothing"
        else:
            for gripper, holding in hand_info.items():
                if len(holding) == 0: continue
                str_buff = ", ".join([f'"{obj}"' for obj in holding.keys()])
                hand_desc += f" {str_buff} and"
            hand_desc = hand_desc[:-4]

        if predicates["agent"]["has_empty_hand"]:
            hand_desc += " and has an empty hand"
        else:
            hand_desc += " and has no empty hand"
        hand_desc += ". "

        scene += hand_desc


        direct_holding = [list(k.keys())[0] for k in hand_info.values()]

        for region, objs in region2objs.items():
            if len(objs) == 0:
                empty.append(region)
            else:
                _objs = []
                for obj in objs:
                    if obj in direct_holding:
                        # scene += f'"{obj}" on hand. '
                        pass
                    else:
                        _objs.append(obj)
                if len(_objs) == 0:
                    continue
                else:
                    str_buff = ", ".join([f'"{_obj}"' for _obj in _objs])
                    scene += f'{str_buff} on "{region}". '

        ## Empty Regions  grouped to shorten repeat of Nothing on##
        if len(empty) == 0:
            pass
        else:
            str_buff = ", ".join([f'"{region}"' for region in empty])
            scene += f"Nothing on {str_buff}. "

        for door in predicates["problem"]["doors"]:
            if predicates["asset"][door]["is_open"]:
                scene += f'"{door}" is open. '
            else:
                scene += f'"{door}" is closed. '

        return scene.strip()


    def make_goal_condition(self, predicate, prompt_keyword: str='Goal Condition'):
        goal_condition = f"### {prompt_keyword} ###\n"
        for obj, region in zip(predicate['problem']['goal']['objects'], predicate['problem']['goal']['regions']):
            str_buff = ", ".join([f'"{_obj}"' for _obj in obj])
            goal_condition += f'{str_buff} {region[1]} "{region[0]}".\n'
        return goal_condition
        

    def make_problem(self, predicates: Dict, action_history: List[str], scenario_id: str) -> str:
        """
        Generates the Goal Condition, History, and State for the problem.
        """

        # Process Goal Condition
        PromptComponents = {}
        PromptComponents["goal_description"] = self.make_goal_condition(predicates)


        ## TEMP (SJ): This should be changed to make the strategy from the prompt config.
        STRATEGY = self.make_strategy(scenario_id)
        if self.prompt_config.get("add_strategy", False):
            PromptComponents["strategy"] = STRATEGY 


        # Process Predicates
        predicate_description_method = self.prompt_config.get("predicate_description_method", "Single")
        if predicate_description_method == "Divided":
            PromptComponents["predicate_prompt"] = self.process_predicates_object(predicates)
        elif predicate_description_method == "Single":
            PromptComponents["predicate_prompt"] = self.process_predicates_individual(predicates)
        else:
            raise NotImplementedError
        

        # Process History
        if self.prompt_config.get("add_history", True):
            PromptComponents["history"] = self.process_history(action_history)

    
        return "\n\n".join(PromptComponents.values())


    def process_history(self, action_history):        
        history_prompt = "### History ###\n"
        if len(action_history) == 0:
            history_prompt += "No action has been taken yet."
        else:
            history_prompt += "["
            for idx, action in enumerate(action_history):
                history_prompt += f"{action}, "
            history_prompt += "]"
        return history_prompt


    def process_predicates_object(self, predicates):
        assert False, "Currently abandoned."
        ### Scene ###
        region2objs = predicates["info"]["region2objs"]
        hand_info = {k:v for k,v in predicates['agent'].items() if '_hand_holding' in k}

        movable_objects = predicates['problem']['is_movable']
        goal = predicates["problem"]["goal"]

        scene = self.make_scene(predicates, region2objs)
        #######

        ### Observation ###
        observations = "### State ###\n"
        observation_scope = self.prompt_config.get("observation_scope", "all")
        objects = []
        if observation_scope == "all":
            objects = movable_objects
        elif observation_scope == "goal":
            for objs in goal["objects"]:
                objects += [obj for obj in objs]
        else:
            raise NotImplementedError

        for obj in objects:
            obs = self.make_obj_observations(obj, predicates, hand_info)
            observations += obs.strip()
            observations += "\n\n"
        output = f"**Scene**\n{scene}\n\n**Observation**\n{observations}"
        return output

    ''' 
    TODO (SJ): Make this part into main functionality for state description.
        - [x] Filter Kitchen Door
        - [ ] Be able to set the scope of objects
        - [x] Not give redundant information on what we are already holding 
        - [ ] Add left, right, front, back information 
    '''     
    def process_predicates_individual(self, predicates):
        # Implement individual-based predicate processing here
        individual_buffer = []
        empty = []
        hand_desc = ""

        hand_info = {k:v for k,v in predicates['agent'].items() if '_hand_holding' in k}
        region2objs = predicates["info"]["region2objs"]

        ## Holding Start ##
        hand_desc += "Agent is holding"
        if all([len(holding) == 0 for holding in hand_info.values()]):
            hand_desc += " nothing"
        else:
            for gripper, holding in hand_info.items():
                if len(holding) == 0: continue
                str_buff = ", ".join([f'"{obj}"' for obj in holding])
                hand_desc += f" {str_buff} and"
            hand_desc = hand_desc[:-4]

        if predicates["agent"]["has_empty_hand"]:
            hand_desc += " and has an empty hand."
        else:
            hand_desc += " and no empty hand."
            
        individual_buffer.append(hand_desc)
        ## Holding End ##

        for region, objs in region2objs.items():
            if len(objs) == 0:
                empty.append(region)
            else:
                str_buff = ", ".join([f'"{obj}"' for obj in objs])
                # individual_buffer.append(f'{str_buff} on "{region}". ')
                for obj in objs:
                    obs = self.make_obj_observations(obj, predicates)
                    obs = obs.replace(f'- "{obj}"', "")
                    obs = obs.replace(f"- ", "")
                    obs = obs.replace("\t", "")
                    obs = obs.replace(":", "")
                    obs = obs.replace(".", "")
                    _obs = obs.split("\n")
                    _obs = [o for o in _obs if len(o) > 1]
                    individual_buffer.append(
                        f'For "{obj}" on "{region}", {", ".join(_obs)}.'
                    )

        ## Empty Regions Start ##
        if len(empty) == 0:
            pass
        else:
            str_buff = ", ".join([f'"{region}"' for region in empty])
            individual_buffer.append(f"Nothing on {str_buff}. ")
        ## Empty Regions End ##


        ## DOOR Start ##
        for door in predicates["problem"]["is_openable"]:
            if predicates["asset"][door]["is_open"]:
                individual_buffer.append(f'"{door}" is open.')
            else:
                individual_buffer.append(f'"{door}" is closed.')
        ## DOOR End ##

        predicate_prompt = "### State ###\n"+"- " + "\n- ".join(individual_buffer)

        return predicate_prompt.strip()
    

    def complete_problem(
        self, predicates: Dict, action_history: List[str], scenario_id: str
    ):
        instance = self.make_problem(predicates, action_history, scenario_id=scenario_id)

        task_query = self.make_task_query()

        return instance + "\n\n" + task_query


    def make_task_query(self):        
        task = "What would be the next action for a plan to satisfy the Goal Condition?"
        return task


    def make_instance(
        self,
        predicates: Dict,
        action_history: List[str] = [],
        scenario_id: str=None,
    ) -> str:
        """
        function that is used to generate a instance including both for few-shot learning and for the problem to solve.
        Default shows better performance in many cases
        args:
            predicates: dictionary of predicates
            action_history: list of actions taken so far
        output:
            instructions: string of instructions
            problem: string of problem
        """

        instructions = self.complete_instructions(predicates)
        problem = self.complete_problem(predicates, action_history, scenario_id=scenario_id)

        instructions, problem = self.split_text(instructions+"\n\n## Problem ##\n"+problem, 
                                                self.prompt_config.get("splitter", "## Problem ##"))

        return instructions, problem
    
    
    def make_response(self, shot_info: Dict) -> str:
        response = "{" + "\n".join([f'"{k}": {shot_info[k]},' for k in self.prompt_config["generation_requirements"]]) + "}"
        return response
    

    def split_text(self, text: str, splitter: str) -> List[str]:
        r"""Split the text into a list of strings.
        args:
            text: string to split
            splitter: string to split the text
        return:
            split_text: list of strings
        """

        splited_text = text.partition(splitter)
        return "".join(splited_text[0]), "".join(splited_text[1:])
    

    def get_scenario_num_from_path(self, path: str) -> str:
        r"""Get the scenario number from the path.
        args:
            path: path to the scenario
        return:
            scenario_num: scenario number
        """
        scenario_step_pattern = re.compile(r'predicates_scenario=(\d+)_step=(\d+).json')
        match = scenario_step_pattern.search(path)
        scenario_id = match.group(1)

        return scenario_id
    
    
    def make_few_shot_examples(self, example_paths: List[str]):
        shots = []
        for idx, exp_path in enumerate(example_paths):
            shot_info = read_json(exp_path)
            scenario_id = self.get_scenario_num_from_path(exp_path)
            instructions, problem = self.make_instance(
                shot_info, shot_info.get("action_history", []), scenario_id=scenario_id
            )

            num = idx + 1
            example_tag = f"\n## Example{num} ##\n"

            shots.append({"role": "user", "content": example_tag + problem.strip()})
            shots.append({"role": "assistant", "content": self.make_response(shot_info).strip()})

        return shots
    

    def generate_prompt(self, predicates: Dict, scenario_id: str=None, history: List[str] = None):
        r"""
        Generate the prompt for the given predicates and history.
        args:
            predicates: dictionary of predicates
            history: list of actions taken so far
            scenario_id: str, id of the scenario
        return:
            output: list of dictionaries with keys 'role' and 'content'

        """
        ## NOTE(SJ) : If we want to use the search experience as the few-shot examples, include the action history when saving the state.
        
        
        ## Step1: We get the examples for the few-shot examples.
        example_paths = self.prompt_config.get("example_paths", [])
        shots = self.make_few_shot_examples(example_paths)


        ## Step2: We make the problem to solve.
        instructions, problem = self.make_instance(
            predicates, action_history=history, scenario_id=scenario_id
        )


        ## Step3: We combine the examples and the problem to solve.
        if len(shots) == 0:
            output = [
                {
                    "role": "user",
                    "content": instructions.strip()
                    + "\n\nThis is the problem you need to solve.\n"
                    + problem.strip(),
                }
            ]
        else:
            output = (
                [
                    {
                        "role": "user",
                        "content": instructions.strip()
                        + "\n\nHere are some examples.\n"
                        + shots[0]["content"],
                    }
                ]
                + shots[1:]
                + [
                    {
                        "role": "user",
                        "content": "This is the problem you need to solve.\n"
                        + problem.strip(),
                    }
                ]
            )

        return output
