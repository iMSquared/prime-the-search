import pprint
from typing import Any, Dict, List, Tuple, Union, Set
import re
import ast
import random
from copy import deepcopy


class EntityCentricState:
    def __init__(self, ):
        self.primary: str = 'properties' 
        self.agg_mode: str = ''
        self.as_dict_or_list: str = ''
        self.occ_agg_mode: str = ''
        self.occ_as_dict_or_list: str = ''


    def __call__(self, predicates, config) -> Any:
        self.set_attributes(config)
        goal, org_predicates = self.organize(predicates, 
                                             config)
        return self.to_string(goal), self.to_string(org_predicates)
    

    def to_string(self, predicates):
        # return pprint.pformat(predicates, indent=4, sort_dicts=False)
        return str(predicates)

    def set_attributes(self, config):
        self.agg_mode: str = config['agg_mode']
        self.as_dict_or_list: str = config['as_dict_or_list']
        self.occ_agg_mode: str = config['occ_agg_mode']
        self.occ_as_dict_or_list: str = config['occ_as_dict_or_list']
        
    def organize_goal(self, predicates):
        conditions = ['position']
        goal = {con:[] for con in conditions}


        for obj, region in zip(predicates['problem']['goal']['objects'], predicates['problem']['goal']['regions']):
            for _obj in obj:
                goal['position'].append({'subject':_obj, 'reference': region[0], 'spatial_relation':region[1]})
        return goal


    def process_occ_pre(self, name, predicates, agg_mode='seperate', as_dict_or_list='dict'):
        processed_occ_pre = []
        occ_pre = predicates['asset'][name]['is_occ_pre']
    
        processed_occ_pre = {'occluders': occ_pre}

        # procesed_oocc_pre = {'action': f'PICK {name}', 'occluders': occ_pre}
        return processed_occ_pre


    ## TODO (SJ): change the organization to new occ manip format
    def process_occ_manip(self, name, predicates, agg_mode='seperate', as_dict_or_list='dict'):
        occ_manip = predicates['asset'][name]['is_occ_manip']
                 

        processed_occ_manip = []
        for region, occluders in occ_manip.items():
            if type(region) == str: region = ast.literal_eval(region)
            processed_occ_manip.append({'reference': region[1], 'spatial_relation': region[0], 'occluders': list(occluders)})
            # processed_occ_manip.append({'action': f'PLACE {name} on {region}', 'occluders': occluders})
            # processed_occ_manip.append({'action': f'PLACE {name} on {region}', 'occluders': occluders})

        return processed_occ_manip
        

    def organize_object(self, name, predicates):
        attribute_name_swap = {'in_region':'at_region', 'is_occ_pre':'occluding_pick', 'is_occ_manip':'occluding_place_at'}
        

        object_attributes = {
            'is_movable': predicates['asset'][name]['is_movable'],
        }

        if object_attributes['is_movable']:
            object_attributes[attribute_name_swap['in_region']] = predicates['asset'][name]['in_region']
            object_attributes[attribute_name_swap['is_occ_pre']] = self.process_occ_pre(name, predicates, self.occ_agg_mode, self.occ_as_dict_or_list)
            object_attributes[attribute_name_swap['is_occ_manip']] = self.process_occ_manip(name, predicates, self.occ_agg_mode, self.occ_as_dict_or_list)

        else:
            pass

        return object_attributes


    def organize_region(self, name, predicates):
        ## TODO (SJ): add relative position of objects in the region        
            ## - [ ] LTR,  

        attribute_name_swap = {'region2objs': 'objects_in_region'}

        region_attributes = {
            'is_region': predicates['asset'][name]['is_region'],
        }
        if region_attributes['is_region']:
            ## TEMP (SJ)
            if 'door' in name: return region_attributes
            region_attributes[attribute_name_swap['region2objs']] = predicates['info']['region2objs'][name]
        else:
            pass

        return region_attributes


    def organize_door(self, name, predicates):
        attribute_name_swap = {'is_open': 'is_open'}
        
        door_attributes = {
            'is_openable': predicates['asset'][name]['is_openable']
        }

        if door_attributes['is_openable']:
            door_attributes[attribute_name_swap['is_open']] = predicates['asset'][name]['is_open']
        else: 
            pass 

        return door_attributes

    def dict_or_list(self, predicates, as_dict_or_list):
        if as_dict_or_list == 'dict':
            return predicates
        elif as_dict_or_list == 'list':
            predicate_list = []
            for key, value in predicates.items():
                # predicate_list.append({'name': key, 'attributes': value})
                predicate_list.append({'name': key, **value})
            return predicate_list

        else:
            raise NotImplementedError

    def filter_by_condition(self, predicates, target_attr, condition):
        return {name: attr for name, attr in predicates.items() if attr.pop(target_attr)==condition}

    

    def organize(self, predicates, config):
        self.set_attributes(config)

        goal = self.organize_goal(predicates)

        output = {
            'agent': {
                'objects_in_hand': predicates['info']['region2objs']['agent_hand'], 
                'is_there_available_hand': predicates['agent']['has_empty_hand']
            },
        }

        

        object_attr_buff, region_attr_buff, door_attr_buff = {}, {}, {}
        for name in predicates['asset'].keys():
            object_attr_buff[name] = self.organize_object(name, predicates)
            region_attr_buff[name] = self.organize_region(name, predicates)
            door_attr_buff[name] = self.organize_door(name, predicates)


        
        if self.agg_mode == 'seperate':
            object_attr_buff, region_attr_buff, door_attr_buff = self.filter_by_condition(object_attr_buff, 'is_movable', True), self.filter_by_condition(region_attr_buff, 'is_region', True), self.filter_by_condition(door_attr_buff, 'is_openable', True)
            output['objects'] = self.dict_or_list(object_attr_buff, self.as_dict_or_list)
            output['regions'] = self.dict_or_list(region_attr_buff, self.as_dict_or_list)
            output['doors'] = self.dict_or_list(door_attr_buff, self.as_dict_or_list)
           
        elif self.agg_mode == 'combined':
            combined_buff = {name: {**object_attr_buff[name], **region_attr_buff[name], **door_attr_buff[name]} for name in object_attr_buff.keys()}
            output['assets'] = self.dict_or_list(combined_buff, self.as_dict_or_list)
        else:
            raise NotImplementedError(f"agg_mode: {self.agg_mode} is not implemented")


        return goal, output
    

class PredicateCentricState:
    def __init__(self):
        self.goal_related = False

    def __call__(self, predicates, config, dump_prompt:bool = False) -> Any:
        # self.set_attributes(config)
        goal, org_predicates = self.organize(predicates, 
                                             **config)
        
        if self.goal_related:
            # self.filter_goal_related_info(org_predicates, predicates)
            org_predicates = self.rearrange_goal_related(org_predicates, predicates)

        return self.to_string(goal, dump_prompt=dump_prompt), self.to_string(org_predicates, dump_prompt=dump_prompt)
    
    def to_string(self, predicates, dump_prompt:bool = False):
        output: str = ""
        if dump_prompt:
            output = pprint.pformat(predicates, indent=4, sort_dicts=False)
        else:
            output = str(predicates)
        return output

    def process_movable_objects(self, predicates, **config):
        output = []
        for obj in predicates['problem']['is_movable']:
            output.append(obj)
        return output
    
    def process_regions(self, predicates, **config):
        output = []
        for region in predicates['problem']['is_region']:
            output.append(region)
        return output
    
    def process_doors(self, predicates, **config):
        output = []
        for door in predicates['problem']['is_openable']:
            output.append(door)
        return output
    
    def process_open(self, predicates, **config):
        output = []
        for door in predicates['problem']['is_openable']:
            if predicates['asset'][door]['is_open']:
                output.append(door)
        return output
    
    def process_closed(self, predicates, **config):
        output = []
        for door in predicates['problem']['is_openable']:
            if not predicates['asset'][door]['is_open']:
                output.append(door)
        return output
    
    def process_occluding_pick(self, predicates, **config):
        output = {}
        # for obj, attr in predicates['info']['is_movable'].items():
        for obj in predicates['problem']['is_movable']:
            attrib = predicates['asset'][obj]
            output[obj] = (len(attrib['is_occ_pre'])!=0, attrib['is_occ_pre'])
            # output[obj] = attrib['is_occ_pre']
            

        return output



    def process_occluding_place(self, predicates, **config):
        output = {}

        for obj in predicates['problem']['is_movable']:
            attrib = predicates['asset'][obj]
            for region, occluders in attrib['is_occ_manip'].items():
                
                output[(obj,region[0],region[1])] = (len(occluders)!=0, occluders)
                # output[(obj,region[0],region[1])] = occluders


        # occ_manip = predicates['asset'][name]['is_occ_manip']
        # processed_occ_manip = []
        # for region, occluders in occ_manip.items():
        #     processed_occ_manip.append({'destination': region, 'occluders': occluders})
            # processed_occ_manip.append({'action': f'PLACE {name} on {region}', 'occluders': occluders})

        return output

    def process_objects_on_region(self, predicates, **config):
        output = {}
        for name, attr in predicates['info']['region2objs'].items():
            if 'agent_hand' in name: continue
            output[name] = {'on': attr}
            output[name].update(predicates['asset'][name]['layout'])
        
        return output



    def holding(self, predicates, **config):
        output = []
        for obj in predicates['info']['region2objs']['agent_hand']:
            output.append(obj)
        return output
    
    def has_empty_hand(self, predicates, **config):
        return predicates['agent']['has_empty_hand']


    def organize(self, predicates, **config):
        goal = {'position':[]}
        for objs, region in zip(predicates['problem']['goal']['objects'], predicates['problem']['goal']['regions']):
            for obj in objs:
                # goal['position'].append((obj, region[1], region[0]))
                goal['position'].append({'subject': obj, 'spatial_relation': region[1], 'reference': region[0]})


        output = {
            'movable_objects': self.process_movable_objects(predicates),
            'regions': self.process_regions(predicates),
            'doors': self.process_doors(predicates),
            'closed': self.process_open(predicates),
            'is_pick_occluded': self.process_occluding_pick(predicates),
            'is_place_occluded': self.process_occluding_place(predicates),
            'objects_on_region': self.process_objects_on_region(predicates),
            'holding': self.holding(predicates),
            'has_empty_hand': self.has_empty_hand(predicates),
        }

        return goal, output
    

    def filter_goal_related(self, predicates: Dict[str, List]):
        # Find goal-related objects and regions
        goal = predicates['problem']['goal'] 
        goal_related_objs: List = []
        for objs in goal['objects']:
            goal_related_objs += objs
        
        goal_references = []
        for objs, ref in zip(goal['objects'], goal['regions']):
            for obj in objs:
                goal_references.append([obj] + ref)

        # initialize output
        output = {
            'info': predicates['info'],
            'problem': predicates['problem'],
        }

        # Step1: Filter holding
        right_hand_holding = [obj for obj in predicates['agent']['right_hand_holding'] if obj in goal_related_objs]
        left_hand_holding = [obj for obj in predicates['agent']['left_hand_holding'] if obj in goal_related_objs]
        agent = deepcopy(predicates['agent'])
        agent['right_hand_holding'] = right_hand_holding
        agent['left_hand_holding'] = left_hand_holding
        
        # Step1.5: Prepare asset predicates
        asset: Dict[str, Union[List, bool]] = deepcopy(predicates['asset'])
        for obj, info in asset.items():
            info['is_occ_pre'] = []
            info['is_occ_manip'] = dict()

        # Step2: Filter pick occluded
        obj_queue = goal_related_objs
        memory = set(goal_related_objs)
        filtered_pick_occluded: List[Tuple[str]] = []
        organized = PredicateCentricState.process_occluding_pick(self, predicates)
        while len(obj_queue) > 0:
            occlusions: Dict[str, List] = {obj: info for obj, info in organized.items() if obj in obj_queue and info[0]}
            obj_queue = []
            # Expand
            for obj, (_, occluders) in occlusions.items():
                for o in occluders:
                    filtered_pick_occluded.append((obj, o))
                    if o not in memory:
                        obj_queue.append(o)
                        memory.add(o)
        # Merge
        for obj, occluder in filtered_pick_occluded:
            asset[obj]['is_occ_pre'].append(occluder)

        del obj_queue, memory, organized

        # Step3: filter place occluded
        obj_queue = goal_related_objs
        memory = set(goal_related_objs)
        filtered_place_occluded: List[Tuple[str]] = []
        organized = PredicateCentricState.process_occluding_place(self, predicates)
        while len(obj_queue) > 0:
            occlusions: Dict[str, List] = {(obj, dir, ref): info for (obj, dir, ref), info in organized.items() if obj in obj_queue and info[0]}
            obj_queue = []
            # Expand
            for (obj, dir, ref), (_, occluders) in occlusions.items():
                if [obj, ref, dir] in goal_references or (obj not in goal_related_objs and dir == 'on'):
                    for o in occluders:
                        filtered_place_occluded.append((obj, dir, ref, o))
                        if o not in memory:
                            obj_queue.append(o)
                            memory.add(o)
        # Merge
        for obj, dir, ref, occluder in filtered_place_occluded:
            if (dir, ref) not in asset[obj]['is_occ_manip']:
                asset[obj]['is_occ_manip'][(dir, ref)] = [occluder]
            else:
                asset[obj]['is_occ_manip'][(dir, ref)].append(occluder)

        del obj_queue, memory, organized

        # Step4: filter relative position
        for obj, info in asset.items():
            if obj not in goal_related_objs:
                info['relative_position'] = {'left': [], 'right': [],'front': [], 'behind': []}

        # Return
        output['agent'] = agent
        output['asset'] = asset

        return output


    def rearrange_goal_related(self, organized: Dict[str, List], unorganized: Dict[str, List], num_item: int=20):
        
        # Find goal-related objects and regions
        goal = unorganized['problem']['goal'] 
        goal_related_objs: List = []
        for objs in goal['objects']:
            goal_related_objs += objs
        
        goal_references = []
        for objs, ref in zip(goal['objects'], goal['regions']):
            for obj in objs:
                goal_references.append([obj] + ref)
        
        # Initialize output
        output = {
            'objects': organized['objects'],
            'regions': organized['regions'],
            'doors': organized['doors'],
            'closed': organized['closed'],
            'has_empty_hand': organized['has_empty_hand']
        }

        # Step1: Filter holding
        holding = [obj for obj in organized['holding'] if obj in goal_related_objs]
        if num_item > 0 and num_item < len(holding):
            holding = random.choices(holding, k=num_item)

        # Step2: Filter object position
        object_positions = {obj: info for obj, info in organized['object_positions'].items() if obj in goal_related_objs}
        if num_item > 0 and num_item < len(object_positions):
            object_positions = random.choices(holding, k=num_item)

        # Step3: Filter pick occluded
        obj_queue = goal_related_objs
        memory = set(goal_related_objs)
        filtered_pick_occluded: List[Tuple[str]] = []
        while len(obj_queue) > 0:
            occlusions: Dict[str, List] = {obj: info for obj, info in organized['is_pick_occluded'].items() if obj in obj_queue and info[0]}
            obj_queue = []
            # Expand
            for obj, (_, occluders) in occlusions.items():
                for o in occluders:
                    filtered_pick_occluded.append((obj, o))
                    if o not in memory:
                        obj_queue.append(o)
                        memory.add(o)
        # N items
        if num_item > 0 and num_item < len(filtered_pick_occluded):
            filtered_pick_occluded = filtered_pick_occluded[:num_item]
        # merge
        is_pick_occluded: Dict[str, List] = dict()
        for occludee, occluder in filtered_pick_occluded:
            if occludee not in is_pick_occluded:
                is_pick_occluded[occludee] = [occluder]
            else:
                is_pick_occluded[occludee].append(occluder)

        del obj_queue, memory

        # Step4: filter place occluded
        obj_queue = goal_related_objs
        memory = set(goal_related_objs)
        filtered_place_occluded: List[Tuple[str]] = []
        while len(obj_queue) > 0:
            occlusions: Dict[str, List] = {(obj, dir, ref): info for (obj, dir, ref), info in organized['is_place_occluded'].items() if obj in obj_queue and info[0]}
            obj_queue = []
            # Expand
            for (obj, dir, ref), (_, occluders) in occlusions.items():
                if [obj, ref, dir] in goal_references or (obj not in goal_related_objs and dir == 'on'):
                    for o in occluders:
                        filtered_place_occluded.append((obj, dir, ref, o))
                        if o not in memory:
                            obj_queue.append(o)
                            memory.add(o)
                # N items
        if num_item > 0 and num_item < len(filtered_place_occluded):
            filtered_place_occluded = filtered_place_occluded[:num_item]
        # merge
        is_place_occluded: Dict[str, List] = dict()
        for obj, dir, ref, occluder in filtered_place_occluded:
            if (obj, dir, ref) not in is_place_occluded:
                is_place_occluded[(obj, dir, ref)] = [occluder]
            else:
                is_place_occluded[(obj, dir, ref)].append(occluder)

        del obj_queue, memory
        
        # Fill out the output
        output['holding'] = holding
        output['object_positions'] = object_positions
        output['is_pick_occluded'] = is_pick_occluded
        output['is_place_occluded'] = is_place_occluded

        return output
        

class PredicateCentricStateV2(PredicateCentricState):
    def __init__(self, shrink=False):
        super().__init__()
        
        self.counterpart = {'right':'left_of', 'left':'right_of', 'front':'behind_of', 'behind':'front_of'}
        self.shrink = shrink


    def process_object_positions(self, predicates, **config):
        output = {}

        for name, objs in predicates['info']['region2objs'].items():
            # if 'agent_hand' in name: continue
            for obj in objs:
                output[obj] = {'on_region': name,
                               'left_of_objects': predicates['asset'][obj]['relative_position']['right'],
                               'right_of_objects': predicates['asset'][obj]['relative_position']['left'],
                               'front_of_objects': predicates['asset'][obj]['relative_position']['behind'],
                               'behind_of_objects': predicates['asset'][obj]['relative_position']['front'],
                               }
        return output

    def process_placement_pose(self, predicates, **config):

        output = {}

        for obj in predicates['problem']['is_movable']:
            attrib = predicates['asset'][obj]
            for region, is_or_not in attrib['has_placement_pose'].items():
                
                output[(obj,region[0],region[1])] = is_or_not

        # occ_manip = predicates['asset'][name]['is_occ_manip']
        # processed_occ_manip = []
        # for region, occluders in occ_manip.items():
        #     processed_occ_manip.append({'destination': region, 'occluders': occluders})
            # processed_occ_manip.append({'action': f'PLACE {name} on {region}', 'occluders': occluders})

        return output

    def organize(self, predicates, **config):
        goal = {'position':[]}
        for objs, region in zip(predicates['problem']['goal']['objects'], predicates['problem']['goal']['regions']):
            for obj in objs:
                # goal['position'].append((obj, region[1], region[0]))
                goal['position'].append({'subject': obj, 'spatial_relation': region[1], 'reference': region[0]})


        output = {
            'objects': self.process_movable_objects(predicates),
            'regions': self.process_regions(predicates),
            'doors': self.process_doors(predicates),
            'closed': self.process_closed(predicates),
            'is_pick_occluded': self.process_occluding_pick(predicates),
            'is_place_occluded': self.process_occluding_place(predicates),
            # 'has_placement_pose': self.process_placement_pose(predicates),
            'object_positions' : self.process_object_positions(predicates),
            'holding': self.holding(predicates),
            'has_empty_hand': self.has_empty_hand(predicates),
        }


        return goal, output
    
    def process_occluding_pick(self, predicates, **config):
        output = super().process_occluding_pick(predicates, **config)

        # Shrink unnecessary predicates with ellipsis
        if self.shrink:
        # Filter
            filtered_output = dict(filter(lambda kv: len(kv[1][1]) > 0, output.items()))
            # Add ellipsis
            if len(filtered_output) < len(output):
                filtered_output["..."] = (False, [])
            output = filtered_output

        return output


    
    def process_occluding_place(self, predicates, **config):
        output = super().process_occluding_place(predicates, **config)

        # Shrink unnecessary predicates with ellipsis
        if self.shrink:
            # Filter
            filtered_output = dict(filter(lambda kv: len(kv[1][1]) > 0, output.items()))
            # Add ellipsis
            if len(filtered_output) < len(output):
                filtered_output[("...", "...", "...")] = (False, [])
            output = filtered_output


        return output

class PDDLStyleState(PredicateCentricState):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.truncate_elements = -1
        self.goal_related = True
        self.python_compatible = True
        

    def __call__(self, predicates, config, dump_prompt: bool = False) -> Any:
        # assert False, "Not implemented"
        goal, org_predicates = self.organize(predicates,**config)
            
        return self.to_string(goal, dump_prompt=dump_prompt), self.to_string(org_predicates, dump_prompt=dump_prompt)

    def flatten_input2list(self, predicates):
        output = []
        if type(predicates) == list:
            output = predicates
        elif type(predicates) == dict:
            for _, value in predicates.items():
                if type(value) == list:
                    output.extend(value)
                elif type(value) == str:
                    output.append(value)
                elif type(value) == dict:
                    for _, v in value.items():
                        output.append(v)
                else:
                    raise NotImplementedError(f"Type: {type(value)} is not implemented")
        else:
            raise NotImplementedError(f"Type: {type(predicates)} is not implemented")
        
        return output
    
    def to_string(self, predicates, dump_prompt: bool = False):
        output: str = ""

        # if type(predicates) == list:
        #     try:
        #         output = " ".join(predicates)
        #     except:
        #         output = str(predicates)
        # else: 
        output = str(predicates)
    
        return output
    
    def truncate_with_ellipsis(self, item, num_item):
        if num_item > 0 and num_item < len(item):
            return item[:num_item] + ["..."]
        else:
            return item


    def make_goals(self, predicates, **config):

        goals = zip(predicates['problem']['goal']['objects'], predicates['problem']['goal']['regions'])
        buffers = []
        for objs, region in goals:
            for obj in objs:
                # buffers.append(direction_pair[region[1]].format(sub=obj, ref=region[0]))
                buffers.append(('AtPosition', obj, region[1], region[0]))
    
        return buffers
    
    def process_closed(self, predicates, **config):
        output = []
        template = "{door}"
        
        for door in predicates["problem"]["is_openable"]:
            if door in predicates["asset"] \
            and "is_open" in predicates["asset"][door] \
            and not predicates["asset"][door]["is_open"]:
                output.append((door, ))
        return output


    def process_occluding_pick(self, predicates, **config):
        output = []
        template = "{obj} {occluder}"

        for obj in predicates["problem"]["is_movable"]:
            if obj in predicates["asset"]:
                attrib = predicates["asset"][obj]
                if "is_occ_pre" in attrib:
                    for occluder in attrib["is_occ_pre"]:
                        output.append((obj, occluder))
                        # output.append(template.format(obj=obj, occluder=occluder))
                        
        return output


    def process_occluding_place(self, predicates, **config):
        output = []
        # template = "{obj} {spatial_relation} {region} {occluder}"
        for obj in predicates["problem"]["is_movable"]:
            if obj in predicates["asset"]:
                attrib = predicates["asset"][obj]
                if "is_occ_manip" in attrib:
                    for region, occluders in attrib["is_occ_manip"].items():
                        for occluder in occluders:
                            output.append((obj, region[0], region[1], occluder))
                            # output.append(template.format(obj=obj, spatial_relation = region[0], region=region[1], occluder=occluder))

        return output


    def process_object_positions(self, predicates, **config):
        output = []
        templateOn = "{obj} on {region}"
        templateLeft = "{obj} left_of {ref}"
        templateRight = "{obj} right_of {ref}"
        templateFront = "{obj} front_of {ref}"
        templateBehind = "{obj} behind_of {ref}"

        for name, objs in predicates["info"]["region2objs"].items():
            if "agent_hand" in name: continue
            for obj in objs:
                # output.append(templateOn.format(obj=obj, region=name))
                output.append((obj, "on", name))
            for obj in objs:
                if obj in predicates["asset"]:
                    for ref in predicates["asset"][obj]["relative_position"]["right"]:
                        # output.append(templateRight.format(obj=obj, ref=ref))
                        output.append((obj, "right_of", ref))
                    for ref in predicates["asset"][obj]["relative_position"]["left"]:
                        # output.append(templateLeft.format(obj=obj, ref=ref))
                        output.append((obj, "left_of", ref))
                    for ref in predicates["asset"][obj]["relative_position"]["behind"]:
                        # output.append(templateBehind.format(obj=obj, ref=ref))
                        output.append((obj, 'behind_of', ref))
                    for ref in predicates['asset'][obj]['relative_position']['front']:
                        # output.append(templateFront.format(obj=obj, ref=ref))
                        output.append((obj, "front_of", ref))

        return output


    def holding(self, predicates, **config):
        output = []
        template = "{obj}"
        for obj in predicates['info']['region2objs']['agent_hand']:
            # output.append(template.format(obj=obj))
            output.append((obj, ))

        return output


    def has_empty_hand(self, predicates, **config):
        output = []
        template = ""
        if predicates['agent']['has_empty_hand']:
            # output.append(template)
            output.append(("",))
            
        return output


    def robot_at(self, predicates, **config):
        output = []
        template = "{region}"
        if predicates['agent']['at']:
            # output.append(template.format(region=predicates['agent']['at']))
            output.append((predicates['agent']['at'],))

        return output
    
    # def process_state_output(self, raw_output):
    #     output = []
    #     for key, value in raw_output.items():
    #         predicate_name = f"('{key}', "
    #         for _value in value:
    #             for _v in _value:
    #                 predicate_name += f"'{_v}', "
    #                 output.append(predicate_name[:-2] + ")")

    #     return output
    
    def process_state_output(self, raw_output):
        output = []
        for key, value in raw_output.items():
            for _value in value:
                _buffer = [key, *_value]
                buffer = [f"{nn}" for nn in _buffer]
                output.append(tuple(buffer))

        return output
    

    def organize(self, predicates, **config):
        self.goal_related = config.get('goal_related', False)
        self.truncate_elements = config.get('truncate_elements', -1)
        self.python_compatible = config.get('python_compatible', True)


        if self.goal_related:
            predicates = self.filter_goal_related(predicates)

        goal = self.make_goals(predicates, **config)



        output = {
            'RobotAt': self.robot_at(predicates),
            'RobotHolding': self.holding(predicates),
            'HandAvailable': self.has_empty_hand(predicates),
            'AtPosition': self.process_object_positions(predicates),
            'PickOccludedBy': self.process_occluding_pick(predicates),
            'PlaceOccludedBy': self.process_occluding_place(predicates), 
            'Closed': self.process_closed(predicates),
        }

        state_output = self.process_state_output(output)

        if self.truncate_elements > 0:
            output = {k: self.truncate_with_ellipsis(v, self.truncate_elements) for k, v in output.items()}


        # output = self.flatten_input2list(output)
        
        return goal, state_output
    



"""
- Sample Pose
- Sample Grasp 
- Sample Arm IK
- Sample Base Path 
"""