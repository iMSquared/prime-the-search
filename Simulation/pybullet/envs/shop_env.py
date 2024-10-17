import os, sys
import math
from typing import Dict, Tuple, List, Union, Optional
import numpy as np
import numpy.typing as npt
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy
import random
import yaml
from math import inf
import json, pickle

import pybullet_data
from Simulation.pybullet.envs.robot import PR2, AttachConstraint
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient, suppress_stdout
from Simulation.pybullet.imm.pybullet_tools.utils import set_point, create_box, Point, \
                                                                         TAN, TRANSPARENT_GREY, YELLOW, PI, \
                                                                         draw_aabb, DARK_GREY, TAN, add_segments


def translate(cord: Tuple[float], shift: Tuple[float]) -> Tuple[float]:
    cord = np.array(cord)
    shift = np.array(shift)

    return (cord + shift).tolist()

@dataclass(frozen=True)
class PickParams:
    points: List[List[float]]
    angles: List[List[float]]
    backward: float
    forward: float


class ShopObjectEntry:

    def __init__(self, bc: BulletClient,
                 urdf_dir_path: str,
                 name: str, 
                 config: Dict, 
                 area: str, 
                 region: str = None,
                 position: npt.ArrayLike = None,
                 orientation: npt.ArrayLike = None,
                 suppress_output: bool = True):
        self.bc: BulletClient = bc
        self.name: str = name
        self.position: List = config[name]["pos"] if position is None else position
        self.orientation: List = config[name]["orn"] if orientation is None else orientation  # Orientation in Euler
        self.is_movable: bool = not config[name]["static"]
        self.is_region: bool = config[name]["IsRegion"]
        self.is_openable: bool = config[name]["IsOpenable"]
        self.is_receptacle: bool = config[name]["IsReceptacle"]

        # Location
        self.area: str = area
        self.region: str = region


        # Only applies to doors
        self.is_open = False

        # Load the asset to the bullet
        if suppress_output:
            with suppress_stdout():
                self.uid = self.bc.loadURDF(
                    fileName        = os.path.join(urdf_dir_path, config[name]["path"]),
                    basePosition    = self.position,
                    baseOrientation = self.bc.getQuaternionFromEuler(self.orientation) if len(self.orientation) == 3 else self.orientation,
                    globalScaling   = config[name]["scale"],
                    useFixedBase    = (not self.is_movable))
        else:
            self.uid = self.bc.loadURDF(
                    fileName        = os.path.join(urdf_dir_path, config[name]["path"]),
                    basePosition    = self.position,
                    baseOrientation = self.bc.getQuaternionFromEuler(self.orientation) if len(self.orientation) == 3 else self.orientation,
                    globalScaling   = config[name]["scale"],
                    useFixedBase    = (not self.is_movable))
        
        self.pick_param = PickParams(**config[name]["pick_param"])

        
        if self.is_region:
            self.robot_offset = config[name]["offset"]
            self.taskspace = config[name]["taskspace"]  # in AABB
            if not self.is_movable: 
                self.region = self.name
            self.entity_group = config[name]["EntityGroup"]
            
            
        else:
            self.robot_offset = None
            self.taskspace = None
            self.entity_group = None

        # receptacle only
        self.num_items = 0
    

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result.bc = self.bc
        result.name = deepcopy(self.name)
        result.position = deepcopy(self.position)
        result.orientation = deepcopy(self.orientation)
        result.is_movable = deepcopy(self.is_movable)
        result.is_region = deepcopy(self.is_region)
        result.is_openable = deepcopy(self.is_openable)
        result.is_receptacle = deepcopy(self.is_receptacle)
        result.area = deepcopy(self.area)
        result.region = deepcopy(self.region)
        result.is_open = deepcopy(self.is_open)
        result.uid = deepcopy(self.uid)
        result.pick_param = deepcopy(self.pick_param)
        result.robot_offset = deepcopy(self.robot_offset)
        result.taskspace = deepcopy(self.taskspace)
        result.entity_group = deepcopy(self.entity_group)
        result.num_items = deepcopy(self.num_items)

        return result    


    ## NOTE (SJ): Considering to move these functions to Predicate class
    @staticmethod
    def filter_region(object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_region == condition, object_dict.items()))
    
    @staticmethod
    def filter_movable(object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_movable == condition, object_dict.items()))
    
    @staticmethod
    def filter_openable(object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_openable == condition, object_dict.items()))

    @staticmethod
    def filter_receptacle(object_dict: Dict[str, "ShopObjectEntry"], condition: bool = True):
        return dict(filter(lambda x: x[1].is_receptacle == condition, object_dict.items()))

    @staticmethod
    def find_default_robot_base_pose(robot: PR2,
                                     region: "ShopObjectEntry", 
                                     config: Dict, 
                                     all_obj: Dict[str, "ShopObjectEntry"]):
        
        base_name = config[region.name]["EntityGroup"][0]
        pos = all_obj[base_name].position
        pos = np.array(pos)
        pos[2] = robot.get_position()[2]
        pos = tuple(pos)
        orn = all_obj[base_name].orientation

        # Compute offset
        if len(np.array(region.robot_offset).shape) > 2:
            distances = np.linalg.norm(np.array(region.robot_offset)[:,0,:] - robot.get_position(), axis=1)
            idx = np.argmin(distances)

            pos_offset, orn_offset = region.robot_offset[idx]

        else:
            pos_offset = region.robot_offset[0]
            orn_offset = region.robot_offset[1]

        if len(orn) == 3:
            orn = region.bc.getQuaternionFromEuler(orn)
        if len(orn_offset) == 3:
            orn_offset = region.bc.getQuaternionFromEuler(orn_offset)


        default_robot_pose = region.bc.multiplyTransforms(pos,
                                                          orn,
                                                          pos_offset,
                                                          orn_offset)

        return default_robot_pose
    

    @staticmethod
    def find_region(obj: "ShopObjectEntry", regions: Dict[str, "ShopObjectEntry"], all_obj: Dict[str, "ShopObjectEntry"], obj_height: float=0.3):
        assert obj.is_movable, "Should only query movable object's"

        position, _ = obj.bc.getBasePositionAndOrientation(obj.uid)
        min_region = (None,-inf)

        for region_name, region_info in regions.items():
            ## TODO (SJ): This is a hack to avoid receptacle
            if "kitchen_door" in region_name:
                continue
            if region_info.is_receptacle:
                continue
            basename = region_info.entity_group[0]
            region_pos = all_obj[basename].position
            region_orn = obj.bc.getQuaternionFromEuler(all_obj[basename].orientation)

            t_min, _ = obj.bc.multiplyTransforms(region_pos,
                                               region_orn,
                                               region_info.taskspace[0],
                                               obj.bc.getQuaternionFromEuler((0,0,0)))
            t_max, _ = obj.bc.multiplyTransforms(region_pos,
                                               region_orn,
                                               region_info.taskspace[1],
                                               obj.bc.getQuaternionFromEuler((0,0,0)))

            taskspace = (t_min, t_max)

            # Check XY
            x = position[0] >= min(taskspace[0][0], taskspace[1][0]) and position[0] <= max(taskspace[0][0], taskspace[1][0])
            y = position[1] >= min(taskspace[0][1], taskspace[1][1]) and position[1] <= max(taskspace[0][1], taskspace[1][1])

            # Check Z
            z = position[2] >= min(taskspace[0][2],taskspace[1][2]) and position[2] <= (max(taskspace[0][2],taskspace[1][2]) + obj_height)

            if x and y and z:
                if min(taskspace[0][2],taskspace[1][2]) > min_region[1]:
                    min_region = (region_name, min(taskspace[0][2],taskspace[1][2]))
            
        return min_region[0]
    
    @staticmethod
    def get_aabb(bc: BulletClient, obj: "ShopObjectEntry"):
        uid = obj.uid
        if obj.is_region:
            aabb = bc.getAABB(uid)
        elif obj.is_movable:
            aabb = bc.getAABB(uid, 0)
    
        return aabb
    
    @staticmethod
    def draw_bbox(bc: BulletClient, obj: "ShopObjectEntry"):
        aabb = ShopObjectEntry.get_aabb(bc, obj)
        draw_aabb(aabb)

    @staticmethod
    def get_obj_margin(bc: BulletClient, obj: "ShopObjectEntry", goal_orn: Tuple[float], ignore_z=True) -> Tuple[float]:
        original_pos, original_orn = bc.getBasePositionAndOrientation(obj.uid)
        bc.resetBasePositionAndOrientation(obj.uid, original_pos, goal_orn)
        
        aabb = np.array(ShopObjectEntry.get_aabb(bc, obj))
        margin = (aabb[1] - aabb[0])/2
        if ignore_z:
            margin[2] = 0

        bc.resetBasePositionAndOrientation(obj.uid, original_pos, original_orn)

        return tuple(margin)
    
    def draw_taskspace(bc: BulletClient, env: "ShopEnv", obj: "ShopObjectEntry", region: "ShopObjectEntry", eps: float=0.02):
        aabb = np.array(region.taskspace)
        margin = ShopObjectEntry.get_obj_margin(bc, obj, obj.orientation)
        bonus = np.array([eps, eps, 0])
        aabb[0] += (margin + bonus)
        aabb[1] -= (margin + bonus)
        
        region_pos = env.all_obj[env.regions[region.name].entity_group[0]].position
        region_orn = env.all_obj[env.regions[region.name].entity_group[0]].orientation
        region_orn = region_orn if len(region_orn) == 4 else bc.getQuaternionFromEuler(region_orn)

        mn = bc.multiplyTransforms(region_pos, region_orn, aabb[0], bc.getQuaternionFromEuler((0,0,0)))[0]
        mx = bc.multiplyTransforms(region_pos, region_orn, aabb[1], bc.getQuaternionFromEuler((0,0,0)))[0]

        draw_aabb((mn, mx))

        return (mn, mx)

class ShopDebugEnv:

    def __init__(self, bc, config):
        self.bc = bc
        self.config = config
        env_params = config["env_params"]["shop_env"]

        # Configs
        CUSTOM_URDF_DIR_PATH: str          = config["project_params"]["custom_urdf_path"]
        

    

        # Path to URDF
        pybullet_data_path = pybullet_data.getDataPath()
        file_path = Path(__file__)
        project_path = file_path.parent.parent
        urdf_dir_path = os.path.join(project_path, CUSTOM_URDF_DIR_PATH)
        self.urdf_dir_path = urdf_dir_path


        # Load environmental URDFs
        self.plane_uid = self.bc.loadURDF(
            fileName        = os.path.join(pybullet_data_path, "plane.urdf"), 
            basePosition    = (0.0, 0.0, 0.0), 
            baseOrientation = bc.getQuaternionFromEuler((0.0, 0.0, 0.0)),
            useFixedBase    = True)
        
        # Load objects information
        self.handles = dict()
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg", "config_shop.yaml"), "r") as f:
            self.shop_config: Dict[str, Dict] = yaml.load(f, Loader=yaml.FullLoader)
        

        # Load Regions
        self.regions_in_kitchen = {k: v for k, v in self.shop_config["kitchen_config"].items() if v["IsRegion"]}
        self.regions_in_hall = {k: v for k, v in self.shop_config["hall_config"].items() if v["IsRegion"]}
        self.regions = {**self.regions_in_kitchen, **self.regions_in_hall}
        self.movable_in_kitchen = {k: v for k, v in self.shop_config["kitchen_config"].items() if not v["static"]}
        self.movable_in_hall = {k: v for k, v in self.shop_config["hall_config"].items() if not v["static"]} 
        self.movable = {**self.movable_in_kitchen, **self.movable_in_hall}
        self.uid_to_name = dict()
        self.name_to_uid = dict()


    def load_region(self, name: str):
        # Determin area
        if name in self.regions_in_kitchen:
            area = "kitchen"
        elif name in self.regions_in_hall:
            area = "hall"
        else:
            print("wrong region name")
            return -1
        
        # Collect related entities
        obj_names = self.shop_config[f"{area}_config"][name]["EntityGroup"]

        # Load entities
        for i, obj in enumerate(obj_names):
            v = self.shop_config[f"{area}_config"][obj]
            if i == 0:
                # offset = deepcopy(v["pos"])
                offset_pos, offset_orn = self.bc.invertTransform(v["pos"], self.bc.getQuaternionFromEuler(v["orn"]))

            try:
                pos, orn = self.bc.multiplyTransforms(offset_pos, offset_orn, v["pos"], self.bc.getQuaternionFromEuler(v["orn"]))

                uid = self.bc.loadURDF(
                    fileName        = os.path.join(self.urdf_dir_path, v["path"]),
                    basePosition    = pos,
                    baseOrientation = orn,
                    globalScaling = v["scale"],
                    useFixedBase    = v["static"])
                self.uid_to_name[uid] = obj
                self.name_to_uid[obj] = uid
            except Exception as e:
                print(f"[KitchenEnvironment] Failed to load {obj}")


    def load_movable(self, name: str):
        # Determin area
        if name in self.movable_in_kitchen:
            area = "kitchen"
        elif name in self.movable_in_hall:
            area = "hall"
        else:
            print("wrong region name")
            return -1

        # Load entities
        v = self.movable[name]
        try:
            uid = self.bc.loadURDF(
                fileName        = os.path.join(self.urdf_dir_path, v["path"]),
                basePosition    = (0, 0, 0),
                baseOrientation = self.bc.getQuaternionFromEuler((0, 0, 0)),
                globalScaling = v["scale"],
                useFixedBase    = v["static"])
            self.uid_to_name[uid] = name
            self.name_to_uid[name] = uid
        except:
            print(f"[KitchenEnvironment] Failed to load {name}")


    def regionAABB(self, name: str):
        # Determin area
        assert name in self.name_to_uid, "Wrong region name"
        
        # Collect related entities
        uid = self.name_to_uid[name]
        aabb = self.bc.getAABB(bodyUniqueId=uid)

        return aabb
    
    def movableAABB(self, name: str):
        # Determin area
        assert name in self.name_to_uid, "Wrong region name"
        
        # Collect related entities
        uid = self.name_to_uid[name]
        aabb = self.bc.getAABB(uid, 0)

        return aabb




class ShopEnv:
    def __init__(self, bc, config, suppress_output: bool=True):
        self.bc = bc
        self.config = config

        # Configs
        CUSTOM_URDF_DIR_PATH: str          = config["project_params"]["custom_urdf_path"]


        # Path to URDF
        pybullet_data_path = pybullet_data.getDataPath()
        file_path = Path(__file__)
        project_path = file_path.parent.parent
        urdf_dir_path = os.path.join(project_path, CUSTOM_URDF_DIR_PATH)
        self.urdf_dir_path = urdf_dir_path


        # Load environmental URDFs
        self.plane_uid = self.bc.loadURDF(
            fileName        = os.path.join(pybullet_data_path, "plane.urdf"), 
            basePosition    = (0.0, 0.0, 0.0), 
            baseOrientation = bc.getQuaternionFromEuler((0.0, 0.0, 0.0)),
            useFixedBase    = True)
        
        # Load objects information
        self.handles: Dict[str, Tuple] = dict()

        scenario = self.config["problem_params"]["scenario"]
        self.scenario_id = str(scenario)
        
        variant = False
        if isinstance(scenario, str) and 'v' in scenario:
            scenario = scenario[1:]
            variant = True

        if scenario is None:
            config_shop_filename = "config_shop.yaml"
        else:
            if variant:
                config_shop_filename = f"scenarios/variants/scenario{scenario}.yaml"
            else:
                config_shop_filename = f"scenarios/scenario{scenario}.yaml"

        
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cfg", config_shop_filename), "r") as f:
            self.shop_config: Dict[str, Dict] = yaml.load(f, Loader=yaml.FullLoader)


        ## TODO (SJ): Fix into config later
        self.target_plan_length = self.shop_config.get("optimal_length", 20)
        door_duration = self.shop_config.get("door_duration", 4)
        self.object_info = {**self.shop_config["kitchen_config"], **self.shop_config["hall_config"]}

        walls, moObst, door = self.set_room(doNAMO=False, suppress_output=suppress_output)

        self.walls_uid = {f"wall_{i}": uid for i, uid in enumerate(walls)}
        self.moObst = {f"moObst_{i}": uid for i, uid in enumerate(moObst)}
        
        kitchen = self.set_kitchen_area(suppress_output=suppress_output)
        hall = self.set_hall_area(suppress_output=suppress_output)

        self.all_obj: Dict[str, ShopObjectEntry] = {**kitchen, **hall, "kitchen_door": door}
        self.rooms = {"kitchen": kitchen, "hall": hall}
        self.regions = ShopObjectEntry.filter_region(self.all_obj)
        self.movable_obj = ShopObjectEntry.filter_movable(self.all_obj)
        self.openable_obj = ShopObjectEntry.filter_openable(self.all_obj)
        self.receptacle_obj = ShopObjectEntry.filter_receptacle(self.all_obj)

        # Stabilize
        for i in range(100):
            self.bc.stepSimulation()

        for name, obj_info in self.movable_obj.items():
            region_name = ShopObjectEntry.find_region(obj_info, self.regions, self.all_obj)
            obj_info.region = region_name

            # update position after stabilization
            obj_info.position, obj_info.orientation = self._get_obj_pose(obj_info.uid)

        self.uid_to_name = {v.uid: k for k, v in self.all_obj.items()}
        self.uid_to_name[None] = None



    def translate(self, cord, shift):
        return [cord[0] + shift[0], cord[1] + shift[1], cord[2] + shift[2]]
    
    
    def set_room(self, roomSize: float=10.0, 
                 pathWidth: float=2.0, 
                 doNAMO : bool=True, 
                 numMo: int=1, 
                 moSize: float=1.1,
                 floor: bool=True,
                 suppress_output: bool=True):
        """
            args: 
                doNAMO  (boolean): whether to add a NAMO task to the hallway path
                numMo   (int): number of movable objects to be generated in the hallway
                moSize  (float): length of movable object [square cube]
            return: 
                dictionary of obstacles to check collision 
        """

        if floor:
            create_box(roomSize, roomSize, 0.002, color=TAN)

        wall_side=0.1 
        wall_height=5

        walls = []

        wall1 = create_box(roomSize + wall_side, wall_side, wall_height, color=DARK_GREY)
        set_point(wall1, Point(y=roomSize/2., z=wall_side/2.))
        walls.append(wall1)

        wall2 = create_box(roomSize + wall_side, wall_side, wall_height, color=DARK_GREY)
        set_point(wall2, Point(y=-roomSize/2., z=wall_side/2.))
        walls.append(wall2)

        wall3 = create_box(wall_side, roomSize + wall_side, wall_height, color=DARK_GREY)
        set_point(wall3, Point(x=roomSize/2., z=wall_side/2.))
        walls.append(wall3)

        wall4 = create_box(wall_side, roomSize + wall_side, wall_height, color=DARK_GREY)
        set_point(wall4, Point(x=-roomSize/2., z=wall_side/2.))
        walls.append(wall4)
        

        pathSide = 2
        wall_x_position = -(1.5/5) * roomSize/2
        wall_length_before_door = roomSize - pathWidth
        
        wall5 = create_box(pathSide, wall_length_before_door, wall_height, color=DARK_GREY)
        set_point(wall5, Point(x=wall_x_position, y=roomSize/2 - wall_length_before_door/2))
        walls.append(wall5)
    
        hallway_center = [wall_x_position, -roomSize/2+pathWidth/2,0]


        moObst = []
        if doNAMO:
            for _ in range(numMo):
                obst = create_box(moSize,moSize,moSize,color=TAN,mass=1)
                set_point(obst, Point(x=hallway_center[0]+random.uniform(-0.7,0.7), y=hallway_center[1]+random.uniform(-0.7,0.7), z=0.25))
                moObst.append(obst)

        door = self.set_door(suppress_output=suppress_output)

        return walls, moObst, door
    

    def set_hall_area(self, doUnload: bool=False, suppress_output: bool=True):
        hall = {}
        for name, config in self.shop_config["hall_config"].items():
            try:
                if "chair" in name:
                    continue

                obj = ShopObjectEntry(self.bc, self.urdf_dir_path, name, self.shop_config["hall_config"], "hall", suppress_output=suppress_output)
                hall[name] = obj

                if "table" not in name:
                    continue

                for _k, _v in {"left_left":[-0.5,-0.40,0.12,0,0,0],"left_right":[-0.6,0.40,0.12,0,0,0],"right_left":[0.6,-0.40,0.12,0,0,-np.pi], "right_right":[0.6,0.40,0.12,0,0,-np.pi]}.items():
                    chair = ShopObjectEntry(self.bc, 
                                            self.urdf_dir_path, 
                                            "chair", 
                                            self.shop_config["hall_config"], 
                                            "hall",
                                            position=translate(config["pos"], _v[0:3]),
                                            orientation=translate(config["orn"], _v[3:]),
                                            suppress_output=suppress_output)
                    chair.name = f"{name}_{_k}_chair"
                    hall[f"{name}_{_k}_chair"] = chair
                
            except Exception as e:
                print(f"Failed to load {name} beacuse {e}!")

        if doUnload:
            table_mat = create_box(0.5,0.7,0.01, color=YELLOW,mass=100)
            set_point(table_mat, Point(x=-0.35,y=1.65,z=1.3))
            self.bc.loadURDF(
                fileName        = os.path.join(self.urdf_dir_path, self.shop_config["spoon_assets"]["KitchenSpoon"]["path"]),
                basePosition    = [-0.25,1.68,1.35],
                baseOrientation = self.bc.getQuaternionFromEuler([PI/2,PI,-PI/5]),
                globalScaling = self.shop_config["spoon_assets"]["KitchenSpoon"]["scale"],
                useFixedBase    = False)
            self.bc.loadURDF(
                fileName        = os.path.join(self.urdf_dir_path, self.shop_config["fork_assets"]["KitchenFork"]["path"]),
                basePosition    = [-0.15,1.35,1.35],
                baseOrientation = self.bc.getQuaternionFromEuler([PI/2,PI/2,-PI/2]),
                globalScaling = self.shop_config["fork_assets"]["KitchenFork"]["scale"],
                useFixedBase    = False)
            
        return hall


    def set_kitchen_area(self, doLoad: bool=True, suppress_output: bool=True):

        kitchen = {}    
        for name, config in self.shop_config["kitchen_config"].items():
            if "kitchen_door" in name:
                continue
            try:
                obj = ShopObjectEntry(self.bc, 
                                      self.urdf_dir_path, 
                                      name, 
                                      self.shop_config["kitchen_config"], 
                                      "kitchen",
                                      suppress_output=suppress_output)
                kitchen[name] = obj
            except Exception as e:
                print(f"[KitchenEnvironment] Failed to load {name}")


        return kitchen
    

    def set_door(self, suppress_output: bool=True):
        door_asset: Dict[str, object] = self.shop_config["kitchen_config"]["kitchen_door"]
        try:
            door = ShopObjectEntry(self.bc, 
                                   self.urdf_dir_path, 
                                   "kitchen_door", 
                                   self.shop_config["kitchen_config"],
                                   "kitchen",
                                   suppress_output=suppress_output)
        except Exception as e:
            print(f"[KitchenEnvironment] Failed to load door")

            return None
        
        door_default_pos = self.shop_config["kitchen_config"]["kitchen_door"]["default_position"]
        door.is_open = False

        self.handles["kitchen_door"] = ("kitchen_door",
                                        door_asset["handle_link_index"],
                                        door_asset["handle_pos"],
                                        door_asset["handle_init_orn"],
                                        door_asset["handle_direction"],
                                        door_default_pos,
                                        door)
        

        ## NOTE (SJ): Part that open, close the door
        if self.shop_config["kitchen_config"]["kitchen_door"]["initial_state"]:
            self.bc.resetJointState(door.uid, door_asset["handle_link_index"], 90/180*math.pi)
            door.is_open = True

        return door
        

    def is_region(self, obj: Union[int, str]) -> bool:
        if isinstance(obj, str):
            try:
                uid = self.movableObj[obj]
            except KeyError:
                print("Not proper object name")
                return False
        else:
            uid = obj

        return (uid in self.regions)

    
    def _get_obj_pose(self, id: Union[str, int]):
        if isinstance(id, str):
            uid = self.all_obj[id]
        else:
            uid = id
        return self.bc.getBasePositionAndOrientation(uid)
    

    def _get_obj_position(self, id: Union[str, int]):
        return self._get_obj_pose(id)[0]
    

    def _get_obj_quaternion(self, id: Union[str, int]):
        return self._get_obj_pose(id)[1]
    

    def _get_obj_euler(self, id: Union[str, int]):
        euler = self.bc.getEulerFromQuaternion(self._get_obj_quaternion(id))
        return euler


    def capture_pybullet_rgbd_image(self, sigma: float = None, 
                                          exec_observation: bool = False) \
                                            -> Tuple[ npt.NDArray, 
                                                      npt.NDArray, 
                                                      Optional[Dict[int, npt.NDArray]] ]:
        """Capture RGB-D image from pybullet

        Args:
            sigma (float): Use config value if not given.
            exec_observation (bool): Capture at high resolution.
            render_nvisii (bool): convert pybullet image to NVISII rendered image.

        Returns:
            depth_array (NDArray): Pixel depth value [H, W]
            rgb_array (NDArray): Pixel RGB value [H, W, 3] ranges within [0, 255].
            uid_seg_mask (Dict[int, NDArray]): Key is UID. Please map UID to GID outside of this function...
            mask_array (NDArray): Pixel uid value [H, W].
        """
        # Define noise
        if sigma == None and exec_observation:
            sigma = self.CAMERA_EXEC_NOISE
        else:
            sigma = self.CAMERA_SIM_NOISE
        
        # Capture
        if exec_observation:
            (w, h, px, px_d, px_id) = self.bc.getCameraImage(
                width            = self.CAMERA_EXEC_WIDTH,
                height           = self.CAMERA_EXEC_HEIGHT,
                viewMatrix       = self.CAMERA_VIEW_MATRIX,
                projectionMatrix = self.CAMERA_EXEC_PROJ_MATRIX,
                renderer         = self.bc.ER_BULLET_HARDWARE_OPENGL)
        else:
            (w, h, px, px_d, px_id) = self.bc.getCameraImage(
                width            = self.CAMERA_SIM_WIDTH,
                height           = self.CAMERA_SIM_HEIGHT,
                viewMatrix       = self.CAMERA_VIEW_MATRIX,
                projectionMatrix = self.CAMERA_SIM_PROJ_MATRIX,
                renderer         = self.bc.ER_BULLET_HARDWARE_OPENGL)
            
        # Segmentation
        mask_array = np.array(px_id, dtype=np.uint8)
        uid_seg_mask = {}
        for uid in self.object_uids:
            uid_seg_mask[uid] = np.where(mask_array==uid, True, False)

        # Reshape list into ndarray(image)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]                 # remove alpha

        if exec_observation:
            rgb_array = rgb_array.astype(np.int32)
            rgb_array += np.random.normal(loc=0, scale=15, size=rgb_array.shape).astype(np.int32)
            rgb_array = rgb_array.clip(0, 255)
            rgb_array = rgb_array.astype(np.uint8)
            # import matplotlib.pyplot as plt
            # plt.imshow(rgb_array)
            # plt.show()
        depth_array = np.array(px_d, dtype=np.float32)
        noise = sigma * np.random.randn(h, w)
        depth_array = depth_array + noise
        
        return depth_array, rgb_array, uid_seg_mask, mask_array


    def reset_dynamics(self):
        """Reset the dynamics of spawned objects."""
        for obj in self.all_obj.values():
            uid = obj.uid
            self.bc.changeDynamics(
                uid, 
                -1, 
                lateralFriction=1.0,
                rollingFriction=0.001,
                spinningFriction=0.001,
                restitution=0.2)
            
    
    def sample_point_from_region(self, region: str, draw=False):

        region = self.regions[region]

        min, max = region.taskspace
        range = np.array(max) - np.array(min)
        rand = np.random.uniform(low=0, high=1, size=len(range))

        sampled = range*rand + np.array(min)

        region_pos = self.all_obj[region.entity_group[0]].position
        region_orn = self.bc.getQuaternionFromEuler(self.all_obj[region.entity_group[0]].orientation)

        pos, orn = self.bc.multiplyTransforms(region_pos,
                                              region_orn,
                                              sampled,
                                              self.bc.getQuaternionFromEuler((0,0,0)))

        pos = np.array(pos)
        pos[2] = region.taskspace[1][2] + region_pos[2] + 0.03
        pos = tuple(pos)

        if draw:
            t_min_world = self.bc.multiplyTransforms(region_pos, 
                                                     region_orn, 
                                                     min, 
                                                     self.bc.getQuaternionFromEuler((0,0,0)))
    
            t_max_world = self.bc.multiplyTransforms(region_pos, 
                                                     region_orn, 
                                                     max, 
                                                     self.bc.getQuaternionFromEuler((0,0,0)))

            self.bc.addUserDebugPoints([t_min_world[0], t_max_world[0]], [[0,0,1]]*2, 3)
            self.bc.addUserDebugPoints([pos], [[1,0,0]], 4)

        return pos, orn
    
    
    def draw_base_limits(self, roomSize: float=10.0, z=1e-1, eps=1e-1, color=(1,0,0), width=2):
        limits = (-roomSize/2.*np.ones(2)+eps, roomSize/2.*np.ones(2)-eps)
        lower, upper = limits

        vertices = [(lower[0], lower[1], z), (lower[0], upper[1], z),
                    (upper[0], upper[1], z), (upper[0], lower[1], z)]
        
        return add_segments(vertices, closed=True, color=color, width=width)



    

# Capturing functionalities
@dataclass
class HandleState:
    uid: int
    link_index: int
    position: Tuple[float]
    is_open: bool

@dataclass(frozen=True)
class EnvCapture:
    robot_arm_state: Tuple[float]              
    robot_base_state: Tuple[float]
    object_states: Dict[str, ShopObjectEntry]
    holding_status: Dict[str, AttachConstraint]
    handle_status: Dict[str, HandleState]
    receptacle_status: Dict[str, AttachConstraint]
    receptacle_holding_info: Tuple[Tuple[float], Tuple[float], Tuple[Tuple[float]]]


def set_env_from_capture(bc: BulletClient,
                         env: ShopEnv,
                         robot: PR2,
                         capture: EnvCapture):
    """
    - Reset robot joint configuration.
    - Remove objects in current simulation.
    - Reload objects according to state in simulation.

    Args:
        bc (BulletClient)
        env (ShopEnv)
        robot (PR2)
        state (ShopState)
    """
    activation_status = capture.holding_status
    obj_in_hands = dict()
    for arm in activation_status.keys():
        obj_in_hands[arm] = env.uid_to_name[activation_status[arm].uid] if activation_status[arm] is not None else None

    for arm in ["left", "right"]:
        robot.release(arm)
    
    # Reposition the objects
    for name, entry in capture.object_states.items():
        uid = env.all_obj[name].uid
        pos = entry.position
        orn = bc.getQuaternionFromEuler(entry.orientation) if len(entry.orientation) == 3 else entry.orientation
        bc.resetBasePositionAndOrientation(uid, pos, orn)

        env.all_obj[name].position = pos
        env.all_obj[name].orientation = orn

        if name in env.handles.keys():
            env.all_obj[name].is_open = entry.is_open


    # Reset robot base
    robot.set_pose(capture.robot_base_state)

    # Reset robot joints
    robot.last_pose = capture.robot_arm_state
    for value_i, joint_i in zip(capture.robot_arm_state, robot.joint_indices_arm):
        bc.resetJointState(robot.uid, joint_i, value_i)

    # Adjust dynamics
    env.reset_dynamics()
    
    # Reactivate grasp
    for arm, attach_info in activation_status.items():
        if attach_info is not None:
            pick_receptacle = (env.uid_to_name[attach_info.uid] in env.receptacle_obj)
            # robot.set_joint_positions(attach_info.joints[1:], attach_info.joint_positions[1:])
            robot.activate(arm, [attach_info.uid], object_pose=attach_info.object_pose, pick_receptacle=pick_receptacle)

    # Restore handle status
    for name, handle_state in capture.handle_status.items():
        bc.resetJointState(handle_state.uid, handle_state.link_index, handle_state.position)
        env.all_obj[name].is_open = handle_state.is_open

    # Restore receptacle related info
    robot.receptacle_status = deepcopy(capture.receptacle_status)
    robot.receptacle_holding_info = deepcopy(capture.receptacle_holding_info)


def capture_shopenv(bc: BulletClient, 
                    env: ShopEnv,
                    robot: PR2) -> EnvCapture:
    
    # Parse robot arm
    robot_arm_state = tuple(robot.get_both_arms_state())    # (torso, left, right)

    # Parse robot base
    robot_base_state = robot.get_pose()


    # Parse handle state
    handle_states = dict()
    for name, handle_info in env.handles.items():
        uid = env.all_obj[name].uid
        link = handle_info[1]
        joint_pos = bc.getJointState(uid, link)[0]
        handle_states[name] = HandleState(uid, link, joint_pos, env.all_obj[name].is_open)

    # Parse object state
    object_states: Dict[str, ShopObjectEntry] = dict()
    for name, object_info in env.movable_obj.items():
        entry = deepcopy(object_info)
        entry.position, entry.orientation = bc.getBasePositionAndOrientation(entry.uid)
        object_states[name] = entry
    
    for name in env.handles.keys():
        entry = deepcopy(env.all_obj[name])
        object_states[name] = entry

    # Parse receptacle related info
    receptacle_status = deepcopy(robot.receptacle_status)
    receptacle_holding_info = deepcopy(robot.receptacle_holding_info)

    # Update holding status
    holding_status = deepcopy(robot.activated)

    capture = EnvCapture(robot_arm_state, 
                         robot_base_state, 
                         object_states,
                         holding_status,
                         handle_states,
                         receptacle_status,
                         receptacle_holding_info)
    
    return capture


def save_predicates(bc: BulletClient,
                    env: ShopEnv,
                    robot: PR2,
                    scenario: int,
                    step: int,
                    predicates: Dict,
                    save_dir: str="saved_predicates", 
                    postprocess: bool=True):
    
    save_dir = os.path.join(os.getcwd(), save_dir)
    save_predicate_dir = os.path.join(save_dir, "predicates")
    save_bullet_dir = os.path.join(save_dir, "bullet_state")
    os.makedirs(save_predicate_dir, exist_ok=True)
    os.makedirs(save_bullet_dir, exist_ok=True)

    filename = f"scenario={scenario}_step={step}"

    # Process predicate (set-->list)
    if postprocess:
        predicates = process_dict_for_json(predicates)

    with open(os.path.join(save_predicate_dir, f"predicates_{filename}.json"), "w") as f:
        json.dump(predicates, f, indent=4)

    capture = capture_shopenv(bc, env, robot)

    with open(os.path.join(save_bullet_dir, f"capture_{filename}.pkl"), "wb") as f:
        pickle.dump(capture, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_predicates(scenario: int, step: int, save_dir: str = "saved_predicates", postprocess=True):
    save_dir = os.path.join(os.getcwd(), save_dir)
    save_predicate_dir = os.path.join(save_dir, "predicates")

    filename = f"_scenario={scenario}_step={step}"

    file_dir = os.path.join(save_predicate_dir, f"predicates{filename}.json")


    with open(os.path.join(save_predicate_dir, f"predicates{filename}.json"), "r") as f:
        predicates = json.load(f)

    if postprocess:
        predicates = process_json_for_predicate(predicates)

    return predicates


def process_dict_for_json(predicate: Dict):
    result = deepcopy(predicate)
    for k, v in predicate["asset"].items():
        if "is_occ_manip" in v:
            for key in v["is_occ_manip"].keys():
                val = result["asset"][k]["is_occ_manip"].pop(key)
                result["asset"][k]["is_occ_manip"][str(key)] = val

        if "has_placement_pose" in v:
            for key in v["is_occ_manip"].keys():
                val = result["asset"][k]["has_placement_pose"].pop(key)
                result["asset"][k]["has_placement_pose"][str(key)] = val

    return result


def process_json_for_predicate(predicate: Dict):
    result = deepcopy(predicate)
    for k, v in predicate["asset"].items():
        if "is_occ_manip" in v:
            for key in v["is_occ_manip"].keys():
                val = result["asset"][k]["is_occ_manip"].pop(key)
                # result["asset"][k]["is_occ_manip"][eval(key)] = val
                result["asset"][k]["is_occ_manip"][eval(key)] = val

        if "has_placement_pose" in v:
            for key in v["has_placement_pose"].keys():
                val = result["asset"][k]["has_placement_pose"].pop(key)
                # result["asset"][k]["is_occ_manip"][eval(key)] = val
                result["asset"][k]["has_placement_pose"][eval(key)] = val

    del predicate

    return result