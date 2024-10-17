
"""
POMDP modelling for the fetching problem
"""

import os
import random
import yaml

import numpy as np
import numpy.typing as npt

from typing import Tuple, Dict, List, TypeVar, Union, NamedTuple


from Simulation.pybullet.mdp.shop.shop_MDP import ShopObjectEntry, ShopGoal, ShopState, make_shop_goal, capture_shopenv_state
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.envs.shop_env import ShopEnv
from Simulation.pybullet.envs.robot import PR2
from Simulation.pybullet.envs.manipulation import PR2Manipulation, PR2SingleArmManipulation
from Simulation.pybullet.envs.navigation import Navigation

DEBUG = False

class SimulatorShop:
    
    def __init__(self, bc: BulletClient,
                       env: ShopEnv,
                       robot: PR2,
                       manip: PR2Manipulation,
                       nav: Navigation,
                       config: Dict):
        """
        args:

        """
        self.bc = bc
        self.env = env 
        self.robot = robot
        self.manip = manip
        self.nav = nav
        self.config = config

        self.AFFORDANCE_CHECK_TRIAL = 1
        self.num_object_state_resample = 3
        self.object_placement_trial = 100
        self.set_initial_gt_trial = 3



    def set_initial_groundtruth(self) -> Tuple[ShopState, ShopGoal]:
        """Make random initial groundtruth.

        Returns:

        """
        robot_pose_candidates = self.config["problem_params"]["robot"]
        object_state_candidates = self.config["problem_params"]["objects"]

        scenario = self.config["problem_params"]["scenario"]
        variant = False
        if isinstance(scenario, str) and 'v' in scenario:
            scenario = scenario[1:]
            variant = True

        if scenario is None:
            # assert False, "scenario must be specified"
            config_shop_filename = "config_shop.yaml"
        else:
            if variant:
                config_shop_filename = f"scenarios/variants/scenario{scenario}.yaml"
            else:
                config_shop_filename = f"scenarios/scenario{scenario}.yaml"

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../cfg", config_shop_filename), "r") as f:
            shop_config: Dict[str, Dict] = yaml.load(f, Loader=yaml.FullLoader)

        if self.config["problem_params"]["scenario"] is None:
            goal_candidates = self.config["problem_params"]["goal"]
        else:
            # assert False, "scenario must be specified"
            goal_candidates = shop_config["goal"]


                                        

        # Capture default configuration
        default_state = capture_shopenv_state(self.bc, self.env, self.robot, "kitchen", None)

        # Generate problem
        valid = False
        trial = 0
        while not valid:
            if trial > self.set_initial_gt_trial:
                break

            # Step 1: Choose the goal
            goal: ShopGoal = self.make_goal(goal_candidates)

            # Step 2: Set the robot
            self.set_robot(robot_pose_candidates)

            # Step 3: Set the objects
            self.set_object_states(object_state_candidates)

            # Step 4: step simulation to stabilize if no contact
            self.nav.stabilize(duration=500)

            # Step 5: check any dropped object
            is_dropped = self.check_dropped_objects()

            valid = not is_dropped

        if not valid:
            return default_state, goal
        
        else:
            return capture_shopenv_state(self.bc, 
                                         self.env, 
                                         self.robot, 
                                         #  self.determine_robot_region(),
                                         "initial_position",
                                         None), goal
    


    def make_goal(self, goal_candidates: List[Dict]):
        
        goal_config = random.choice(goal_candidates)

        return make_shop_goal(goal_config)

        
    def set_robot(self, robot_pose_candidates: List[Tuple]):

        robot_pose = random.choice(robot_pose_candidates)

        pos = robot_pose[0]
        orn = self.bc.getQuaternionFromEuler(robot_pose[1])

        self.robot.set_pose((pos, orn))


    def determine_robot_region(self):
        position = self.robot.get_pose()[0]

        # fill in the region criterion
        if position[0] < -2:
            return "kitchen"
        else:
            return "hall"


    def set_object_states(self, object_state_candidates: List[List[str]]):
        
        settled = False
        valid = False
        i = 0
        while not settled:
            if i > self.num_object_state_resample:
                break
            
            object_state = random.choice(object_state_candidates)

            # Put all the moving objects in the air first to avoid collision
            for (name, _) in object_state:
                pos = self.env.movable_obj[name].position
                pos = np.array(pos)
                pos[2] = 5
                orn = self.env.movable_obj[name].orientation
                if len(orn) == 3:
                    orn = self.bc.getQuaternionFromEuler(orn)
                self.bc.resetBasePositionAndOrientation(self.env.movable_obj[name].uid, pos, orn)

            for (name, region) in object_state:
                obj = self.env.all_obj[name]
                r = self.env.regions[region]
                obj.region = region
                obj.area = r.area
                obj_height = self.get_object_height(obj)

                j = 0
                valid = False
                while not valid:
                    if j > self.object_placement_trial:
                        break

                    # sample object pose from the task space
                    sampled = self.sample_from_region(*r.taskspace)
                    pos, orn = self.bc.multiplyTransforms(r.position,
                                                        self.bc.getQuaternionFromEuler(r.orientation),
                                                        sampled,
                                                        self.bc.getQuaternionFromEuler((0,0,0)))

                    region_pos = self.env.all_obj[r.entity_group[0]].position
                    region_orn = self.bc.getQuaternionFromEuler(self.env.all_obj[r.entity_group[0]].orientation)

                    pos = np.array(pos)
                    pos[2] = r.taskspace[1][2] + region_pos[2] + obj_height/2 + 0.03
                    pos = tuple(pos)

                    if DEBUG:
                        t_min, t_max = r.taskspace
                        t_min_world = self.bc.multiplyTransforms(region_pos, region_orn, 
                                                    t_min, self.bc.getQuaternionFromEuler((0,0,0)))
                
                        t_max_world = self.bc.multiplyTransforms(region_pos, region_orn, 
                                                            t_max, self.bc.getQuaternionFromEuler((0,0,0)))

                        self.bc.addUserDebugPoints([t_min_world[0], t_max_world[0]], [[0,0,1]]*2, 3)

                    self.bc.resetBasePositionAndOrientation(obj.uid, pos, orn)
                    obj.position = pos
                    obj.orientation = self.bc.getEulerFromQuaternion(orn)

                    # Collision check
                    valid = not self.check_close_objects(obj)
                    j += 1
                if not valid:
                    break
            i += 1
            if not valid:
                continue
            settled = True

        return valid

            
    def check_close_objects(self, object_info: ShopObjectEntry, threshold: float=0.01) -> bool:

        is_close = False
        for name, entry in self.env.movable_obj.items():
            if object_info.name == name:
                continue
            if object_info.region != entry.region:
                continue

            closest_points = self.bc.getClosestPoints(object_info.uid, entry.uid, threshold, -1, -1)
            if len(closest_points) > 0:
                is_close = True
                break

        return is_close


    def check_dropped_objects(self, threshold: float=0.3):

        is_dropped = False
        for name, entry in self.env.movable_obj.items():
            pos = self.bc.getBasePositionAndOrientation(entry.uid)
            is_dropped = pos[0][2] < threshold

            if is_dropped:
                break

        return is_dropped


    def sample_from_region(self, min: npt.ArrayLike, max: npt.ArrayLike, margin: float = 0.05) -> npt.ArrayLike:
        assert len(min)==len(max), "Number of dimension must be same"

        range = np.array(max) - np.array(min) - np.array([2*margin, 2*margin, 0])
        rand = np.random.uniform(low=0, high=1, size=len(range))

        return range*rand + np.array(min) + np.array([margin, margin, 0])


    def get_object_height(self, object_info: ShopObjectEntry):
        uid = object_info.uid
        if object_info.is_movable:
            try:
                aabb = self.bc.getAABB(uid, 0)
            except Exception as e:
                aabb = self.bc.getAABB(uid)
        else:
            aabb = self.bc.getAABB(uid)

        return aabb[1][2] - aabb[0][2]


