import math
import os 
import time
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import pickle
import uuid
from pathlib import Path

from contextlib import contextmanager
from typing import List, Tuple, Union, Callable, Dict, Iterable
from Simulation.pybullet.envs.robot import Robot, UR5Suction, PR2
from Simulation.pybullet.imm.pybullet_tools.pr2_utils import PR2_GROUPS
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient

from Simulation.pybullet.imm.pybullet_util.common import get_link_pose, get_relative_transform
from Simulation.pybullet.imm.pybullet_util.collision import NavigationCollision, LinkPair
from Simulation.pybullet.imm.motion_planners.rrt_connect import birrt
from Simulation.pybullet.imm.motion_planners.prm import DegreePRM

from Simulation.pybullet.envs.shop_env import ShopEnv, EnvCapture, set_env_from_capture, capture_shopenv




@contextmanager
def dream(bc: BulletClient, env: ShopEnv, robot: PR2) -> EnvCapture:
    try:
        capture = capture_shopenv(bc, env, robot)
        yield capture
    finally:
        set_env_from_capture(bc, env, robot, capture)



def interpolate_trajectory(cur: List, 
                           goal: List, 
                           action_duration: float, 
                           control_dt: float) -> Tuple[npt.NDArray, ...]:
    '''
    This function returns linear-interpolated (dividing straight line)
    trajectory between current and goal pose.
    Acc, Jerk is not considered.
    '''
    # Interpolation steps
    steps = math.ceil(action_duration/control_dt)
    
    # Calculate difference
    delta = [ goal[i] - cur[i] for i in range(len(cur)) ]
    
    # Linear interpolation
    trajectory: Tuple[npt.NDArray, ...] = ([
        np.array([
            cur[j] + ( delta[j] * float(i)/float(steps) ) 
            for j in range(len(cur))
        ])
        for i in range(1, steps+1)
    ])

    return trajectory





class Navigation():

    def __init__(self, bc         : BulletClient,
                       env        : ShopEnv, 
                       robot      : Robot,
                       config     : dict,
                       planner_type: str="prm"): 
        
        self.config      = config
        self.bc          = bc
        self.env         = env
        self.robot       = robot
        
        # Sim params
        sim_params = config["sim_params"]
        self.DEBUG_GET_DATA = self.config['project_params']['debug']['get_data']
        self.delay          = sim_params["delay"]
        self.control_dt     = 1. / sim_params["control_hz"]
       
        self._set_navigation_params()

        # Default RRT callbacks
        distance_fn, sample_fn, extend_fn, collsion_fn = self.__get_default_functions()
        self.distance_fn          = distance_fn
        self.sample_fn            = sample_fn
        self.extend_fn            = extend_fn
        self.default_collision_fn = collsion_fn

        # Empty collision function
        allow_list = [v.uid for v in self.env.movable_obj.values()]
        allow_list += [self.env.all_obj[name].uid for name in self.env.handles.keys()]
        self.empty_prm_collision_fn \
            = self._define_pr2_navigation_collision_fn(None, allow_uid_list=allow_list)

        self.planner_type = planner_type

        self.empty_prm_verticies = {}
        self.empty_prm_edges = []
        self.default_prm_vertices = {}
        self.default_prm_edges = []

        if planner_type in ["prm", "PRM"]:
            self.motion_planner = self._get_prm_planner()
            #Grow roadmaps
            # self.reset_empty_roadmap()
            # self.reset_roadmap()
            self.default_open_pickle_filename = config["navigation_params"]["pickle"]["default_open"]
            self.default_close_pickle_filename = config["navigation_params"]["pickle"]["default_close"]
            self.empty_pickle_filename = config["navigation_params"]["pickle"]["empty"]

        elif planner_type in ["rrt", "RRT"]:
                pass
                # self.motion_planner = "rrt"   TODO (dlee): get RRT here

        

    ### Initialization
    def _get_prm_planner(self, target_degree=4, connect_distance=float('inf')):
        motion_planner = DegreePRM(self.distance_fn, self.extend_fn, self.default_collision_fn,
                                   target_degree=target_degree, connect_distance=connect_distance)

        return motion_planner
    

    def _set_navigation_params(self):
        navigation_params = self.config["navigation_params"]
        self.base_limits = navigation_params["base_limits"]
        self.circular_limits = (-np.pi, np.pi)
        self.resolutions = navigation_params["resolutions"]
        self.trials = navigation_params["trials"]
        self.num_samples = navigation_params["num_samples"]
        self.delay = navigation_params["delay"]


    def __get_default_functions(self) -> Tuple[Callable, Callable, Callable, Callable]:
        """Init default sample, extend, distance, collision function

        Returns:
            distance_fn
            sample_fn
            extend_fn
            collision_fn
        """
        def difference_fn(q0: np.ndarray, q1: np.ndarray):
            dx, dy = np.array(q1[:2]) - np.array(q0[:2])
            dtheta = (q1[2] - q0[2]) % (2*np.pi) - np.pi

            return dx, dy, dtheta

        def distance_fn(q0: np.ndarray, q1: np.ndarray, weights=np.ones(3)):
            difference = np.array(difference_fn(q0, q1))
            return np.sqrt(np.dot(weights, difference * difference))
        

        def sample_fn(debug=False):
            """node sampling function in cartesian space"""
            
            x, y = np.random.uniform(*self.base_limits)
            theta = np.random.uniform(*self.circular_limits)

            if debug:
                self.bc.addUserDebugPoints([(x, y, 0.001)], [(1, 0, 0)], pointSize=3)
                _, _, z = self.robot.get_position()
                orn_q = self.bc.getQuaternionFromEuler((0, 0, theta))
                self.robot.set_pose(((x, y, z),orn_q))

            return (x, y, theta)
        
        def extend_fn(q1, q2):
            dq = difference_fn(q1, q2)
            n = int(np.max(np.abs(dq[:-1]))/self.resolutions)+1

            return q1 + np.linspace(0, 1, num=n)[:, None] * dq
        


        default_collision_fn = NavigationCollision(bc=self.bc, 
                                                   robot_id=self.robot.uid,
                                                   attachlist=[],
                                                   allowlist=[],
                                                   touchlist=[],
                                                   tol={},
                                                   touch_tol=0.005)


        return distance_fn, sample_fn, extend_fn, default_collision_fn


    def deepcopy_with_pickle(self, data):
        ## NOTE(SJ): if the given data is only consisted of arrays using numpy instead of pickle could be much faster
        # new_data = pickle.loads(pickle.dumps(data, -1))
        new_data = deepcopy(data)
        return new_data


    def reset_empty_roadmap(self, use_pickle:bool=False):
        self.motion_planner.vertices = {}
        self.motion_planner.edges = []

        if use_pickle:
            dirname = os.path.dirname(__file__)
            self.load_prm_nodes(os.path.join(dirname, self.empty_pickle_filename))
        else:

            self.motion_planner.collision_fn = self.empty_prm_collision_fn
            # Grow roadmap
            with dream(self.bc, self.env, self.robot):
                samples = []
                while len(samples) < self.num_samples:
                    sample = tuple(self.sample_fn())
                    if self.motion_planner.collision_fn(sample)[0]:
                        continue
                    samples.append(sample)
                    self.motion_planner.grow(samples)

        self.empty_prm_verticies = self.deepcopy_with_pickle(self.motion_planner.vertices)
        self.empty_prm_edges = self.deepcopy_with_pickle(self.motion_planner.edges)


    def reset_roadmap(self, use_pickle: bool=False, door_open=True):
        assert self.planner_type in ["prm", "PRM"], "Motion planner should be PRM"
        
        self.motion_planner.vertices = {}
        self.motion_planner.edges = []

        if use_pickle:
            dirname = os.path.dirname(__file__)
            if door_open:
                self.load_prm_nodes(os.path.join(dirname, self.default_open_pickle_filename))
            else:
                self.load_prm_nodes(os.path.join(dirname, self.default_close_pickle_filename))

        else:

            self.motion_planner.collision_fn = self.default_collision_fn
            # Grow roadmap
            with dream(self.bc, self.env, self.robot):
                samples = []
                while len(samples) < self.num_samples:
                    sample = tuple(self.sample_fn())
                    if self.motion_planner.collision_fn(sample)[0]:
                        continue
                    samples.append(sample)
                    # self.bc.addUserDebugPoints([(sample[0], sample[1], 0.001)], [(1, 0, 0)], pointSize=3)
                self.motion_planner.grow(samples)

        self.default_prm_vertices = self.deepcopy_with_pickle(self.motion_planner.vertices)
        self.default_prm_edges = self.deepcopy_with_pickle(self.motion_planner.edges)


    ### Set roadmaps
    def set_empty_roadmap(self, reset_roadmap: bool=True):
        # NOTE (dlee): should empty PRM deepcopied everytime?
        if reset_roadmap:
            self.motion_planner.vertices = deepcopy(self.empty_prm_verticies)
            self.motion_planner.edges = deepcopy(self.empty_prm_edges)
        self.motion_planner.collision_fn = self.empty_prm_collision_fn


    def set_default_roadmap(self, reset_roadmap: bool=True):
        if reset_roadmap:
            self.motion_planner.vertices = deepcopy(self.default_prm_vertices)
            self.motion_planner.edges = deepcopy(self.default_prm_edges)
        self.motion_planner.collision_fn = self.default_collision_fn


    def _SE3fromSE2(self, se2: npt.NDArray):
        assert len(se2) == 3, "SE2 must be (x, y, theta)"
        _, _, z = self.robot.get_position()
        pos = (se2[0], se2[1], z)
        orn_q = self.bc.getQuaternionFromEuler((0, 0, se2[2]))

        return (pos, orn_q) 

    def _SE2fromSE3(self, se3: Tuple[Iterable[float]]):
        x, y, _ = se3[0]
        orn = se3[1]
        if len(orn) == 3:
            orn = self.bc.getEulerFromQuaternion(orn)

        return (x, y, orn[2])


    def stabilize(self, debug=False, duration=100):
        i = 0
        while True:
            if not debug:
                i += 1
            joints = self.robot.torso_joints + self.robot.left_arm_joints + self.robot.right_arm_joints + self.robot.joints_from_names(PR2_GROUPS["left_gripper"]) + self.robot.joints_from_names(PR2_GROUPS["right_gripper"])
            curr_joint_poses = self.robot.get_joint_positions(joints)
            self.bc.stepSimulation()
            self.robot.set_joint_positions(joints, curr_joint_poses)
            if i > duration:
                break

    ### Node utils
    def draw_navigation_prm_nodes(self, draw_points=False, draw_empty=False):
        if draw_points:
            vertices = self.motion_planner.vertices if not draw_empty else self.empty_prm_verticies
            points = [(*v[:2], 0.01) for v in vertices]

            self.bc.addUserDebugPoints(points, [[1,0,0]]*len(points), 3)

        edges = self.motion_planner.edges if not draw_empty else self.empty_prm_edges

        for edge in edges:
            x1 = (*edge.v1.q[:2], 0.01)
            x2 = (*edge.v2.q[:2], 0.01)
            self.bc.addUserDebugLine(x1, x2, [0,0,1])


    def save_prm_nodes(self, filename: str="_navigation_prm_nodes.pkl"):
        save_dir = os.path.dirname(__file__)

        data = {"edges": self.motion_planner.edges, "vertices": self.motion_planner.vertices}
        with open(os.path.join(save_dir, "default" + filename), "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        data = {"edges": self.empty_prm_edges, "vertices": self.empty_prm_verticies}
        with open(os.path.join(save_dir, "empty" + filename), "wb") as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    
    def load_prm_nodes(self, filename: str):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.motion_planner.edges = data["edges"]
        self.motion_planner.vertices = data["vertices"]


    ### Attachments
    def _get_attach_xfms(self)-> Dict[LinkPair, npt.ArrayLike]:
        attachlist: List[LinkPair] = []
        for arm, info in self.robot.activated.items():
            if info is None:
                continue
            attach_pair = LinkPair(
                    body_id_a=self.robot.uid,
                    link_id_a=self.robot.gripper_link_indices[arm][0],
                    body_id_b=info.uid,
                    link_id_b=-1)
            
            attachlist.append(attach_pair)

        for name, info in self.robot.receptacle_status.items():
            attach_pair = LinkPair(
                    body_id_a=self.robot.uid,
                    link_id_a=self.robot.gripper_link_indices[arm][0],
                    body_id_b=info[0].uid,
                    link_id_b=-1)
            attachlist.append(attach_pair)

        attach_xfms = {
            C: get_relative_transform(self.bc,
                                      C.body_id_a,
                                      C.link_id_a,
                                      C.link_id_b,
                                      C.body_id_b,
                                      inertial=False)
            for C in attachlist}
        
        return attach_xfms


    def _update_attached_bodies(self, attach_xfms: Dict[LinkPair, npt.ArrayLike]):
        bc = self.bc
        # Update transforms of attached bodies.
        for C, xfm in attach_xfms.items():
            # NOTE(ycho): as long as we're consistent
            # about `inertial` keyword, we're fine.
            pose = get_link_pose(bc, C.body_id_a, C.link_id_a)
            pose = bc.multiplyTransforms(pose[0], pose[1],
                                         xfm[0], xfm[1])
            bc.resetBasePositionAndOrientation(
                C.body_id_b, pose[0], pose[1])


    ### Simulation
    def move(self, traj: List[npt.NDArray], physics:bool=False, simple:bool=True):
        attach_xfms = self._get_attach_xfms()

        if simple and len(traj) > 0:
            self.robot.set_pose(self._SE3fromSE2(traj[-1]))
            self._update_attached_bodies(attach_xfms)

        else:
            for bq in traj:
                if physics:

                    for i in range(int(self.delay/self.control_dt)):
                        self.bc.stepSimulation()
                else:
                    self.robot.set_pose(self._SE3fromSE2(bq))
                    self._update_attached_bodies(attach_xfms)
                    if self.delay != 0:
                        time.sleep(self.delay)
            
        # Wait until control completes
        # self.wait(steps=240)
        
    
    def wait(self, steps=-1):
        """General wait function for hold or stabilization.

        Args:
            steps (int): Defaults to -1.
        """
        while steps != 0: 
            self.robot.update_arm_control()
            self.bc.stepSimulation()
            if self.delay:
                time.sleep(self.delay)
            steps -= 1
    

    def _define_pr2_navigation_collision_fn(self, grasp_uid: Union[int, List[int]], 
                                        allow_uid_list: List[int], 
                                        debug=False) -> NavigationCollision:
        """ This function returns the collision function that allows the grasped object to
        1. collide with the robot body
        2. be attached at robot finger
        3. touch with the cabinet.
        
        Args:
            grasp_uid: This object will be move together with the end effector.
            allow_uid_list: Uids of object to allow the contact.
        """
        attachlist = []
        allowlist = []
        touchlist = []

        # Attached object will be moved together when searching the path
        self.robot:PR2

        # Grasping single object
        if isinstance(grasp_uid, int):
            arm = self.robot.main_hand
            attach_pair = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=self.robot.gripper_link_indices[arm][0],
                body_id_b=grasp_uid,
                link_id_b=-1)
            
            attachlist.append(attach_pair)

            # Allow pair is not commutative
            allow_pair_grasp_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=grasp_uid,
                link_id_b=None)
            allow_pair_grasp_b = LinkPair(
                body_id_a=grasp_uid,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            
            allowlist.append(allow_pair_grasp_a)
            allowlist.append(allow_pair_grasp_b)
        # Grasping multiple objects
        elif isinstance(grasp_uid, list):
            for i, uid in enumerate(grasp_uid):
                if i == 0:
                    arm = self.robot.main_hand
                else:
                    arm = self.robot.get_other_arm()
                attach_pair = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=self.robot.gripper_link_indices[arm][0],
                body_id_b=uid,
                link_id_b=-1)
            
                attachlist.append(attach_pair)

                # Allow pair is not commutative
                allow_pair_grasp_a = LinkPair(
                    body_id_a=self.robot.uid,
                    link_id_a=None,
                    body_id_b=uid,
                    link_id_b=None)
                allow_pair_grasp_b = LinkPair(
                    body_id_a=uid,
                    link_id_a=None,
                    body_id_b=self.robot.uid,
                    link_id_b=None)
                
                allowlist.append(allow_pair_grasp_a)
                allowlist.append(allow_pair_grasp_b)
                

        # Optional allow pair
        for allow_uid_a in allow_uid_list:
            for allow_uid_b in allow_uid_list:
                if allow_uid_a == allow_uid_b:
                    continue
                allow_pair_ab = LinkPair(   # ba will be accessed within double loop. dont worry.
                    body_id_a=allow_uid_a,
                    link_id_a=None,
                    body_id_b=allow_uid_b,
                    link_id_b=None)
                allowlist.append(allow_pair_ab)
            allow_pair_a_robot = LinkPair(
                body_id_a=allow_uid_a,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            allow_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=allow_uid_a,
                link_id_b=None)
            allowlist.append(allow_pair_a_robot)
            allowlist.append(allow_pair_robot_a)
        # Touch pair is not commutative. One should define bidirectional pair.
        # NOTE(ssh): Let's also allow touch between the objects defaultly.
        for _, touch_uid_a in self.env.all_obj.items():
            for _, touch_uid_b in self.env.all_obj.items():
                if touch_uid_a == touch_uid_b:
                    continue
                touch_pair_ab = LinkPair(   # ba will be accessed within double loop. dont worry.
                    body_id_a=touch_uid_a.uid,
                    link_id_a=None,
                    body_id_b=touch_uid_b.uid,
                    link_id_b=None)
                touchlist.append(touch_pair_ab)
            touch_pair_a_robot = LinkPair(
                body_id_a=touch_uid_a.uid,
                link_id_a=None,
                body_id_b=self.robot.uid,
                link_id_b=None)
            touch_pair_robot_a = LinkPair(
                body_id_a=self.robot.uid,
                link_id_a=None,
                body_id_b=touch_uid_a.uid,
                link_id_b=None)
            touchlist.append(touch_pair_a_robot)
            touchlist.append(touch_pair_robot_a)
        
        # Compose collision fn
        collision_fn = NavigationCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            allowlist    = allowlist,
            attachlist   = attachlist,
            touchlist    = touchlist,
            tol          = {},
            touch_tol    = -0.005)   #TODO: config
        # Debug
        if debug:
            for _, uid in self.env.movable_obj.items():
                contact2 = np.array(self.bc.getContactPoints(bodyA=grasp_uid, bodyB=uid), dtype=object)
                for c in contact2:
                    print(f"Object {uid} contact: {c[8]}")

        return collision_fn
    

    ### Distance
    def get_node_distance(self, q0: Iterable[float], q1: Iterable[float]) -> float:
        
        assert len(q0) == len(q1) == 3
        
        q0 = np.array(q0)
        q1 = np.array(q1)

        dx, dy, _ = np.abs(q0-q1)
        d = np.linalg.norm((dx, dy))

        return float(d)
    
    def get_traj_distance(self, traj: Iterable[Iterable[float]]) -> float:
        total_distance = 0
        for i, q in enumerate(traj[:-1]):
            if len(q) != 3:
                continue
            q_next = traj[i+1]
            d = self.get_node_distance(q, q_next)
            total_distance += d
        
        return total_distance

    ### Motion plan
    def motion_plan(self, start: npt.ArrayLike, 
                          goal: npt.ArrayLike,
                          holding_obj_uid: Union[int, Tuple[int], None], 
                          allow_uid_list: List[int] = [],
                          ignore_movable: bool=False) -> Union[List[npt.NDArray], None]:
        """Get a motion plan trajectory.

        Args:
            start (npt.ArrayLike): Source joint position
            goal (npt.ArrayLike): Destination joint position
            holding_obj_uid (Union[int, None]): Uid of the holding object.
            allow_uid_list (List[int]): asdf
            use_interpolation (str, Optional): Flag for force motion planning method
        Returns:
            Union[List[npt.ArrayLike], None]: Generated trajectory. None when no feasible trajectory is found.
        """

        # Compute motion plan
        if self.planner_type in ["interpolation"]:
            trajectory = interpolate_trajectory(
                cur             = start, 
                goal            = goal, 
                action_duration = 0.5, 
                control_dt      = self.control_dt)
        elif self.planner_type in ["RRT", "rrt", "birrt"]:
            # Try RRT
            with dream(self.bc, self.env, self.robot):
                # Moving with grapsed object
                if np.any([constraint is not None for _, constraint in self.robot.activated]):
                    collision_fn = self._define_pr2_navigation_collision_fn(allow_uid_list, debug=False)
                    # Get RRT path using constraints
                    trajectory, contacts = birrt(
                        start,
                        goal,
                        self.distance_fn,
                        self.sample_fn,
                        self.extend_fn,
                        collision_fn,
                        max_solutions=1,
                        restarts=self.trials)
                # Moving without grasped object
                else:
                    # Get RRT path using default constraints
                    trajectory, contacts = birrt(
                        start,
                        goal,
                        self.distance_fn,
                        self.sample_fn,
                        self.extend_fn,
                        self.default_collision_fn,
                        max_solutions=1,
                        restarts=self.trials)
        elif self.planner_type in ["PRM", "prm"]:
            if ignore_movable:
                self.set_empty_roadmap()
            else:
                self.set_default_roadmap()
            with dream(self.bc, self.env, self.robot):
                # Moving with grapsed object
                start = tuple(start)
                goal = tuple(goal)
                # samples = [start, goal] + [tuple(self.sample_fn()) for _ in range(self.num_samples)]
                samples = [start, goal]

                if ignore_movable:
                    collision_fn = self.empty_prm_collision_fn
                else:
                    if holding_obj_uid is not None:
                        attachlist = holding_obj_uid
                        if self.robot.is_holding_receptacle():
                            receptacle_uid = self.robot.activated[self.robot.get_other_arm()].uid
                            obj_on_receptacle_uids = [attach_info.uid for (attach_info, _, _) in self.robot.receptacle_status.values()]
                            attachlist = [holding_obj_uid] +  obj_on_receptacle_uids + [receptacle_uid]
                        collision_fn = self._define_pr2_navigation_collision_fn(attachlist, allow_uid_list, debug=False)

                    # Moving without grasped object
                    else:
                        if self.robot.is_holding_receptacle():
                            receptacle_uid = self.robot.activated[self.robot.get_other_arm()].uid
                            obj_on_receptacle_uids = [attach_info.uid for (attach_info, _, _) in self.robot.receptacle_status.values()]
                            attachlist = obj_on_receptacle_uids + [receptacle_uid]
                            collision_fn = self._define_pr2_navigation_collision_fn(attachlist, allow_uid_list, debug=False)
                        else:
                            collision_fn = self.default_collision_fn
                        
                self.motion_planner.collision_fn = collision_fn
            
                self.motion_planner.grow(samples=samples)
                
            trajectory = self.motion_planner(start, goal)

            # NOTE (dlee): This is because of PRM which requires the input to be in tuple for hashing.
            self.motion_planner[start].clear()
            self.motion_planner[goal].clear()

            if trajectory is not None:
                trajectory = np.array(trajectory)
            contacts = []
        else:
            raise ValueError("Mode should be interpolation, rrt, or prm")
        

        return trajectory, contacts

