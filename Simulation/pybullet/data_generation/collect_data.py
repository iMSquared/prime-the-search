import json
import numpy as np
import numpy.typing as npt
import os
import paramiko
from typing import Dict, Tuple, List, Union, Optional
import pathlib
import pickle

from Simulation.pybullet.mdp.MDP_framework import HistoryEntry, TerminationT, PhaseT, PHASE_EXECUTION, PHASE_ROLLOUT, PHASE_SIMULATION
from Simulation.pybullet.mdp.shop.shop_MDP import ShopContinuousAction, ShopState, ShopAgent
from Simulation.pybullet.mdp.online_planner_framework import DiscreteNode

def time_stamp() -> str:
    """Get the unqiue time stamp for a process or file"""
    from datetime import datetime
    now = datetime.now()
    current_date = now.date()
    month = current_date.month
    day = current_date.day
    current_time = now.time()
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    millis = current_time.microsecond / 1000
    TIME_STAMP = f"data_{month}m{day}d{hour}h{minute}m{second}s{f'{millis}'.zfill(3)}ms"

    return TIME_STAMP

def dump_pickle(data: object, path: str):
    """Dump pickle... just a wrapper for better looking."""
    with open(path, "wb") as f:
        pickle.dump(data, f)

def dump_json(data: object, path: str):
    """Dump json... just a wrapper for better looking."""
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def log_single_episode(episode_termination: TerminationT,
                       episode_agent_history: Tuple[HistoryEntry],
                       episode_list_time_taken_per_planning: List[float],
                       episode_total_sim_success_count: int,
                       episode_total_sim_count: int,
                       episode_n: int,
                       PROCESS_TIME_STAMP: str,
                       POLICY: str,
                       VALUE: str,
                       NUM_SIMS_PER_PLAN: int,
                       EXP_LOG_DIR: str):
    """Log a single execution information"""

    # Make log directory if not exist.
    exp_log_dir_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), EXP_LOG_DIR)
    if not os.path.exists(exp_log_dir_path):
        os.mkdir(exp_log_dir_path)

    # Compose filename
    exp_log_fname = PROCESS_TIME_STAMP
    exp_log_fname += f"_exp{episode_n}"
    if POLICY == "guided" or POLICY == "llm":
        exp_log_fname += f"_{POLICY}_policy"
    if VALUE:
        exp_log_fname += f"_{VALUE}_value"
    exp_log_fname += ".json"

    # Reuse some information from trajectory dict..
    episode_termination = episode_termination
    episode_action_history = [str(entry.action) for entry in episode_agent_history]
    episode_observation_history = [str(entry.observation) for entry in episode_agent_history]
    episode_total_reward = sum([entry.reward for entry in episode_agent_history])

    # Dump json log!
    ## TODO(SJ) : to dump_json
    with open(os.path.join(exp_log_dir_path, exp_log_fname), "w") as f:
        exp_data = {
            "termination": episode_termination,
            "total_reward": episode_total_reward,
            "list_time_taken_per_planning": episode_list_time_taken_per_planning,
            "episode_action_history": episode_action_history,
            "episode_observation_history": episode_observation_history,
            "total_sim_success_count": episode_total_sim_success_count,
            "total_sim_count": episode_total_sim_count,
            "use_guided_policy": POLICY,
            "use_guided_value": VALUE,
            "num_particles": NUM_SIMS_PER_PLAN,}
        json.dump(exp_data, f, indent=4)

    dump_json()


def log_shop_single_episode(tree: DiscreteNode,
                            episode_termination: TerminationT,
                            episode_agent_history: Tuple[HistoryEntry],
                            episode_list_time_taken_per_planning: List[float],
                            episode_total_sim_success_count: int,
                            episode_total_sim_count: int,
                            episode_n: int,
                            PROCESS_TIME_STAMP: str,
                            POLICY: str,
                            VALUE: str,
                            NUM_SIMS_PER_PLAN: int,
                            EXP_LOG_DIR: str):
    """Log a single execution information"""

    # Make log directory if not exist.
    os.makedirs(os.path.join(EXP_LOG_DIR, "tree"), exist_ok=True)

    # Compose filename
    exp_log_fname = PROCESS_TIME_STAMP
    exp_log_fname += f"_exp{episode_n}"
    if POLICY == "guided" or POLICY == "llm":
        exp_log_fname += f"_{POLICY}_policy"
    if VALUE:
        exp_log_fname += f"_{VALUE}_value"
    tree_fname = exp_log_fname + ".pickle"
    exp_log_fname += ".json"

    # Reuse some information from trajectory dict..
    episode_termination = episode_termination
    episode_action_history = [str(entry.action) for entry in episode_agent_history if entry.action is not None]
    # episode_state_history = [entry.state.state_description for entry in episode_agent_history if entry.state is not None]
    episode_total_reward = sum([entry.reward for entry in episode_agent_history if entry.reward is not None])
    ## TODO(SJ) : map to llm_proc in save_mdp_routine, gt_uid_to_class, gt_uid_to_color
    # episode_bullet_state_filenames = [entry.state.bullet_filename for entry in episode_agent_history if entry.state is not None]

    # Dump json log!
    with open(os.path.join(EXP_LOG_DIR, exp_log_fname), "w") as f:
        exp_data = {
            "termination": episode_termination,
            "total_reward": episode_total_reward,
            "list_time_taken_per_planning": episode_list_time_taken_per_planning,
            "episode_action_history": episode_action_history,
            # "episode_state_history": episode_state_history,
            # "episode_bullet_state_filenames": episode_bullet_state_filenames,
            "total_sim_success_count": episode_total_sim_success_count,
            "total_sim_count": episode_total_sim_count,
            "use_guided_policy": POLICY,
            "use_guided_value": VALUE,
            "num_particles": NUM_SIMS_PER_PLAN,}
        json.dump(exp_data, f, indent=4)

    save_tree(os.path.join(EXP_LOG_DIR, "tree", tree_fname), tree)



def convert_shop_state_to_transferrable_data(state: ShopState) -> Dict[str, Dict]:
    """Common routine for processing the state to a transferrable data format.
    This function only leaves the foreground masked image and depth.

    Args:
        state (ShopState)

    Returns:
        masked_depth_image (npt.NDArray): Foreground depth image, shape=(H, W)
        masked_rgb_image (npt.NDArray): Foreground RGB image, shape=(H, W, 3)
        grasp_contact (bool): Grasp contact
    """

    state_dict = dict()

    # Object state
    obj_state = dict()
    for name, entry in state.object_states.items():
        pos = entry.position
        orn = entry.orientation
        region = entry.region
        obj_state[name] = dict()
        obj_state[name]["pose"] = (pos, orn)
        obj_state[name]["region"] = region

    # string description
    # state_dict["description"] = state.state_description
    state_dict["obj_state"] = obj_state

    return state_dict

##############################################################################################################


def collect_shop_trajectory_data(
    agent: ShopAgent, history: Tuple[HistoryEntry, ...], termination: TerminationT
) -> Dict[str, object]:
    """Collect the trajectory data into a dictionary.

    Args:
        agent (FetchingAgent): Fetching agent instance
        history (Tuple[HistoryEntry]): A trajectory tuple
        termination (TerminationT): Termination condition of the trajectory.
    Returns:
        Dict: Some formatted dictionary.
    """
    goal_condition = agent.goal_condition.__dict__

    # Saving execution and simulation histories in the trajectory separately.
    DATA_COLLECTOR = {
        PHASE_EXECUTION: {
            "action": [],
            "state": [],
            "reward": [],
        },
        PHASE_SIMULATION: {
            "action": [],
            "state": [],
            "reward": [],
        },
        PHASE_ROLLOUT: {
            "action": [],
            "state": [],
            "reward": [],
        },
    }

    # Formatting history
    for i, history_entry in enumerate(history):
        action: ShopContinuousAction = history_entry.action
        state: ShopState = history_entry.state
        reward: float = history_entry.reward
        phase: PhaseT = history_entry.phase

        obj_state = convert_shop_state_to_transferrable_data(state)

        if i == 0:
            DATA_COLLECTOR[PHASE_EXECUTION]["state"].append(obj_state)
            # data_exec_rgbd.append(state.rgbd)
            # data_exec_state.append(obj_state)
            continue

        # Formatting action
        formatted_action = {
            "action_type":      action.discrete_action.type,
            "action_region":    action.region
                                if action.is_feasible() else None,
            "action_pos":       action.pos 
                                if action.is_feasible() else None,
            "action_orn":       action.orn 
                                if action.is_feasible() else None,
            "action_lang":      action.discrete_action.lang_command 
                                if action.is_feasible() else None,
        }
            



        if phase in [PHASE_EXECUTION, PHASE_SIMULATION, PHASE_ROLLOUT]:
            DATA_COLLECTOR[phase]["action"].append(formatted_action)
            DATA_COLLECTOR[phase]["state"].append(obj_state)
            DATA_COLLECTOR[phase]["reward"].append(reward)
        else:
            raise ValueError("Not a valid phase type.")

    data = {
        "goal_condition": goal_condition,
        "exec_action": DATA_COLLECTOR[PHASE_EXECUTION]["action"],
        "exec_state": DATA_COLLECTOR[PHASE_EXECUTION]["state"],
        "exec_reward": DATA_COLLECTOR[PHASE_EXECUTION]["reward"],
        "sim_action": DATA_COLLECTOR[PHASE_SIMULATION]["action"],
        "sim_state": DATA_COLLECTOR[PHASE_SIMULATION]["state"],
        "sim_reward": DATA_COLLECTOR[PHASE_SIMULATION]["reward"],
        "rollout_action": DATA_COLLECTOR[PHASE_ROLLOUT]["action"],
        "rollout_state": DATA_COLLECTOR[PHASE_ROLLOUT]["state"],
        "rollout_reward": DATA_COLLECTOR[PHASE_ROLLOUT]["reward"],
        "termination": termination,
    }

    return data


def mkdir_data_save_path(dataset_save_path: str):
    # List of subpaths to be created under dataset_save_path
    subpaths = [
        "groundtruth_sequence",
        "tree_sequence",
        "exec_dataset/json_data",
        "exec_dataset/npz_data",
        "sim_dataset/json_data",
        "sim_dataset/npz_data",
    ]

    # Iterate over each subpath and create it
    for subpath in subpaths:
        os.makedirs(os.path.join(dataset_save_path, subpath), exist_ok=True)


def mkdir_data_save_path(dataset_save_path: str):
    # List of subpaths to be created under dataset_save_path
    subpaths = [
        "groundtruth_sequence",
        "tree_sequence",
        "exec_dataset/json_data",
        "exec_dataset/npz_data",
        "sim_dataset/json_data",
        "sim_dataset/npz_data",
    ]

    # Iterate over each subpath and create it
    for subpath in subpaths:
        os.makedirs(os.path.join(dataset_save_path, subpath), exist_ok=True)

def mkdir_data_save_path_sftp(sftp: paramiko.SFTPClient, dataset_save_path_sftp: str):
    """Triple checking the directory exists."""
    subpaths = [
        "exec_dataset",
        "exec_dataset/json_data",
        "exec_dataset/npz_data",
        "sim_dataset",
        "sim_dataset/json_data",
        "sim_dataset/npz_data",
        "groundtruth_sequence",
        "tree_sequence"
    ]

    for subpath in subpaths:
        path = os.path.join(dataset_save_path_sftp, subpath)
        mkdir_sftp(sftp, path)


def save_trajectory_data_json_numpy(
    json_file_path: str, numpy_file_path: str, data: Dict[str, object]
):
    """Save trajectory data into json and numpy format.
    It detects the number of execution and simulation history from the `data` automatically.
    """

    # Dump json
    json_data = {
        "goal_condition": data["goal_condition"],
        "exec_action": data["exec_action"],
        "exec_reward": data["exec_reward"],
        "sim_action": data["sim_action"],
        "sim_reward": data["sim_reward"],
        "rollout_action": data["rollout_action"],
        "rollout_reward": data["rollout_reward"],
        "termination": data["termination"],
    }

    dump_json(json_data, json_file_path)

    # Dump numpy
    numpy_data = {}
    numpy_data["init_observation_depth"] = data['init_observation'][0]
    numpy_data["init_observation_rgb"] = data['init_observation'][1]
    numpy_data["init_observation_grasp"] = data['init_observation'][2]


    for idx, rgbd in enumerate(data["exec_observation"]):
        numpy_data[f"exec_observation_{idx}_depth"] = rgbd[0]
        numpy_data[f"exec_observation_{idx}_rgb"] = rgbd[1]
        numpy_data[f"exec_observation_{idx}_grasp"] = rgbd[2]
    
    for idx, rgbd in enumerate(data["sim_observation"]):
        numpy_data[f"sim_observation_{idx}_depth"] = rgbd[0]
        numpy_data[f"sim_observation_{idx}_rgb"] = rgbd[1]
        numpy_data[f"sim_observation_{idx}_grasp"] = rgbd[2]

    for idx, rgbd in enumerate(data["rollout_observation"]):
        numpy_data[f"rollout_observation_{idx}_depth"] = rgbd[0]
        numpy_data[f"rollout_observation_{idx}_rgb"] = rgbd[1]
        numpy_data[f"rollout_observation_{idx}_grasp"] = rgbd[2]



    np.savez(numpy_file_path, **numpy_data)


def save_mdp_trajectory_data_json_numpy(
    json_file_path: str, numpy_file_path: str, data: Dict[str, object]
):
    """Save trajectory data into json and numpy format.
    It detects the number of execution and simulation history from the `data` automatically.
    """

    # Dump json
    json_data = {
        "goal_condition":   data["goal_condition"],
        "exec_action":      data["exec_action"],
        "exec_state":       data["exec_state"],
        "exec_reward":      data["exec_reward"],
        "sim_action":       data["sim_action"],
        "sim_state":        data["sim_state"],
        "sim_reward":       data["sim_reward"],
        "rollout_action":   data["rollout_action"],
        "rollout_state":    data["rollout_state"],
        "rollout_reward":   data["rollout_reward"],
        "termination":      data["termination"],
    }

    dump_json(json_data, json_file_path)

    # Dump numpy
    numpy_data = {}
    for idx, rgbd in enumerate(data["exec_rgbd"]):
        numpy_data[f"exec_state_{idx}_rgbd"] = rgbd

    for idx, rgbd in enumerate(data["sim_rgbd"]):
        numpy_data[f"sim_state_{idx}_rgbd"] = rgbd

    for idx, rgbd in enumerate("rollout_rgbd"):
        numpy_data[f"rollout_state_{idx}_rgbd"] = rgbd

    np.savez(numpy_file_path, **numpy_data)
    return


def collect_tree(agent: ShopAgent):
    curr_node: DiscreteNode = agent.tree

    # find the real root
    while curr_node.parent is not None:
        curr_node = curr_node.parent

    return curr_node


def save_tree(tree_save_dir_path: str, tree: DiscreteNode):

    with open(tree_save_dir_path, "wb") as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)