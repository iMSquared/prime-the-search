import os, sys
import time
import wandb
from typing import Dict, Tuple, List
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["TQDM_DISABLE"]="1"

# PyBullet
from Simulation.pybullet.envs.common import init_new_bulletclient_pr2

# MDP
from Simulation.pybullet.mdp.MDP_framework import Environment, BlackboxModel, TERMINATION_SUCCESS, TERMINATION_FAIL
from Simulation.pybullet.mdp.MCTS import MCTS, LLMAssistedMCTS, LLMMCTS, ReAct, SayCan, Reflexion
from Simulation.pybullet.mdp.shop.shop_MDP import ShopTransitionModel, ShopRewardModel, ShopAgent
from Simulation.pybullet.mdp.shop.shop_problem_generator import  SimulatorShop
from Simulation.pybullet.mdp.shop.policy.llm_assisted_policy import ShopLLMassistedRolloutPolicy
from Simulation.pybullet.mdp.shop.policy.llm_assisted_llm_policy import ShopLLMassistedLLMPolicy, ShopLLMassistedPDDLPolicy
from Simulation.pybullet.mdp.shop.value.h_count import HCountValue
from Simulation.pybullet.predicate.predicate_shop import ShopPredicateManager

# Collecting data
from data_generation.collect_data import time_stamp, collect_shop_trajectory_data
from Simulation.pybullet.utils.exp_utils import *


def episode(config: Dict) -> Tuple:
    """Lifetime of this main function is one episode.

    Args:
        config (Dict): Configuration file
        episode_n (int): The number of current episode.
    Returns:
        Too much data...
    """

    # Configuration
    POLICY    = config["project_params"]["overridable"]["policy"]
    VALUE     = config["project_params"]["overridable"]["value"]
    ROLLOUT   = config["project_params"]["overridable"]["rollout"]
    COLLECT_DATA         = config["project_params"]["overridable"]["collect_data"]
    PLAN_MAX_DEPTH       = config["plan_params"]["max_depth"] + 1
    PLAN_DISCOUNT_FACTOR = config["plan_params"]["discount_factor"]
    TIME_LIMIT          = config["plan_params"]["time_limit"]

    # debug_data_reset()


    ProblemGenerator = SimulatorShop

    SimTransitionModel = ShopTransitionModel
    SimRewardModel = ShopRewardModel

    ExecutionTransitionModel = ShopTransitionModel
    ExecutionRewardModel = ShopRewardModel

    ## Initialization procedure

    # Connect to a new bullet client
    bc, shop_env, robot, manip, nav = init_new_bulletclient_pr2(config, stabilize=True, reset_roadmap=False)

    # Predicate manager
    predicate_manager = ShopPredicateManager(config, shop_env)

    # Setup 1: intialize Problem Generator and Belief Generator
    prob_generator = ProblemGenerator(bc, shop_env, robot, manip, nav, config)
    
    # Setup 2: Problem setup.
    initial_state, goal = prob_generator.set_initial_groundtruth()

    # Setup 3: Initialize blackbox models for simulation (planner)
    sim_transition_model   = SimTransitionModel(bc, shop_env, robot, manip, nav, config, goal, False, predicate_manager=predicate_manager)
    sim_reward_model       = SimRewardModel(bc, shop_env, robot, goal, manip, nav ,config, False, predicate_manager=predicate_manager)

    print("Initialized Problem")
    del prob_generator

    # Setup 5: Environment models
    exec_transition_model  = ExecutionTransitionModel(bc, shop_env, robot, manip, nav, config, goal, True, predicate_manager=predicate_manager)
    exec_reward_model      = ExecutionRewardModel(bc, shop_env, robot, goal, manip, nav, config, True, predicate_manager=predicate_manager)   
    
    # Setup 6: initialize guiding models
    if POLICY == "llm_next":
        policy_model         = ShopLLMassistedLLMPolicy(bc, shop_env, robot, manip, nav, config, predicate_manager=predicate_manager)
    elif POLICY == "llm_plan":
        policy_model   = ShopLLMassistedPDDLPolicy(bc, shop_env, robot, manip, nav, config, predicate_manager=predicate_manager)
    else:
        policy_model         = ShopLLMassistedRolloutPolicy(bc, shop_env, robot, manip, nav, config, predicate_manager=predicate_manager)

    if ROLLOUT == "random":
        rollout_policy_model = ShopLLMassistedRolloutPolicy(bc, shop_env, robot, manip, nav, config, predicate_manager=predicate_manager)
    elif ROLLOUT == "llm_plan":
        rollout_policy_model = ShopLLMassistedPDDLPolicy(bc, shop_env, robot, manip, nav, config, predicate_manager=predicate_manager)
    else:
        rollout_policy_model = None

    if VALUE == "hcount":
        value_model = HCountValue(bc, shop_env, robot, manip, nav, config)
    else:
        value_model = None

    # Setup 7: initialize agent
    blackbox_model = BlackboxModel(sim_transition_model, sim_reward_model)
    agent          = ShopAgent(bc, shop_env, robot, manip, nav, config, blackbox_model, 
                                   policy_model, rollout_policy_model, value_model, goal)
    
    # Setup 8: initialization environment
    env            = Environment(exec_transition_model, exec_reward_model, initial_state)
    
    # Setup 9: initialization of planner
    if config['baseline'] in ['SAYCAN']:
        planner = SayCan(agent, env, config)
    elif config['baseline'] in ['LLM_MCTS']:
        planner = LLMMCTS(agent, env, config)
    elif config['baseline'] in ["ONE_SHOT", "NO_UCT", "STaLM",]:
        planner = LLMAssistedMCTS(agent, env, config)
    elif config["baseline"] in ["ReAct"]:
        planner = ReAct(agent, env, config)
    elif config["baseline"] in ["Reflexion"]:  
        planner = Reflexion(agent, env, config)
    else: # "SAYCAN","MCTS_UCT", "LLM_MCTS", 
        planner        = MCTS(agent, env, config)

    # Log initial state
    agent.update(None, initial_state, None)
    
    print("Starting the episode...")


    # Plan+Execution loop
    #   Debugging data
    total_reward = 0.0
    episode_tree_sequence        = []
    episode_groundtruth_sequence = []
    #   Profiling data
    episode_num_sim_total       = 0
    episode_num_sim_success     = 0
    episode_time_taken_per_step = []
    episode_sim_trajs_total     = []

    start_planning_time = time.time()
    trajectory_table = []
    time_chart = []

    while (len(agent.history) < PLAN_MAX_DEPTH) and (time.time() - start_planning_time < TIME_LIMIT):
        # =====
        # Simulation (planning)
        # =====
        # Update pybullet client
        bc.disconnect()
        del bc, shop_env, robot, manip, nav
        bc, shop_env, robot, manip, nav = init_new_bulletclient_pr2(config, stabilize=False, reset_roadmap=False)
        
        exec_transition_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        exec_reward_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        sim_transition_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        sim_reward_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)

        policy_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        if rollout_policy_model is not None:
            rollout_policy_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        if value_model is not None:
            value_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        agent.set_new_bulletclient(bc, shop_env, robot, manip, nav)

        # Plan to the agent's goal
        ## NOTE: planner plan uses class variable agent from  agent.set_new_bulletclient
        next_action, time_taken, num_sim_total, num_sim_success, sim_trajs = planner.plan()

        # Collect debug data
        episode_num_sim_total   += num_sim_total
        episode_num_sim_success += num_sim_success
        episode_time_taken_per_step.append(time_taken)

        #   Planning data
        if COLLECT_DATA:
            episode_sim_trajs_total += sim_trajs
            episode_tree_sequence.append(agent.tree)
        episode_groundtruth_sequence.append(env.current_state)    # State before the execution. This is very useful even for logging.

        # =====
        # Execution
        # =====
        # Update pybullet client
        bc.disconnect()
        del bc, shop_env, robot, manip
        bc, shop_env, robot, manip, nav = init_new_bulletclient_pr2(config, stabilize=False, reset_roadmap=False)
        sim_transition_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        sim_reward_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        exec_transition_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        exec_reward_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        policy_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        if rollout_policy_model is not None:
            rollout_policy_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        if value_model is not None:
            value_model.set_new_bulletclient(bc, shop_env, robot, manip, nav)
        agent.set_new_bulletclient(bc, shop_env, robot, manip, nav)

        # Restore the ground truth state in simulation
        print("In execution...")
        agent.imagine_state(env.current_state)
        
        # Execution in real world
        reward, termination = env.execute(next_action)
        total_reward = reward + PLAN_DISCOUNT_FACTOR * total_reward

        try: 
            trajectory_table.append([str(next_action.discrete_action),
                            str(next_action.is_feasible()),
                            time_taken,
                            agent._success_plan_found,
                            termination,
                            reward])
        except:
            trajectory_table.append(["N/A",
                            "N/A",
                            time_taken,
                            agent._success_plan_found,
                            termination,
                            reward])
        time_chart.append([time.time() - start_planning_time, getattr(agent, 'ACC_LLM_CALL_TIME', 0), getattr(agent, 'LLM_CALL_CNT', 0)])
        
        # Logging & Data collection
        execution_history = ""
        infeasible_action_cnt = 0 # This is for preventing waste of LLM in case of deadlocks.
        for i in range(len(agent.history)):
            if i == 0: continue
            if agent.history[i].action is None: continue
            if agent.history[i].action.is_feasible() == False:
                infeasible_action_cnt += 1
            execution_history += f"Action {i}: {str(agent.history[i].action)}\n"
        logger.info(f"[previous actions]\n {execution_history}[next_action at depth {len(agent.history)}] {next_action}")
        LLOG.depth_idx += 1
        LLOG.sim_idx = 0

        if next_action is None:
            if agent._success_plan_found:
                termination = TERMINATION_SUCCESS
            else:
                termination = TERMINATION_FAIL
        else:
            # Update history and belief state
            # Update search tree (clean or reuse whatever...)
            agent.update(next_action, env.current_state, reward)
            planner.update(agent, next_action, env.current_state)


        # Check termination condition!
        if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):
            break


    LLOG.depth_idx = 0
    logger.info(f"Termination: {termination}")

    # Finalizing the planning!
    bc.disconnect()
    # Always collect execution data. It is useful for debugging.
    episode_groundtruth_sequence.append(env.current_state)
    episode_agent_history = agent.history
    ## Collecting data on execution
    episode_exec_traj = collect_shop_trajectory_data(
        agent       = agent,
        history     = agent.history,
        termination = termination)
    if not COLLECT_DATA:
        episode_sim_trajs_total = None
        episode_tree_sequence   = None

    # TODO (dlee): remove this
    tree = None

    logger.info(f"ToT LLM CALLS: {getattr(agent, 'LLM_CALL_CNT', 0)}")

    return termination, total_reward, \
            episode_num_sim_total, episode_num_sim_success, \
            episode_time_taken_per_step, \
            episode_agent_history, \
            episode_sim_trajs_total, episode_exec_traj, \
            episode_tree_sequence, episode_groundtruth_sequence, tree, trajectory_table, time_chart


def main(config      : Dict, 
         num_episodes: int): 
    """Project main"""

    # Load OpenAI token
    load_openai_token()


    # Set configs.
    NUM_SIMS_PER_PLAN     : int  = config["plan_params"]["num_sims"]
    PROCESS_TIME_STAMP    : str  = time_stamp()
    POLICY     : bool = config["project_params"]["overridable"]["policy"]
    VALUE      : bool = config["project_params"]["overridable"]["value"]
    COLLECT_DATA          : bool = config["project_params"]["overridable"]["collect_data"]
    DATASET_SAVE_PATH     : str  = config["project_params"]["overridable"]["default_dataset_save_path"]
    EXP_LOG_DIR           : str  = config["project_params"]["overridable"]["default_exp_log_dir_path"]
    print(f"NUM_SIMS_PER_PLAN     : {NUM_SIMS_PER_PLAN}")
    print(f"PROCESS_TIME_STAMP    : {PROCESS_TIME_STAMP}")
    print(f"USE_GUIDED_POLICY     : {POLICY}")
    print(f"USE_GUIDED_VALUE      : {VALUE}")
    print(f"COLLECT_DATA          : {COLLECT_DATA}")
    print(f"DATASET_SAVE_PATH     : {DATASET_SAVE_PATH}")
    print(f"EXP_LOG_DIR           : {EXP_LOG_DIR}")

    os.makedirs(os.path.join(EXP_LOG_DIR, "BulletState"), exist_ok=True)
    run_tags, run_name = tags_for_logging(config)

    ## Some High Level Metrics
    # Repeat executions
    num_exec_success = 0
    num_sim          = 0
    num_sim_success  = 0
    total_time       = 0.0
    total_time_success = 0.0

    master_run_name = run_name+PROCESS_TIME_STAMP
    master_run = wandb.init(
        config=config,
        tags =run_tags,
        name = master_run_name,
        project=config["project_params"]["project_name"]
    )
    table_columns = ['action', 'is_feasible', 'time_taken', 'success_found', 'termination', 'reward', 'Episode', 'Depth']
    table_data = wandb.Table(columns=table_columns)
    time_table_data = wandb.Table(columns=['Total_time_taken', 'LLM_TAKE_TIME', 'LLM_CALL_CNT', 'Episode', 'Depth'])

    for episode_n in range(num_episodes):
        experiment_logger = logger.add(config["project_params"]["overridable"]["default_exp_log_dir_path"] + f"/experiment_{episode_n+1}.log", 
                                       level="TRACE",
                                       enqueue=True,
                                       diagnose=True,)
        logger.info(f"Episode {episode_n+1} / {num_episodes}")
        LLOG.episode_idx = episode_n+1

        # One episode
        episode_termination, episode_total_reward, episode_num_sim_total, episode_num_sim_success, episode_time_taken_per_step, \
            _, _, _, _, _, _, trajectory_table, time_chart = episode(config)

        ### Logging and data collection ###
        # Log one execution
        total_time_taken = sum(episode_time_taken_per_step)
        for idx, t in enumerate(episode_time_taken_per_step):
            logger.info(f"Planning time at depth {idx}: {t}")

        logger.info(f"Execution Result of {episode_n+1}: {episode_termination}, time taken - {total_time_taken}s, total reward - {episode_total_reward}")
        for i, t in enumerate(episode_time_taken_per_step):
            print(f"Planning time at depth {i}: {t}")
        # Log success count
        if episode_termination == TERMINATION_SUCCESS:
            num_exec_success += 1
            total_time_success += total_time_taken

        num_sim += episode_num_sim_total
        num_sim_success += episode_num_sim_success
        total_time += total_time_taken
        
        for idx, t in enumerate(trajectory_table):
            t.append(episode_n+1)
            t.append(idx+1)
            table_data.add_data(*t)

        for idx, _time_chart in enumerate(time_chart):
            _time_chart.append(episode_n+1)
            _time_chart.append(idx+1)
            time_table_data.add_data(*_time_chart)

        master_run.log({
            f"trajectory {episode_n+1}": wandb.Table(data=trajectory_table, columns=['action', 'is_feasible', 'time_taken', 'success_found', 'termination', 'reward', 'Episode', 'Depth']),
            f"time chart {episode_n+1}": wandb.Table(data=time_chart, columns=['Total_time_taken', 'LLM_TAKE_TIME', 'LLM_CALL_CNT', 'Episode', 'Depth']),
            "LLM call count": time_chart[-1][-3],
            'episode_success': 1 if episode_termination == TERMINATION_SUCCESS else 0,
            'total_time_taken': total_time_taken,})

        logger.log("LLM_PARSED", f"Episode simulation success: {episode_num_sim_success}/{episode_num_sim_total}")
        logger.info(f"Total execution success: {num_exec_success} / {episode_n+1}", num_exec_success, episode_n+1)
        logger.info(f"Total simulation success: {num_sim_success} / {num_sim}", num_sim_success, num_sim)
        logger.remove(experiment_logger)
        
    master_run.log({"SUCCESS_RATE": num_exec_success/num_episodes, 
            "AVERAGE_TIME_FOR_SUCCESS": 0 if num_exec_success == 0 else total_time_success/num_exec_success,
            "AVERAGE_TIME": total_time/num_episodes,})
    master_run.finish()



if __name__=="__main__":

    params = get_params()
    config = override_configs(params)
    dump_config(config)

    # main...
    main(config       = config, 
         num_episodes = params.num_episodes)
    
