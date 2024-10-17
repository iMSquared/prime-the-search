import pybullet as pb
import math

from typing import Dict, Tuple

from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.imm.pybullet_tools.utils import connect
from Simulation.pybullet.envs.shop_env import ShopEnv, ShopDebugEnv
from Simulation.pybullet.envs.robot import PR2
from Simulation.pybullet.envs.manipulation import PR2Manipulation
from Simulation.pybullet.envs.navigation import Navigation


def init_new_bulletclient_pr2(config      : Dict,
                          force_direct: bool = False,
                          stabilize   : bool = True,
                          reset_roadmap: bool = True,
                          suppress_output: bool = True) \
        -> Tuple[ BulletClient, ShopEnv, PR2, PR2Manipulation, Navigation]:
    """Some common routines of bullet client initialization.
    This function does not destroy previous client.

    Args:
        config (Dict): Configuration file
        force_direct (bool): Force the pybullet to be in direct mode. Used in the belief update.
        stabilize (bool): When true, steps several times to stabilize the environment.
        
    Returns:
        bc (BulletClient): New client
        binpick_env (BinpickEnvPrimitive): New env
        robot (PR2): New robot
    """

    # Configuration
    DEBUG_SHOW_GUI         = config["project_params"]["debug"]["show_gui"]
    CONTROL_HZ             = config["sim_params"]["control_hz"]
    GRAVITY                = config["sim_params"]["gravity"]
    CAMERA_DISTANCE        = config["sim_params"]["debug_camera"]["distance"]
    CAMERA_YAW             = config["sim_params"]["debug_camera"]["yaw"]
    CAMERA_PITCH           = config["sim_params"]["debug_camera"]["pitch"]
    CAMERA_TARGET_POSITION = config["sim_params"]["debug_camera"]["target_position"]

    # Connect bullet client

    if DEBUG_SHOW_GUI and not force_direct:
        # sim_id = connect(use_gui=True)
        sim_id =  pb.connect(pb.GUI)
    else:
        sim_id = connect(use_gui=False)
    bc = BulletClient(sim_id)
    print("env sim id: ", bc.sim_id)
    # Sim params
    CONTROL_DT = 1. / CONTROL_HZ
    bc.setTimeStep(CONTROL_DT)
    bc.setGravity(0, 0, GRAVITY)
    bc.resetDebugVisualizerCamera(
        cameraDistance       = CAMERA_DISTANCE, 
        cameraYaw            = CAMERA_YAW, 
        cameraPitch          = CAMERA_PITCH, 
        cameraTargetPosition = CAMERA_TARGET_POSITION )
    bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING, 1)

    # Simulation initialization
    env = ShopEnv(bc, config, suppress_output=suppress_output)
    # return bc, env, None, None, None
    robot       = PR2(bc, config, suppress_output=suppress_output)

    if stabilize:
        # run physics for 1 second
        for _ in range(int(CONTROL_HZ)):
            bc.stepSimulation()
    nav         = Navigation(bc, env, robot, config)

    # Set up roadmap
    nav.reset_empty_roadmap(use_pickle=(not reset_roadmap))
    nav.reset_roadmap(use_pickle=(not reset_roadmap), door_open=env.all_obj["kitchen_door"].is_open)
    manip       = PR2Manipulation(bc, env, robot, nav, config)
    

    return bc, env, robot, manip, nav

