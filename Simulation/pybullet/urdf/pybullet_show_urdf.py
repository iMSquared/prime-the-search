import pybullet as p
import time
import math
from datetime import datetime
from time import sleep

p.connect(p.GUI)


# robot_id = p.loadURDF("/commonsense_robotics/Simulation/pybullet/urdf/small_shelf/shelf.urdf", useFixedBase=True)
# robot_id = p.loadURDF("/home/im2/Projects/iai_maps/iai_kitchen_defs/room/allInOne.urd.xacro", useFixedBase=False)
robot_id = p.loadSDF("/home/im2/Projects/commonsense_robotics/Simulation/pybullet/urdf/kitchen_scene.sdf")





# robot_id = p.loadURDF("/home/im2/Projects/commonsense_robotics/Simulation/pybullet/urdf/rooms_environment.urdf", useFixedBase=True)

p.setGravity(0, 0, 0)

while True:

    p.stepSimulation()
