#!/usr/bin/env python3
"""Example script for motion planning with a robot arm in pybullet."""

import os
import sys
import pybullet as pb
import pybullet_data
import numpy as np
from typing import List, Tuple
from contextlib import contextmanager
import logging
import time

from imm.pybullet_util.typing_extra import Tuple3, TranslationT, QuaternionT
from imm.pybullet_util.bullet_client import BulletClient
from imm.pybullet_util.common import (
    get_joint_limits, get_joint_positions,
    get_link_pose)
from imm.pybullet_util.collision import ContactBasedCollision

from imm.motion_planners.rrt_connect import birrt


@contextmanager
def imagine(bc: BulletClient) -> int:
    """Temporarily change the simulation, but restore the state afterwards.

    NOTE(ycho): Do not abuse this function! saveState() is quite expensive.
    """
    try:
        state_id = bc.saveState()
        yield state_id
    finally:
        bc.restoreState(state_id)
        bc.removeState(state_id)


def normalized_angle(x: float) -> float:
    """Map angle in range (-pi,pi)."""
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def wrap_to_joint_limits(q: Tuple[float, ...],
                         q_lim: Tuple[Tuple[float, ...],
                                      Tuple[float, ...]]) -> Tuple[float, ...]:
    """Naive utility function to ensure `q` remains within joint limits."""
    q = normalized_angle(np.asanyarray(q))
    q_lim = np.asanyarray(q_lim)

    lo = q < q_lim[0]
    q[lo] += (2 * np.pi)
    hi = q >= q_lim[1]
    q[hi] -= (2 * np.pi)
    return q


def ik_fun(bc: BulletClient, robot_id: int, ee_id: int,
           pos: TranslationT, orn: QuaternionT, *args, **kwds) -> List[float]:
    """Inverse kinematics wrapper around pybullet numerical IK."""
    return bc.calculateInverseKinematics(
        robot_id, ee_id, pos, orn, *args, **kwds)


def add_cubes(bc: BulletClient, rng: np.random.Generator,
              num_cubes: int, radius: Tuple3[float]) -> List[int]:
    """Add a bunch of cubes around the workspace."""
    cube_mass: float = 0.1
    cube_color: Tuple4[float] = (1, 0, 0, 1)
    pos_lo: Tuple3[float] = (-1, -1, 0)
    pos_hi: Tuple3[float] = (+1, +1, 2)

    if num_cubes <= 0:
        return

    col_id = bc.createCollisionShape(bc.GEOM_BOX,
                                     halfExtents=radius)
    vis_id = bc.createVisualShape(
        bc.GEOM_BOX,
        halfExtents=radius,
        rgbaColor=cube_color
    )

    body_ids = []
    for _ in range(num_cubes):
        body_id = bc.createMultiBody(cube_mass, col_id, vis_id)
        while True:
            pos = rng.uniform(pos_lo, pos_hi)
            orn = bc.getQuaternionFromEuler(
                rng.uniform(-np.pi, np.pi, size=3))
            bc.resetBasePositionAndOrientation(body_id, pos, orn)
            bc.performCollisionDetection()
            if len(bc.getContactPoints(body_id)) > 0:
                continue
            break
        body_ids.append(body_id)
    return body_ids


def main():
    # Parameters for running this example.
    seed: int = 0
    num_cubes: int = 16
    cube_radius: Tuple3[float] = (0.05, 0.05, 0.05)
    ee_pos_lo: Tuple3[float] = (-1, -1, 0)
    ee_pos_hi: Tuple3[float] = (+1, +1, +2)
    min_translation: float = 1.0
    min_rotation: float = np.deg2rad(30)
    max_pos_tol: float = 0.05
    max_orn_tol: float = np.deg2rad(5)
    num_ik_iter: int = 1024
    max_ik_residual: float = 0.01
    delay: float = 0.03
    log_level: str = 'WARN'
    line_color: Tuple3[float] = (0, 0, 1)
    line_width: float = 4
    line_lifetime: float = 5.0

    # Configure logging.
    logging.root.setLevel(log_level)
    logging.basicConfig()
    # Connect to pybullet simulator.
    sim_id: int = pb.connect(pb.GUI)
    if sim_id < 0:
        raise ValueError('Failed to connect to simulator!')
    bc = BulletClient(sim_id)
    # Configure random seed.
    rng = np.random.default_rng(seed)

    # Load scene.
    bc.setAdditionalSearchPath(
        pybullet_data.getDataPath())
    plane_id: int = bc.loadURDF('plane.urdf', (0, 0, -0.3))
    robot_id: int = bc.loadURDF('kuka_iiwa/model.urdf', (0, 0, 0),
                                baseOrientation=(0, 0, 0, 1))
    ee_id: int = 6  # NOTE(ycho): hardcoded end-effector joint for kuka robot
    joint_ids = list(range(6))
    num_joints = bc.getNumJoints(robot_id)
    joint_limits = get_joint_limits(bc, robot_id, joint_ids)
    add_cubes(bc, rng, num_cubes, cube_radius)

    # Loop through a bunch of motions.
    while True:
        # Query valid target.
        bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING, 0)
        q_src = get_joint_positions(bc, robot_id, range(6))
        src_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)
        while True:
            # Sample target EE pose.
            pos = rng.uniform(ee_pos_lo, ee_pos_hi)
            orn = bc.getQuaternionFromEuler(
                rng.uniform(-np.pi, np.pi, size=3))

            # Try to enforce a large-ish motion.
            d_pos = np.subtract(pos, src_ee_pose[0])
            d_ang = bc.getAxisAngleFromQuaternion(
                bc.getDifferenceQuaternion(orn, src_ee_pose[1]))
            if np.linalg.norm(d_pos) < min_translation:
                logging.debug('skip due to small translation')
                continue
            if abs(normalized_angle(d_ang[1])) < min_rotation:
                logging.debug('skip due to small rotation')
                continue

            # Run inverse kinematics.
            q_dst = ik_fun(bc, robot_id, ee_id, pos, orn,
                           maxNumIterations=num_ik_iter,
                           residualThreshold=max_ik_residual,
                           )[:6]

            # Ensure within joint limits.
            q_dst = wrap_to_joint_limits(q_dst, joint_limits)
            if (np.any(q_dst < joint_limits[0])
                    or np.any(q_dst >= joint_limits[1])):
                logging.debug('skip due to joint limit violation')
                continue
            for i, v in zip(range(6), q_dst):
                bc.resetJointState(robot_id, i, v)

            # Check EE pose at specified joint positions.
            # NOTE(ycho): calculateInverseKinematics() operates on
            # the link coordinate, not CoM(inertial) coordinate.
            dst_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)

            d_pos = np.subtract(pos, dst_ee_pose[0])
            d_ang = bc.getAxisAngleFromQuaternion(
                bc.getDifferenceQuaternion(orn, dst_ee_pose[1]))
            if np.linalg.norm(d_pos) >= max_pos_tol:
                logging.debug('skip due to positional deviation')
                continue
            if abs(normalized_angle(d_ang[1])) >= max_orn_tol:
                logging.debug('skip due to orientation deviation')
                continue

            bc.performCollisionDetection()
            if len(bc.getContactPoints(robot_id)) > 0:
                logging.debug('skip due to collision')
                continue
            break
        bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING, 1)

        # Reset joint to original state.
        for i, v in zip(range(6), q_src):
            bc.resetJointState(robot_id, i, v)

        # Compute motion plan.
        collision_fn = ContactBasedCollision(bc,
                                             robot_id, joint_ids,
                                             [], [], joint_limits, {})

        def distance_fn(q0: np.ndarray, q1: np.ndarray):
            return np.linalg.norm(np.subtract(q1, q0))

        def sample_fn():
            return rng.uniform(joint_limits[0], joint_limits[1])

        def extend_fn(q0: np.ndarray, q1: np.ndarray):
            dq = np.subtract(q1, q0)  # Nx6
            return q0 + np.linspace(0, 1)[:, None] * dq

        with imagine(bc):
            q_trajectory = birrt(
                q_src,
                q_dst,
                distance_fn,
                sample_fn,
                extend_fn,
                collision_fn)

        # Disable default joints.
        bc.setJointMotorControlArray(
            robot_id,
            joint_ids,
            bc.VELOCITY_CONTROL,
            targetVelocities=np.zeros(len(joint_ids)),
            forces=np.zeros(
                len(joint_ids)))

        # Execute the trajectory.
        prv_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)
        for q in q_trajectory:
            bc.setJointMotorControlArray(robot_id, joint_ids,
                                         bc.POSITION_CONTROL,
                                         targetPositions=q)
            bc.stepSimulation()
            cur_ee_pose = get_link_pose(bc, robot_id, ee_id, inertial=False)
            bc.addUserDebugLine(
                prv_ee_pose[0],
                cur_ee_pose[0],
                line_color,
                line_width, line_lifetime)
            prv_ee_pose = cur_ee_pose
            if len(bc.getContactPoints(robot_id)) > 0:
                raise ValueError('Invalid motion plan resulted in collision!')
            time.sleep(delay)


if __name__ == '__main__':
    main()
