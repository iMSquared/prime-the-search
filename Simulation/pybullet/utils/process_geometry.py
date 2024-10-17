import numpy as np
from Simulation.pybullet.imm.pybullet_util.bullet_client import BulletClient
from Simulation.pybullet.imm.pybullet_util.typing_extra import TranslationT, EulerT
from typing import Optional, Tuple, List

def matrix_visual_to_com(bc:BulletClient, uid: int):
    """ Get transformation from vesh to link frame

    Args:
        bc (BulletClient): bullet client id
        uid (int): URDF instance id
    """

    # NOTE(ssh): 
    #   In URDF, the origin tag works with the following order
    #   1. rpy 
    #   2. xyz
    #   3. scaling 


    # Parse base link info
    base_link_info = bc.getVisualShapeData(uid)[0]  # [0] is base link

    # Create the transformation matrix
    # local visual frame(mesh design) -> link frame(center of mass)
    v2l_pos = base_link_info[5]
    v2l_orn = np.reshape( bc.getMatrixFromQuaternion(base_link_info[6]), 
                          (3, 3) )
    v2l_scale = base_link_info[3]

    scaling_matrix = np.array([[v2l_scale[0], 0, 0, 0],
                               [0, v2l_scale[1], 0, 0],
                               [0, 0, v2l_scale[2], 0],
                               [0, 0, 0, 1]])

    translation_matrix = np.array([[1, 0, 0, v2l_pos[0]],
                                   [0, 1, 0, v2l_pos[1]],                                   
                                   [0, 0, 1, v2l_pos[2]],
                                   [0, 0, 0, 1]]) 
                                   
    rotation_matrix = np.array([[v2l_orn[0,0], v2l_orn[0,1], v2l_orn[0,2], 0],
                                [v2l_orn[1,0], v2l_orn[1,1], v2l_orn[1,2], 0],    
                                [v2l_orn[2,0], v2l_orn[2,1], v2l_orn[2,2], 0],
                                [0, 0, 0, 1]])

    T_v2l = np.matmul(translation_matrix, rotation_matrix)
    T_v2l = np.matmul(T_v2l, scaling_matrix)


    return T_v2l


def matrix_com_to_base(bc:BulletClient, uid: int):
    '''
    Get transformation matrix 
    from the local link frame (aka.`center of mass`) frame
    to `URDF link base` frame.
    
    params
    - bc: Bullet clinet
    - uid: Object uid to generate point cloud

    returns
    - pcd: A point cloud
    - T_l2b: A transformation matrix from link frame(center of mass) to URDF link base frame. 
    '''
    # Parse dynamics info
    dynamics_info = bc.getDynamicsInfo(uid, -1)

    # Create the transformation matrix
    # URDF link base frame -> link frame(center of mass)
    cm_pos = dynamics_info[3]
    cm_orn = np.reshape(
        bc.getMatrixFromQuaternion(dynamics_info[4]), 
        (3, 3))

    translation_matrix = np.array([[1, 0, 0, cm_pos[0]],
                                   [0, 1, 0, cm_pos[1]],                                   
                                   [0, 0, 1, cm_pos[2]],
                                   [0, 0, 0, 1]]) 
    rotation_matrix = np.array([[cm_orn[0,0], cm_orn[0,1], cm_orn[0,2], 0],
                                [cm_orn[1,0], cm_orn[1,1], cm_orn[1,2], 0],    
                                [cm_orn[2,0], cm_orn[2,1], cm_orn[2,2], 0],
                                [0, 0, 0, 1]])

    # Transformation matrix (link -> visual)
    T_b2l = np.matmul(translation_matrix, rotation_matrix)
    # Transformation matrix (visual -> link): This is what we need.
    T_l2b = np.linalg.pinv(T_b2l)

    return T_l2b



def matrix_base_to_world(bc:BulletClient, uid:int):
    '''
    Get transformation matrix 
    from the local `urdf base` frame
    to world frame.
    '''
    # NOTE(ssh): hardcoded for readability
    obj_pos, obj_orn = bc.getBasePositionAndOrientation(uid)                    # Get (x, y, z) position and (quaternion) orientation
    obj_orn = np.reshape(bc.getMatrixFromQuaternion(obj_orn), (3, 3))           # Convert (quaternion) to (rotation matrix)
    translation_matrix = np.array([[1, 0, 0, obj_pos[0]],                       # Convert (obj_pos) to (homogeneous translation)
                                   [0, 1, 0, obj_pos[1]],                                   
                                   [0, 0, 1, obj_pos[2]],
                                   [0, 0, 0, 1]])  
    rotation_matrix = np.array([[obj_orn[0,0], obj_orn[0,1], obj_orn[0,2], 0],  # Convert (obj_orn) to (homogeneous rotation)
                                [obj_orn[1,0], obj_orn[1,1], obj_orn[1,2], 0],    
                                [obj_orn[2,0], obj_orn[2,1], obj_orn[2,2], 0],
                                [0, 0, 0, 1]])
    T_world = np.matmul(translation_matrix, rotation_matrix)                    # We have to rotate first.

    return T_world



def matrix_world_to_camera(view_matrix:np.ndarray):
    """
    Get transformation from the camera frame to world frame (hardcoded for readability)
    """
    view_matrix = np.transpose(np.reshape(view_matrix, (4, 4)))         # Transposing the view matrix for notation reason...
    coord_swap_matrix = np.array([[0, 0, -1, 0],                        # Coordinate swap: view_matrix has -z axis for depth
                                  [-1, 0, 0, 0],                        # Swaping to x for depth  (-z -> x), (-x -> y), (y -> z)
                                  [0, 1, 0, 0],
                                  [0, 0, 0, 1]])
    T_w2c = np.matmul(coord_swap_matrix, view_matrix)

    return T_w2c


def matrix_camera_to_world(view_matrix:np.ndarray):
    """
    Get transformation from the world frame to camera frame (hardcoded for readability)
    """

    T_w2c = matrix_world_to_camera(view_matrix)
    T_c2w = np.linalg.pinv(T_w2c)

    return T_c2w



def draw_coordinate(bc: BulletClient, 
                    target_pos: Optional[TranslationT]=None, 
                    target_orn_e: Optional[EulerT]=None, 
                    parent_object_unique_id: Optional[int] = None, 
                    parent_link_index: Optional[int] = None, 
                    line_uid_xyz: Optional[Tuple[int, int, int]] = None,
                    brightness: float = 1.0) -> Tuple[int, int, int]:
    """Draw coordinate frame

    Args:
        bc (BulletClient): PyBullet client
        target_pos (Optional[TranslationT], optional): Position of local frame in global frame
        target_orn_e (Optional[Tuple3], optional): Orientation of local frame in global frame
        parent_object_unique_id (Optional[int], optional): Local frame? Defaults to None.
        parent_link_index (Optional[int], optional): Local frame? Defaults to None.
        line_uid (Tuple[int, int, int], optional): Replace uid. Defaults to None.
        brightness (float): Color brightness

    Returns:
        line_uid_xyz (Tuple[int, int, int]): Line uid
    """

    origin_pos = [0.0, 0.0, 0.0]
    x_pos = [0.1, 0.0, 0.0]
    y_pos = [0.0, 0.1, 0.0]
    z_pos = [0.0, 0.0, 0.1]
    origin_orn_e = [0.0, 0.0, 0.0]


    if parent_object_unique_id is not None:
        if line_uid_xyz is not None:
            line_uid_x = bc.addUserDebugLine(origin_pos, x_pos, [1*brightness, 0, 0], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[0])
            line_uid_y = bc.addUserDebugLine(origin_pos, y_pos, [0, 1*brightness, 0], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[1])
            line_uid_z = bc.addUserDebugLine(origin_pos, z_pos, [0, 0, 1*brightness], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[2])
        else:
            line_uid_x = bc.addUserDebugLine(origin_pos, x_pos, [1*brightness, 0, 0], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
            line_uid_y = bc.addUserDebugLine(origin_pos, y_pos, [0, 1*brightness, 0], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
            line_uid_z = bc.addUserDebugLine(origin_pos, z_pos, [0, 0, 1*brightness], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
    else:
        target_origin_pos, target_origin_orn_q = bc.multiplyTransforms(target_pos, bc.getQuaternionFromEuler(target_orn_e),
                                                                       origin_pos, bc.getQuaternionFromEuler(origin_orn_e)) 
        target_x_pos, target_x_orn_q = bc.multiplyTransforms(target_pos, bc.getQuaternionFromEuler(target_orn_e),
                                                             x_pos, bc.getQuaternionFromEuler(origin_orn_e))
        target_y_pos, target_y_orn_q = bc.multiplyTransforms(target_pos, bc.getQuaternionFromEuler(target_orn_e),
                                                             y_pos, bc.getQuaternionFromEuler(origin_orn_e))
        target_z_pos, target_z_orn_q = bc.multiplyTransforms(target_pos, bc.getQuaternionFromEuler(target_orn_e),
                                                             z_pos, bc.getQuaternionFromEuler(origin_orn_e))
        
        if line_uid_xyz is not None:
            line_uid_x = bc.addUserDebugLine(target_origin_pos, target_x_pos, [1*brightness, 0, 0], 
                                            lineWidth = 0.01,
                                            replaceItemUniqueId = line_uid_xyz[0])
            line_uid_y = bc.addUserDebugLine(target_origin_pos, target_y_pos, [0, 1*brightness, 0], 
                                            lineWidth = 0.01,
                                            replaceItemUniqueId = line_uid_xyz[1])
            line_uid_z = bc.addUserDebugLine(target_origin_pos, target_z_pos, [0, 0, 1*brightness], 
                                            lineWidth = 0.01,
                                            replaceItemUniqueId = line_uid_xyz[2])
        else:
            line_uid_x = bc.addUserDebugLine(target_origin_pos, target_x_pos, [1*brightness, 0, 0], 
                                            lineWidth = 0.01)
            line_uid_y = bc.addUserDebugLine(target_origin_pos, target_y_pos, [0, 1*brightness, 0], 
                                            lineWidth = 0.01)
            line_uid_z = bc.addUserDebugLine(target_origin_pos, target_z_pos, [0, 0, 1*brightness], 
                                            lineWidth = 0.01)

    return (line_uid_x, line_uid_y, line_uid_z)



def l3_norm(position1: TranslationT, position2: TranslationT):
    """Simple l3 norm function"""
    norm = np.linalg.norm(np.array(position1)-np.array(position2))
    return norm



def random_sample_array_from_config(center: List[float], 
                                    half_ranges: List[float]) -> List[float]:
    """Randomly sample a array given center and half ranges"""

    if len(center) != len(half_ranges):
        raise ValueError("Length mismatch")
    
    rand = np.random.uniform(low=-1, high=1, size=len(center))
    sampled = center + rand*half_ranges

    return sampled.tolist()


def random_sample_region_from_config(taskspace_x_range: List[float],
                                     taskspace_y_range: List[float],
                                     taskspace_z_range: List[float],
                                     grid: List[int],
                                     region: int):
    
    x_unit = (taskspace_x_range[1] - taskspace_x_range[0])/grid[0]
    y_unit = (taskspace_y_range[1] - taskspace_y_range[0])/grid[1]
    unit = np.array([x_unit, y_unit])
    xy_min = np.array([taskspace_x_range[0], taskspace_y_range[0]])

    rand = np.random.uniform(size=(2,))

    xy = np.array([region//grid[0], region%grid[0]]) + rand
    
    coord = unit*xy + xy_min

    # Sample z
    z = np.random.uniform(low=taskspace_z_range[0], high=taskspace_z_range[1])

    return coord.tolist() + [z]