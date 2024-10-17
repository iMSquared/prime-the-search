import open3d as o3d
import pickle
import numpy as np



o3d_mesh = o3d.io.read_triangle_mesh("non_target/cuboid12/cuboid12.obj")
o3d_mesh.compute_vertex_normals()


o3d_pcd = o3d_mesh.sample_points_poisson_disk(5000)
print("Recompute the normal of the downsampled point cloud")
o3d_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=6, max_nn=50))
o3d_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 10.]))
normals = np.asarray(o3d_pcd.normals)
normals = -normals  # Outward
o3d_pcd.normals = o3d.utility.Vector3dVector(normals)
o3d_pcd.normalize_normals()


frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5, origin=[0, 0, 0])
o3d.visualization.draw_geometries([o3d_pcd, frame],
                                  point_show_normal=True)


new_data = {}
new_data["points"] = np.asarray(o3d_pcd.points)
new_data["normals"] = np.asarray(o3d_pcd.normals)
with open("non_target/cuboid12/cuboid_point_cloud_fixed.pickle", "wb") as f:
    pickle.dump(new_data, f)







# with open("./cuboid12_point_cloud.pickle", "rb") as f:
#     data = pickle.load(f)


# print(data["points"])
# print(data["normals"])


# o3d_pcd = o3d.geometry.PointCloud()
# o3d_pcd.points = o3d.utility.Vector3dVector(np.asarray(data["points"]))


# print("Recompute the normal of the downsampled point cloud")
# o3d_pcd.estimate_normals(
#     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=6, max_nn=50))
# o3d_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 10.]))
# normals = np.asarray(o3d_pcd.normals)
# normals = -normals  # Outward
# o3d_pcd.normals = o3d.utility.Vector3dVector(normals)
# o3d_pcd.normalize_normals()

# frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=5, origin=[0, 0, 0])


# o3d.visualization.draw_geometries([o3d_pcd, frame],
#                                   point_show_normal=True)


# new_data = {}
# new_data["points"] = np.asarray(o3d_pcd.points)
# new_data["normals"] = np.asarray(o3d_pcd.normals)

# with open("./cuboid12_point_cloud_fixed.pickle", "wb") as f:
#     pickle.dump(new_data, f)