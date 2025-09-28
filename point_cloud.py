import numpy as np
import open3d as o3d

def depth_to_point_cloud(depth_map, color_image):
    h, w = depth_map.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    zz = depth_map

    # Flatten arrays
    points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Add colors
    colors = np.array(color_image).reshape(-1, 3) / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize point cloud
    o3d.visualization.draw_geometries([pcd])

    return pcd

def point_cloud_to_mesh(pcd):
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh.compute_vertex_normals()

    # Visualize mesh
    o3d.visualization.draw_geometries([mesh])

    return mesh
