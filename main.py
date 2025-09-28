import os
from depth_estimation import estimate_depth
from point_cloud import depth_to_point_cloud, point_cloud_to_mesh
import open3d as o3d

# Paths
image_path = "images/house.jpeg"  # Replace with your image
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Depth estimation
depth_map, img = estimate_depth(image_path)

# Step 2: Point cloud
pcd = depth_to_point_cloud(depth_map, img)

# Step 3: Mesh generation
mesh = point_cloud_to_mesh(pcd)

# Step 4: Save outputs
pcd_file = os.path.join(output_dir, "house_pointcloud.ply")
mesh_file = os.path.join(output_dir, "house_mesh.ply")
o3d.io.write_point_cloud(pcd_file, pcd)
o3d.io.write_triangle_mesh(mesh_file, mesh)

print(f"Point cloud saved at: {pcd_file}")
print(f"Mesh saved at: {mesh_file}")
