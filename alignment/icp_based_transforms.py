import open3d as o3d
import numpy as np
import os
import sys
import trimesh
from scipy.spatial import cKDTree
import json
from colorama import Fore, Back, Style
import argparse
import tqdm

json_root = "F:/AAA_Kaggle/Aligned/record.json"
# 这个json用来记录所有物体的操作记录，以及每次操作的fitness，采样点数、阈值设置、chamfer_distance、最终的transformation matrix
if not os.path.exists(json_root):
    with open(json_root, 'w') as f:
        json.dump({}, f)

def load_mesh(file_path):
    # 使用trimesh读，转成open3d的mesh
    mesh = trimesh.load(file_path)
    mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices), triangles=o3d.utility.Vector3iVector(mesh.faces))  # 为什么分开传入就不行？？
    return mesh

def load_obj_as_pointcloud(file_path):
    mesh = trimesh.load(file_path)
    return np.array(mesh.vertices)

def apply_transformation(pc, transform_matrix):
    homogenous_pc = np.hstack((pc, np.ones((pc.shape[0], 1))))
    transformed_pc = homogenous_pc.dot(transform_matrix.T)
    return transformed_pc[:, :3]

def chamfer_distance(pc1, pc2):
    kdtree1 = cKDTree(pc1)
    kdtree2 = cKDTree(pc2)

    distances_1, _ = kdtree1.query(pc2)
    distances_2, _ = kdtree2.query(pc1)

    chamfer_dist = np.mean(distances_1) + np.mean(distances_2)
    return chamfer_dist

def get_point_cloud(mesh, number_of_points=100000):
    # 将mesh转成open3d的mesh，然后采样点
    # mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices), triangles=o3d.utility.Vector3iVector(mesh.faces))
    if number_of_points > len(mesh.vertices):
        print(len(mesh.vertices))
        return mesh.sample_points_uniformly(number_of_points=len(mesh.vertices))
    return mesh.sample_points_uniformly(number_of_points=100000)

def align_point_clouds(source_pcd, target_pcd, threshold=0.5, radius=0.2, max_nn=300, max_iteration=2000):
    # Preprocessing
    voxel_size = 1e-6
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # print("source_down.points: ", source_down.points)
    # print("target_down.points: ", target_down.points)

    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=300))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=300))

    # Apply ICP for fine alignment
    threshold = 0.5 # 调的高一些稍微
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
        )
    
    # 打印最终的距离
    print("Final fitness: ", reg_p2p.fitness)
    
    return reg_p2p.transformation

def transform_mesh(mesh, transformation):
    mesh.transform(transformation)
    return mesh

def get_transformation_matrix(center, tx, ty, tz, roll, pitch, yaw):
    # Convert degrees to radians
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)

    # Create transformation matrices for rotation
    R_x = np.array([[1, 0, 0, 0],
                    [0, np.cos(roll), -np.sin(roll), 0],
                    [0, np.sin(roll), np.cos(roll), 0],
                    [0, 0, 0, 1]])
                    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
                    [0, 1, 0, 0],
                    [-np.sin(pitch), 0, np.cos(pitch), 0],
                    [0, 0, 0, 1]])
                    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                    [np.sin(yaw), np.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))

    # Translation matrices
    T1 = np.array([[1, 0, 0, -center[0]],
                   [0, 1, 0, -center[1]],
                   [0, 0, 1, -center[2]],
                   [0, 0, 0, 1]])
                   
    T2 = np.array([[1, 0, 0, center[0]],
                   [0, 1, 0, center[1]],
                   [0, 0, 1, center[2]],
                   [0, 0, 0, 1]])

    T_translate = np.array([[1, 0, 0, tx],
                            [0, 1, 0, ty],
                            [0, 0, 1, tz],
                            [0, 0, 0, 1]])
    
    # Combined transformation matrix
    transformation_matrix = np.dot(T2, np.dot(R, T1))
    transformation_matrix = np.dot(T_translate, transformation_matrix)
    return transformation_matrix

def main():
    parser = argparse.ArgumentParser(description="Align two .obj files interactively and save the transformation matrix.")
    parser.add_argument('-Food_id', type=str, help="Food id")
    parser.add_argument('-Load_Last', type=int, default=-1 , required=False, help="Load last transformation matrix")
    parser.add_argument('-number_of_points', type=int, default=100000, required=False, help="Number of points to sample")
    parser.add_argument('-threshold', type=float, default=0.5, required=False, help="Threshold for ICP")
    parser.add_argument('-radius', type=float, default=0.2, required=False, help="Radius for normal estimation")
    parser.add_argument('-max_nn', type=int, default=300, required=False, help="Max number of nearest neighbors for normal estimation")
    parser.add_argument('-qx', type=float, default=5, required=False, help="Rotation angle in degrees for each key press")
    parser.add_argument('-max_iter', type=int, default=200000, required=False, help="Max iteration for ICP")
    parser.add_argument('-align_times', type=int, default=1, required=False, help="Align times")
    parser.add_argument('-vis', action='store_true', help="Visualize the alignment process")
    args = parser.parse_args()
    Food_id = args.Food_id
    Load_Last = args.Load_Last
    qx_init = args.qx
    
    # 查看json里是否有物体的操作记录，没有则创建一个
    json_file = open(json_root, 'r')
    json_data = json.load(json_file)
    json_file.close()
    if Food_id not in json_data:
        json_data[Food_id] = []
        
    current_record = {}
    current_record["sample_points"] = args.number_of_points
    current_record["threshold"] = args.threshold
    current_record["radius"] = args.radius
    current_record["max_nn"] = args.max_nn
    
    # Load meshes
    print("Load meshe gt...")
    original_mesh1 = load_mesh(f"F:/AAA_Kaggle/Food_Rec/ground_truth_object_v2/{Food_id}.obj")
    print("Load meshe phase2...")
    original_mesh2 = load_mesh(f"F:/AAA_Kaggle/Food_Rec/ININ_VIAUN_Phase2/{Food_id}.obj")
    
    mesh1 = original_mesh1
    mesh2 = original_mesh2
    
    trimesh_mesh1 = trimesh.load(f"F:/AAA_Kaggle/Food_Rec/ground_truth_object_v2/{Food_id}.obj")
    trimesh_mesh2 = trimesh.load(f"F:/AAA_Kaggle/Food_Rec/ININ_VIAUN_Phase2/{Food_id}.obj")
    
    print(len(mesh1.vertices), len(mesh2.vertices))
    aligned_root = "F:/AAA_Kaggle/Aligned"

    ####################### 一阶段：交互式对齐 #######################

    # Get point clouds
    pcd1 = get_point_cloud(mesh1, args.number_of_points)
    pcd2 = get_point_cloud(mesh2, args.number_of_points)
    
    # 赋予两个mesh不同的颜色
    # pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    Init_transforms = None

    # 初始转换矩阵，从mesh2到mesh1，先对坐标轴取反
    if Load_Last != -1:
        print(f"Load {Load_Last} Tranforms:")
        if json_data[Food_id][Load_Last]["transformation"] is not None:
            Init_transforms = np.array(json_data[Food_id][Load_Last]["transformation"])
    else:
        # 直接导入最后一次的transformation matrix
        print("Load Mini Tranforms:")
        # 根据每个的chanfer distance来选择最小的transformation
        min_chamfer_distance = 100
        for record in json_data[Food_id]:
            if record["chamfer_distance"] < min_chamfer_distance:
                min_chamfer_distance = record["chamfer_distance"]
                Init_transforms = np.array(record["transformation"])
    
    if Init_transforms is None:
        Init_transforms = np.eye(4)
        center1 = pcd1.get_center()
        center2 = pcd2.get_center() * 1
        Init_translations = np.array([[1, 0, 0, center1[0] - center2[0]], [0, 1, 0, center1[1] - center2[1]], [0, 0, 1, center1[2] - center2[2]], [0, 0, 0, 1]])
        Init_transforms = np.dot(Init_translations, Init_transforms)
    print("Init_transforms:")
    print(Init_transforms)
    pcd2.transform(Init_transforms)

    # 使用VisualizerWithKeyCallback进行交互
    transformation = np.eye(4)
    if args.vis:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd2)
        
        center = pcd2.get_center()

        def update_transformation(vis, tx=0, ty=0, tz=0, roll=0, pitch=0, yaw=0):
            nonlocal transformation
            T = get_transformation_matrix(center, tx, ty, tz, roll, pitch, yaw)
            pcd2.transform(T)
            transformation = np.dot(T, transformation)
            vis.update_geometry(pcd2)
            vis.poll_events()
            vis.update_renderer()
            
        def align_to_center(vis):
            nonlocal transformation
            center1 = pcd1.get_center()
            center2 = pcd2.get_center()
            T_align = np.eye(4)
            T_align[0:3, 3] = center1 - center2
            pcd2.transform(T_align)
            transformation = np.dot(T_align, transformation)
            vis.update_geometry(pcd2)
            vis.poll_events()
            vis.update_renderer()

        vis.register_key_callback(ord("Q"), lambda vis: update_transformation(vis, yaw=qx_init))
        vis.register_key_callback(ord("W"), lambda vis: update_transformation(vis, pitch=qx_init))
        vis.register_key_callback(ord("E"), lambda vis: update_transformation(vis, roll=qx_init))
        vis.register_key_callback(ord("A"), lambda vis: update_transformation(vis, yaw=-qx_init))
        vis.register_key_callback(ord("S"), lambda vis: update_transformation(vis, pitch=-qx_init))
        vis.register_key_callback(ord("D"), lambda vis: update_transformation(vis, roll=-qx_init))
        vis.register_key_callback(ord("R"), lambda vis: update_transformation(vis, tx=0.001))
        vis.register_key_callback(ord("T"), lambda vis: update_transformation(vis, ty=0.001))
        vis.register_key_callback(ord("Y"), lambda vis: update_transformation(vis, tz=0.001))
        vis.register_key_callback(ord("F"), lambda vis: update_transformation(vis, tx=-0.001))
        vis.register_key_callback(ord("G"), lambda vis: update_transformation(vis, ty=-0.001))
        vis.register_key_callback(ord("H"), lambda vis: update_transformation(vis, tz=-0.001))
        vis.register_key_callback(ord("C"), align_to_center)

        vis.run()
        vis.destroy_window()

    ####################### 二阶段：ICP对齐 #######################
    # Align 次数
    pbar = tqdm.tqdm(total=args.align_times)
    
    min_transformation = None
    min_chamfer_distance = 100
    
    print("Start to Align...")
    
    for i in range(args.align_times):
        # Align point clouds
        alignment_transformation = align_point_clouds(pcd2, pcd1, args.threshold, args.radius, args.max_nn, args.max_iter)
        
        # 将两个transformations相乘
        final_transformation = np.dot(alignment_transformation, transformation)
        final_transformation = np.dot(final_transformation, Init_transforms)
        
        # 计算chamfer distance
        pc1 = np.array(trimesh_mesh1.vertices)
        pc2 = np.array(trimesh_mesh2.vertices)
        pc2 = apply_transformation(pc2, final_transformation)
        distance = chamfer_distance(pc1, pc2)
        print("Chamfer distance: ", distance)
        
        if distance < min_chamfer_distance:
            min_chamfer_distance = distance
            min_transformation = final_transformation
            
        # 结束重新采集一批点云
        print("Sample meshe phase2 again...")
        pcd1 = get_point_cloud(original_mesh1, args.number_of_points)
        mesh2 = load_mesh(f"F:/AAA_Kaggle/Food_Rec/ININ_VIAUN_Phase2/{Food_id}.obj")
        # 对pcd2进行变换，在当前最短的transformation下进行
        if min_transformation is not None:
            mesh2 = transform_mesh(mesh2, min_transformation)
        pcd2 = get_point_cloud(mesh2, args.number_of_points)
        
        # o3d.visualization.draw_geometries([pcd1, pcd2])

        pbar.update(1)
    
    # 保存
    mesh1 = load_mesh(f"F:/AAA_Kaggle/Food_Rec/ground_truth_object_v2/{Food_id}.obj")
    mesh2 = load_mesh(f"F:/AAA_Kaggle/Food_Rec/ININ_VIAUN_Phase2/{Food_id}.obj")
    transformed_mesh2 = transform_mesh(mesh2, final_transformation)

    # 将两个mesh对齐后的结果保存到文件，创建一个新的mesh文件保存两个mesh，制定两个mesh不同的颜色
    mesh1.paint_uniform_color([1, 0.706, 0])
    transformed_mesh2.paint_uniform_color([0, 0.651, 0.929])
    aligned_mesh = mesh1 + transformed_mesh2
    aligned_file = os.path.join(aligned_root, f"{Food_id}.obj")
    o3d.io.write_triangle_mesh(aligned_file, aligned_mesh)
    print("Save aligned mesh to: ", aligned_file)
    
    current_record["transformation"] = min_transformation.tolist()
    current_record["chamfer_distance"] = min_chamfer_distance
    
    print(Fore.RED + "Final transformation:")
    print(min_transformation)
    print(Fore.RED + "Final chamfer distance: ", min_chamfer_distance)
    
    # 保存记录
    json_data[Food_id].append(current_record)
    with open(json_root, 'w') as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    main()