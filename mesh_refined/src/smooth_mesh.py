import trimesh
import numpy as np
import sys
import os

def smooth_mesh_trimesh(mesh, iterations=20, lambda_factor=0.4):
    """
    使用Trimesh库中的拉普拉斯平滑算法平滑网格。
    
    Parameters:
        mesh (trimesh.Trimesh): 输入的网格。
        iterations (int): 平滑迭代次数。
        lambda_factor (float): 控制平滑强度的系数。
    
    Returns:
        trimesh.Trimesh: 平滑后的网格。
    """
    for _ in range(iterations):
        # 获取顶点邻居
        neighbors = mesh.vertex_neighbors
        # 创建一个新的顶点数组
        new_vertices = mesh.vertices.copy()
        for i, vertex in enumerate(mesh.vertices):
            # 获取当前顶点的邻居顶点的平均位置
            neighbor_vertices = mesh.vertices[neighbors[i]]
            average_position = neighbor_vertices.mean(axis=0)
            # 更新顶点位置
            new_vertices[i] = vertex + lambda_factor * (average_position - vertex)
        mesh.vertices = new_vertices
    return mesh

def main(input_mesh_file, output_mesh_file, iterations):
    # 读取现有网格
    mesh = trimesh.load_mesh(input_mesh_file)

    # 平滑网格
    smooth_mesh_result = smooth_mesh_trimesh(mesh, iterations)

    # 保存平滑后的网格
    smooth_mesh_result.export(output_mesh_file)
    print(f"Smoothed mesh saved to {output_mesh_file}")

if __name__ == "__main__":
    input_mesh_file = sys.argv[1]
    output_mesh_dir = sys.argv[2]
    lambda_factor = sys.argv[3]
    iterations = 10
    output_mesh_file = os.path.join(output_mesh_dir, f"smoothed_{os.path.basename(input_mesh_file)}")
    main(input_mesh_file, output_mesh_file, iterations)
