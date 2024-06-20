import os
import cv2
import numpy as np
import sys
import open3d as o3d
import json
import matplotlib.pyplot as plt


def quat2rot(qua):
    qw, qx, qy, qz = qua[0], qua[1], qua[2], qua[3]
    rot = np.array([[1-2*qy**2-2*qz**2, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
                    [2*qx*qy+2*qz*qw, 1-2*qx**2-2*qz**2, 2*qy*qz-2*qx*qw],
                    [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx**2-2*qy**2]])
    return rot

def get_RT(qua, t):
    rot = quat2rot(qua)
    RT_44 = np.eye(4)
    RT_44[:3, :3] = rot
    RT_44[:3, 3] = t
    return RT_44


def reat_image_txt(image_txt_file, mask_dir):
    with open(image_txt_file, "r") as f:
        lines = f.readlines()
    
    image_dict = {}
    for i in range(4, len(lines), 2):
        line = lines[i]
        line = line.strip().split(" ")
        image_id = int(line[0])
        qua = [float(x) for x in line[1:5]]
        t = [float(x) for x in line[5:8]]
        image_name = line[9]
        mask_name = image_name.split(".")[0] + "_segmented_mask.jpg"
        mask_name = os.path.join(mask_dir, mask_name)
        # 获取RT矩阵
        r_t = get_RT(qua, t)
        # 存储信息
        image_dict[image_id] = {"r_t": r_t, "mask_name": mask_name}
        
    # 根据image_id排序
    image_dict = dict(sorted(image_dict.items(), key=lambda x: x[0]))
    
    return image_dict

def read_intrinsic(intirnsic_file):
    '''
    格式：# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# Number of cameras: 1
1 PINHOLE 1440 1920 1500.7290293890524 1500.7270949774258 720 960
    只有最后一行有用
    '''
    
    with open(intirnsic_file, "r") as f:
        lines = f.readlines()
        
    line = lines[-1].split(" ")
    fx = float(line[4])
    fy = float(line[5])
    cx = float(line[6])
    cy = float(line[7])
    
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    return intrinsic_matrix

def judge_one_point(mask, u, v, threshold):
    '''
    判断一个点是否在mask内
    '''
    # 判断index是否越界
    if u - threshold < 0 or u + threshold >= mask.shape[1] or v - threshold < 0 or v + threshold >= mask.shape[0]:
        return False
    # 以当前点为中心的阈值范围内的所有像素点是否在mask内
    for i in range(-threshold, threshold+1):
        for j in range(-threshold, threshold+1):
            if mask[v+i, u+j] == 255:
                return True
    return False

if __name__ == "__main__":
    
    # 输入变量
    data_root = sys.argv[1]
    scene_id = int(sys.argv[2])
    
    # 处理路径
    data_dir = os.path.join(data_root, str(scene_id))
    image_txt_file = os.path.join(data_dir, "dense_colmap_masked/sparse/0/images.txt")
    intirnsic_file = os.path.join(data_dir, "dense_colmap_masked/sparse/0/cameras.txt")
    ply_mesh_file = os.path.join(data_dir, "dense/meshed-delaunay.ply")
    # ply_mesh_file = os.path.join(data_dir, "bbox_mesh.ply")
    mask_img_root = os.path.join(data_dir, "mask")
    dist_threshold = 3
    based_on_lasht_frame_num = 200 # 基于最后几帧的mask进行筛选
    jump = 10    # 每隔几帧进行筛选
    
    jumps = jump
    
    # 读取txt文集
    image_dict = reat_image_txt(image_txt_file, mask_img_root)
    
    # 读取内参矩阵
    intrinsic_matrix = read_intrinsic(intirnsic_file)
    
    # 读取ply文件
    original_mesh = o3d.io.read_triangle_mesh(ply_mesh_file)
    
    # 开始遍历所有的mask文件，并将不在mask内的mesh face删除
    new_triangles = []
    
    # 原始数据
    vertices = np.array(original_mesh.vertices)
    triangles = np.array(original_mesh.triangles)
    out_flag = np.zeros(len(triangles))

    for key, value in image_dict.items():
        print(key)
        if key <= len(image_dict) - based_on_lasht_frame_num:
            continue
        
        jumps -= 1
        if jumps != 0:
            continue
        else:
            jumps = jump
        
        mask_img = cv2.imread(value["mask_name"], 0)
        r_t = value["r_t"]
        
        print("Processing image: ", value["mask_name"])
        print("Current triangles: ", len(out_flag) - np.sum(out_flag))
        
        # 开始遍历每个face
        for i, triangle in enumerate(triangles):
            if out_flag[i] == 1:
                continue
            # 获取三个点
            point1 = vertices[triangle[0]]
            point2 = vertices[triangle[1]]
            point3 = vertices[triangle[2]]
            
            # 将三个点投影到相机坐标系
            point1 = np.dot(r_t, np.array([point1[0], point1[1], point1[2], 1]))[:3]
            point2 = np.dot(r_t, np.array([point2[0], point2[1], point2[2], 1]))[:3]
            point3 = np.dot(r_t, np.array([point3[0], point3[1], point3[2], 1]))[:3]
            
            # 将三个点投影到图像坐标系
            point1 = np.dot(intrinsic_matrix, point1) / point1[2]
            point2 = np.dot(intrinsic_matrix, point2)  / point2[2]
            point3 = np.dot(intrinsic_matrix, point3)   / point3[2]
            
            # 检查index是否越界
            u1, v1 = int(point1[0]), int(point1[1])
            u2, v2 = int(point2[0]), int(point2[1])
            u3, v3 = int(point3[0]), int(point3[1])
            if u1 < 0 or u1 >= mask_img.shape[1] or v1 < 0 or v1 >= mask_img.shape[0]:
                out_flag[i] = 1
                continue
            if u2 < 0 or u2 >= mask_img.shape[1] or v2 < 0 or v2 >= mask_img.shape[0]:
                out_flag[i] = 1
                continue
            if u3 < 0 or u3 >= mask_img.shape[1] or v3 < 0 or v3 >= mask_img.shape[0]:
                out_flag[i] = 1
                continue
            
            # 判断三个点为中心的阈值范围小中心的所有像素点是否在mask内
            if judge_one_point(mask_img, u1, v1, dist_threshold) and judge_one_point(mask_img, u2, v2, dist_threshold) and judge_one_point(mask_img, u3, v3, dist_threshold):
                continue
            else:
                out_flag[i] = 1
    
    # 将没有out_flag的face添加到new_triangles中
    for i, triangle in enumerate(triangles):
        if out_flag[i] == 0:
            new_triangles.append(triangle)
    
    # 创建一个新的mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = original_mesh.vertices
    new_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # 保存，名字加上阈值
    output_ply = os.path.join(data_dir, "dense/meshed-delaunay_masked_dist_" + str(dist_threshold) + ".ply")
    o3d.io.write_triangle_mesh(output_ply, new_mesh)
        
                
