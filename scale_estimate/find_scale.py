import os
import cv2
import struct
import torch
import numpy as np
import open3d as o3d

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

class Generate_Real_Scale():
    def __init__(self, system_param):
        self.system_param = system_param
        self.data_root = self.system_param["data_root"]
        self.save_root = self.system_param["save_root"]
        self.data_name = self.system_param["data_name"] + "/dense/"
        self.colmap_root = self.system_param["colmap_pth"]
        self.ply_name = self.system_param["ply_name"]
        self.chess_T = self.system_param["chess_T"]
        self.chess_h = self.system_param["chess_h"]
        self.chess_w = self.system_param["chess_w"]
        self.std_dist = self.system_param["std_size"]
        self.downsample = self.system_param["downsample"]
        self.device = "cuda"
        # 图片存储路径
        self.image_root = os.path.join(Path(self.data_root), Path(self.data_name), Path("images"))
        # 点云文件的存储路径
        self.ply_root = os.path.join(Path(self.data_root), Path(self.data_name), Path(self.ply_name))
        # 读取图片的信息
        # 棋盘点，相应的图片，相应的图片名称
        self.chess_pts, self.chess_img, self.chess_name = self.load_images(self.image_root, chess_T=self.chess_T)
        # 读取colmap中的文件信息
        # 矩阵，矩阵，字典，字典
        # 字典的键值是图片名称
        self.intr_mat, self.intr_mat_inv, self.pose_c2w, self.pose_w2c = self.load_colmap_bin(self.colmap_root, self.chess_name)
        # exit()
        # 读取点云文件
        self.pts_3d, self.colors = self.load_cloud_points(self.ply_root, downsample_ratio=self.downsample)
        # 空间点重投影
        # [图片数量, N, 2]
        reproj_pts = self.reproject2pixel(self.pts_3d, self.intr_mat, self.pose_w2c)
        # 寻找空间中重投影的点是chessboard角点的点
        near_pts_3d = self.find_space_pts_for_chessboard(self.chess_pts, reproj_pts, self.chess_name)
        # 计算尺度
        ave_radio, max_radio, min_radio = self.generate_scale_info(near_pts_3d, self.std_dist)

        print("AVE:{}".format(ave_radio))
        print("MAX:{}".format(max_radio))
        print("MIN:{}".format(min_radio))

        self.show_merge_3d_pts(self.pts_3d, self.colors, near_pts_3d.reshape(-1, 3))
        # exit()
        # 显示重投影结果
        # idx 表示显示第几张图
        self.show_reproj_results(reproj_pts, self.chess_img, self.colors, idx=0)

    # 将空间点投影到图片中
    # 使用GPU加速
    def reproject2pixel(self, w_pts, intr_mat, w2c_pose):
        # [1, N, 3]
        tensor_wpts = torch.from_numpy(w_pts)[None, ...].to(self.device)
        # [N, 4, 4]
        tenosr_pose = self.align_pose_info(w2c_pose)
        # [1, 3, 3]
        tensor_mat = torch.from_numpy(intr_mat)[None, ...].to(self.device)
        # 重投影 [1, N, 4]
        hom_wpts = self.world2hom(tensor_wpts)
        proj_cam_pts = self.world2cam(hom_wpts, tenosr_pose)
        proj_pixel_pts = self.cam2pix(proj_cam_pts, tensor_mat)

        return proj_pixel_pts

    def load_images(self, img_folder, chess_T):
        name_list = os.listdir(img_folder)[:50]
        name_list.sort()
        img_path_list = [os.path.join(Path(self.image_root), Path(name)) for name in name_list]
        # 读取图片并检测chessboard，能检测到的就会有自己的dict键值，后面的字典是相应的图片路径
        all_chess_pts, all_chess_img_pth, all_chess_name = self.chessboard_detection(img_path_list, name_list, chess_T)
        # 只读取有效的图片
        img_list = []
        for cur_pth in all_chess_img_pth:
            img = cv2.imread(cur_pth)
            img_list += [img]
        
        return all_chess_pts, img_list, all_chess_name

    def chessboard_detection(self, chessboard_pth, name_list, chess_T):
        corners_dict = {}
        useful_path = []
        useful_name = []
        for pth, name in zip(chessboard_pth, name_list):
            cur_img = cv2.imread(pth)
            gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(gray, chess_T, 255, cv2.THRESH_BINARY)[1]
            # 保存binary图像
            print(pth.replace("images", "binary"))
            cv2.imwrite(pth.replace("images", "binary"), binary)
            chessboard_size = (self.chess_w, self.chess_h)
            ret, corners = cv2.findChessboardCorners(binary, chessboard_size, None)
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                corners_dict[name] = corners
                useful_path += [pth]
                useful_name += [name]
                # 绘制并保存
                cv2.drawChessboardCorners(cur_img, chessboard_size, corners, ret)
                cv2.imwrite(pth.replace("images", "chessboard"), cur_img)

        if len(useful_path) < 15:
            print("可以检测到棋盘格的数量少于15张！请重新调整阈值 chess_T ！")
            print("当前有效的棋盘格图片:{}张！共{}张！".format(len(useful_path), len(name_list)))
            exit()
        else:
            print("检测到有效的棋盘格图片:{}张！共{}张！".format(len(useful_path), len(name_list)))

        return corners_dict, useful_path, useful_name

    def load_colmap_bin(self, bin_file_root, ava_name_list):
        camera_info = os.path.join(Path(self.data_root), Path(self.data_name), Path(bin_file_root), Path("cameras.bin"))
        images_info = os.path.join(Path(self.data_root), Path(self.data_name), Path(bin_file_root), Path("images.bin"))
        # 打开文件
        # 注意，凡是读取过程的函数(f.read())不能够省略
        # 因为这个读取过程会移动文件中的指针，影响数据读取的位置
        # 读取相机参数
        with open(camera_info, 'rb') as f:
            # 获取相机和图像的信息
            # 相机数量
            num_cameras = struct.unpack('L', f.read(8))[0]
            # struct.unpack用于从2进制文件中解析成Python数据格式
            camera_id, camera_type, w, h = struct.unpack('IiLL', f.read(24))
            # 360_v2中的模型是简单针孔,focal,u0,v0
            # 支持的参数数量
            num_params = 4
            # 转换为矩阵
            fx, fy, ux, uy = struct.unpack('d' * num_params, f.read(8 * num_params))
            intr_mat = np.array([[fx, 0,  ux],
                                 [0, fy,  uy],
                                 [0,  0,  1]])

            intr_mat_inv = np.linalg.inv(intr_mat)

        # 打开文件，读取图片位姿信息
        with open(images_info, 'rb') as f:
            # 获取图像的数量
            # struct.unpack用于从2进制文件中解析成Python数据格式
            # 'L'表示无符号长整型数据
            num_images = struct.unpack('L', f.read(8))[0]
            image_struct = struct.Struct('<I 4d 3d I')
            # 位姿缓存
            pose_c2w = {}
            pose_w2c = {}
            # 对每一张图像进行处理
            for _ in range(num_images):
                # 读取指定大小的数据，并且移动数据读取指针
                data = image_struct.unpack(f.read(image_struct.size))
                # 获取当前图片的名称
                name = b''.join(c for c in iter(lambda: f.read(1), b'\x00')).decode()
                # 如果当前位姿在棋盘格检测点中
                if name in ava_name_list:
                    quaternion = (np.array(data[1:5]))
                    rot = self.quaternion2rot(quaternion)
                    trans = np.array(data[5:8]).reshape(3, 1)
                    w2c = np.concatenate([np.concatenate([rot, trans], axis=1), np.array([0,0,0,1]).reshape(1,4)], axis=0)
                    # 转为c2w
                    c2w = np.linalg.inv(w2c)
                    pose_c2w[name] = c2w
                    pose_w2c[name] = w2c

                # 跳过一部分不需要的数据
                num_points2D = struct.unpack('Q', f.read(8))[0]
                f.read(num_points2D*24)

        return intr_mat, intr_mat_inv, pose_c2w, pose_w2c 

    def load_cloud_points(self, ply_pth, downsample_ratio=1):
        pcd = o3d.io.read_point_cloud(ply_pth)
        # 打印将采样前的点云个数
        print("The number of points before downsampling is:", len(pcd.points))
        # 降采样
        pcd = pcd.random_down_sample(downsample_ratio)
        points = np.asarray(pcd.points)
        print("The number of points after downsampling is:", len(pcd.points))

        # 保存一下将采样后的点云
        o3d.io.write_point_cloud(self.ply_root.replace(".ply", "_downsample.ply"), pcd)
        # 判断是否有颜色信息
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        else:
            colors = np.ones(points.shape)
        # visualizeyy
        # o3d.visualization.draw_geometries([pcd])

        return points, colors     

    def find_space_pts_for_chessboard(self, detect_pts, reproj_pts, chess_name):
        detect_pts_list = []
        choiced_pts_all = []
        # 世界坐标系中点的空间坐标
        # [N, 3]
        tensor_wpts = torch.from_numpy(self.pts_3d).to(self.device) 
        # 处理监测点
        for img_name in chess_name:
            chess_pts = detect_pts[img_name]
            detect_pts_list += [torch.tensor(chess_pts)]
        # [N, 12, 2]
        detect_pts_list = torch.stack(detect_pts_list, 0)[..., 0, :].to(self.device)
        # 循环处理
        for ii in range(detect_pts_list.shape[0]):
            cur_pts = detect_pts_list[ii]
            cur_repts = reproj_pts[ii]
            dist_mat = torch.sum((cur_pts[:, None] - cur_repts) ** 2, dim=2)
            min_distances, indices = torch.min(dist_mat, dim=1)
            # [12, 3]
            choiced_3d_pts = tensor_wpts[indices]
            choiced_pts_all += [choiced_3d_pts]
        # 合并
        choiced_pts_all = torch.stack(choiced_pts_all, 0)

        return choiced_pts_all

    # 对齐所有的图像信息
    # c2w_pose:字典
    def align_pose_info(self, c2w_pose):
        mat_list = []
        for name in self.chess_name:
            cur_mat = torch.from_numpy(c2w_pose[name]).to(self.device)
            mat_list += [cur_mat]
        mat_list = torch.stack(mat_list)

        return mat_list
        
    # 转换为矩阵
    # quaternion: [4]                    
    def quaternion2rot(self, quaternion):
        return np.eye(3) + 2 * np.array((
          (-quaternion[2] * quaternion[2] - quaternion[3] * quaternion[3],
            quaternion[1] * quaternion[2] - quaternion[3] * quaternion[0],
            quaternion[1] * quaternion[3] + quaternion[2] * quaternion[0]),
          ( quaternion[1] * quaternion[2] + quaternion[3] * quaternion[0],
           -quaternion[1] * quaternion[1] - quaternion[3] * quaternion[3],
            quaternion[2] * quaternion[3] - quaternion[1] * quaternion[0]),
          ( quaternion[1] * quaternion[3] - quaternion[2] * quaternion[0],
            quaternion[2] * quaternion[3] + quaternion[1] * quaternion[0],
           -quaternion[1] * quaternion[1] - quaternion[2] * quaternion[2])))

    # [Batch, HW, 3]->[Batch, HW, 4]
    def world2hom(self, world_cord):
        X_hom = torch.cat([world_cord, torch.ones_like(world_cord[...,:1])], dim=-1)
        return X_hom        

    # world_cord: [batch, ..., 4]
    # pose: [batch, ..., 4, 4]
    def world2cam(self, world_cord, w2c_pose):
        cam_cord = w2c_pose @ world_cord.transpose(-2, -1)
        return cam_cord
        
    # intr_mat: [batch, ..., 3, 3]    
    def cam2pix(self, cam_cord, intr_mat):
        hom_intr_mat = torch.cat([intr_mat, torch.zeros_like(intr_mat[...,:1])], dim=-1)
        pix_cord = hom_intr_mat @ cam_cord
        pix_cord = pix_cord[...,:2,:]/pix_cord[...,2:,:]
        pix_cord = pix_cord.transpose(-2, -1)
        return pix_cord
    
    # 重投影图像点
    def show_reproj_results(self, reproj_pts, img_list, color_pts, idx, save=True):
        # [H, W, 3]
        choice_img = img_list[idx]
        # 筛选有效点
        choice_repts = np.array(reproj_pts[idx].cpu())
        pts_x_min, pts_x_max = 0, choice_img.shape[1]-1
        pts_y_min, pts_y_max = 0, choice_img.shape[0]-1
        mask = (choice_repts[:, 0] >= pts_x_min) & (choice_repts[:, 0] <= pts_x_max) &\
               (choice_repts[:, 1] >= pts_y_min) & (choice_repts[:, 1] <= pts_y_max)
        choice_repts = choice_repts[mask].astype(np.int32)
        # 筛选RGB像素
        choice_colors = color_pts[mask]
        # 将筛选点绘制在空白图片上
        blank_img = np.ones_like(choice_img)*255
        blank_img[choice_repts[:, 1], choice_repts[:, 0]] = choice_colors
        # 合并结果，查看对比图像
        cat_img = np.concatenate([choice_img, blank_img], axis=1)
        if save:
            save_pth = os.path.join(Path(self.save_root), Path(self.data_name))
            
            os.makedirs(save_pth, exist_ok=True)
            file_Pth = os.path.join(Path(save_pth), Path(self.chess_name[idx]))
            print(file_Pth)
            cv2.imwrite(file_Pth, cat_img)
    
    # 根据每张图片算出的空间坐标点计算尺寸
    def generate_scale_info(self, near_pts_3d, std_dist):
        mid_dist = []
        # 循环处理数据
        for ii in range(near_pts_3d.shape[0]):
            cur_pts = near_pts_3d[ii]
            diff = cur_pts[:, None, :] - cur_pts[None, :, :]
            distances = torch.sqrt(torch.sum(diff**2, dim=2))
            # 将对角元素设置为无穷大
            distances.fill_diagonal_(float('inf'))
            # 获取每个点距离最近点的距离，也就是棋盘格的相邻距离
            min_dist = torch.min(distances, dim=1)[0]
            sorted_dist, sorted_idx  = torch.sort(min_dist)
            # 取中值
            mid_sort = sorted_dist[int(sorted_dist.shape[0]/2)]
            mid_dist += [float(mid_sort.cpu())]
        # 计算均值，最大范围，最小范围
        ave_radio = std_dist / float(torch.tensor(mid_dist).mean())
        max_radio = std_dist / (min(mid_dist) + 1e-6)
        min_radio = std_dist / max(mid_dist)
        
        return ave_radio, max_radio, min_radio

    # 展示投影到3D空间的点
    def show_merge_3d_pts(self, ply_pts_3d, ply_colors, nearst_pts_3d):
        # 自己的计算的最近的点设置为红色
        colors = np.array([1.0, 0.0, 0.0])
        colors = np.tile(colors, [nearst_pts_3d.shape[0], 1])

        # 融合点信息
        np_3d_pts = np.array(nearst_pts_3d.cpu())
        merge_pts_3d = np.concatenate([ply_pts_3d, np_3d_pts], axis=0)
        merge_colors_3d = np.concatenate([ply_colors, colors], axis=0)

        # 构建空的o3d格式数据，将点和颜色添加进去
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merge_pts_3d)
        pcd.colors = o3d.utility.Vector3dVector(merge_colors_3d)
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # 添加点云到可视化器
        vis.add_geometry(pcd)
        # 设置点的大小
        opt = vis.get_render_option()
        opt.point_size = 5  # 设置点的显示尺寸，根据需要调整
        # 运行可视化
        vis.run()
        vis.destroy_window()

    def show_3d_pts(self):
        pass
        
if __name__ == "__main__":

    system_param = {"data_root":"I:/MTF_Challenge_Phase_1_Backup/Data_and_code/My_MTF_Data",
                    "save_root":"I:/MTF_Challenge_Phase_1_Backup/Data_and_code/Find_Scale",
                    "data_name": "8",
                    "colmap_pth": "sparse",
                    "ply_name": "fused.ply",
                    "chess_h": 3,
                    "chess_w": 4,
                    "chess_T": 220,
                    "downsample": 0.3,
                    "std_size": 0.012}

    # system_param = {"data_root":"/home/gaoyu/CVPR_Work/Apriltag_Calib",
    #                 "save_root":"/home/gaoyu/CVPR_Work/image_save",
    #                 "data_name": "apriltag_cube",
    #                 "colmap_pth": "sparse",
    #                 "ply_name": "fused.ply",
    #                 "chess_h": 8,
    #                 "chess_w": 11,
    #                 "chess_T": 150,
    #                 "std_size": 0.03}
     
    datascale = Generate_Real_Scale(system_param)