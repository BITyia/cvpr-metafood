'''
对root下某个场景的某个文件夹scale，并保存到特定文件夹下。

'''

import trimesh
import os

scale_factor_path = "/home/yangdianyi/MTF_Challenge/Data_and_code/Find_Scale/scale.txt"
root_dir = "/home/yangdianyi/MTF_Challenge/AAA_Submit/Phase_1"

# scale factor是一个txt文件，第一行对应第一个场景的缩放因子，第二行对应第二个场景的缩放因子，需要读成一个字典，只去AVE的值
'''
txt样子示例：
1 AVE:0.06005869402570996 MAX:0.07142360251277807 MIN:0.0571694044755275
2 AVE:0.0818297412227297 MAX:0.09212620592522533 MIN:0.07893139172389738
8 AVE:0.06849613913800846 MAX:0.07179182781302863 MIN:0.06669381912084305
9 AVE:0.05929247969232063 MAX:0.06278312533966118 MIN:0.05804306485666181
10 AVE:0.058236827845618164 MAX:0.061441033056423046 MIN:0.05720817601345222
'''
scale_factor_dict = {}
with open(scale_factor_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        id = line.split()[0]
        ave = line.split()[1].split(':')[1]
        scale_factor_dict[id] = float(ave)
        
        
# 读取root下的所有场景
for file in os.listdir(root_dir):
    if file.split('_')[0] == "bbox" or file.split('_')[0] == "mesh":
        scene_id = file.split('.')[0].split('_')[-1]
        scale = scale_factor_dict[scene_id]
        print(f"Scene {scene_id} scale factor: {scale}")
        # 读取场景
        mesh_before = trimesh.load(os.path.join(root_dir, file))
        # scale
        mesh_after = mesh_before.apply_scale(scale)
        print('The Volume of the mesh after scaling is: ', round(mesh_after.volume*1e6, 2))
        # 保存
        mesh_after.export(os.path.join(root_dir, f"{scene_id}.obj"))
        
        
# scene_ids = [1,2,5,6,7,8,9,11,13,15]
# ply_name = "smoothed_fixed_meshed-delaunay_masked.ply"

# for id in scene_ids:
#     masked_colmap_ply_path = os.path.join(root_dir, f"{id}/{ply_name}")
#     ply = trimesh.load(masked_colmap_ply_path)
#     scale_factor = scale_factor_dict[str(id)]
#     ply2 = ply.apply_scale(scale_factor)
#     ply2.export(masked_colmap_ply_path)
#     print(f"Exported {masked_colmap_ply_path}")