'''
快速计算某个物体的体积

'''

import trimesh
import os
import sys

scale_factor_path = "/home/yangdianyi/MTF_Challenge/Data_and_code/Find_Scale/scale.txt"

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
        
        
scene_ids = sys.argv[1]
ply_name = sys.argv[2]

mesh = trimesh.load(ply_name)
mesh2 = mesh.apply_scale(scale_factor_dict[str(scene_ids)])
volume = round(mesh2.volume * 1e6, 2)

print(f"Volume of {ply_name} is {volume} mm^3")
