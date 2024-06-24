
This repository contains the code implementation of the technical report for the Physically Informed 3D Food Reconstruction Challenge at CVPR 2024 MetaFood Workshop.


The structure of repository:
```
.
├── colmap_process
│   ├── colmap_dense_for_all.sh                     # 生成稠密点云
│   ├── colmap_masked_dense_for_all.sh              # 依据掩码信息生成物体级别的稠密点云
│   ├── filter_mesh_based_mask.sh                   # 依据掩码信息对mesh进行投影过滤
│   ├── gen_colmap_for_all.sh                       # 生成1~15物体的colmap信息
│   ├── gen_delaunayMesh.sh                         # 依据delaunayMesh模型从点云生成mesh
│   └── src
│       ├── add_mask_and_depth.py
│       ├── colmap2nerf.py
│       ├── colmap_depth_process.py
│       ├── filter_mesh_based_on_mask.py
│       ├── filter_pose_from_images_txt.py
│       ├── food_colmap_images.py
│       ├── mask_original.py
│       └── transform_colmap_camera.py
├── diffusionerf
├── nerf2mesh
├── ZeroNVS
├── mesh_refined
│   ├── fix_all_mesh.sh                             # 消除mesh空洞
│   ├── quick_fix_smooth_cal.sh                     # 对单个物体直接全部的refine操作并计算体积
│   ├── smooth_all_mesh.sh                          # 平滑mesh表面
│   └── src
│       ├── mesh_fixer.py
│       ├── quick_apply_scale.py
│       ├── quick_get_volumn.py
│       └── smooth_mesh.py
└── scale_estimate
│   ├── find_scale.py                               # 得到单个物体的scale估计结果。
│   └── scale_results.txt                           # 1~15全部scale估计结果
│── alignment
│   ├── align_gradient_descent.py                   # 使用梯度下降方法对齐。
│   ├──  
```

## Get Started

### Installation

If you wish to use DiffusionNeRF or NeRF2Mesh for multi-view 3D reconstruction, please refer to their respective setup instructions.
## Usage
### scale estimation

### 3d reconstruction

### mesh refinement