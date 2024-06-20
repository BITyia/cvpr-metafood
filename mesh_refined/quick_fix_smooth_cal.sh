#!/bin/sh

# 输入一个ply文件，进行fix smooth 并计算体积
PYTHON_FIX_FILE="../src/mesh_fixer.py"
PYTHON_SMOOTH_FILE="../src/smooth_mesh.py"
PYTHON_CAL_VOLUME="../src/quick_get_volumn.py"

scene_id=$1
input_ply=$2

# fix，传入两个参数，一个是输入的ply文件，一个是输出的ply文件夹路径，后者用前者的文件夹路径
save_dir=$(dirname $input_ply)
python $PYTHON_FIX_FILE $input_ply $save_dir

# smooth 传入的参数是输入的ply文件，前者是在文件名字前面加上了fixed_
input_ply_fixed=$save_dir/fixed_$(basename $input_ply)
python $PYTHON_SMOOTH_FILE $input_ply_fixed $save_dir 0.2

# 计算体积
input_ply_smmothed=$save_dir/smoothed_$(basename $input_ply_fixed)
python $PYTHON_CAL_VOLUME $scene_id $input_ply_smmothed


