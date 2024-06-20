#!/bin/sh

DATASET_ROOT_DIR="../../5_16_ner2mesh_fixed"
SAVE_ROOT_DIR="../../5_16_nerf2mesh_smoothed"
INDEX_START=1
INDEX_END=15

mkdir -p $SAVE_ROOT_DIR

PY_SMOOTH_MESH="../src/smooth_mesh.py"

for i in $(seq $INDEX_START $INDEX_END)
do
    # 跳过9
    # if [ $i -eq 9 ]; then
    #     continue
    # fi

    DATASET_PATH=$DATASET_ROOT_DIR/$i
    
    # input_path=$DATASET_PATH/dense/meshed-delaunay.ply
    input_path=$DATASET_ROOT_DIR/fixed_mesh_$i.ply

    if [ ! -f $input_path ]; then
        echo "File $input_path not found!"
        continue
    fi
    # 修复mesh
    python $PY_SMOOTH_MESH $input_path $SAVE_ROOT_DIR 0.4
done

