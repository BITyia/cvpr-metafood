#!/bin/sh

DATASET_ROOT_DIR="../../MTF_Challenge/5_16_ner2mesh"
SAVE_ROOT_DIR="../../5_16_ner2mesh_fixed"
INDEX_START=1
INDEX_END=15

mkdir -p $SAVE_ROOT_DIR

PY_FIX_MESH="../src/mesh_fixer.py"

for i in $(seq $INDEX_START $INDEX_END)
do
    # 跳过9
    # if [ $i -eq 9 ]; then
    #     continue
    # fi

    DATASET_PATH=$DATASET_ROOT_DIR/$i
    
    # input_path=$DATASET_PATH/dense/meshed-delaunay.ply
    input_path=$DATASET_ROOT_DIR/mesh_$i.ply

    if [ ! -f $input_path ]; then
        echo "File $input_path not found!"
        continue
    fi
    # 修复mesh
    python $PY_FIX_MESH $input_path $SAVE_ROOT_DIR
done

