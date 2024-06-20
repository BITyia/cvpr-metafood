#!/bin/sh


DATASET_ROOT_DIR="../../My_MTF_Data"
SCENE_INDEX=10
PYTHON_SCRIPT="./src/filter_mesh_based_mask.py"

echo "!!!!!!!!!!!!!!!!!!!!!!!Start!!!!!!!!!!!!!!!!!!!!!!!"

echo "Processing dataset $SCENE_INDEX"

python $PYTHON_SCRIPT $DATASET_ROOT_DIR $SCENE_INDEX