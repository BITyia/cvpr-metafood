#!/bin/sh

DATASET_ROOT_DIR="../../My_MTF_Data"
INDEX_START=10
INDEX_END=10

echo "!!!!!!!!!!!!!!!!!!!!!!!Start!!!!!!!!!!!!!!!!!!!!!!!"

for i in $(seq $INDEX_START $INDEX_END)
do
    # 跳过9
    # if [ $i -eq 9 ]; then
    #     continue
    # fi
    echo "Processing dataset $i"

    DATASET_PATH=$DATASET_ROOT_DIR/$i

    input_path=$DATASET_PATH/dense_colmap_masked/
    output_path=$DATASET_PATH/dense_colmap_masked/meshed-delaunay.ply
    # input_path=$DATASET_PATH/dense/
    # output_path=$DATASET_PATH/dense/meshed-delaunay.ply

    colmap delaunay_mesher \
    --input_path $input_path \
    --output_path $output_path\
    --DelaunayMeshing.max_proj_dist 0.01
done
