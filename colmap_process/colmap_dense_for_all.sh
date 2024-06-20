#!/bin/sh

DATASET_ROOT_DIR="../../My_MTF_Data"
INDEX_START=9
INDEX_END=9

echo "!!!!!!!!!!!!!!!!!!!!!!!Start!!!!!!!!!!!!!!!!!!!!!!!"

for i in $(seq $INDEX_START $INDEX_END)
do
    DATASET_PATH=$DATASET_ROOT_DIR/$i
    # 开始进行稠密重建
    echo "Start to process $DATASET_PATH"
    mkdir -p $DATASET_PATH/dense

    echo " @@@@@@@@@@@@@@@@@@Image undistortion@@@@@@@@@@@@@@@@@@"
    colmap image_undistorter \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse/0 \
    --output_path $DATASET_PATH/dense \
    --output_type COLMAP \
    --max_image_size 2000

    # 生成稠密点云
    echo " @@@@@@@@@@@@@@@@@@PatchMatch Stereo@@@@@@@@@@@@@@@@@@"
    colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true      
    
    # 融合
    echo " @@@@@@@@@@@@@@@@@@Stereo Fusion@@@@@@@@@@@@@@@@@@"
    colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense/fused.ply

    # 转出txt
    echo " @@@@@@@@@@@@@@@@@@Model Converter@@@@@@@@@@@@@@@@@@"
    colmap model_converter \
    --input_path $DATASET_PATH/dense/sparse \
    --output_path $DATASET_PATH/dense/sparse \
    --output_type TXT

    echo "Finish processing $DATASET_PATH"

done