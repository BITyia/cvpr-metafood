#!/bin/sh

DATASET_ROOT_DIR="../../My_MTF_Data"
INDEX_START=9
INDEX_END=9

PY_MASKED_ORIGINAL=../src/mask_original.py
PY_FILTER_IMAGE_TXT=../src/filter_pose_from_images_txt.py
PY_TRANSFORM_COLMAP_CAMERA=../src/transform_colmap_camera.py

echo "!!!!!!!!!!!!!!!!!!!!!!!Start!!!!!!!!!!!!!!!!!!!!!!!"

for i in $(seq $INDEX_START $INDEX_END)
do
    DATASET_PATH=$DATASET_ROOT_DIR/$i
    # 生成mask文件
    original_image_path=$DATASET_PATH/images
    mask_image_path=$DATASET_PATH/mask
    masked_image_path=$DATASET_PATH/masked_original
    mkdir -p $masked_image_path
    python $PY_MASKED_ORIGINAL $original_image_path $mask_image_path $masked_image_path

    # 创建colmap_masked文件夹，并转成特殊的txt
    colmap_original_txt=$DATASET_PATH/colmap/sparse/0
    colmap_masked_txt_folder=$DATASET_PATH/dense_colmap_masked/sparse/0
    mkdir -p $colmap_masked_txt_folder
    python $PY_FILTER_IMAGE_TXT $colmap_original_txt/images.txt $colmap_masked_txt_folder/images.txt
    cp $colmap_original_txt/cameras.txt $colmap_masked_txt_folder/cameras.txt
    touch $colmap_masked_txt_folder/points3D.txt

    # 抽取图像特征
    db_path=$DATASET_PATH/dense_colmap_masked/database.db
    colmap feature_extractor --database_path $db_path --image_path $masked_image_path

    # 通过transform_colmap_camera 转成colmap格式的db
    colmap_cameras_txt=$colmap_masked_txt_folder/cameras.txt
    python $PY_TRANSFORM_COLMAP_CAMERA $colmap_cameras_txt $db_path

    # 进行特征匹配
    echo " @@@@@@@@@@@@@@@@@@Feature Matching@@@@@@@@@@@@@@@@@@"
    colmap exhaustive_matcher --database_path $db_path

    # 三角测量
    echo " @@@@@@@@@@@@@@@@@@Point Triangulation@@@@@@@@@@@@@@@@@@"
    triangu_out_path=$DATASET_PATH/dense_colmap_masked/triangulated/sparse/
    mkdir -p $triangu_out_path
    colmap point_triangulator --database_path $db_path --image_path $masked_image_path --input_path $DATASET_PATH/dense_colmap_masked/sparse/0 --output_path $triangu_out_path

    # 导出临时txt文件，将其拷贝到sparse/0下
    echo " @@@@@@@@@@@@@@@@@@Model Converter@@@@@@@@@@@@@@@@@@"
    colmap model_converter \
    --input_path $DATASET_PATH/dense_colmap_masked/triangulated/sparse \
    --output_path $DATASET_PATH/dense_colmap_masked/triangulated/sparse \
    --output_type TXT

    cp $DATASET_PATH/dense_colmap_masked/triangulated/sparse/points3D.txt $DATASET_PATH/dense_colmap_masked/sparse/0/points3D.txt
    rm -rf $DATASET_PATH/dense_colmap_masked/triangulated
    
    # 开始进行稠密重建 去畸变->
    echo "@@@@@@@@@@@@@@@@@@Image undistortion@@@@@@@@@@@@@@@@@@"
    colmap image_undistorter \
    --image_path $DATASET_PATH/masked_original \
    --input_path $DATASET_PATH/dense_colmap_masked/sparse/0 \
    --output_path $DATASET_PATH/dense_colmap_masked \
    --output_type COLMAP \
    --max_image_size 2000

    # 生成稠密点云
    echo " @@@@@@@@@@@@@@@@@@PatchMatch Stereo@@@@@@@@@@@@@@@@@@"
    colmap patch_match_stereo \
    --workspace_path $DATASET_PATH/dense_colmap_masked \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true

    # 融合
    echo " @@@@@@@@@@@@@@@@@@Stereo Fusion@@@@@@@@@@@@@@@@@@"
    colmap stereo_fusion \
    --workspace_path $DATASET_PATH/dense_colmap_masked \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path $DATASET_PATH/dense_colmap_masked/fused.ply

    echo "Finish processing $DATASET_PATH"

done