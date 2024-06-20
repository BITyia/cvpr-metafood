#!/bin/sh

DATASET_ROOT_DIR="../../My_MTF_Data"
DSTINATION_ROOT_DIR="../../New_Transforms"
INDEX_START=1
INDEX_END=15

PY_DATASET_TO_COLAMP ="../src/food_colmap_images.py"
PY_COLMAP_TO_NERF="../src/colmap2nerf.py"
PY_CONVET_DEPTH="../src/convert_depth.py"
PY_ADD_MASK_DEPTH_JSON="../src/add_mask_and_depth.py"

echo "!!!!!!!!!!!!!!!!!!!!!!!Start!!!!!!!!!!!!!!!!!!!!!!!"

for i in $(seq $INDEX_START $INDEX_END)
do
    DATASET_DIR=$DATASET_ROOT_DIR/$i
    DESTINATION_DIR=$DSTINATION_ROOT_DIR/$i

    mkdir -p $DESTINATION_DIR

    echo "Processing dataset $i"

    # 1. 创建一个新的文件夹:colmap_process/input/。然后将数据集路径下的Original文件夹下的所有图片拷贝到input文件夹下。
    mkdir -p $DESTINATION_DIR/original/
    #mkdir -p $DESTINATION_DIR/mask/
    cp -r $DATASET_DIR/Original/* $DESTINATION_DIR/original/
    #cp -r $DATASET_DIR/Mask/* $DESTINATION_DIR/mask/
    sleep 1
    echo "!!!!!!!!!!!!!!!!!!!!!!!Copy Original images Done!!!!!!!!!!!!!!!!!!!!!!!"

    # 2. 运行 COLMAP_GEN_DATA_FILE 脚本
    python $PY_DATASET_TO_COLAMP --source_path $DESTINATION_DIR --camera PINHOLE
    echo "!!!!!!!!!!!!!!!!!!!!!!!COLMAP_GEN_DATA_FILE Done!!!!!!!!!!!!!!!!!!!!!!!"
    sleep 5

    # 3. 运行 COLMAP_TO_NERF 脚本;
    python $PY_COLMAP_TO_NERF --out $DESTINATION_DIR/transforms.json --text $DATASET_DIR/colmap/sparse/0/ --keep_colmap_coords
    echo "!!!!!!!!!!!!!!!!!!!!!!!COLMAP_TO_NERF Done!!!!!!!!!!!!!!!!!!!!!!!"

    # 4. 转depth
    # python $PY_CONVET_DEPTH $DATASET_DIR/Depth/ $DESTINATION_DIR/depth/
    # echo "!!!!!!!!!!!!!!!!!!!!!!!End!!!!!!!!!!!!!!!!!!!!!!!"

    # 5. 生成mask+depth 的json文件
    python $PY_ADD_MASK_DEPTH_JSON --depth --mask --root_path $DESTINATION_DIR

    # 6. 生成depth的json文件
    python $PY_ADD_MASK_DEPTH_JSON --depth --root_path $DESTINATION_DIR

    # 7. 生成mask的json文件
    python $PY_ADD_MASK_DEPTH_JSON --mask --root_path $DESTINATION_DIR
    
done