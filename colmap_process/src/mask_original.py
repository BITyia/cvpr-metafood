''''
输入：原始图片路径 + mask路径
输出：原始图片的mask路径

原始图像路径下的图片名字：frame_0000_original.jpg
对应的mask路径下的图片名字：frame_0000_original_segmented_mask.jpg

mask为白色区域为有效区域，黑色区域为无效区域

'''

import os
import cv2
import numpy as np
import sys

original_image_path = sys.argv[1]   # 原始图像路径
mask_path = sys.argv[2]             # mask路径
save_path = sys.argv[3]             # 保存路径，保存的名字和原始图像名字一致

original_image_files = os.listdir(original_image_path)
mask_files = os.listdir(mask_path)

original_image_files = sorted(original_image_files, key=lambda x: int(x.split(".")[0]))
mask_files = sorted(mask_files)

for i, original_image_file in enumerate(original_image_files):
    if i == 0:
        continue
    print(original_image_file)
    if not original_image_file.endswith(".jpg"):
        continue
    original_image_file_name = original_image_file.split(".")[0]
    mask_file = mask_files[i]
    if mask_file not in mask_files:
        continue
    original_image = cv2.imread(os.path.join(original_image_path, original_image_file))
    mask = cv2.imread(os.path.join(mask_path, mask_file))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 检查二者图像大小是否一致
    assert original_image.shape[:2] == mask.shape[:2]

    # 将mask中的白色区域作为有效区域
    mask = mask > 0
    
    # 取出原始图像中的有效区域
    masked_image = original_image.copy()
    masked_image[~mask] = 0

    # 将无效区域设置为透明
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2BGRA)
    masked_image[:, :, 3] = mask.astype(np.uint8) * 255

    # 保存图片
    save_image_name = original_image_file_name + ".png"
    cv2.imwrite(os.path.join(save_path, save_image_name), masked_image)

