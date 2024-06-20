'''

    # Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 30, mean observations per image: 641.86666666666667
30 0.9980893920104259 -0.031994708078830654 0.052638787527799463 0.0048023180665171736 0.20608434155934313 0.25161879285829464 1.2333677862442347 1 frame_0029_original.jpg
758.77703857421875 2.7062447071075439 -1 758.77703857421875 2.7062447071075439 -1 1148.4383544921875 2.4697628021240234 -1 352.58352661132812 7.1289982795715332 -1 352.58352661132812 7.1289982795715332 -1 387.39962768554688 19.230798721313477 -1 387.39962768554688 19.230798721313477 -1 388.802490234375 22.210065841674805 -1 781.72698974609375 24.125720977783203 -1 397.67745971679688 26.71160888671875 -1 771.
'''

import os
import argparse
import sys

original_txt_path = sys.argv[1]  # 原始txt路径
new_txt_path = sys.argv[2]      # 新的txt路径

with open(original_txt_path, "r") as f:
    lines = f.readlines()
    with open(new_txt_path, "w") as new_f:
        # 写入前4行
        for i in range(4):
            new_f.write(lines[i])
            print(lines[i])
        for i in range(4, len(lines), 2):
            new_f.write(lines[i])
            # 后面空一行
            new_f.write("\n")
            # print(lines[i])

print("Done!")

