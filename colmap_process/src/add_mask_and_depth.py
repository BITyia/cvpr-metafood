import os
import json
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser(description='Add mask and depth path to json file')
parser.add_argument('--depth', action='store_true', help='add depth path to json file')
parser.add_argument('--mask', action='store_true', help='add mask path to json file')
parser.add_argument('--root_path', type=str, help='root path of the dataset')

root_path = parser.parse_args().root_path
json_path = root_path + "/transforms.json"
save_path = ""
mask_path = "./" + "mask"
depth_path = "./" + "depth"

def main(mask=True, depth=True):
    with open(json_path, 'r') as f:
        data = json.load(f)

    frames = data["frames"]
    for frame in frames:
        file_path = frame["file_path"]
        id = file_path.split("_")[-2]
        mask_file_path = os.path.join(mask_path, "frame_" + id + "_original_segmented_mask.jpg")
        depth_file_path = os.path.join(depth_path, "frame_" + id + "_depth.png")
        if mask:
          frame["mask_path"] = mask_file_path
        if depth:
          frame["depth_file_path"] = depth_file_path

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        print("Add mask and depth path to json file successfully!")


if __name__ == "__main__":
    depth = parser.parse_args().depth
    mask = parser.parse_args().mask

    if depth and mask:
        save_path = root_path + "/transforms_depth_mask.json"
    elif depth:
        save_path = root_path + "/transforms_depth.json"
    elif mask:
        save_path = root_path + "/transforms_mask.json"

    main(mask, depth)
