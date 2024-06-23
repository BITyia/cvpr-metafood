import os
from pathlib import Path
from typing import List, Optional

import itertools
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .circle_path import make_circle_path
from .llff import make_llff_train_test_split
from .utils import get_rays, srgb_to_linear


class MultiLoader:
    # Helper which wraps a loader and repeats it a set number of times
    def __init__(self, loader: DataLoader, num_repeats: int):
        self._loader = loader
        self._num_repeats = num_repeats

    def __len__(self):
        return self._num_repeats * len(self._loader)

    def __iter__(self):
        return itertools.chain(*[self._loader for _ in range(self._num_repeats)])

    @property
    def _data(self):
        # Ugly use of private member variable :(
        return self._loader._data

    @property
    def batch_size(self):
        return self._loader.batch_size


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''

    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()

        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.mode = opt.mode # colmap, blender, llff
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = True

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        elif self.mode == 'llff':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
            transform['frames'] = sorted(transform['frames'], key=lambda f: f['file_path'])
            if self.mode == 'llff':
                train_idxs, test_idxs = make_llff_train_test_split(
                    num_images_in_scene=len(transform['frames']), target_num_train_images=opt.num_train_poses
                )

            frame_idxs = test_idxs if type in ('test',) else train_idxs
            transform['frames'] = [transform['frames'][idx] for idx in frame_idxs]
            print('Frames now has len', len(transform['frames']))
            print('Images:', [f['file_path'] for f in transform['frames']])
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        # read images
        frames = transform["frames"]

        # for colmap, manually interpolate a test set.
        if type == 'circle_path':
            print('Making circle path')
            pose = np.array(frames[0]['transform_matrix'], dtype=np.float32)  # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale)
            poses = make_circle_path(start_pose=pose,
                                     circle_size=0.15,
                                     num_poses=n_test)
            self.poses = [torch.tensor(p, dtype=torch.float) for p in poses]

            self.segmentation = []
            self.image_filenames = [f'circle_path_{idx}' for idx in range(len(self.poses))]
            self.images = [np.zeros((self.H, self.W, 3)) for _ in range(len(self.poses))]

        elif type == 'interpolate':
            print('Making interpolative trajectory')
            self.poses = []
            self.images = None
            self.image_filenames = []
            for f0, f1 in zip(frames[:-1], frames[1:]):
                print('Interpolating between frames', f0, f1)
                pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
                pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
                rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                slerp = Slerp([0, 1], rots)

                for i in range(n_test):
                    ratio = i / n_test
                    pose = np.eye(4, dtype=np.float32)
                    pose[:3, :3] = slerp(ratio).as_matrix()
                    pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                    self.poses.append(pose)
                    self.image_filenames.append(f'interp_{f0["file_path"]}_{f1["file_path"]}_{i}')

            self.poses = [torch.tensor(p, dtype=torch.float) for p in self.poses]

            self.segmentation = []
            self.images = [np.zeros((self.H, self.W, 3)) for _ in range(len(self.poses))]

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames

            self.poses = []
            self.images = []
            self.segmentation = []
            self.image_filenames = []


            for f in tqdm.tqdm(frames, desc=f'Loading {type} data:'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and f_path[-4:] != '.png':
                    f_path += '.png' # so silly...

                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                assert image is not None, f'Unable to find image at {f_path}'
                if downscale != 1 and self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                H, W, C = image.shape
                H = H // downscale
                W = W // downscale

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
                self.image_filenames.append(f['file_path'])

        print('No segmentation found for dataset.')
        self.segmentation = None

        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]

        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
        print('Loaded intrinsics:', self.intrinsics)

        analyse_poses(self.poses)

    @property
    def fov_x(self):
        return 2. * np.arctan(self.W / (2. * self.intrinsics[0]))

    @property
    def fov_y(self):
        return 2. * np.arctan(self.H / (2. * self.intrinsics[1]))

    def collate(self, index):

        B = len(index) # always 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]

        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'rays_d_cam_z': rays['rays_d_cam_z'],
            'pose_c2w': poses,
            'intrinsics': self.intrinsics,
            'image_filename': self.image_filenames[index[0]],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            results['images_full'] = images
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
            results['inds'] = rays['inds']

        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader


def analyse_poses(poses):
    # Print out some information about where the camera centres are
    # Can be useful in helping to set the bounds
    # Poses should be camera-to-world and 4x4
    camera_centres = [np.array(p[:3, 3]) for p in poses]
    camera_centres = np.array(camera_centres)
    bounds_min = np.min(camera_centres, axis=0)
    bounds_max = np.max(camera_centres, axis=0)
    print('Pose bounds:', bounds_min, bounds_max)
    print('Avg of min and max bounds:', 0.5*bounds_min + 0.5*bounds_max)


def get_typical_deltas_between_poses(poses):
    # Get avg deltas between poses
    poses = np.array([np.array(p) for p in poses])
    delta_positions = []
    delta_orientations = []
    for pose, next_pose in zip(poses[:-1], poses[1:]):
        delta_position = next_pose[:3, 3] - pose[:3, 3]
        delta_orientation = Rotation.from_matrix(pose[:3, :3]) * Rotation.from_matrix(next_pose[:3, :3]).inv()
        delta_positions.append(delta_position)
        delta_orientations.append(delta_orientation)

    mean_delta_pos = np.mean(np.linalg.norm(delta_positions, axis=1))
    mean_delta_ori = np.mean([np.linalg.norm(rot.as_rotvec()) for rot in delta_orientations])
    return mean_delta_pos, mean_delta_ori
