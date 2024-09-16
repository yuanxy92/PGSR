#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import copy
from collections import deque

from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal

from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import CubicSpline

def look_at_to_rt(eye, target, up):
    # Calculate the forward, right, and up vectors.
    up = -up
    zaxis = (target - eye)
    zaxis /= np.linalg.norm(zaxis)
    
    xaxis = np.cross(up, zaxis)
    xaxis /= np.linalg.norm(xaxis)
    
    yaxis = np.cross(zaxis, xaxis)
    yaxis /= np.linalg.norm(yaxis)

    # Create the rotation matrix (3x3).
    rotation_matrix = np.vstack((xaxis, yaxis, zaxis)).T

    # Create the translation vector (3x1).
    translation_vector = -np.dot(eye, rotation_matrix)

    # Combine the rotation and translation into an RT matrix (4x4).
    rt_matrix = np.identity(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation_vector

    rt_matrix = rt_matrix.astype(np.float32)
    rotation_matrix = rotation_matrix.astype(np.float32)
    translation_vector = translation_vector.astype(np.float32)
    
    return rt_matrix, rotation_matrix, translation_vector

# function to load cameras
class ViewpointCamera:
    FoVx: np.array
    FoVy: np.array
    image_width: int
    image_height: int
    world_view_transform: np.array
    full_proj_transform: np.array
    camera_center: np.array
    R: np.array
    T: np.array

    def __init__(self, image_width=1280, image_height=720, fx = 783.5623272369992, fy = 775.0592003023257):
        self.image_width = image_width
        self.image_height = image_height
        # "fy": 775.0592003023257, "fx": 783.5623272369992
        self.FoVx = focal2fov(fx, self.image_width)
        self.FoVy = focal2fov(fy, self.image_height)

    def load_extrinsic(self, eye, target, up, znear, zfar):
        self.camera_center = eye
        RT, self.R, self.T = look_at_to_rt(eye, target, up)
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        self.world_view_transform = getWorld2View2(self.R, self.T, trans, scale)
        # self.world_view_transform = RT
        self.world_view_transform = torch.tensor(self.world_view_transform).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load_extrinsic2(self, R, T, znear, zfar):
        trans = np.array([0.0, 0.0, 0.0])
        scale = 1.0
        self.world_view_transform = getWorld2View2(R, T, trans, scale)
        # self.world_view_transform = RT
        self.world_view_transform = torch.tensor(self.world_view_transform).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class CameraInterpolation:
    # Define a function to interpolate rotation matrices using SLERP
    def interpolate_extrinsic_matrices(Rs, Ts, inter_num):
        interpolated_matrices = []
        t_orig = np.linspace(0.0, len(Rs) - 1, num=len(Rs), endpoint=True)
        t_values = np.linspace(0.0, len(Rs) - 1, num=inter_num, endpoint=True)
        # Interpolate rotation matrices 
        rotations = Rotation.from_matrix(Rs)
        rot_spline = RotationSpline(t_orig, rotations)
        interpolated_rotations = rot_spline(t_values)
        interpolated_matrices = interpolated_rotations.as_matrix()
        # Interpolate translation matrices 
        Ts_mat = np.stack(Ts)
        spline1 = CubicSpline(t_orig, Ts_mat[:, 0])
        spline2 = CubicSpline(t_orig, Ts_mat[:, 1])
        spline3 = CubicSpline(t_orig, Ts_mat[:, 2])
        interpolated_positions = np.array([
            spline1(t_values),
            spline2(t_values),
            spline3(t_values)
        ]).T
        return interpolated_matrices, interpolated_positions
    
def interpolate_cameras(cameras, inter_num):
    Rs = []
    Ts = []
    for idx in range(len(cameras)):
        Rs.append(cameras[idx].R)
        Ts.append(cameras[idx].T)
    Rs_inter, Ts_inter = CameraInterpolation.interpolate_extrinsic_matrices(Rs, Ts, inter_num=inter_num)
    views = []
    for idx in range(Rs_inter.shape[0]):
        view = ViewpointCamera()
        view.load_extrinsic2(Rs_inter[idx, :, :], Ts_inter[idx, :], 0.01, 100)
        views.append(view)
    return views


def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, 
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")

    makedirs(render_path, exist_ok=True)

    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # gt, _ = view.get_image()
        out = render(view, gaussians, pipeline, background, app_model=app_model, return_depth_normal=False)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        if name == 'test':
            # torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, f"{idx:04d}.jpg"), rendering_np)
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, use_depth_filter : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        # app_model = AppModel()
        # app_model.load_weights(scene.model_path)
        # app_model.eval()
        # app_model.cuda()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        bg_color = [1,1,1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        print(f"TSDF voxel_size {voxel_size}")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4.0*voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        camera_trajectories = interpolate_cameras(scene.getTrainCameras(), 100)

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, camera_trajectories, scene, gaussians, pipeline, background, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter)
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.use_depth_filter)