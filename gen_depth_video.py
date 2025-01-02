import numpy as np
import cv2
import glob
import os
from fast_bilateral_solver import *

def solve_image_ldl3(A11, A12, A13, A22, A23, A33, b1, b2, b3):
    # An unrolled LDL solver for a 3x3 symmetric linear system.
    d1 = A11
    L12 = A12/d1
    d2 = A22 - L12*A12
    L13 = A13/d1
    L23 = (A23 - L13*A12)/d2
    d3 = A33 - L13*A13 - L23*L23*d2
    y1 = b1
    y2 = b2 - L12*y1
    y3 = b3 - L13*y1 - L23*y2
    x3 = y3/d3
    x2 = y2/d2 - L23*x3
    x1 = y1/d1 - L12*x2 - L13*x3
    return x1, x2, x3

def planar_filter(Z, filt, eps):
    # Solve for the plane at each pixel in `Z`, where the plane fit is computed
    # by using `filt` (a function that blurs something of the same size and shape
    # as `Z` by taking a linear non-negative combination of inputs) to weight
    # pixels in Z, and `eps` regularizes the output to be fronto-parallel.
    # Returns (Zx, Zy, Zz), which is a plane parameterization for each pixel:
    # the derivative wrt x and y, and the offset (which can itself be used as
    # "the" filtered output).

    # Note: This isn't the same code as in the paper. I flipped x and y to match
    # a more pythonic (x, y) convention, and I had to flip a sign on the output
    # slopes to make the unit tests pass(this may be a bug in the paper's math).
    # Also, I decided to not regularize the "offset" component of the plane fit,
    # which means that setting eps -> infinity gives the output (0, 0, filt(Z)).
    xy_shape = np.array(Z.shape[-2:])
    xy_scale = 2 / np.mean(xy_shape-1)  # Scaling the x, y coords to be in ~[0, 1]
    x, y = np.meshgrid(*[(np.arange(s) - (s-1)/2) * xy_scale for s in xy_shape], indexing='ij')
    [F1, Fx, Fy, Fz, Fxx, Fxy, Fxz, Fyy, Fyz] = [
        filt(t) for t in [
        np.ones_like(x), x, y, Z, x**2, x*y, x*Z, y**2, y*Z]]
    A11 = F1*x**2 - 2*x*Fx + Fxx + eps**2
    A22 = F1*y**2 - 2*y*Fy + Fyy + eps**2
    A12 = F1*y*x - x*Fy - y*Fx + Fxy
    A13 = F1*x - Fx
    A23 = F1*y - Fy
    A33 = F1# + eps**2
    b1 = Fz*x - Fxz
    b2 = Fz*y - Fyz
    b3 = Fz
    Zx, Zy, Zz = solve_image_ldl3(A11, A12, A13, A22, A23, A33, b1, b2, b3)
    return -Zx*xy_scale, -Zy*xy_scale, Zz

# A simple linear blur filter. This can be whatever, provided it averages the
# input images by averaging its inputs with non-negative weights.
def blur(X, alpha):
    # Do an exponential decay filter on the outermost two dimensions of X.
    # Equivalent to convolving an image with a Laplacian blur.
    Y = X.copy()
    for i in range(Y.shape[-1]-1):
        Y[...,i+1] += alpha * Y[...,i]

    for i in range(Y.shape[-1]-1)[::-1]:
        Y[...,i] += alpha * Y[...,i+1]

    for i in range(Y.shape[-2]-1):
        Y[...,i+1,:] += alpha * Y[...,i,:]

    for i in range(Y.shape[-2]-1)[::-1]:
        Y[...,i,:] += alpha * Y[...,i+1,:]
    return Y

def load_single_view_depth_results(view_id):
    sv_dir = '/data/hdd/Data/SkinSight_video/20241227'
    sv_depth = np.load(os.path.join(sv_dir, f'1215_{view_id}_pred.npy'))
    return sv_depth

# Parameters
input_folder = "/data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr/train/ours_40000/renders_depth"  # Path to your npy files
fps = 30 # Frames per second

# load single view depth 
sv_depth_npy_name = os.path.join(f"/data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr/train/sv_depth_all_views.npy")
if not os.path.isfile(sv_depth_npy_name):
    sv_depth_images_all_views = []
    for view_id in range(4):
        sv_depth = load_single_view_depth_results(view_id)
        sv_depth2 = np.zeros((512, 512, sv_depth.shape[2]))
        for t in range(sv_depth.shape[2]):
            sv_depth2[:, :, t] = cv2.resize(sv_depth[t, :, :].astype(np.float32), (512, 512))
        sv_depth_images_all_views.append(sv_depth2)
    sv_depth_images_all_views_stack = np.stack(sv_depth_images_all_views, axis=-1).astype(np.float32)
    np.save(sv_depth_npy_name, sv_depth_images_all_views_stack)
else:
    sv_depth_images_all_views_stack = np.load(sv_depth_npy_name)

# Load depth images and find global min and max for colormap normalization
depth_images_all_views = []
for view_id in range(4):
    depth_images = []
    for frameidx in range(75, 843):
        npyname = os.path.join(input_folder, f'{frameidx:04d}', f'{view_id}.npy')
        if frameidx % 50 == 0:
            print(npyname)
        depth = np.load(npyname)
        depth_images.append(depth)
    depth_stack = np.stack(depth_images, axis=-1)
    depth_images_all_views.append(depth_stack)

# Convert list to 3D matrix (H, W, T, view)
depth_images_all_views_stack = np.stack(depth_images_all_views, axis=-1)
np.save(os.path.join(f"/data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr/train/depth_all_views.npy"), depth_images_all_views_stack)

# Calculate global min and max for the colormap
min_depth = np.min(depth_images_all_views_stack)
max_depth = 20
print(f'min depth: {min_depth}, max depth: {max_depth}')

# fast bilateral filter
grid_params = {
    'sigma_luma' : 1.5, # Brightness bandwidth
    'sigma_chroma': 0.1, # Color bandwidth
    'sigma_spatial': 2.5 # Spatial bandwidth
}
bs_params = {
    'lam': 4, # The strength of the smoothness parameter
    'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
    'cg_tol': 1e-5, # The tolerance on the convergence in PCG
    'cg_maxiter': 25 # The number of PCG iterations
}

for view_id in range(4):
    depth_rgb_frames = []
    rgb_frames = []
    depany_frames = []
    depth_rgb_fbs_frames = []
    output_combine_video = f"/data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr/train/rgbdepth_video_view_{view_id}.mp4"      # Output video file
    output_fbs_combine_video = f"/data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr/train/rgbdepth_fbs_video_view_{view_id}.mp4"      # Output video file

    depth_stack = depth_images_all_views_stack[:, :, :, view_id]
    for i in range(depth_stack.shape[-1]):
        normalized_depth_f = (depth_stack[:, :, i] - min_depth) / (max_depth - min_depth)
        normalized_depth = (normalized_depth_f * 255).astype(np.uint8)  # Scale to 0-255
        depth_rgb_frame = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)  # Apply jet colormap
        depth_rgb_frames.append(depth_rgb_frame)
        
        # read input rgb images
        rgb_name = f'/data/hdd/Data/SkinSight_video/UVC_cam_undis/images/{(i + 75):04d}/{view_id}.png'
        rgb_frame = cv2.imread(rgb_name)
        rgb_frames.append(rgb_frame)
        
        # read depth anything results
        depthanything_name = f'/data/hdd/Data/SkinSight_video/UVC_cam_undis/depths/{(i + 75):04d}/{view_id}.png'
        depany_frame = cv2.imread(depthanything_name)
        depany_frames.append(depany_frame)

        # apply fast bilateral solver
        reference = depany_frame
        target = normalized_depth_f
        im_shape = reference.shape[:2]
        confidence = np.ones(im_shape, np.float32)
        grid = BilateralGrid(reference, **grid_params)
        t = target.reshape(-1, 1).astype(np.double)
        c = confidence.reshape(-1, 1).astype(np.double)
        tc_filt = grid.filter(t * c)
        c_filt = grid.filter(c)
        output_filter = (tc_filt / c_filt).reshape(im_shape)
        normalized_depth_fbs = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)

        # apply depth to color transform
        normalized_depth_fbs_8U = (normalized_depth_fbs * 255).astype(np.uint8)  # Scale to 0-255
        depth_rgb_fbs_frame = cv2.applyColorMap(normalized_depth_fbs_8U, cv2.COLORMAP_JET)  # Apply jet colormap
        depth_rgb_fbs_frames.append(depth_rgb_fbs_frame)

        if i % 50 == 0:
            print(f'Finish depth image processing for view {view_id} and frame {i}')

    
    # Get frame size from the first image
    frame_height, frame_width, _ = depth_rgb_frames[0].shape

    # # Write video using OpenCV (rgb and depth combined)
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video_writer = cv2.VideoWriter(output_combine_video, fourcc, fps, (frame_width * 2, frame_height))
    # for frameidx in range(len(depth_rgb_frames)):
    #     rgb_frame_ = rgb_frames[frameidx]
    #     depth_frame_ = depth_rgb_frames[frameidx]
    #     video_writer.write(cv2.hconcat([rgb_frame_, depth_frame_]))
    # video_writer.release()

    # Write video using OpenCV (rgb and depth fbs combined)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_fbs_combine_video, fourcc, fps, (frame_width * 3, frame_height))
    for frameidx in range(len(depth_rgb_frames)):
        rgb_frame_ = rgb_frames[frameidx]
        depth_frame_ = depth_rgb_frames[frameidx]
        depth_frame_fbs_ = depth_rgb_fbs_frames[frameidx]
        video_writer.write(cv2.hconcat([rgb_frame_, depth_frame_, depth_frame_fbs_]))
    video_writer.release()

    print(f"Video saved to {output_combine_video} and {output_fbs_combine_video}")
