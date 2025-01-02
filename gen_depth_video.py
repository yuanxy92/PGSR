import numpy as np
import cv2
import glob
import os

# Parameters
input_folder = "/data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr/train/ours_40000/renders_depth"  # Path to your npy files
fps = 30 # Frames per second

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

# Normalize and apply colormap
for view_id in range(4):
    depth_rgb_frames = []
    rgb_frames = []
    output_video = f"/data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr/train/depth_video_view_{view_id}.mp4"      # Output video file
    output_combine_video = f"/data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr/train/rgbdepth_video_view_{view_id}.mp4"      # Output video file
    depth_stack = depth_images_all_views_stack[:, :, :, view_id]
    for i in range(depth_stack.shape[-1]):
        normalized_depth = (depth_stack[:, :, i] - min_depth) / (max_depth - min_depth)
        normalized_depth = (normalized_depth * 255).astype(np.uint8)  # Scale to 0-255
        depth_rgb_frame = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)  # Apply jet colormap
        depth_rgb_frames.append(depth_rgb_frame)
        # read input rgb images
        rgb_name = f'/data/hdd/Data/SkinSight_video/UVC_cam_undis/images/{(i + 75):04d}/{view_id}.png'
        rgb_frames.append(cv2.imread(rgb_name))
    
    # Get frame size from the first image
    frame_height, frame_width, _ = depth_rgb_frames[0].shape

    # Write video using OpenCV (depth only)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    for frame in depth_rgb_frames:
        video_writer.write(frame)
    video_writer.release()

    # Write video using OpenCV (rgb and depth combined)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_combine_video, fourcc, fps, (frame_width * 2, frame_height))
    for frameidx in range(len(depth_rgb_frames)):
        rgb_frame_ = rgb_frames[frameidx]
        depth_frame_ = depth_rgb_frames[frameidx]
        video_writer.write(cv2.hconcat([rgb_frame_, depth_frame_]))
    video_writer.release()

    print(f"Video saved to {output_video} and {output_combine_video}")
