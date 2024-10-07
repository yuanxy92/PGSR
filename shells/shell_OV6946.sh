#!/bin/bash

# python train.py -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon/colmap \
#     -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon/pgsr_depth \
#     --max_abs_split_points 0 --opacity_cull_threshold 0.05 
# python render.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon/pgsr_depth \
#     --max_depth 60.0 --voxel_size 0.025 --iteration 2000

# python train.py -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/convex_recon/colmap \
#     -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/convex_recon/pgsr_depth \
#     --max_abs_split_points 0 --opacity_cull_threshold 0.05 
# python render.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/convex_recon/pgsr_depth \
#     --max_depth 60.0 --voxel_size 0.025 --iteration 2000

# python train.py -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/planar_recon/colmap \
#     -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/planar_recon/pgsr_depth \
#     --max_abs_split_points 0 --opacity_cull_threshold 0.05 
# python render.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/planar_recon/pgsr_depth \
#     --max_depth 60.0 --voxel_size 0.025 --iteration 2000

# for i in $(seq -w 000 014);
# do
#     python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241002_2/${i}_recon/colmap \
#         -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241002_2/${i}_recon/pgsr_depth_normal \
#         --max_abs_split_points 0 --opacity_cull_threshold 0.05 
#     python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241002_2/${i}_recon/pgsr_depth_normal \
#         --max_depth 60.0 --voxel_size 0.05 --iteration 2000
# done

# for i in $(seq -w 000 018);
# do
#     python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241003_1/${i}_recon/colmap \
#         -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241003_1/${i}_recon/pgsr_depth_normal \
#         --max_abs_split_points 0 --opacity_cull_threshold 0.05 
#     python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241003_1/${i}_recon/pgsr_depth_normal \
#         --max_depth 60.0 --voxel_size 0.05 --iteration 2000

#     python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241003_1/${i}_sr_recon/colmap \
#         -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241003_1/${i}_sr_recon/pgsr_depth_normal \
#         --max_abs_split_points 0 --opacity_cull_threshold 0.05 
#     python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241003_1/${i}_sr_recon/pgsr_depth_normal \
#         --max_depth 60.0 --voxel_size 0.05 --iteration 2000
# done

for i in $(seq -w 000 004);
do
    python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_2/${i}_recon/colmap \
        -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_2/${i}_recon/pgsr_depth_normal \
        --max_abs_split_points 0 --opacity_cull_threshold 0.05 
    python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_2/${i}_recon/pgsr_depth_normal \
        --max_depth 60.0 --voxel_size 0.05 --iteration 2000

    python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_2/${i}_sr_recon/colmap \
        -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_2/${i}_sr_recon/pgsr_depth_normal \
        --max_abs_split_points 0 --opacity_cull_threshold 0.05 
    python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_2/${i}_sr_recon/pgsr_depth_normal \
        --max_depth 60.0 --voxel_size 0.05 --iteration 2000
done

for i in $(seq -w 000 005);
do
    python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_3/${i}_recon/colmap \
        -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_3/${i}_recon/pgsr_depth_normal \
        --max_abs_split_points 0 --opacity_cull_threshold 0.05 
    python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_3/${i}_recon/pgsr_depth_normal \
        --max_depth 60.0 --voxel_size 0.05 --iteration 2000

    python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_3/${i}_sr_recon/colmap \
        -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_3/${i}_sr_recon/pgsr_depth_normal \
        --max_abs_split_points 0 --opacity_cull_threshold 0.05 
    python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_3/${i}_sr_recon/pgsr_depth_normal \
        --max_depth 60.0 --voxel_size 0.05 --iteration 2000
done

for i in $(seq -w 000 012);
do
    python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4/${i}_recon/colmap \
        -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4/${i}_recon/pgsr_depth_normal \
        --max_abs_split_points 0 --opacity_cull_threshold 0.05 
    python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4/${i}_recon/pgsr_depth_normal \
        --max_depth 60.0 --voxel_size 0.05 --iteration 2000

    python train.py -s /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4/${i}_sr_recon/colmap \
        -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4/${i}_sr_recon/pgsr_depth_normal \
        --max_abs_split_points 0 --opacity_cull_threshold 0.05 
    python render.py -m /home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4/${i}_sr_recon/pgsr_depth_normal \
        --max_depth 60.0 --voxel_size 0.05 --iteration 2000
done