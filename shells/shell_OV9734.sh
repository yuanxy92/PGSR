#!/bin/bash
for i in {1..7}
do
    python train.py -s /home/luvision/project/Code/data/Aurora/Fig_3/20240911_OV9734/shape_$i/images_recon/colmap \
        -m /home/luvision/project/Code/data/Aurora/Fig_3/20240911_OV9734/shape_$i/images_recon/pgsr_depth \
        --max_abs_split_points 0 --opacity_cull_threshold 0.05 
    python render.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240911_OV9734/shape_$i/images_recon/pgsr_depth \
        --max_depth 60.0 --voxel_size 0.025 --iteration 2000
done