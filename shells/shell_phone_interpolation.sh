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


python train.py -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon/colmap \
    -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon/pgsr_depth_normal \
    --max_abs_split_points 0 --opacity_cull_threshold 0.05 
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 1000
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 2000
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 3000

python train.py -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/convex_recon/colmap \
    -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/convex_recon/pgsr_depth_normal \
    --max_abs_split_points 0 --opacity_cull_threshold 0.05 
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/convex_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 1000
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/convex_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 2000
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/convex_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 3000

python train.py -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/planar_recon/colmap \
    -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/planar_recon/pgsr_depth_normal \
    --max_abs_split_points 0 --opacity_cull_threshold 0.05 
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/planar_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 1000
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/planar_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 2000
python render_interpolation.py -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/planar_recon/pgsr_depth_normal \
    --max_depth 60.0 --voxel_size 0.025 --iteration 3000