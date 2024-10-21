#!/bin/bash

python render_interpolation.py \
    -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/concave_recon_smooth/colmap \
    -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/concave_recon_smooth/pgsr_depth_normal \
    --max_depth 300.0 --voxel_size 0.05 --iteration 50001
python render_interpolation.py \
    -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/concave_recon_smooth/colmap \
    -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/concave_recon_smooth/pgsr_depth_normal \
    --max_depth 300.0 --voxel_size 0.05 --iteration 50002

# python render_interpolation.py \
#     -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/planar_recon_smooth/colmap \
#     -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/planar_recon_smooth/pgsr_depth_normal \
#     --max_depth 300.0 --voxel_size 0.05 --iteration 50001
# python render_interpolation.py \
#     -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/planar_recon_smooth/colmap \
#     -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/planar_recon_smooth/pgsr_depth_normal \
#     --max_depth 300.0 --voxel_size 0.05 --iteration 50002

# python render_interpolation.py \
#     -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/convex_recon_smooth/colmap \
#     -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/convex_recon_smooth/pgsr_depth_normal \
#     --max_depth 300.0 --voxel_size 0.05 --iteration 50001
# python render_interpolation.py \
#     -s /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/convex_recon_smooth/colmap \
#     -m /home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/smooth/convex_recon_smooth/pgsr_depth_normal \
#     --max_depth 300.0 --voxel_size 0.05 --iteration 50002