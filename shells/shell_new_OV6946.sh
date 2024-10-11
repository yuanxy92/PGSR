#!/bin/bash
iterations1=3000
iterations2=3000
voxel_size=0.8

# for i in $(seq -w 020 024);
# do
#     rootdir=/home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241004_4/${i}_recon
#     python dust3r_depth_refine.py ${rootdir}
#     python train.py -s ${rootdir}/colmap \
#         -m ${rootdir}/pgsr_depth_normal \
#         --max_abs_split_points 0 --opacity_cull_threshold 0.1 \
#         --lambda_l1_depth 0.01 \
#         --normal_weight 0.01 \
#         --single_view_weight 0.1 \
#         --multi_view_weight_from_iter 8000 \
#         --iterations ${iterations1} \
#         --multi_view_ncc_weight 0 \
#         --multi_view_geo_weight 0.1
#     python render.py -m ${rootdir}/pgsr_depth_normal \
#         --max_depth 300.0 --voxel_size 0.5 --iteration ${iterations1}
# done

for i in $(seq -w 101 101);
do
    rootdir=/home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241010_2/${i}_recon
    python dust3r_depth_refine.py ${rootdir}
    python train.py -s ${rootdir}/colmap \
        -m ${rootdir}/pgsr_depth_normal \
        --max_abs_split_points 0 --opacity_cull_threshold 0.1 \
        --lambda_l1_depth 0.01 \
        --normal_weight 0.01 \
        --single_view_weight 0.1 \
        --multi_view_weight_from_iter 8000 \
        --iterations ${iterations1} \
        --multi_view_ncc_weight 0 \
        --multi_view_geo_weight 0.1
    python render.py -m ${rootdir}/pgsr_depth_normal \
        --max_depth 300.0 --voxel_size 0.5 --iteration ${iterations1}
done