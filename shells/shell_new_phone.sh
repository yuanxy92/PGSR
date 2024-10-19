#!/bin/bash
iterations1=8000
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

# for i in $(seq -w 001 001);
# do
#     rootdir=/home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241010_2/${i}_recon
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
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 500
#     python render.py -m ${rootdir}/pgsr_depth_normal \
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 3000
#     python render.py -m ${rootdir}/pgsr_depth_normal \
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 5000
# done

# for i in $(seq -w 004 004);
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
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 500
#     python render.py -m ${rootdir}/pgsr_depth_normal \
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 3000
#     python render.py -m ${rootdir}/pgsr_depth_normal \
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 5000
# done

# for i in $(seq -w 003 003);
# do
#     rootdir=/home/luvision/project/Code/data/Aurora/Fig_4/Softgripper_demo/capture_20241014_hand_1/${i}_recon
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
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 500
#     python render.py -m ${rootdir}/pgsr_depth_normal \
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 3000
#     python render.py -m ${rootdir}/pgsr_depth_normal \
#         --max_depth 1000.0 --voxel_size 0.05 --iteration 5000
# done

rootdir=/home/luvision/project/Code/data/Aurora/Fig_3/20240914_phone2/concave_recon_smooth
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
    --multi_view_geo_weight 0.1 \
    --normal_scale 1.0
python render.py -m ${rootdir}/pgsr_depth_normal \
    --max_depth 300.0 --voxel_size 0.01 --iteration 500
python render.py -m ${rootdir}/pgsr_depth_normal \
    --max_depth 300.0 --voxel_size 0.01 --iteration 3000
python render.py -m ${rootdir}/pgsr_depth_normal \
    --max_depth 300.0 --voxel_size 0.01 --iteration 5000

