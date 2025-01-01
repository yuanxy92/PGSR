python render.py -m /data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr \
    --max_depth 10.0 \
    --voxel_size 0.05 \
    --use_depth_filter
    # --iterations 40000

python train.py -s /data/hdd/Data/SkinSight_video/UVC_cam_undis \
    -m /data/hdd/Data/SkinSight_video/UVC_cam_undis/pgsr \
    --max_abs_split_points 0 \
    --opacity_cull_threshold 0.05 

