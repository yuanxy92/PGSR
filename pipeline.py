#pipeline
import os
#render pts
#monodepths
#guided gt imgs
data1 = '004_recon_colmap'
voxel_size = 0.8
iterations1 = 3000
iterations2 = 3000
output1 = '1004'


os.system(f'python iterative_fitting_all.py ./data/{data1}/depths ./data/{data1}/mono_depths')

os.system(f'python weight_guided_filter.py ./data/{data1} 1 1')



os.system(f'python train.py -s ./data/{data1} -m output/{output1} --max_abs_split_points 0 --opacity_cull_threshold 0.1 \
--lambda_l1_depth 0.01 --normal_weight 0.01 --single_view_weight 0.1 --multi_view_weight_from_iter 8000 --iterations {iterations1} --multi_view_ncc_weight 0 --multi_view_geo_weight 0.1')
os.system(f'python render.py -m ./output/{output1} --max_depth 6000 --voxel_size {voxel_size} --eval --iteration {iterations1}')


