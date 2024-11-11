import open3d as o3d
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
from models import __models__
from utils import *
from datasets import middlebury_loader as mb
from datasets import KITTI2015loader as kt2015
from torchvision import transforms
from PIL import Image


parser = argparse.ArgumentParser(description='Efficient Stereo Matching through Cross-Attention Integration of Contextual and Geometric Features')
parser.add_argument('--model', default='CCAStereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--scene_index', type=int, default=150, help='index of selected scene to plot')
parser.add_argument('--datapath', default='G:/Datasets/KITTI_2015', help='data path')
parser.add_argument('--testlist',default='./filenames/kitti15_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./pretrained/kitti.ckpt',help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()


all_limg, all_rimg, all_ldisp, test_limg, test_rimg, test_ldisp = kt2015.kt2015_loader(args.datapath+'/testing')
test_limg = all_limg + test_limg
test_rimg = all_rimg + test_rimg
limg = Image.open(test_limg[args.scene_index]).convert('RGB')
rimg = Image.open(test_rimg[args.scene_index]).convert('RGB')
w, h = limg.size
m = 32
wi, hi = (w // m + 1) * m, (h // m + 1) * m
limg = limg.crop((w - wi, h - hi, w, h))
rimg = rimg.crop((w - wi, h - hi, w, h))
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
left = transform(limg)
right = transform(rimg)
left = left.unsqueeze(0).cuda()
right = right.unsqueeze(0).cuda()
focal_length = 721 
baseline = 0.54
    
# model
model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda().eval()

#load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

with torch.no_grad():
    disp_ests = model(left,right)[-1]

disp_est = tensor2numpy(disp_ests)[0]
disp_est[disp_est == 0] = 0.1
depth_map = (focal_length * baseline) / disp_est
depth_map[np.isnan(depth_map)] = 0

crop = 15
rgb_image = np.ascontiguousarray(np.array(limg)[crop:-crop,crop:-crop])
depth_image = np.ascontiguousarray(np.asarray(depth_map, dtype=np.float32)[crop:-crop,crop:-crop])

# Convert numpy arrays to Open3D Image objects
o3d_rgb = o3d.geometry.Image(rgb_image)
o3d_depth = o3d.geometry.Image(depth_image)

# Create RGBDImage from the color and depth images
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d_rgb, 
    o3d_depth, 
    convert_rgb_to_intensity=False
)

# Define the camera intrinsic parameters

intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(
    1242,          
    375,           
    focal_length,  
    focal_length,  
    6.071928e+02,  
    1.852157e+02   
)


# Create a point cloud from the RGBD image
point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, 
    intrinsic
)

# Flip the point cloud
point_cloud.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])