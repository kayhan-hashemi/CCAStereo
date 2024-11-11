# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm



if __name__ =='__main__':
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    
    parser = argparse.ArgumentParser(description='Efficient Stereo Matching through Cross-Attention Integration of Contextual and Geometric Features')
    parser.add_argument('--model', default='CCAStereo', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    
    parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath', default='', help='data path')
    parser.add_argument('--testlist',default='./filenames/sceneflow_test.txt', help='testing list')
    
    parser.add_argument('--loadckpt', default='./pretrained/sceneflow.ckpt',help='load the weights from a specific checkpoint')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    
    parser.add_argument('--image_output_path', default='./outputs/sceneflow/')
    parser.add_argument('--save_image_output', default=False)


    # parse arguments, set seeds
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    
    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    test_dataset = StereoDataset(args.datapath, args.testlist, False)
    TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=16, drop_last=False)
    
    # model, optimizer
    model = __models__[args.model](args.maxdisp)
    model = nn.DataParallel(model)
    model.cuda()
    
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    
    
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in tqdm(enumerate(TestImgLoader),total=len(TestImgLoader)):
        
        model.eval()
        imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        disp_gt = disp_gt.cuda()
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        with torch.no_grad():
            disp_ests = model(imgL, imgR)
            
        scalar_outputs = {}
        scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
        scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
        scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
        scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    
        if args.save_image_output:
            path = args.image_output_path+'disp_est'+str(batch_idx)+'.png'
            depth_colored = colorize_depth_maps(disp_ests[0]/args.maxdisp).squeeze()
            save_image(depth_colored,path)
    
        avg_test_scalars.update(tensor2float(scalar_outputs))
        del scalar_outputs
    
    avg_test_scalars = avg_test_scalars.mean()
    
    print("######EPE =",round(avg_test_scalars['EPE'][0],3))
    print("######D1% =",round(avg_test_scalars['D1'][0]*100,2))
    print("####Bad1% =",round(avg_test_scalars['Thres1'][0]*100,2))
    print("####Bad2% =",round(avg_test_scalars['Thres2'][0]*100,2))
    print("####Bad3% =",round(avg_test_scalars['Thres3'][0]*100,2))