from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import time
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
from torchvision.utils import save_image


if __name__=='__main__':
    # cudnn.benchmark = True
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    parser = argparse.ArgumentParser(description='Efficient Stereo Matching through Cross-Attention Integration of Contextual and Geometric Features')
    parser.add_argument('--model', default='CCAStereo', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--dataset', default='kitti', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--datapath_12', default='', help='data path')
    parser.add_argument('--datapath_15', default='', help='data path')
    parser.add_argument('--testlist',default='', help='testing list')
    parser.add_argument('--loadckpt', default='./pretrained/kitti.ckpt',help='load the weights from a specific checkpoint')
    # parse arguments
    args = parser.parse_args()
    
    # dataset, dataloader
    StereoDataset = __datasets__[args.dataset]
    test_dataset = StereoDataset(args.datapath_12, args.datapath_15, args.testlist, False)
    TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)
    
    # model, optimizer
    model = __models__[args.model](args.maxdisp)
    model = nn.DataParallel(model)
    model.cuda()
    
    ###load parameters
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    
    save_dir = './outputs/kitti'
    
    def test():
        os.makedirs(save_dir, exist_ok=True)
        for batch_idx, sample in enumerate(TestImgLoader):
            torch.cuda.synchronize()
            start_time = time.time()
            disp_est_np = tensor2numpy(test_sample(sample))
            torch.cuda.synchronize()
            print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                    time.time() - start_time))
            top_pad_np = tensor2numpy(sample["top_pad"])
            right_pad_np = tensor2numpy(sample["right_pad"])
            left_filenames = sample["left_filename"]
    
            for disp_est, top_pad, right_pad, fn in zip(disp_est_np, top_pad_np, right_pad_np, left_filenames):
                assert len(disp_est.shape) == 2
                disp_est = np.array(disp_est[top_pad:, :-right_pad], dtype=np.float32)
    
                fn = os.path.join(save_dir, fn.split('/')[-1])
                print("saving to", fn, disp_est.shape)
    
                depth_colored = colorize_depth_maps(disp_est).squeeze()
                save_image(depth_colored,fn)
    
    
    
    # test one sample
    @make_nograd_func
    def test_sample(sample):
        model.eval()
        disp_ests = model(sample['left'].cuda(), sample['right'].cuda())
        return disp_ests[-1]

    test()
    