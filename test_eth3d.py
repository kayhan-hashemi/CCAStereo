import torch
import torch.nn as nn
from torchvision import transforms
import os
from tqdm import  trange
from PIL import Image
import numpy as np
import argparse
from models import __models__
from datasets import ETH3D_loader as et
from datasets.readpfm import readPFM
from torchvision.utils import save_image
from utils import *
import cv2


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Efficient Stereo Matching through Cross-Attention Integration of Contextual and Geometric Features')
parser.add_argument('--model', default='CCAStereo', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default='', help='data path')
parser.add_argument('--loadckpt', default='./pretrained/sceneflow.ckpt',help='load the weights from a specific checkpoint')

# parse arguments
args = parser.parse_args()


all_limg, all_rimg, all_disp, all_mask = et.et_loader(args.datapath)


model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()
model.eval()

os.makedirs('./outputs/eth3d/', exist_ok=True)

state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])


pred_mae = 0
pred_op = 0
for i in trange(len(all_limg)):
    limg = Image.open(all_limg[i]).convert('RGB')
    rimg = Image.open(all_rimg[i]).convert('RGB')

    w, h = limg.size
    wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32
    limg = limg.crop((w - wi, h - hi, w, h))
    rimg = rimg.crop((w - wi, h - hi, w, h))

    limg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
    rimg_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)
    limg_tensor = limg_tensor.unsqueeze(0).cuda()
    rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

    disp_gt, _ = readPFM(all_disp[i])
    disp_gt = np.ascontiguousarray(disp_gt, dtype=np.float32)
    disp_gt[disp_gt == np.inf] = 0
    gt_tensor = torch.FloatTensor(disp_gt).unsqueeze(0).unsqueeze(0).cuda()

    occ_mask = np.ascontiguousarray(Image.open(all_mask[i]))

    with torch.no_grad():

        pred_disp  = model(limg_tensor, rimg_tensor)[-1]

        pred_disp = pred_disp[:, hi - h:, wi - w:]

    predict_np = pred_disp.squeeze().cpu().numpy()

    op_thresh = 1
    mask = (disp_gt > 0) & (occ_mask == 255)
    # mask = disp_gt > 0
    error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))

    pred_error = np.abs(predict_np * mask.astype(np.float32) - disp_gt * mask.astype(np.float32))
    pred_op += np.sum(pred_error > op_thresh) / np.sum(mask)
    pred_mae += np.mean(pred_error[mask])

    ########save

    filename = os.path.join('./output/ETH3D/', all_limg[i].split('/')[-2]+all_limg[i].split('/')[-1])
    depth_colored = colorize_depth_maps(cv2.convertScaleAbs(predict_np*4*356, alpha=0.01)).squeeze()
    save_image(depth_colored,filename)



print('#### EPE' , pred_mae / len(all_limg))
print("#### >1.0", pred_op*100 / len(all_limg))