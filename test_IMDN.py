import argparse
import torch
import os
import numpy as np
import utils
import skimage.color as sc
import cv2
from model import architecture
# Testing settings

parser = argparse.ArgumentParser(description='IMDN')
parser.add_argument("--input_folder", type=str, default='Test_Datasets/Set5_LR/x2/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=str, default='results/Set5/x2')
parser.add_argument("--checkpoint", type=str, default='checkpoints/IMDN_x2.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.input_folder
ext = '.png'

filelist = utils.get_list(filepath, ext=ext)

model = architecture.IMDN(upscale=opt.upscale_factor)
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:
    im_l = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    if len(im_l.shape) < 3:
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = model(im_input)
        end.record()
        torch.cuda.synchronize()

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
        
    out_filename = os.path.join(opt.output_folder, os.path.basename(imname))

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)
    cv2.imwrite(out_filename, out_img[:, :, [2, 1, 0]])
