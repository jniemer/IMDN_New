import argparse
import torch
from torch import onnx
import os
import numpy as np
import utils
from models.IMDN import IMDN
# Testing settings

parser = argparse.ArgumentParser(description='IMDN')
parser.add_argument("--checkpoint", type=str, default='checkpoints/model.pth',
                    help='Checkpoint file to load')
parser.add_argument("--size", type=int, default=256,
                    help='image size')
parser.add_argument("--upscale", type=int, default=4,
                    help='upscaling factor')
opt = parser.parse_args()

if (os.path.exists(opt.checkpoint)):
    device = torch.device('cpu')
    model = IMDN(in_nc=3, out_nc=3, nc=64, nb=8, upscale=opt.upscale)
    #model_dict = utils.load_state_dict()
    model.load_state_dict(torch.load(opt.checkpoint, map_location=device), strict=True)
    
    dummy_input = torch.randn(1, 3, opt.size, opt.size, device='cpu')
    input_names = [ "imagedata" ]
    output_names = [ "image_upscaled" ]

    torch.onnx.export(model, dummy_input, "test.onnx", input_names= input_names, output_names= output_names)
else:
    print(f'File {opt.checkpoint} does not exist!')