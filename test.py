import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

from utils import XRayDataset, test_loop, retrieve_head
import configs.base as base_config
from model import CustomResNet

parser = argparse.ArgumentParser(description ='Train a simple CNN head + MLP')
parser.add_argument("--head",
					type = str,
					help ='name of CNN head you wish to use')
parser.add_argument("--ckpt",
					type = str,
					help ='name of weights you wish to load, weights must be located in \/weights.')
parser.add_argument("--data",
					type = str,
					help = 'path to your data folder, which has \/train, \/val and \/test folders',
					default = 'data')
parser.add_argument("--gpu",
					action = 'store_true',
					help = 'use gpu for training and inference')
parser.add_argument("--resize",
					type = int,
					help = 'size of images to resize to',
					default = 224)

args = parser.parse_args()

transform = torchvision.transforms.Resize((args.resize, args.resize))

if args.gpu:
	device = 'cuda'
else:
	device = 'cpu'

#############
# Load Data #
#############
test = XRayDataset(args.data + '/test', transform = transform)
test_dataloader = DataLoader(test, batch_size=1, shuffle=True)

##############
# Load model #
##############
head = retrieve_head(args.head, pretrained = False)
model = CustomResNet(head)
if args.gpu:
	model.to(device)
model.load_state_dict(torch.load('weights/'+args.ckpt))

##############
# Test Model #
##############
test_loop(test_dataloader, model, device)

