import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt

from utils import XRayDataset, EarlyStopper, train_loop, val_loop, retrieve_head
import configs.base as base_config
from model import CustomResNet

parser = argparse.ArgumentParser(description ='Train a simple CNN head + MLP')
parser.add_argument("--head",
					type = str,
					help ='name of CNN head you wish to use')
parser.add_argument("--pretrained",
					action = 'store_true',
					help ='use pretrained weights')
parser.add_argument("--data",
					type = str,
					help = 'path to your data folder, which has \/train, \/val and \/test folders',
					default = 'data')
parser.add_argument("--gpu",
					action = 'store_true',
					help = 'use gpu for training and inference')
parser.add_argument("--batchsize",
					type = int,
					help = 'size of batches',
					default = 16)
parser.add_argument("--epochs",
					type = int,
					help = 'number of epochs to train',
					default = 50)
parser.add_argument("--resize",
					type = int,
					help = 'size of images to resize to',
					default = 224)
parser.add_argument("--xflip",
					type = float,
					help = 'chance of random horizontal flip for images',
					default = 0)
parser.add_argument("--yflip",
					type = float,
					help = 'chance of random vertical flip for images',
					default = 0)
parser.add_argument("--rotate",
					type = int,
					help = 'max range of angles to randomly rotate for images',
					default = 0)

args = parser.parse_args()

#################################
# Initialize Data Augmentations #
#################################
transforms = []
transforms.append(torchvision.transforms.RandomHorizontalFlip(args.xflip))
transforms.append(torchvision.transforms.RandomVerticalFlip(args.yflip))
transforms.append(torchvision.transforms.RandomRotation(args.rotate))
transforms.append(torchvision.transforms.Resize((args.resize, args.resize)))
transforms = nn.Sequential(*transforms)

if args.gpu:
	device = 'cuda'
else:
	device = 'cpu'

#############
# Load Data #
#############
train = XRayDataset(args.data + '/train', transform = transforms)
val = XRayDataset(args.data + '/val', transform = transforms)

train_dataloader = DataLoader(train, batch_size=args.batchsize, shuffle=True)
val_dataloader = DataLoader(val, batch_size=args.batchsize, shuffle=True)

################
# Define Model #
################
if args.pretrained:
	head = retrieve_head(args.head, pretrained = True)
else:
	head = retrieve_head(args.head, pretrained = False)
model = CustomResNet(head)
if args.gpu:
	model.to(device)
	
####################
# Training Configs #
####################
ckpt_name = (args.head+'_xflip_'+str(args.xflip)
			+ '_yflip_'+str(args.yflip)
			+ '_rot_'+str(args.rotate)
			+ '.pt')
if args.pretrained:
	ckpt_name = 'pretrained_'+ ckpt_name
loss = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
stopper = EarlyStopper(patience = 8, delta = 0, path = 'weights/'+ckpt_name)

#################
# Training Loop #
#################
val_losses = []
train_losses = []

print(f"Epoch 0\n-------------------------------")
val_loss = val_loop(val_dataloader, model, loss, device)

for t in range(args.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(train_dataloader, model, loss, optimizer, device)
    val_loss = val_loop(val_dataloader, model, loss, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    stop = stopper(val_loss, model)
    if stop:
        break
print("Done Training!")

###################
# Save loss plots #
###################
epochs = list(range(len(val_losses)))
plt.plot(epochs, val_losses, label = 'validation loss')
plt.plot(epochs, train_losses, label = 'train loss')
plt.legend()
plt.savefig('plots/' + ckpt_name[:-3] + '.png')
