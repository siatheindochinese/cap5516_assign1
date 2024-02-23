# UCF CAP 5516 Spring 2024 Programming Assignment 1

## 0) Install relevant packages
Install all relevant packages with `pip install -r requirements.txt`.

## 1) Dataset
Download the XRay pneumonia classification dataset from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and extract the \/train, \/test and \/val folder into the \/data folder in this repostory.

## 2) Training
Run `python train.py` to train a model. You will need to specify some flags:

- `--head` name of the CNN head you wish to use, either `resnet18`, `resnet34` or `resnet50`.
- `--pretrained` use ImageNet (v1) pretrained weights for your CNN head.
- `--data` path to your data folder, by default it is just \/data. Follow Step 1 outlined above.
- `--gpu` specify whether to use a GPU, by default it uses a GPU.
- `--batchsize` specify the batch size you wish to use for training.
- `--epochs` specify the number of epochs you wish to train for.
- `--resize` specify the dimension of images to resize to (both width and height will be the same), default is 224.
- `--xflip` (for data augmentation) specify the probability that your image will flip horizontally, default is 0.
- `--yflip` (for data augmentation) specify the probability that your image will flip vertically, default is 0.
- `--rotate` (for data augmentation) specify the range of angles that your image will randomly rotate, default is 0.

Example:
`python train.py --head resnet18 --pretrained --xflip 0.5 --yflip 0.5 --rot 360 --gpu --epochs 50 --resize 224 --batchsize 256`

All training is done using early stopping. The `EarlyStopper` class can be found in `utils.py`.

Training is done on one RTX 4090 with 24GB vRAM. If you have a weaker GPU, please reduce the batch size with `--batchsize`.

All trained weights can be downloaded from `https://drive.google.com/drive/folders/1GYiqLQYXoqg-g_Ql0k4A6Sb3i3rBwya5` if you do not wish to train all models from scratch.

## 3) Testing
Run `python test.py` to train a model. You will need to specify some flags:

- `--head` name of the CNN head you wish to use, either `resnet18`, `resnet34` or `resnet50`.
- `--ckpt` name of weights you wish to load, weights must be located in `weights`.
- `--data` path to your data folder, by default it is just \/data. Follow Step 1 outlined above.
- `--gpu` specify whether to use a GPU, by default it uses a GPU.
- `--resize` specify the dimension of images to resize to (both width and height will be the same), default is 224.

Example:
`python test.py --head resnet18 --ckpt pretrained_resnet18_xflip_0.5_yflip_0.5_rot_360.pt --gpu --resize 224`

## 4) Mass Training and Testing

Training and tesing for all model configurations can be done by running `sh train.sh` followed by `sh test.sh`.