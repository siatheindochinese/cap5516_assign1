# UCF CAP 5516 Spring 2024 Programming Assignment 1

## 0) Install relevant packages
Install all relevant packages with `pip install -r requirements.txt`.

## 1) Dataset
Download the XRay pneumonia classification dataset from https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia and extract the \/train, \/test and \/val folder into the \/data folder in this repostory.

## 2) Training
Run `python train.py` to train a model. You will need to specify some flags

- `--head` name of the CNN head you wish to use, either `resnet18`, `resnet34` or `resnet50`.
- `--pretrained` use ImageNet (v1) pretrained weights for your CNN head.
- `--data` path to your data folder, by default it is just \/data. Follow Step 1 outlined above.
- `--gpu` specify whether to use a GPU, by default it uses a GPU.
- `--batchsize` specify the batch size you wish to use for training.
- `--epochs` specify the number of epochs you wish to train for
- `--resize` specify the dimension of images to resize to (both width and height will be the same), default is 224.
- `--xflip` (for data augmentation) specify the probability that your image will flip horizontally, default is 0.
- `--yflip` (for data augmentation) specify the probability that your image will flip vertically, default is 0.
- `--rotate` (for data augmentation) specify the range of angles that your image will randomly rotate, default is 0.

All training is done using early stopping. The `EarlyStopper` class can be found in `utils.py`.

Training is done on one RTX 4090 with 24GB vRAM. If you have a weaker GPU, please reduce the batch size with `--batchsize`.

## 3) Testing
tba