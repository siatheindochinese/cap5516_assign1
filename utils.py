import os
import random

import torch
import torchvision
from torch.utils.data import Dataset

class XRayDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dirs = []
        self.labels = []

        normal = os.listdir(os.path.join(img_dir,'NORMAL'))
        normal = list(map(lambda x: os.path.join(img_dir, 'NORMAL', x), normal))
        self.img_dirs.extend(normal); self.labels.extend([0]*len(normal))

        pneumonia = os.listdir(os.path.join(img_dir,'PNEUMONIA'))
        pneumonia = list(map(lambda x: os.path.join(img_dir, 'PNEUMONIA', x), pneumonia))
        self.img_dirs.extend(pneumonia); self.labels.extend([1]*len(pneumonia))

        self.idxmap = list(range(len(self.img_dirs)))
        random.shuffle(self.idxmap)

        self.transform = transform

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        mapped_idx = self.idxmap[idx]
        img_path = self.img_dirs[mapped_idx]
        image = torchvision.io.read_image(img_path)
        label = self.labels[mapped_idx]
        if self.transform:
            image = self.transform(image)
        if image.shape[0] != 3:
            image = image.repeat(3, 1, 1)
        return image, label

class EarlyStopper:
    def __init__(self, window = 5, tolerance = 0.01):
        self.window = window
        self.tol = tolerance
        self.tally = 0
        self.valprev = None

    def __call__(self, val_loss):
        if self.valprev is None:
            self.valprev = val_loss
        if val_loss > self.valprev and abs(val_loss - self.valprev) > self.tol:
            self.tally += 1
        else:
            self.tally = 0
        
        self.valprev = val_loss

        if self.tally > self.window:
            return True
        else:
            return False
            
def retrieve_head(name, pretrained = True):
    if pretrained:
        if name == 'resnet18':
            return torchvision.models.resnet18(weights='IMAGENET1K_V1')
        elif name == 'resnet34':
            return torchvision.models.resnet34(weights='IMAGENET1K_V1')
        elif name == 'resnet50':
            return torchvision.models.resnet50(weights='IMAGENET1K_V1')
    else:
        if name == 'resnet18':
            return torchvision.models.resnet18()
        elif name == 'resnet34':
            return torchvision.models.resnet34()
        elif name == 'resnet50':
            return torchvision.models.resnet50()
            
def train_loop(dataloader, model, loss_fn, optimizer, gpu='cuda'):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    bs = size // num_batches
    train_loss= 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.float(), y
        if gpu:
            X, y = X.to(gpu), y.to(gpu)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y.float().unsqueeze(1))

        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * bs + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batches
    print(f"Train Error: \n Avg loss: {train_loss:>8f} \n")

    return train_loss

def val_loop(dataloader, model, loss_fn, gpu='cuda'):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.float(), y
            if gpu:
                X, y = X.to(gpu), y.to(gpu)
            pred = model(X)
            val_loss += loss_fn(pred, y.float().unsqueeze(1)).item()
            correct += ((pred.squeeze() > 0.5) == (y == 1)).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Val Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    return val_loss
