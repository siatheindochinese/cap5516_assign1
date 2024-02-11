import torch.nn as nn

class CustomResNet(nn.Module):
    def __init__(self, head):
        super(CustomResNet, self).__init__()
        out_len = list(head.children())[-1].weight.shape[1]
        sliced = list(head.children())[:-1]
        self.cnn = nn.Sequential(*sliced)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(out_len,256), nn.ReLU(), nn.Linear(256,1), nn.Sigmoid())

    def forward(self, x):
        x = self.cnn(x).squeeze()
        x = self.mlp(x)
        return x
