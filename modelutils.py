import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, in_c, out_c, kernel_size=3, padding=1):
    super(ConvBlock, self).__init__()
    self.net = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
        nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_c),
        nn.Dropout2d()
    )
    if in_c != out_c:
      self.skip = nn.Conv2d(in_c, out_c, kernel_size=1)
    else:
      self.skip = nn.Identity()
    self.final = nn.ReLU()
  def forward(self, x):
    res = self.net(x)
    y = self.skip(x) + res
    return self.final(y)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.net = nn.Sequential(
        ConvBlock(1, 8),
        ConvBlock(8, 16),
        nn.MaxPool2d((1, 2)),
        ConvBlock(16, 32),
        ConvBlock(32, 64),
        nn.MaxPool2d((1, 2)),
        ConvBlock(64, 64),
        ConvBlock(64, 128),
        nn.MaxPool2d((1, 2)),
        ConvBlock(128, 128),
        nn.AvgPool2d((7, 33))
    )
    self.final = nn.Linear(128, 88)
  def forward(self, x):
    y = self.net(x)
    y = y.squeeze(2).squeeze(2)
    return self.final(y)

def load_model(path):
    return torch.load(path, map_location=torch.device('cpu'))