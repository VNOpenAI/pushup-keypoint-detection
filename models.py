import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet169, resnet50
import torchvision.transforms.functional as functional
import torch.nn.functional as F

class Efficient_head(nn.Module):
    def __init__(self, pre_model, n_kps=7):
        super(Efficient_head, self).__init__()
        self.pre_model = pre_model
        self.last_conv = nn.Conv2d(120, 3*n_kps, (1,1), 1)
        self.output = nn.Sigmoid()
    def forward(self, x):
        x = self.pre_model._conv_stem(x)
        x = self.pre_model._bn0(x)
        for block in self.pre_model._blocks[:16]:
          x = block(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class ResNeSt_head(nn.Module):
    def __init__(self, pre_model, n_kps=7):
        super(ResNeSt_head, self).__init__()
        self.pre_model = pre_model
        self.last_conv = nn.Conv2d(1024, 3*n_kps, (1,1), 1)
        self.output = nn.Sigmoid()
    def forward(self, x):
        x = self.pre_model.conv1(x)
        x = self.pre_model.bn1(x)
        x = self.pre_model.relu(x)
        x = self.pre_model.maxpool(x)
        x = self.pre_model.layer1(x)
        x = self.pre_model.layer2(x)
        x = self.pre_model.layer3(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

def build_detection_based_model(model_name, n_kps=7):
  if model_name == 'efficient':
    pre_model = EfficientNet.from_pretrained('efficientnet-b2')
    for param in pre_model.parameters():
        param.requires_grad = True
    model = Efficient_head(pre_model, n_kps)
    return model
  elif model_name == 'resnest':
    pre_model = resnest50(pretrained=True)
    for param in pre_model.parameters():
        param.requires_grad = True
    model = ResNeSt_head(pre_model, n_kps)
    return model
  else:
    print('Not support this model!')

def build_regression_based_model(model_name, n_kps=7):
    if model_name == 'efficient'
        model = EfficientNet.from_pretrained('efficientnet-b2')
        for param in model.parameters():
            param.requires_grad = True
        in_feature = model._fc.in_features
        model._fc = nn.Sequential(nn.Linear(in_feature, 2*n_kps, bias=True), nn.Sigmoid())
        return model 
    elif model_name == 'resnest'
        model = resnest50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        in_feature = model._fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_feature, 2*n_kps, bias=True), nn.Sigmoid())
        return model
    else:
        print('Not support this model!')