import torch, time, os
import torch.nn as nn
from torchvision.models import densenet169, resnet50
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest50

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
    if model_name == 'efficient':
        model = EfficientNet.from_pretrained('efficientnet-b2')
        for param in model.parameters():
            param.requires_grad = True
        in_feature = model._fc.in_features
        model._fc = nn.Sequential(nn.Linear(in_feature, 2*n_kps, bias=True), nn.Sigmoid())
        return model 
    elif model_name == 'resnest':
        model = resnest50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        in_feature = model._fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_feature, 2*n_kps, bias=True), nn.Sigmoid())
        return model
    else:
        print('Not support this model!')

class SHPE_model():
    def __init__(self, pb_type, model_name, loss_func, optimizer, loader_dict, n_kps=7, lr_sch=None, epochs=120, ckp_dir='./checkpoint'):
        if pb_type == 'detection':
            self.model = build_detection_based_model(model_name, n_kps)
        elif pb_type == 'regression':
            self.model = build_regression_based_model(model_name, n_kps)
        else:
            raise Exception("not support this pb_type!!!")
        self.criterion = loss_func
        self.optimizer = optimizer
        if 'train' not in list(loader_dict.keys()):
            raise Exception("missing \'train\' keys in loader_dict!!!")
        self.loader_dict = loader_dict
        self.lr_sch = lr_sch
        self.epochs = epochs
        self.ckp_dir = ckp_dir
    def train(self):
        best_loss = 80.0
        if os.path.exists(self.ckp_dir):
            shutil.rmtree(self.ckp_dir)
        os.mkdir(self.ckp_dir)
        modes = list(self.loader_dict.keys())
        for epoch in range(self.epochs):
            s="Epoch [{}/{}]:".format(epoch+1, self.epochs)
            start = time.time()
            for mode in modes:
                running_loss = 0.0
                ova_len = self.loader_dict[mode].dataset.n_data
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for i, data in enumerate(self.loader_dict[mode]):
                    imgs, labels = data[0].to(device), data[1].to(device)
                    preds = self.model(imgs)
                    loss = self.criterion(preds, labels)
                    if mode == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    iter_len = imgs.size()[0]
                    preds = (preds > 0.5).float()
                    running_loss += loss.item()*iter_len
                running_loss /= ova_len
                s += "{}_loss {:.3f} -".format(mode, running_loss)
            end = time.time()
            s = s[:-1] + "({:.1f}s)".format(end-start)
            print(s)
            if running_loss < best_loss or (epoch+1)%10==0:
                best_loss = running_loss
                torch.save(self.model.state_dict(), os.path.join(=self.checkpoint_dir,'epoch'+str(epoch+1)+'.pt'))
                print('new checkpoint saved!')
            if self.lr_sch is not None:
                self.lr_sch.step()
                print('current lr: {:.4f}'.format(self.lr_sch.get_lr()[0]))