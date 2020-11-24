import torch, time, os, shutil
import torch.nn as nn
from torchvision.models import densenet169, resnet50
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest50
import numpy as np

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
    def __init__(self, pb_type='detection', model_name='resnest', n_kps=7):
        self.pb_type = pb_type
        self.model_name = model_name
        self.n_kps = n_kps
        if pb_type == 'detection':
            self.model = build_detection_based_model(model_name, n_kps)
        elif pb_type == 'regression':
            self.model = build_regression_based_model(model_name, n_kps)
        else:
            raise Exception("not support this pb_type!!!")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def train(self, loader_dict, loss_func, optimizer, lr=3e-4, use_lr_sch=False, epochs=120, ckp_dir='./checkpoint'):
        criterion = loss_func
        optimizer = optimizer(self.model.parameters(),lr)
        if 'train' not in list(loader_dict.keys()):
            raise Exception("missing \'train\' keys in loader_dict!!!")
        if use_lr_sch:
            lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 80, 1/3)
        else:
            lr_sch = None
        best_loss = 80.0
        if os.path.exists(ckp_dir):
            shutil.rmtree(ckp_dir)
        os.mkdir(ckp_dir)
        modes = list(loader_dict.keys())
        for epoch in range(epochs):
            s="Epoch [{}/{}]:".format(epoch+1, epochs)
            start = time.time()
            for mode in modes:
                running_loss = 0.0
                ova_len = loader_dict[mode].dataset.n_data
                if mode == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                for i, data in enumerate(loader_dict[mode]):
                    imgs, labels = data[0].to(self.device), data[1].to(self.device)
                    preds = self.model(imgs)
                    loss = criterion(preds, labels)
                    if mode == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
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
                torch.save(self.model.state_dict(), os.path.join(ckp_dir,'epoch'+str(epoch+1)+'.pt'))
                print('new checkpoint saved!')
            if lr_sch is not None:
                lr_sch.step()
                print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))
    def load_ckp(self, ckp_path):
        checkpoint=torch.load(ckp_path)
        self.model.load_state_dict(checkpoint)
    def evaluate(self, loader):
        self.model.eval()
        with torch.no_grad() as tng:
            ova_len = loader.dataset.n_data
            for i, data in enumerate(loader):
                imgs, targets = data[0].to(self.device), data[1].to(self.device)
                preds = self.model(imgs)
                preds = preds.cpu().numpy()
                targets = targets.cpu().numpy()
                if self.pb_type == 'regression':
                    err = np.abs(preds-targets)
                    ova_loss += np.sum(err)
                else:
                    heatmaps = preds[:,:self.n_kps]
                    flatten_hm = heatmaps.reshape((heatmaps.shape[0], self.n_kps, -1))
                    flat_vectx = preds[:,self.n_kps:2*self.n_kps].reshape((heatmaps.shape[0], self.n_kps, -1))
                    flat_vecty = preds[:,2*self.n_kps:].reshape((heatmaps.shape[0], self.n_kps, -1))
                    flat_max = np.argmax(flatten_hm, axis=-1)
                    max_mask = flatten_hm == np.expand_dims(np.max(flatten_hm, axis=-1), axis=-1)
                    cxs = flat_max%heatmaps.shape[-2]
                    cys = flat_max//heatmaps.shape[-2]
                    ovxs = np.sum(flat_vectx*max_mask, axis=-1)
                    ovys = np.sum(flat_vectx*max_mask, axis=-1)
                    xs_p = (cxs*15+ovxs)/heatmaps.shape[-1]
                    ys_p = (cys*15+ovys)/heatmaps.shape[-2]

                    heatmaps = targets[:,:self.n_kps]
                    flatten_hm = heatmaps.reshape((heatmaps.shape[0], self.n_kps, -1))
                    flat_vectx = preds[:,self.n_kps:2*self.n_kps].reshape((heatmaps.shape[0], self.n_kps, -1))
                    flat_vecty = preds[:,2*self.n_kps:].reshape((heatmaps.shape[0], self.n_kps, -1))
                    flat_max = np.argmax(flatten_hm, axis=-1)
                    max_mask = flatten_hm == np.expand_dims(np.max(flatten_hm, axis=-1), axis=-1)
                    cxs = flat_max%heatmaps.shape[-2]
                    cys = flat_max//heatmaps.shape[-2]
                    ovxs = np.sum(flat_vectx*max_mask, axis=-1)
                    ovys = np.sum(flat_vectx*max_mask, axis=-1)
                    xs_t = (cxs*15+ovxs)/heatmaps.shape[-1]
                    ys_t = (cys*15+ovys)/heatmaps.shape[-2]

                    ova_loss = np.sum(np.abs(xs_t-xs_p) + np.abs(ys_t-ys_p))
        return ova_loss/(ova_len*14)