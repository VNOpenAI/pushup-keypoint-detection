import torch
import torch.nn as nn
from torchvision.models import densenet169, resnet50
from torchvision.models import shufflenet_v2_x1_0, mobilenet_v2, shufflenet_v2_x1_5
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from resnest.torch import resnest50
import numpy as np

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class conv_block(nn.Module):
  def __init__(self, in_c, out_c, filters=3, strides=1, padding=1, norm='mvn', reps=2, use_depthwise=True):
    super(conv_block, self).__init__()
    self.in_c = in_c
    self.out_c = out_c
    self.convs = nn.ModuleList()
    in_conv = self.in_c
    for i in range(reps):
      if use_depthwise:
        self.convs.append(depthwise_separable_conv(in_conv, self.out_c))
      else:
        self.convs.append(nn.Conv2d(in_conv, self.out_c, filters, strides, padding=padding))
      if norm == 'mvn':
        self.convs.append(MVN())
      elif norm == 'bn':
        self.convs.append(nn.BatchNorm2d(self.out_c))
      elif norm == 'mvn+bn':
        self.convs.append(MVN())
        self.convs.append(nn.BatchNorm2d(self.out_c))
      self.convs.append(nn.ReLU(inplace=True))
      in_conv = self.out_c
  def forward(self, x):
    for layer in self.convs:
      x = layer(x)
    return x

class Efficient_encode(nn.Module):
    def __init__(self, pre_model, n_kps=7):
        super(Efficient_encode, self).__init__()
        # self.pre_model = pre_model
        self._conv_stem = pre_model._conv_stem
        self._bn0 = pre_model._bn0
        self._blocks = pre_model._blocks[:16]
        self.last_conv = nn.Conv2d(120, 3*n_kps, (1,1), 1)
        self.output = nn.Sigmoid()
    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._blocks(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class ResNeSt_encode(nn.Module):
    def __init__(self, pre_model, n_kps=7):
        super(ResNeSt_encode, self).__init__()
        # self.pre_model = pre_model
        self.conv1 = pre_model.conv1
        self.bn1 = pre_model.bn1
        self.relu = pre_model.relu
        self.maxpool = pre_model.maxpool
        self.layer1 = pre_model.layer1
        self.layer2 = pre_model.layer2
        self.layer3 = pre_model.layer3
        self.last_conv = nn.Conv2d(1024, 3*n_kps, (1,1), 1)
        self.output = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class ShuffleNet_deconv(nn.Module):
    def __init__(self, pre_model, n_deconvs=2, use_depthwise = False):
        super(ShuffleNet_deconv, self).__init__()
        self.conv1 = pre_model.conv1
        self.maxpool = pre_model.maxpool
        self.stage2 = pre_model.stage2
        self.stage3 = pre_model.stage3
        self.stage4 = pre_model.stage4
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if n_deconvs == 2:
            self.decode =  nn.Sequential(
                                        conv_block(696, 232, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                        conv_block(348, 116, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise)
                                        )
        elif n_deconvs == 3:
            self.decode =  nn.Sequential(
                                        conv_block(696, 232, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                        conv_block(348, 116, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                        conv_block(140, 70, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                        )
        else:
            raise Exception("Not support this number of deconv blocks!!!!")
        self.last_conv = nn.Conv2d(self.decode[-1].out_channels, 21, (1,1), 1)
        self.output = nn.Sigmoid()
    def forward(self, x):
        e = [x]
        e.append(self.conv1(x))
        e.append(self.maxpool(e[-1]))
        e.append(self.stage2(e[-1]))
        e.append(self.stage3(e[-1]))
        e.append(self.stage4(e[-1]))
        for i, block in enumerate(self.decode):
            x = self.upsampling(e[-i-1])
            conc = torch.cat([x, e[-i-2]], dim = 1)
            x = block(conc)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class MobileNet_deconv(nn.Module):
    def __init__(self, pre_model, n_deconvs=2, use_depthwise = False):
        super(MobileNet_deconv, self).__init__()
        self.eblock_1 = pre_model.features[:2]
        self.eblock_2 = pre_model.features[2:4]
        self.eblock_3 = pre_model.features[4:7]
        self.eblock_4 = pre_model.features[7:14]
        self.eblock_5 = pre_model.features[14:-1]
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if n_deconvs == 2:
            self.decode =  nn.Sequential(
                                        conv_block(416, 128, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                        conv_block(160, 64, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise)
                                        )
        elif n_deconvs == 3:
            self.decode =  nn.Sequential(
                                        conv_block(416, 128, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                        conv_block(160, 64, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                        conv_block(88, 32, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise)
                                        )
        else:
            raise Exception("Not support this number of deconv blocks!!!!")
        self.last_conv = nn.Conv2d(self.decode[-1].out_c, 21, (1,1), 1)
        self.output = nn.Sigmoid()
    def forward(self, x):
        e = [x]
        e.append(self.eblock_1(e[-1]))
        e.append(self.eblock_2(e[-1]))
        e.append(self.eblock_3(e[-1]))
        e.append(self.eblock_4(e[-1]))
        e.append(self.eblock_5(e[-1]))
        for i, block in enumerate(self.decode):
            x = self.upsampling(e[-i-1])
            conc = torch.cat([x, e[-i-2]], dim = 1)
            x = block(conc)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class ResNeSt_deconv(nn.Module):
    def __init__(self, pre_model, n_deconvs=2, use_depthwise = False):
        super(ResNeSt_deconv, self).__init__()
        self.encoder = nn.Sequential(
                                    nn.Sequential(pre_model.conv1, pre_model.bn1, pre_model.relu),
                                    pre_model.maxpool, pre_model.layer1, pre_model.layer2, pre_model.layer3
                                    )
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if n_deconvs == 2:
            self.decode =  nn.Sequential(
                                    conv_block(1536, 512, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                    )
        elif n_deconvs == 3:
            self.decode =  nn.Sequential(
                                    conv_block(1536, 512, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                    conv_block(768, 256, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                    )
        else:
            raise Exception("Not support this number of deconv blocks!!!!")
        self.last_conv = nn.Conv2d(self.decode[-1].out_c, 21, (1,1), 1)
        self.output = nn.Sigmoid()

    def forward(self, x):
        e = [x]
        for eblock in self.encoder:
            e.append(eblock(eblock(e[-1])))
        
        for i, deblock in enumerate(self.decode):
            x = self.upsampling(e[-i-1])
            conc = torch.cat([x, e[-i-2]], dim = 1)
            x = deblock(conc)

        x = self.last_conv(x)
        x = self.output(x)

        return x

def build_detection_based_model(model_name, n_kps=7, pretrained=True, n_deconvs=2, use_depthwise=True):
    if model_name == 'efficient_encode':
        pre_model = EfficientNet.from_pretrained('efficientnet-b2')
        for param in pre_model.parameters():
            param.requires_grad = True
        model = Efficient_encode(pre_model, n_kps)
        return model
    elif model_name == 'resnest_encode':
        pre_model = resnest50(pretrained=pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        model = ResNeSt_encode(pre_model, n_kps)
        return model
    elif model_name == 'mobile_deconv':
        pre_model = mobilenet_v2(pretrained=pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        model = MobileNet_deconv(pre_model, n_deconvs=n_deconvs, use_depthwise = use_depthwise)
        return model
    elif model_name == 'shuffle_deconv':
        pre_model = shufflenet_v2_x1_0(pretrained=pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        model = ShuffleNet_deconv(pre_model, n_deconvs=n_deconvs, use_depthwise = use_depthwise)
        return model
    elif model_name == 'resnest_deconv':
        pre_model = resnest50(pretrained=pretrained)
        for param in pre_model.parameters():
            param.requires_grad = True
        model = ResNeSt_deconv(pre_model, n_deconvs=n_deconvs, use_depthwise = use_depthwise)
        return model
    else:
        print('Not support this model!')

def build_regression_based_model(model_name, n_kps=7, pretrained=True):
    if model_name == 'efficient':
        model = EfficientNet.from_pretrained('efficientnet-b2')
        for param in model.parameters():
            param.requires_grad = True
        in_feature = model._fc.in_features
        model._fc = nn.Sequential(nn.Linear(in_feature, 2*n_kps, bias=True), nn.Sigmoid())
        return model 
    elif model_name == 'resnest':
        model = resnest50(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = True
        in_feature = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(in_feature, 2*n_kps, bias=True), nn.Sigmoid())
        return model
    else:
        print('Not support this model!')

def preprocessed_img_test(img, img_size):
    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(img_size[:2]),
                                transforms.ToTensor()
    ])
    oh, ow = img.shape[:2]
    if oh > ow:
        new_img = np.zeros((oh, oh, 3), np.uint8)
        cl = ((oh-ow)//2)
        new_img[:,cl:cl+ow] = img
        clx = cl
        cly = 0
    else:
        new_img = np.zeros((ow, ow, 3), np.uint8)
        cl = ((ow-oh)//2)
        new_img[cl:cl+oh,:] = img
        clx = 0
        cly = cl
    new_img = trans(new_img)
    new_img = torch.unsqueeze(new_img, 0)
    return new_img, max([oh, ow]), clx, cly

def heatmap2coor(hp_preds, n_kps = 7, img_size=(225,225), stride=15):
    heatmaps = hp_preds[:,:n_kps]
    flatten_hm = heatmaps.reshape((heatmaps.shape[0], n_kps, -1))
    flat_vectx = hp_preds[:,n_kps:2*n_kps].reshape((heatmaps.shape[0], n_kps, -1))
    flat_vecty = hp_preds[:,2*n_kps:].reshape((heatmaps.shape[0], n_kps, -1))
    flat_max = torch.argmax(flatten_hm, dim=-1)
    max_mask = flatten_hm == torch.unsqueeze(torch.max(flatten_hm, dim=-1)[0], dim=-1)
    cxs = flat_max%(heatmaps.shape[-2])
    cys = flat_max//(heatmaps.shape[-2])
    ovxs = torch.sum(flat_vectx*max_mask, dim=-1)*stride
    ovys = torch.sum(flat_vecty*max_mask, dim=-1)*stride
    xs_p = (cxs*stride+ovxs)/img_size[1]
    ys_p = (cys*stride+ovys)/img_size[0]
    hp_preds = torch.stack([xs_p, ys_p], dim=-1)
    return hp_preds