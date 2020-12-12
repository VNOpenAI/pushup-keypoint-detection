import torch.nn as nn
from resnest.torch import resnest50
from torchvision.models import shufflenet_v2_x1_0, mobilenet_v2
import torch

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
  def __init__(self, in_c, out_c, filters=3, strides=1, padding=1, norm='mvn', reps=2, use_depthwise=False):
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

class ResNeSt_head(nn.Module):
    def __init__(self, pre_model, use_depthwise = False):
        super(ResNeSt_head, self).__init__()
        self.pre_model = pre_model
        # self.pre_model.layer3 = nn.Identity()
        self.pre_model.layer4 = nn.Identity()
        self.pre_model.avgpool = nn.Identity()
        self.pre_model.fc = nn.Identity()
        self.last_conv = nn.Conv2d(512, 21, (1,1), 1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decode =  nn.Sequential(
                                     conv_block(1536, 512, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                    #  conv_block(256, 512, norm='bn', filters=3, reps=1),
                                    #  conv_block(512, 256, norm='bn', filters=1, reps=1)
                                     )
        self.output = nn.Sigmoid()
    def forward(self, x):
        x = self.pre_model.conv1(x)
        x = self.pre_model.bn1(x)
        x = self.pre_model.relu(x)
        x = self.pre_model.maxpool(x)
        x = self.pre_model.layer1(x)
        x = self.pre_model.layer2(x)
        aa = self.pre_model.layer3(x)
        up1 = self.upsampling(aa)
        conc1 = torch.cat([up1, x], dim = 1)
        # print(conc1.shape)
        x = self.decode(conc1)
        # x = self.pre_model.layer4(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class ResNeSt2_head(nn.Module):
    def __init__(self, pre_model, use_depthwise = False):
        super(ResNeSt2_head, self).__init__()
        self.pre_model = pre_model
        # self.pre_model.layer3 = nn.Identity()
        self.pre_model.layer4 = nn.Identity()
        self.pre_model.avgpool = nn.Identity()
        self.pre_model.fc = nn.Identity()
        self.last_conv = nn.Conv2d(256, 21, (1,1), 1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decode =  nn.Sequential(
                                     conv_block(1536, 512, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                     conv_block(768, 256, norm='bn', filters=3, reps=1, use_depthwise = use_depthwise),
                                    #  conv_block(512, 256, norm='bn', filters=1, reps=1)
                                     )
        self.output = nn.Sigmoid()
    def forward(self, x):
        x = self.pre_model.conv1(x)
        x = self.pre_model.bn1(x)
        x = self.pre_model.relu(x)
        x = self.pre_model.maxpool(x)
        e1 = self.pre_model.layer1(x)
        e2= self.pre_model.layer2(x)
        e3 = self.pre_model.layer3(x)
        up1 = self.upsampling(e3)
        conc1 = torch.cat([up1, e2], dim = 1)
        # print(conc1.shape)
        x = self.decode(conc1)
        conc2 = torch.cat([x, e1], dim=1)
        x = self.decode[1](conc2)
        # # x = self.pre_model.layer4(x)
        x = self.last_conv(x)
        # x = self.pre_model.layer4(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class ShuffleNet_head(nn.Module):
    def __init__(self, pre_model, use_depthwise = False):
        super(ShuffleNet_head, self).__init__()
        self.conv1 = pre_model.conv1
        self.maxpool = pre_model.maxpool
        self.stage2 = pre_model.stage2
        self.stage3 = pre_model.stage3
        self.stage4 = pre_model.stage4
        self.conv5 = pre_model.conv5
        self.last_conv = nn.Conv2d(116, 21, (1,1), 1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decode =  nn.Sequential(
                                     conv_block(696, 232, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                     conv_block(348, 116, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise)
                                     )
        self.output = nn.Sigmoid()
    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.maxpool(e1)
        e3 = self.stage2(e2)
        e4 = self.stage3(e3)
        e5 = self.stage4(e4)
        # x = self.conv5(x)
        x = self.upsampling(e5)
        conc1 = torch.cat([x, e4], dim = 1)
        # print(conc1.shape)
        x = self.decode[0](conc1)
        x = self.upsampling(x)
        conc2 = torch.cat([x, e3], dim=1)
        x = self.decode[1](conc2)
        # # x = self.pre_model.layer4(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class ShuffleNet2_head(nn.Module):
    def __init__(self, pre_model, use_depthwise = False):
        super(ShuffleNet2_head, self).__init__()
        self.conv1 = pre_model.conv1
        self.maxpool = pre_model.maxpool
        self.stage2 = pre_model.stage2
        self.stage3 = pre_model.stage3
        self.stage4 = pre_model.stage4
        self.conv5 = pre_model.conv5
        self.last_conv = nn.Conv2d(70, 21, (1,1), 1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decode =  nn.Sequential(
                                     conv_block(696, 232, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                     conv_block(348, 116, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                     conv_block(140, 70, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                    #  conv_block(70, 70, norm='bn', filters=3, reps=2)
                                     )
        self.output = nn.Sigmoid()
    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.maxpool(e1)
        e3 = self.stage2(e2)
        e4 = self.stage3(e3)
        e5 = self.stage4(e4)
        # x = self.conv5(x)
        x = self.upsampling(e5)
        conc1 = torch.cat([x, e4], dim = 1)
        # print(conc1.shape)
        x = self.decode[0](conc1)
        x = self.upsampling(x)
        conc2 = torch.cat([x, e3], dim=1)
        x = self.decode[1](conc2)
        x = self.upsampling(x)
        conc3 = torch.cat([x, e2], dim=1)
        x = self.decode[2](conc3)
        # x = self.upsampling(x)
        # x = self.decode[3](x)
        # # x = self.pre_model.layer4(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class MobileNet_head(nn.Module):
    def __init__(self, pre_model, use_depthwise = False):
        super(MobileNet_head, self).__init__()
        self.eblock_1 = pre_model.features[:2]
        self.eblock_2 = pre_model.features[2:4]
        self.eblock_3 = pre_model.features[4:7]
        self.eblock_4 = pre_model.features[7:14]
        self.eblock_5 = pre_model.features[14:-1]
        self.last_conv = nn.Conv2d(64, 21, (1,1), 1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decode =  nn.Sequential(
                                     conv_block(416, 128, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                     conv_block(160, 64, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise)
                                     )
        self.output = nn.Sigmoid()
    def forward(self, x):
        e1 = self.eblock_1(x)
        e2 = self.eblock_2(e1)
        e3 = self.eblock_3(e2)
        e4 = self.eblock_4(e3)
        e5 = self.eblock_5(e4)
        x = self.upsampling(e5)
        conc1 = torch.cat([x, e4], dim = 1)
        x = self.decode[0](conc1)
        x = self.upsampling(x)
        conc2 = torch.cat([x, e3], dim=1)
        x = self.decode[1](conc2)
        # # x = self.pre_model.layer4(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

class MobileNet2_head(nn.Module):
    def __init__(self, pre_model, use_depthwise = False):
        super(MobileNet2_head, self).__init__()
        self.eblock_1 = pre_model.features[:2]
        self.eblock_2 = pre_model.features[2:4]
        self.eblock_3 = pre_model.features[4:7]
        self.eblock_4 = pre_model.features[7:14]
        self.eblock_5 = pre_model.features[14:-1]
        self.last_conv = nn.Conv2d(32, 21, (1,1), 1)
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decode =  nn.Sequential(
                                     conv_block(416, 128, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                     conv_block(160, 64, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise),
                                     conv_block(88, 32, norm='bn', filters=3, reps=2, use_depthwise = use_depthwise)
                                     )
        self.output = nn.Sigmoid()
    def forward(self, x):
        e1 = self.eblock_1(x)
        e2 = self.eblock_2(e1)
        e3 = self.eblock_3(e2)
        e4 = self.eblock_4(e3)
        e5 = self.eblock_5(e4)
        x = self.upsampling(e5)
        conc1 = torch.cat([x, e4], dim = 1)
        x = self.decode[0](conc1)
        x = self.upsampling(x)
        conc2 = torch.cat([x, e3], dim=1)
        x = self.decode[1](conc2)
        x = self.upsampling(x)
        conc3 = torch.cat([x, e2], dim=1)
        x = self.decode[2](conc3)
        # # x = self.pre_model.layer4(x)
        x = self.last_conv(x)
        x = self.output(x)
        return x

def build_model(model_name, use_depthwise=False, pretrained=False):
  if model_name == 'efficient':
    pre_model = EfficientNet.from_pretrained('efficientnet-b2')
    for param in pre_model.parameters():
        param.requires_grad = True
    model = Efficient_head(pre_model)
    return model
  elif model_name == 'resnest':
    pre_model = resnest50(pretrained=pretrained)
    for param in pre_model.parameters():
        param.requires_grad = True
    model = ResNeSt_head(pre_model)
    return model
  elif model_name == 'mobile':
    pre_model = mobilenet_v2(pretrained=pretrained)
    for param in pre_model.parameters():
        param.requires_grad = True
    model = MobileNet_head(pre_model, use_depthwise = use_depthwise)
    return model
  elif model_name == 'mobile2':
    pre_model = mobilenet_v2(pretrained=pretrained)
    for param in pre_model.parameters():
        param.requires_grad = True
    model = MobileNet2_head(pre_model, use_depthwise = use_depthwise)
    return model
  elif model_name == 'shuffle':
    pre_model = shufflenet_v2_x1_0(pretrained=pretrained)
    for param in pre_model.parameters():
        param.requires_grad = True
    model = ShuffleNet_head(pre_model, use_depthwise = use_depthwise)
    return model
  elif model_name == 'shuffle2':
    pre_model = shufflenet_v2_x1_0(pretrained=pretrained)
    for param in pre_model.parameters():
        param.requires_grad = True
    model = ShuffleNet2_head(pre_model, use_depthwise = use_depthwise)
    return model
  else:
    print('Not support this model!')
