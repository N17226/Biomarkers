
import PIL
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torchvision.transforms import transforms, InterpolationMode
import xlsxwriter
import h5py
import numpy as np
num = 0

import torch
import torch.nn as nn
# from torchsummaryX import summary
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
from torch.autograd import Variable  # 输入必须是tensor，因此肯定是requires_grad = False, 不必删除


def returnCAM(feature_conv, weight_softmax, class_idx):
    # 类激活图上采样到 256 x 256
    size_upsample = (512, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    a=weight_softmax[class_idx].unsqueeze(0)
    b=feature_conv.reshape((nc, h * w))
    cam = torch.mm(a, b)
    # print(cam.shape)		#
    cam = cam.reshape(h, w) #
    # 特征图上所有元素归一化到 0-1
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    return cam_img

class VGGtr(nn.Module):

    def __init__(self, features, num_class=2):
        super(VGGtr,self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_class)


    def forward(self, x1,x2,b):
      output1 = self.features(x1)
      output2 = self.features(x2)

      feat1 = output1
      feat2 = output2
      N1, C1, H1, W1 = feat1.shape
      N2, C2, H2, W2 = feat2.shape
      output1 = self.avgpool(output1)
      output2 = self.avgpool(output2)
      # output = self.maxpool(output)
      output1 = output1.view(output1.size(0), -1)
      output2 = output2.view(output2.size(0), -1)
      output=torch.cat((output1, output2), dim=1)
      output = self.classifier(output)
    # self.is_train实例化时已经设置好了，即使后面再设置这个属性，但是在forward中依然是True。可以在forward参数里面进行设置，设置默认参数，测试时再设置。
      params1 = list(self.parameters())
      fc_weights1 = params1[-2].data
      fc_weights2 = params1[-2].data
      fc_weights1 = np.squeeze(fc_weights1[:,:512].view(2, C1, 1, 1))
      fc_weights2 = np.squeeze(fc_weights2[:, 512:].view(2, C2, 1, 1))
      # fc_weights = Variable(fc_weights, requires_grad=False)  # 让这个权重 再次使用 不必更新
      _value1, preds_fianl1 = output.max(1)
      CAM1=[]
      CAM2 = []

      for i in range(b):
          CAMs1 = returnCAM(feat1[i].unsqueeze(0), fc_weights1, preds_fianl1[i])
          CAM1.append(CAMs1)
      for ii in range(b):
          CAMs2 = returnCAM(feat2[ii].unsqueeze(0), fc_weights2, preds_fianl1[i])
          CAM2.append(CAMs2)

      return output,CAM1,CAM2


def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel =2
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # layers += [SELayer(input_channel)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)


def vgg16_bn():
    return VGGtr(make_layers(cfg['D'], batch_norm=True))


