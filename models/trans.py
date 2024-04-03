import numpy as np
import torch
import torch.nn as nn
# from torchsummaryX import summary
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}
def returnCAM(feature_conv, weight_softmax, class_idx):
    # 类激活图上采样到 256 x 256
    size_upsample = (512, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    # 将权重赋给卷积层：这里的weigh_softmax.shape为(1000, 512)
    # 				feature_conv.shape为(1, 512, 13, 13)
    # weight_softmax[class_idx]由于只选择了一个类别的权重，所以为(1, 512)
    # feature_conv.reshape((nc, h * w))后feature_conv.shape为(512, 169)
    a=weight_softmax[class_idx].unsqueeze(0)
    b=feature_conv.reshape((nc, h * w))
    cam = torch.mm(a, b)
    # print(cam.shape)		# 矩阵乘法之后，为各个特征通道赋值。输出shape为（1，169）
    cam = cam.reshape(h, w) # 得到单张特征图
    # 特征图上所有元素归一化到 0-1
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    # for tt in range(16):
    #     for yyy in range(8):
    #         if cam_img[tt,yyy]<0.3:
    #             cam_img[tt,yyy]=0
    #         else:
    #             cam_img[tt, yyy] = 1

    return cam_img


class VGG_trans(nn.Module):

    def __init__(self, features, num_class=2):
        super(VGG_trans,self).__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_token = nn.Parameter(torch.zeros(4,512, 1))
        self.classifier = nn.Linear(512, num_class)
    def forward(self, x1,x2,b):
      output1 = self.features(x1)
      output2 = self.features(x2)
      # y1=self.convad2d(y1)
      # y2= self.convad2d(y2)
      feat1 = output1
      feat2 = output2
      N1, C1, H1, W1 = feat1.shape
      N2, C2, H2, W2 = feat2.shape
      # output1 = self.avgpool(output1)
      # output2 = self.avgpool(output2)
      output1 = output1.view(output1.size(0),output1.size(1),-1)
      output2 = output2.view(output2.size(0),output2.size(1), -1)
      output1 = torch.cat((output1, self.cls_token), dim=-1)
      output2 = torch.cat((output2, self.cls_token), dim=-1)



      # output = self.maxpool(output)

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
      # print(preds_fianl,'fianl')
      # fc_weights=fc_weights[0]
      for i in range(b):
          CAMs1 = returnCAM(feat1[i].unsqueeze(0), fc_weights1, preds_fianl1[i])
          CAM1.append(CAMs1)
      for ii in range(b):
          CAMs2 = returnCAM(feat2[ii].unsqueeze(0), fc_weights2, preds_fianl1[i])
          CAM2.append(CAMs2)

      #
      # CAMs2 = returnCAM(feat[1].unsqueeze(0), fc_weights, preds_fianl[1])
      # print(preds_fianl)
      # attention
      # feat = feat.unsqueeze(1)  # N * 1 * C * H * W
      # hm = feat * fc_weights  # [n, 1,c, h,w] * [1, num_labels,c, 1,1]
      # hm = hm.sum(1)  # N * self.num_labels * H * W
      #
      # heatmap = hm
      return output,CAM1,CAM2

def make_layers_trans(cfg, batch_norm=False):
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
def trans():
    return VGG_trans(make_layers_trans(cfg['D'], batch_norm=True))