"""resnext in pytorch



[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.

    Aggregated Residual Transformations for Deep Neural Networks
    https://arxiv.org/abs/1611.05431
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
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
    #         if cam_img[tt,yyy] < 0.4:
    #             cam_img[tt,yyy] = 0
    #         else:
    #             cam_img[tt, yyy] = 1

    # plt.figure()
    # plt.imshow(cam_img.unsqueeze(-1).cpu().detach().numpy(),cmap='gray')
    # plt.show()
    # print(cam_img.size())
    # resize_all=transforms.Resize((512, 256),interpolation=InterpolationMode.BILINEAR)
    # print(resize_all.size)
    # cam_img = resize_all(cam_img)
    # cam_img = np.float32(255 * cam_img.cpu().detach().numpy())
    # 再将元素更改到　0-255
    # heatmap = cv2.applyColorMap(cv2.resize(cam_img, (256, 512)), cv2.COLORMAP_JET)
    # cam_img = np.uint8(255 * cam_img.cpu().detach().numpy())
    # output_cam.append(cv2.resize(cam_img, size_upsample))
    # heatmap = cv2.applyColorMap(cv2.resize(cam_img, (256, 512)), cv2.COLORMAP_JET)
    # heatmap = cv2.applyColorMap(np.uint8(255 * cam_img.cpu()), cv2.COLORMAP_JET)
    # heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    # b, g, r = heatmap.split(1)
    # heatmap = torch.cat([r, g, b])
    return cam_img
#only implements ResNext bottleneck c


#"""This strategy exposes a new dimension, which we call “cardinality”
#(the size of the set of transformations), as an essential factor
#in addition to the dimensions of depth and width."""
CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64

#"""The grouped convolutional layer in Fig. 3(c) performs 32 groups
#of convolutions whose input and output channels are 4-dimensional.
#The grouped convolutional layer concatenates them as the outputs
#of the layer."""

class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        C = CARDINALITY #How many groups a feature map was splitted into

        #"""We note that the input/output width of the template is fixed as
        #256-d (Fig. 3), We note that the input/output width of the template
        #is fixed as 256-d (Fig. 3), and all widths are dou- bled each time
        #when the feature map is subsampled (see Table 1)."""
        D = int(DEPTH * out_channels / BASEWIDTH) #number of channels per group
        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, C * D, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * D, C * D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False),
            nn.BatchNorm2d(C * D),
            nn.ReLU(inplace=True),
            nn.Conv2d(C * D, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        return F.relu(self.split_transforms(x) + self.shortcut(x))

class ResNext(nn.Module):

    def __init__(self, block, num_blocks, class_names=2):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 128, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 256, 2)
        # self.conv5 = self._make_layer(block, num_blocks[3], 512, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 128, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 2)

    def forward(self, x1,x2,y1,y2,b):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        output1 = self.conv5(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = self.conv4(x2)
        output2 = self.conv5(x2)
        y1 = self.conv1(y1)
        y1 = self.conv2(y1)
        y1 = self.conv3(y1)
        y1 = self.conv4(y1)
        output11 = self.conv5(y1)
        y2 = self.conv1(y2)
        y2 = self.conv2(y2)
        y2 = self.conv3(y2)
        y2 = self.conv4(y2)
        output21 = self.conv5(y2)

        feat1 = output1
        feat2 = output2
        feat11 = output11
        feat21 = output21
        N1, C1, H1, W1 = feat1.shape
        N2, C2, H2, W2 = feat2.shape
        # N1, C1, H1, W1 = output1_1.shape
        # N2, C2, H2, W2 = output2_1.shape
        output1 = self.avg(output1)
        output2 = self.avg(output2)
        output11 = self.avg(output11)
        output21 = self.avg(output21)
        # output = self.maxpool(output)
        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)
        output11 = output11.view(output11.size(0), -1)
        output21 = output21.view(output21.size(0), -1)
        output = torch.cat((output1, output2), dim=1)
        output22 = torch.cat((output11, output21), dim=1)
        output = torch.cat((output, output22), dim=1)
        output = self.classifier(output)
        # self.is_train实例化时已经设置好了，即使后面再设置这个属性，但是在forward中依然是True。可以在forward参数里面进行设置，设置默认参数，测试时再设置。
        params1 = list(self.parameters())
        fc_weights1 = params1[-2].data
        fc_weights2 = params1[-2].data
        fc_weights3 = params1[-2].data
        fc_weights4 = params1[-2].data
        fc_weights1 = np.squeeze(fc_weights1[:, :512].view(2, C1, 1, 1))
        fc_weights2 = np.squeeze(fc_weights2[:, 512:1024].view(2, C2, 1, 1))
        fc_weights3 = np.squeeze(fc_weights3[:, 1024:1536].view(2, C2, 1, 1))
        fc_weights4 = np.squeeze(fc_weights4[:, 1536:].view(2, C2, 1, 1))

        # fc_weights = Variable(fc_weights, requires_grad=False)  # 让这个权重 再次使用 不必更新
        _value1, preds_fianl1 = output.max(1)
        CAM1 = []
        CAM2 = []
        CAM3 = []
        CAM4 = []
        # print(preds_fianl,'fianl')
        # fc_weights=fc_weights[0]
        for i in range(b):
            CAMs1 = returnCAM(feat1[i].unsqueeze(0), fc_weights1, preds_fianl1[i])
            CAM1.append(CAMs1)
        for ii in range(b):
            CAMs2 = returnCAM(feat2[ii].unsqueeze(0), fc_weights2, preds_fianl1[i])
            CAM2.append(CAMs2)
        for iii in range(b):
            CAMs3 = returnCAM(feat11[ii].unsqueeze(0), fc_weights3, preds_fianl1[i])
            CAM3.append(CAMs3)
        for iiii in range(b):
            CAMs4 = returnCAM(feat21[ii].unsqueeze(0), fc_weights4, preds_fianl1[i])
            CAM4.append(CAMs4)


        return output, CAM1, CAM2, CAM3, CAM4

    def _make_layer(self, block, num_block, out_channels, stride):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride

        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

def resnext50():
    """ return a resnext50(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 6, 3])

def resnext101():
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 23, 3])

def resnext152():
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 36, 3])



