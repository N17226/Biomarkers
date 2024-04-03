"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

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
class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=2):
        super().__init__()
        self.prelayer = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout2d(p=0.4)
        self.classifier = nn.Linear(4096, num_class)

    def forward(self, x1,x2,y1,y2,b):
        x1 = self.prelayer(x1)
        x1 = self.maxpool(x1)
        x1 = self.a3(x1)
        x1 = self.b3(x1)
        x1 = self.maxpool(x1)
        x1 = self.a4(x1)
        x1 = self.b4(x1)
        x1 = self.c4(x1)
        x1 = self.d4(x1)
        x1 = self.e4(x1)
        x1 = self.maxpool(x1)
        x1 = self.a5(x1)
        output1 = self.b5(x1)

        x2 = self.prelayer(x2)
        x2 = self.maxpool(x2)
        x2 = self.a3(x2)
        x2 = self.b3(x2)
        x2 = self.maxpool(x2)
        x2 = self.a4(x2)
        x2 = self.b4(x2)
        x2 = self.c4(x2)
        x2 = self.d4(x2)
        x2 = self.e4(x2)
        x2 = self.maxpool(x2)
        x2 = self.a5(x2)
        output2 = self.b5(x2)

        y1 = self.prelayer(y1)
        y1 = self.maxpool(y1)
        y1 = self.a3(y1)
        y1 = self.b3(y1)
        y1 = self.maxpool(y1)
        y1 = self.a4(y1)
        y1 = self.b4(y1)
        y1 = self.c4(y1)
        y1 = self.d4(y1)
        y1 = self.e4(y1)
        y1 = self.maxpool(y1)
        y1 = self.a5(y1)
        output11 = self.b5(y1)

        y2 = self.prelayer(y2)
        y2 = self.maxpool(y2)
        y2 = self.a3(y2)
        y2 = self.b3(y2)
        y2 = self.maxpool(y2)
        y2 = self.a4(y2)
        y2 = self.b4(y2)
        y2 = self.c4(y2)
        y2 = self.d4(y2)
        y2 = self.e4(y2)
        y2 = self.maxpool(y2)
        y2 = self.a5(y2)
        output21 = self.b5(y2)


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
        # fc_weights1 = np.squeeze(fc_weights1[:, :512].view(2, C1, 1, 1))
        # fc_weights2 = np.squeeze(fc_weights2[:, 512:1024].view(2, C2, 1, 1))
        # fc_weights3 = np.squeeze(fc_weights3[:, 1024:1536].view(2, C2, 1, 1))
        # fc_weights4 = np.squeeze(fc_weights4[:, 1536:].view(2, C2, 1, 1))
        fc_weights1 = np.squeeze(fc_weights1[:, :1024].view(2, C1, 1, 1))
        fc_weights2 = np.squeeze(fc_weights2[:, 1024:2048].view(2, C2, 1, 1))
        fc_weights3 = np.squeeze(fc_weights3[:, 2048:3072].view(2, C2, 1, 1))
        fc_weights4 = np.squeeze(fc_weights4[:, 3072:].view(2, C2, 1, 1))

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
        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        # x = self.avgpool(x)
        # x = self.dropout(x)
        # x = x.view(x.size()[0], -1)
        # x = self.linear(x)

        # return x

def googlenet():
    return GoogleNet()


