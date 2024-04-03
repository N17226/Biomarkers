""" inceptionv3 in pytorch


[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna

    Rethinking the Inception Architecture for Computer Vision
    https://arxiv.org/abs/1512.00567v3
"""

import torch
import torch.nn as nn
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

class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

#same naive inception module
class InceptionA(nn.Module):

    def __init__(self, input_channels, pool_features):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BasicConv2d(input_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1x1 -> 5x5(same)
        branch5x5 = self.branch5x5(x)
        #branch5x5 = self.branch5x5_2(branch5x5)

        #x -> 1x1 -> 3x3 -> 3x3(same)
        branch3x3 = self.branch3x3(x)

        #x -> pool -> 1x1(same)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch5x5, branch3x3, branchpool]

        return torch.cat(outputs, 1)

#downsample
#Factorization into smaller convolutions
class InceptionB(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = BasicConv2d(input_channels, 384, kernel_size=3, stride=2)

        self.branch3x3stack = nn.Sequential(
            BasicConv2d(input_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2)
        )

        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x - > 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 3x3 -> 3x3(downsample)
        branch3x3stack = self.branch3x3stack(x)

        #x -> avgpool(downsample)
        branchpool = self.branchpool(x)

        #"""We can use two parallel stride 2 blocks: P and C. P is a pooling
        #layer (either average or maximum pooling) the activation, both of
        #them are stride 2 the filter banks of which are concatenated as in
        #figure 10."""
        outputs = [branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

#Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, input_channels, channels_7x7):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 192, kernel_size=1)

        c7 = channels_7x7

        #In theory, we could go even further and argue that one can replace any n × n
        #convolution by a 1 × n convolution followed by a n × 1 convolution and the
        #computational cost saving increases dramatically as n grows (see figure 6).
        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch7x7stack = nn.Sequential(
            BasicConv2d(input_channels, c7, kernel_size=1),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 192, kernel_size=1),
        )

    def forward(self, x):

        #x -> 1x1(same)
        branch1x1 = self.branch1x1(x)

        #x -> 1layer 1*7 and 7*1 (same)
        branch7x7 = self.branch7x7(x)

        #x-> 2layer 1*7 and 7*1(same)
        branch7x7stack = self.branch7x7stack(x)

        #x-> avgpool (same)
        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, input_channels):
        super().__init__()

        self.branch3x3 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2)
        )

        self.branch7x7 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x):

        #x -> 1x1 -> 3x3(downsample)
        branch3x3 = self.branch3x3(x)

        #x -> 1x1 -> 1x7 -> 7x1 -> 3x3 (downsample)
        branch7x7 = self.branch7x7(x)

        #x -> avgpool (downsample)
        branchpool = self.branchpool(x)

        outputs = [branch3x3, branch7x7, branchpool]

        return torch.cat(outputs, 1)


#same
class InceptionE(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.branch1x1 = BasicConv2d(input_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(input_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3stack_1 = BasicConv2d(input_channels, 448, kernel_size=1)
        self.branch3x3stack_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(input_channels, 192, kernel_size=1)
        )

    def forward(self, x):

        #x -> 1x1 (same)
        branch1x1 = self.branch1x1(x)

        # x -> 1x1 -> 3x1
        # x -> 1x1 -> 1x3
        # concatenate(3x1, 1x3)
        #"""7. Inception modules with expanded the filter bank outputs.
        #This architecture is used on the coarsest (8 × 8) grids to promote
        #high dimensional representations, as suggested by principle
        #2 of Section 2."""
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        # x -> 1x1 -> 3x3 -> 1x3
        # x -> 1x1 -> 3x3 -> 3x1
        #concatenate(1x3, 3x1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)

        branchpool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]

        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.Conv2d_1a_3x3 = BasicConv2d(2, 32, kernel_size=3, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)

        #naive inception module
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        #downsample
        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        #downsample
        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(512)

        #6*6 feature size
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d()
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x1,x2,y1,y2,b):

        #32 -> 30
        x1 = self.Conv2d_1a_3x3(x1)
        x1 = self.Conv2d_2a_3x3(x1)
        x1 = self.Conv2d_2b_3x3(x1)
        x1 = self.Conv2d_3b_1x1(x1)
        x1 = self.Conv2d_4a_3x3(x1)
        x1 = self.Mixed_5b(x1)
        x1 = self.Mixed_5c(x1)
        x1 = self.Mixed_5d(x1)
        x1 = self.Mixed_6a(x1)
        x1 = self.Mixed_6b(x1)
        x1 = self.Mixed_6c(x1)
        x1 = self.Mixed_6d(x1)
        x1 = self.Mixed_6e(x1)
        x1 = self.Mixed_7a(x1)
        x1 = self.Mixed_7b(x1)
        output1 = self.Mixed_7c(x1)

        x2 = self.Conv2d_1a_3x3(x2)
        x2 = self.Conv2d_2a_3x3(x2)
        x2 = self.Conv2d_2b_3x3(x2)
        x2 = self.Conv2d_3b_1x1(x2)
        x2 = self.Conv2d_4a_3x3(x2)
        x2 = self.Mixed_5b(x2)
        x2 = self.Mixed_5c(x2)
        x2 = self.Mixed_5d(x2)
        x2 = self.Mixed_6a(x2)
        x2 = self.Mixed_6b(x2)
        x2 = self.Mixed_6c(x2)
        x2 = self.Mixed_6d(x2)
        x2 = self.Mixed_6e(x2)
        x2 = self.Mixed_7a(x2)
        x2 = self.Mixed_7b(x2)
        output2 = self.Mixed_7c(x2)

        y1 = self.Conv2d_1a_3x3(y1)
        y1 = self.Conv2d_2a_3x3(y1)
        y1 = self.Conv2d_2b_3x3(y1)
        y1 = self.Conv2d_3b_1x1(y1)
        y1 = self.Conv2d_4a_3x3(y1)
        y1 = self.Mixed_5b(y1)
        y1 = self.Mixed_5c(y1)
        y1 = self.Mixed_5d(y1)
        y1 = self.Mixed_6a(y1)
        y1 = self.Mixed_6b(y1)
        y1 = self.Mixed_6c(y1)
        y1 = self.Mixed_6d(y1)
        y1 = self.Mixed_6e(y1)
        y1 = self.Mixed_7a(y1)
        y1 = self.Mixed_7b(y1)
        output11 = self.Mixed_7c(y1)

        y2 = self.Conv2d_1a_3x3(y2)
        y2 = self.Conv2d_2a_3x3(y2)
        y2 = self.Conv2d_2b_3x3(y2)
        y2 = self.Conv2d_3b_1x1(y2)
        y2 = self.Conv2d_4a_3x3(y2)
        y2 = self.Mixed_5b(y2)
        y2 = self.Mixed_5c(y2)
        y2 = self.Mixed_5d(y2)
        y2 = self.Mixed_6a(y2)
        y2 = self.Mixed_6b(y2)
        y2 = self.Mixed_6c(y2)
        y2 = self.Mixed_6d(y2)
        y2 = self.Mixed_6e(y2)
        y2 = self.Mixed_7a(y2)
        y2 = self.Mixed_7b(y2)
        output21 = self.Mixed_7c(y2)
        #30 -> 30


        #30 -> 14
        #Efficient Grid Size Reduction to avoid representation
        #bottleneck


        #14 -> 14
        #"""In practice, we have found that employing this factorization does not
        #work well on early layers, but it gives very good results on medium
        #grid-sizes (On m × m feature maps, where m ranges between 12 and 20).
        #On that level, very good results can be achieved by using 1 × 7 convolutions
        #followed by 7 × 1 convolutions."""


        #14 -> 6
        #Efficient Grid Size Reduction


        #6 -> 6
        #We are using this solution only on the coarsest grid,
        #since that is the place where producing high dimensional
        #sparse representation is the most critical as the ratio of
        #local processing (by 1 × 1 convolutions) is increased compared
        #to the spatial aggregation."""
        feat1 = output1
        feat2 = output2
        feat11 = output11
        feat21 = output21
        N1, C1, H1, W1 = feat1.shape
        N2, C2, H2, W2 = feat2.shape
        # N1, C1, H1, W1 = output1_1.shape
        # N2, C2, H2, W2 = output2_1.shape
        output1 = self.avgpool(output1)
        output2 = self.avgpool(output2)
        output11 = self.avgpool(output11)
        output21 = self.avgpool(output21)
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



        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return output, CAM1, CAM2, CAM3, CAM4

        #6 -> 1
        # x = self.avgpool(x)
        # x = self.dropout(x)
        #
        #
        #
        #
        #
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        # return x


def inceptionv3():
    return InceptionV3()



