"""xception in pytorch


[1] François Chollet

    Xception: Deep Learning with Depthwise Separable Convolutions
    https://arxiv.org/abs/1610.02357
"""
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
def returnCAM1(feature_conv, weight_softmax, class_idx):
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
    for tt in range(16):
        for yyy in range(8):
            if cam_img[tt,yyy]<0.3:
                cam_img[tt,yyy]=0
            else:
                cam_img[tt, yyy] = 1

    return cam_img
class SeperableConv2d(nn.Module):

    #***Figure 4. An “extreme” version of our Inception module,
    #with one spatial convolution per output channel of the 1x1
    #convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            bias=False,
            **kwargs
        )

        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class EntryFlow(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 3,stride=2,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )

        #no downsampling
        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        #no downsampling
        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1,stride=2),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut

        return x

class MiddleFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)

        return shortcut + residual

class MiddleFlow(nn.Module):
    def __init__(self, block):
        super().__init__()

        #"""then through the middle flow which is repeated eight times"""
        self.middel_block = self._make_flow(block, 8)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())

        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv2d(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )

        self.conv = nn.Sequential(
            SeperableConv2d(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2d(1536, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            # SeperableConv2d(1536, 2048, 3, padding=1),
            # nn.BatchNorm2d(2048),
            # nn.ReLU(inplace=True)
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        # output = self.avgpool(output)

        return output

class Xception(nn.Module):

    def __init__(self, block, num_class=2):
        super().__init__()
        self.entry_flow = EntryFlow()
        self.middel_flow = MiddleFlow(block)
        self.exit_flow = ExitFLow()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(
        #     nn.Linear(2048, num_class),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(128, num_class),
        # )
        self.classifier = nn.Linear(512, num_class)
    # def forward(self, x1,x2,y1,y2,b):
    #     x1 = self.entry_flow(x1)
    #     x1 = self.middel_flow(x1)
    #     output1 = self.exit_flow(x1)
    #
    #     x2 = self.entry_flow(x2)
    #     x2 = self.middel_flow(x2)
    #     output2 = self.exit_flow(x2)
    #
    #     y1 = self.entry_flow(y1)
    #     y1 = self.middel_flow(y1)
    #     output11 = self.exit_flow(y1)
    #
    #     y2 = self.entry_flow(y2)
    #     y2 = self.middel_flow(y2)
    #     output21 = self.exit_flow(y2)
    #
    #     feat1 = output1
    #     feat2 = output2
    #     feat11 = output11
    #     feat21 = output21
    #     N1, C1, H1, W1 = feat1.shape
    #     N2, C2, H2, W2 = feat2.shape
    #     # N1, C1, H1, W1 = output1_1.shape
    #     # N2, C2, H2, W2 = output2_1.shape
    #     output1 = self.avgpool(output1)
    #     output2 = self.avgpool(output2)
    #     output11 = self.avgpool(output11)
    #     output21 = self.avgpool(output21)
    #     # output = self.maxpool(output)
    #     output1 = output1.view(output1.size(0), -1)
    #     output2 = output2.view(output2.size(0), -1)
    #     output11 = output11.view(output11.size(0), -1)
    #     output21 = output21.view(output21.size(0), -1)
    #     output = torch.cat((output1, output2), dim=1)
    #     output22 = torch.cat((output11, output21), dim=1)
    #     output = torch.cat((output, output22), dim=1)
    #     output = self.classifier(output)
    #     # self.is_train实例化时已经设置好了，即使后面再设置这个属性，但是在forward中依然是True。可以在forward参数里面进行设置，设置默认参数，测试时再设置。
    #     params1 = list(self.parameters())
    #     fc_weights1 = params1[-2].data
    #     fc_weights2 = params1[-2].data
    #     fc_weights3 = params1[-2].data
    #     fc_weights4 = params1[-2].data
    #     fc_weights1 = np.squeeze(fc_weights1[:, :512].view(2, C1, 1, 1))
    #     fc_weights2 = np.squeeze(fc_weights2[:, 512:1024].view(2, C2, 1, 1))
    #     fc_weights3 = np.squeeze(fc_weights3[:, 1024:1536].view(2, C2, 1, 1))
    #     fc_weights4 = np.squeeze(fc_weights4[:, 1536:].view(2, C2, 1, 1))
    #
    #     # fc_weights = Variable(fc_weights, requires_grad=False)  # 让这个权重 再次使用 不必更新
    #     _value1, preds_fianl1 = output.max(1)
    #     CAM1 = []
    #     CAM2 = []
    #     CAM3 = []
    #     CAM4 = []
    #     # print(preds_fianl,'fianl')
    #     # fc_weights=fc_weights[0]
    #     for i in range(b):
    #         CAMs1 = returnCAM(feat1[i].unsqueeze(0), fc_weights1, preds_fianl1[i])
    #         CAM1.append(CAMs1)
    #     for ii in range(b):
    #         CAMs2 = returnCAM(feat2[ii].unsqueeze(0), fc_weights2, preds_fianl1[i])
    #         CAM2.append(CAMs2)
    #     for iii in range(b):
    #         CAMs3 = returnCAM(feat11[ii].unsqueeze(0), fc_weights3, preds_fianl1[i])
    #         CAM3.append(CAMs3)
    #     for iiii in range(b):
    #         CAMs4 = returnCAM(feat21[ii].unsqueeze(0), fc_weights4, preds_fianl1[i])
    #         CAM4.append(CAMs4)
    #
    #
    #
    #     # x = x.view(x.size(0), -1)
    #     # x = self.fc(x)
    #
    #     return output, CAM1, CAM2, CAM3, CAM4
    def forward(self, x,b):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        output = self.exit_flow(x)
        feat = output


        N1, C1, H1, W1 = feat.shape
        # N1, C1, H1, W1 = output1_1.shape
        # N2, C2, H2, W2 = output2_1.shape
        output = self.avgpool(output)

        # output = self.maxpool(output)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        # self.is_train实例化时已经设置好了，即使后面再设置这个属性，但是在forward中依然是True。可以在forward参数里面进行设置，设置默认参数，测试时再设置。
        params1 = list(self.parameters())
        fc_weights = params1[-2].data
        fc_weights = np.squeeze(fc_weights[:, :512].view(2, C1, 1, 1))
        # fc_weights = Variable(fc_weights, requires_grad=False)  # 让这个权重 再次使用 不必更新
        _value1, preds_fianl1 = output.max(1)
        CAM = []
        # print(preds_fianl,'fianl')
        # fc_weights=fc_weights[0]
        for i in range(b):
            CAMs = returnCAM1(feat[i].unsqueeze(0), fc_weights, preds_fianl1[i])
            CAM.append(CAMs)




        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return output,CAM
    # def forward(self, x1,x2,b):
    #     x1 = self.entry_flow(x1)
    #     x1 = self.middel_flow(x1)
    #     output1 = self.exit_flow(x1)
    #
    #     x2 = self.entry_flow(x2)
    #     x2 = self.middel_flow(x2)
    #     output2 = self.exit_flow(x2)
    #
    #     feat1 = output1
    #     feat2 = output2
    #
    #     N1, C1, H1, W1 = feat1.shape
    #     N2, C2, H2, W2 = feat2.shape
    #     # N1, C1, H1, W1 = output1_1.shape
    #     # N2, C2, H2, W2 = output2_1.shape
    #     output1 = self.avgpool(output1)
    #     output2 = self.avgpool(output2)
    #
    #     # output = self.maxpool(output)
    #     output1 = output1.view(output1.size(0), -1)
    #     output2 = output2.view(output2.size(0), -1)
    #     output = torch.cat((output1, output2), dim=1)
    #     output = self.classifier(output)
    #     # self.is_train实例化时已经设置好了，即使后面再设置这个属性，但是在forward中依然是True。可以在forward参数里面进行设置，设置默认参数，测试时再设置。
    #     params1 = list(self.parameters())
    #     fc_weights1 = params1[-2].data
    #     fc_weights2 = params1[-2].data
    #     fc_weights1 = np.squeeze(fc_weights1[:, :512].view(2, C1, 1, 1))
    #     fc_weights2 = np.squeeze(fc_weights2[:, 512:].view(2, C2, 1, 1))
    #     # fc_weights = Variable(fc_weights, requires_grad=False)  # 让这个权重 再次使用 不必更新
    #     _value1, preds_fianl1 = output.max(1)
    #     CAM1 = []
    #     CAM2 = []
    #     # print(preds_fianl,'fianl')
    #     # fc_weights=fc_weights[0]
    #     for i in range(b):
    #         CAMs1 = returnCAM(feat1[i].unsqueeze(0), fc_weights1, preds_fianl1[i])
    #         CAM1.append(CAMs1)
    #     for ii in range(b):
    #         CAMs2 = returnCAM(feat2[ii].unsqueeze(0), fc_weights2, preds_fianl1[i])
    #         CAM2.append(CAMs2)
    #
    #
    #
    #     # x = x.view(x.size(0), -1)
    #     # x = self.fc(x)
    #
    #     return output,CAM1,CAM2


class SeperableConv2d_train(nn.Module):

    #***Figure 4. An “extreme” version of our Inception module,
    #with one spatial convolution per output channel of the 1x1
    #convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            bias=False,
            **kwargs
        )

        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class EntryFlow_train(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 32, 3,stride=2,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_residual = nn.Sequential(
            SeperableConv2d_train(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d_train(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d_train(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d_train(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )

        #no downsampling
        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d_train(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d_train(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        #no downsampling
        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1,stride=2),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut

        return x

class MiddleFLowBlock_train(nn.Module):

    def __init__(self):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d_train(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d_train(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d_train(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)

        return shortcut + residual

class MiddleFlow_train(nn.Module):
    def __init__(self, block):
        super().__init__()

        #"""then through the middle flow which is repeated eight times"""
        self.middel_block = self._make_flow(block, 8)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())

        return nn.Sequential(*flows)


class ExitFLow_train(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d_train(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv2d_train(728, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2),
            nn.BatchNorm2d(1024)
        )

        self.conv = nn.Sequential(
            SeperableConv2d_train(1024, 1536, 3, padding=1),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeperableConv2d_train(1536, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            # SeperableConv2d(1536, 2048, 3, padding=1),
            # nn.BatchNorm2d(2048),
            # nn.ReLU(inplace=True)
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        output = self.conv(output)
        # output = self.avgpool(output)

        return output
class Xception_train(nn.Module):

    def __init__(self, block, num_class=2):
        super().__init__()
        self.entry_flow = EntryFlow_train()
        self.middel_flow = MiddleFlow_train(block)
        self.exit_flow = ExitFLow_train()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Sequential(
        #     nn.Linear(2048, num_class),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(128, num_class),
        # )
        self.classifier = nn.Linear(1024, num_class)
    # def forward(self, x1,x2,y1,y2,b):
    #     x1 = self.entry_flow(x1)
    #     x1 = self.middel_flow(x1)
    #     output1 = self.exit_flow(x1)
    #
    #     x2 = self.entry_flow(x2)
    #     x2 = self.middel_flow(x2)
    #     output2 = self.exit_flow(x2)
    #
    #     y1 = self.entry_flow(y1)
    #     y1 = self.middel_flow(y1)
    #     output11 = self.exit_flow(y1)
    #
    #     y2 = self.entry_flow(y2)
    #     y2 = self.middel_flow(y2)
    #     output21 = self.exit_flow(y2)
    #
    #     feat1 = output1
    #     feat2 = output2
    #     feat11 = output11
    #     feat21 = output21
    #     N1, C1, H1, W1 = feat1.shape
    #     N2, C2, H2, W2 = feat2.shape
    #     # N1, C1, H1, W1 = output1_1.shape
    #     # N2, C2, H2, W2 = output2_1.shape
    #     output1 = self.avgpool(output1)
    #     output2 = self.avgpool(output2)
    #     output11 = self.avgpool(output11)
    #     output21 = self.avgpool(output21)
    #     # output = self.maxpool(output)
    #     output1 = output1.view(output1.size(0), -1)
    #     output2 = output2.view(output2.size(0), -1)
    #     output11 = output11.view(output11.size(0), -1)
    #     output21 = output21.view(output21.size(0), -1)
    #     output = torch.cat((output1, output2), dim=1)
    #     output22 = torch.cat((output11, output21), dim=1)
    #     output = torch.cat((output, output22), dim=1)
    #     output = self.classifier(output)
    #     # self.is_train实例化时已经设置好了，即使后面再设置这个属性，但是在forward中依然是True。可以在forward参数里面进行设置，设置默认参数，测试时再设置。
    #     params1 = list(self.parameters())
    #     fc_weights1 = params1[-2].data
    #     fc_weights2 = params1[-2].data
    #     fc_weights3 = params1[-2].data
    #     fc_weights4 = params1[-2].data
    #     fc_weights1 = np.squeeze(fc_weights1[:, :512].view(2, C1, 1, 1))
    #     fc_weights2 = np.squeeze(fc_weights2[:, 512:1024].view(2, C2, 1, 1))
    #     fc_weights3 = np.squeeze(fc_weights3[:, 1024:1536].view(2, C2, 1, 1))
    #     fc_weights4 = np.squeeze(fc_weights4[:, 1536:].view(2, C2, 1, 1))
    #
    #     # fc_weights = Variable(fc_weights, requires_grad=False)  # 让这个权重 再次使用 不必更新
    #     _value1, preds_fianl1 = output.max(1)
    #     CAM1 = []
    #     CAM2 = []
    #     CAM3 = []
    #     CAM4 = []
    #     # print(preds_fianl,'fianl')
    #     # fc_weights=fc_weights[0]
    #     for i in range(b):
    #         CAMs1 = returnCAM(feat1[i].unsqueeze(0), fc_weights1, preds_fianl1[i])
    #         CAM1.append(CAMs1)
    #     for ii in range(b):
    #         CAMs2 = returnCAM(feat2[ii].unsqueeze(0), fc_weights2, preds_fianl1[i])
    #         CAM2.append(CAMs2)
    #     for iii in range(b):
    #         CAMs3 = returnCAM(feat11[ii].unsqueeze(0), fc_weights3, preds_fianl1[i])
    #         CAM3.append(CAMs3)
    #     for iiii in range(b):
    #         CAMs4 = returnCAM(feat21[ii].unsqueeze(0), fc_weights4, preds_fianl1[i])
    #         CAM4.append(CAMs4)
    #
    #
    #
    #     # x = x.view(x.size(0), -1)
    #     # x = self.fc(x)
    #
    #     return output, CAM1, CAM2, CAM3, CAM4
    # def forward(self, x,b):
    #     x = self.entry_flow(x)
    #     x = self.middel_flow(x)
    #     output = self.exit_flow(x)
    #     feat = output
    #
    #
    #     N1, C1, H1, W1 = feat.shape
    #     # N1, C1, H1, W1 = output1_1.shape
    #     # N2, C2, H2, W2 = output2_1.shape
    #     output = self.avgpool(output)
    #
    #     # output = self.maxpool(output)
    #     output = output.view(output.size(0), -1)
    #     output = self.classifier(output)
    #     # self.is_train实例化时已经设置好了，即使后面再设置这个属性，但是在forward中依然是True。可以在forward参数里面进行设置，设置默认参数，测试时再设置。
    #     params1 = list(self.parameters())
    #     fc_weights = params1[-2].data
    #     fc_weights = np.squeeze(fc_weights[:, :512].view(2, C1, 1, 1))
    #     # fc_weights = Variable(fc_weights, requires_grad=False)  # 让这个权重 再次使用 不必更新
    #     _value1, preds_fianl1 = output.max(1)
    #     CAM = []
    #     # print(preds_fianl,'fianl')
    #     # fc_weights=fc_weights[0]
    #     for i in range(b):
    #         CAMs = returnCAM(feat[i].unsqueeze(0), fc_weights, preds_fianl1[i])
    #         CAM.append(CAMs)
    #
    #
    #
    #
    #     # x = x.view(x.size(0), -1)
    #     # x = self.fc(x)
    #
    #     return output,CAM
    def forward(self, x1,x2,b):
        x1 = self.entry_flow(x1)
        x1 = self.middel_flow(x1)
        output1 = self.exit_flow(x1)

        x2 = self.entry_flow(x2)
        x2 = self.middel_flow(x2)
        output2 = self.exit_flow(x2)

        feat1 = output1
        feat2 = output2

        N1, C1, H1, W1 = feat1.shape
        N2, C2, H2, W2 = feat2.shape
        # N1, C1, H1, W1 = output1_1.shape
        # N2, C2, H2, W2 = output2_1.shape
        output1 = self.avgpool(output1)
        output2 = self.avgpool(output2)

        # output = self.maxpool(output)
        output1 = output1.view(output1.size(0), -1)
        output2 = output2.view(output2.size(0), -1)
        output = torch.cat((output1, output2), dim=1)
        output = self.classifier(output)
        # self.is_train实例化时已经设置好了，即使后面再设置这个属性，但是在forward中依然是True。可以在forward参数里面进行设置，设置默认参数，测试时再设置。
        params1 = list(self.parameters())
        fc_weights1 = params1[-2].data
        fc_weights2 = params1[-2].data
        fc_weights1 = np.squeeze(fc_weights1[:, :512].view(2, C1, 1, 1))
        fc_weights2 = np.squeeze(fc_weights2[:, 512:].view(2, C2, 1, 1))
        # fc_weights = Variable(fc_weights, requires_grad=False)  # 让这个权重 再次使用 不必更新
        _value1, preds_fianl1 = output.max(1)
        CAM1 = []
        CAM2 = []
        # print(preds_fianl,'fianl')
        # fc_weights=fc_weights[0]
        for i in range(b):
            CAMs1 = returnCAM(feat1[i].unsqueeze(0), fc_weights1, preds_fianl1[i])
            CAM1.append(CAMs1)
        for ii in range(b):
            CAMs2 = returnCAM(feat2[ii].unsqueeze(0), fc_weights2, preds_fianl1[i])
            CAM2.append(CAMs2)



        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return output,CAM1,CAM2

def xception():
    return Xception(MiddleFLowBlock)


def xception_train():
    return Xception_train(MiddleFLowBlock_train)
