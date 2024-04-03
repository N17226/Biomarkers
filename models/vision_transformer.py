import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
from torchvision.transforms import transforms
# from visualizer import get_local


# from torchsummaryX import summary
cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

import h5py
import numpy as np

def Thresh_and_blur3(img):  # 设定阈值
    # blurred = cv2.GaussianBlur(gradient, (13, 13),0)
    # (_, thresh) = cv2.threshold(blurred, gradient.max()-30, 1, cv2.THRESH_BINARY)
    # athdMEAN = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 1)
    ret, otsu = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret2, dst = cv2.threshold(img, 0.6*ret, 255, cv2.THRESH_BINARY)
    # print(ret)
    return dst



def returnCAM_att(cam,pre):
    # 类激活图上采样到 256 x 256
    cam=cam[pre,:,:]
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    resize_all = transforms.Resize((512, 256))
    cam_img=resize_all(cam_img.unsqueeze(0)).squeeze(0).cpu()
    cam_img = (cam_img - cam_img.min()) / max((cam_img.max() - cam_img.min()), 1)
    return cam_img

def returnCAM_att2(cam,pre):
    # 类激活图上采样到 256 x 256

    # print()
    cam=cam[pre,:,:]
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    resize_all = transforms.Resize((512, 256))
    cam_img=resize_all(cam_img.unsqueeze(0))
    cam_img = Thresh_and_blur3(np.uint8(255 * cam_img.squeeze(0).cpu()))
    cam_img = (cam_img - cam_img.min()) / max((cam_img.max() - cam_img.min()), 1)
    cam_img = np.float32(cam_img)
    cam_img = torch.tensor(cam_img)
    return cam_img




class MLP(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
class Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5      # 根号d，缩放因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # print(q, '22222')
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attention = attn
        attn = self.attn_drop(attn)
        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attention

class SAB_Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(SAB_Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))

        return x,self.attn(self.norm1(x))[1]


class Crossview_Attention(nn.Module):
    def __init__(self,
                 dim,  # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Crossview_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5      # 根号d，缩放因子
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    # @get_local('attn')
    def forward(self,x):

        x1, x2 = x[0].squeeze(0), x[1].squeeze(0)
        B, N, C = x1.shape
        qkv1 = self.qkv(x1).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv2 = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q1, k1, v1 = qkv1[0], qkv1[1], qkv1[2]  # make torchscript happy (cannot use tensor as tuple)
        q2, k2, v2 = qkv2[0], qkv2[1], qkv2[2]
        attn = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attention=attn
        attn = self.attn_drop(attn)
        x = (attn @ v1).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,attention

class Crossview_Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Crossview_Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn= Crossview_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self,x):
        x1, x2 = x[0].squeeze(0),x[1].squeeze(0)
        x1=self.norm1(x1).unsqueeze(0)
        x2 = self.norm1(x2).unsqueeze(0)
        x = torch.cat((x1, x2), dim=0)
        a=self.attn(x)
        x1 = x1.squeeze(0) + self.attn(x)[0]
        x1 = x1 + self.mlp(self.norm2(x1))

        return x1,self.attn(x)[1]

class VGG_vit(nn.Module):

    def __init__(self, features, num_class=2,B=2,embed_dim=512,num_patches=128,drop_ratio=0.5
                 ,depth=8, num_heads=16, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False,
                 attn_drop_ratio=0., drop_path_ratio=0.5, norm_layer=None,
                 act_layer=None):
        super(VGG_vit,self).__init__()
        self.features = features
        self.cls_token = nn.Parameter(torch.zeros(B, 1, 512))
        self.cls_token2 = nn.Parameter(torch.zeros(B, 1, 512))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches +1, embed_dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches +1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.SAB_blocks = nn.Sequential(*[
            SAB_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
        ])

        self.Cross_view_blocks = nn.Sequential(*[
            Crossview_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
        ])

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(512, num_class)
        self.trans_cls_head = nn.Conv2d(embed_dim, num_class, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
#####
    def forward(self, x1,x2,b):
        attn_weights1=[]
        attn_weights2 = []
        output1 = self.features(x1)
        output2 = self.features(x2)
        N1, C1, H1, W1 = output1.shape
        N2, C2, H2, W2 = output2.shape
        output1 = output1.view(N1,C1,-1).transpose(1, 2)
        output2 = output2.view(N2,C2, -1).transpose(1, 2)
        cls_tokens1 = self.cls_token
        cls_tokens2 = self.cls_token2
        output1 = torch.cat((cls_tokens1, output1), dim=1)
        output2 = torch.cat((cls_tokens2, output2), dim=1)
        output1 = self.pos_drop(output1 + self.pos_embed)
        output2 = self.pos_drop(output2 + self.pos_embed2)
        output1,att11 = self.SAB_blocks(output1)
        output2,att21 = self.SAB_blocks(output2)
        output1,att12 = self.SAB_blocks(output1)
        output2,att22 = self.SAB_blocks(output2)
        output1 = output1.unsqueeze(0)
        output2 = output2.unsqueeze(0)
        attn_weights1.append(att11)
        attn_weights1.append(att12)
        attn_weights2.append(att21)
        attn_weights2.append(att22)
        output1 = self.norm(output1)
        output2=self.norm(output2)
        output11 = torch.cat((output1, output2), dim=0)
        output21 = torch.cat((output2, output1), dim=0)
        output11,att13 = self.Cross_view_blocks(output11)
        output21, att23 = self.Cross_view_blocks(output21)
        output11 = self.norm(output11)
        output21 = self.norm(output21)
        attn_weights1.append(att13)
        attn_weights2.append(att23)
###left
        attn_weights1 = torch.stack(attn_weights1)
        attn_weights1 = torch.mean(attn_weights1, dim=2)
        residual_att = torch.eye(attn_weights1.size(2)).unsqueeze(0).unsqueeze(1).to(
            attn_weights1.get_device())
        aug_att_mat = attn_weights1 + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)


        joint_attns = torch.zeros(aug_att_mat.size()).to(attn_weights1.get_device())
        joint_attns[0] = aug_att_mat[0]
        joint_attns_all = torch.zeros(aug_att_mat.size()).to(attn_weights1.get_device())
        joint_attns_all[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size()[0]):
            joint_attns_all[n] = torch.matmul(aug_att_mat[n], joint_attns[n - 1])

        x_t = output11
        x_patch = x_t[:, 1:]
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, 16, 8, c])
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        x_patch = self.trans_cls_head(x_patch)
        x_logits = self.pooling(x_patch).flatten(1)

        feature_map = x_patch.detach().clone()
        n, c, h, w = feature_map.shape

        # cams = joint_attns_all[0][:, 0, 1:].reshape([n, h, w]).unsqueeze(1)   ###another way to generate Semantic-agnostic Attention Map
        cams = attn_weights1.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)#####Semantic-aware Maps
        tscams = cams *feature_map

####right
        attn_weights2 = torch.stack(attn_weights2)
        attn_weights2= torch.mean(attn_weights2, dim=2)
        residual_att2 = torch.eye(attn_weights2.size(2)).unsqueeze(0).unsqueeze(1).to(
            attn_weights2.get_device())  # 12 * B * N * N
        aug_att_mat2 = attn_weights2 + residual_att2
        aug_att_mat2 = aug_att_mat2 / aug_att_mat2.sum(dim=-1).unsqueeze(-1)

        joint_attns2 = torch.zeros(aug_att_mat2.size()).to(attn_weights2.get_device())
        joint_attns2[0] = aug_att_mat2[0]
        joint_attns_all2 = torch.zeros(aug_att_mat2.size()).to(attn_weights2.get_device())
        joint_attns_all2[0] = aug_att_mat2[0]
        for n2 in range(1, aug_att_mat2.size()[0]):
            joint_attns_all2[n2] = torch.matmul(aug_att_mat2[n2], joint_attns2[n2 - 1])

        x_t2 = output21
        x_patch2 = x_t2[:, 1:]
        n1, p1, c1= x_patch2.shape
        x_patch2 = torch.reshape(x_patch2, [n1, 16, 8, c1])
        x_patch2 = x_patch2.permute([0, 3, 1, 2]).contiguous()
        x_patch2 = self.trans_cls_head(x_patch2)
        x_logits2 = self.pooling(x_patch2).flatten(1)

        feature_map2 = x_patch2.detach().clone()
        n3, c3, h3, w3 = feature_map2.shape
        # cams2 = joint_attns_all2[0][:, 0, 1:].reshape([n3, h3, w3]).unsqueeze(1)    ###another way to generate Semantic-agnostic Attention Map
        cams2 = attn_weights2.sum(0)[:, 0, 1:].reshape([n3, h3, w3]).unsqueeze(1)   ###Semantic-aware Maps
        tscams2 = cams2 * feature_map2


        output_left=x_logits
        output_right=x_logits2
        _value1, preds_fianl1 = output_left.max(1)
        _value2, preds_fianl2 = output_right.max(1)
        # a=cams[0]
        CAM1=[]
        CAM2 = []

        for i in range(b):

            CAMs_l = returnCAM_att(tscams[i],preds_fianl1[i])
            CAMs_r = returnCAM_att(tscams2[i],preds_fianl2[i])
            CAM1.append(CAMs_l)
            CAM2.append(CAMs_r)
        return output_left,output_right,CAM1,CAM2



class VGG_vit_FZFX(nn.Module):

    def __init__(self, features, num_class=2,B=2,embed_dim=512,num_patches=128,drop_ratio=0.5
                 ,depth=8, num_heads=16, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False,
                 attn_drop_ratio=0., drop_path_ratio=0.5, norm_layer=None,
                 act_layer=None):
        super(VGG_vit_FZFX,self).__init__()
        self.features = features
        self.cls_token = nn.Parameter(torch.zeros(B, 1, 512))
        self.cls_token2 = nn.Parameter(torch.zeros(B, 1, 512))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches +1, embed_dim))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches +1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.SAB_blocks = nn.Sequential(*[
            SAB_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
        ])

        self.Cross_view_blocks = nn.Sequential(*[
            Crossview_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
        ])

        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(512, num_class)
        self.trans_cls_head = nn.Conv2d(embed_dim, num_class, kernel_size=3, stride=1, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
#####
    def forward(self, x1,x2,b):
        attn_weights1=[]
        attn_weights2 = []
        output1 = self.features(x1)
        output2 = self.features(x2)
        N1, C1, H1, W1 = output1.shape
        N2, C2, H2, W2 = output2.shape
        output1 = output1.view(N1,C1,-1).transpose(1, 2)
        output2 = output2.view(N2,C2, -1).transpose(1, 2)
        cls_tokens1 = self.cls_token
        cls_tokens2 = self.cls_token2
        output1 = torch.cat((cls_tokens1, output1), dim=1)
        output2 = torch.cat((cls_tokens2, output2), dim=1)
        output1 = self.pos_drop(output1 + self.pos_embed)
        output2 = self.pos_drop(output2 + self.pos_embed2)
        output1,att11 = self.SAB_blocks(output1)
        output2,att21 = self.SAB_blocks(output2)
        output1,att12 = self.SAB_blocks(output1)
        output2,att22 = self.SAB_blocks(output2)
        output1 = output1.unsqueeze(0)
        output2 = output2.unsqueeze(0)
        attn_weights1.append(att11)
        attn_weights1.append(att12)
        attn_weights2.append(att21)
        attn_weights2.append(att22)
        output1 = self.norm(output1)
        output2=self.norm(output2)
        output11 = torch.cat((output1, output2), dim=0)
        output21 = torch.cat((output2, output1), dim=0)
        output11,att13 = self.Cross_view_blocks(output11)
        output21, att23 = self.Cross_view_blocks(output21)
        output11 = self.norm(output11)
        output21 = self.norm(output21)
        attn_weights1.append(att13)
        attn_weights2.append(att23)
###left
        attn_weights1 = torch.stack(attn_weights1)
        attn_weights1 = torch.mean(attn_weights1, dim=2)
        residual_att = torch.eye(attn_weights1.size(2)).unsqueeze(0).unsqueeze(1).to(
            attn_weights1.get_device())
        aug_att_mat = attn_weights1 + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)


        joint_attns = torch.zeros(aug_att_mat.size()).to(attn_weights1.get_device())
        joint_attns[0] = aug_att_mat[0]
        joint_attns_all = torch.zeros(aug_att_mat.size()).to(attn_weights1.get_device())
        joint_attns_all[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size()[0]):
            joint_attns_all[n] = torch.matmul(aug_att_mat[n], joint_attns[n - 1])

        x_t = output11
        x_patch = x_t[:, 1:]
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, 16, 8, c])
        x_patch = x_patch.permute([0, 3, 1, 2]).contiguous()
        x_patch = self.trans_cls_head(x_patch)
        x_logits = self.pooling(x_patch).flatten(1)

        feature_map = x_patch.detach().clone()
        n, c, h, w = feature_map.shape

        # cams = joint_attns_all[0][:, 0, 1:].reshape([n, h, w]).unsqueeze(1)   ###another way to generate Semantic-agnostic Attention Map
        cams = attn_weights1.sum(0)[:, 0, 1:].reshape([n, h, w]).unsqueeze(1)#####Semantic-aware Maps
        tscams = cams *feature_map

####right
        attn_weights2 = torch.stack(attn_weights2)
        attn_weights2= torch.mean(attn_weights2, dim=2)
        residual_att2 = torch.eye(attn_weights2.size(2)).unsqueeze(0).unsqueeze(1).to(
            attn_weights2.get_device())  # 12 * B * N * N
        aug_att_mat2 = attn_weights2 + residual_att2
        aug_att_mat2 = aug_att_mat2 / aug_att_mat2.sum(dim=-1).unsqueeze(-1)

        joint_attns2 = torch.zeros(aug_att_mat2.size()).to(attn_weights2.get_device())
        joint_attns2[0] = aug_att_mat2[0]
        joint_attns_all2 = torch.zeros(aug_att_mat2.size()).to(attn_weights2.get_device())
        joint_attns_all2[0] = aug_att_mat2[0]
        for n2 in range(1, aug_att_mat2.size()[0]):
            joint_attns_all2[n2] = torch.matmul(aug_att_mat2[n2], joint_attns2[n2 - 1])

        x_t2 = output21
        x_patch2 = x_t2[:, 1:]
        n1, p1, c1= x_patch2.shape
        x_patch2 = torch.reshape(x_patch2, [n1, 16, 8, c1])
        x_patch2 = x_patch2.permute([0, 3, 1, 2]).contiguous()
        x_patch2 = self.trans_cls_head(x_patch2)
        x_logits2 = self.pooling(x_patch2).flatten(1)

        feature_map2 = x_patch2.detach().clone()
        n3, c3, h3, w3 = feature_map2.shape
        # cams2 = joint_attns_all2[0][:, 0, 1:].reshape([n3, h3, w3]).unsqueeze(1)    ###another way to generate Semantic-agnostic Attention Map
        cams2 = attn_weights2.sum(0)[:, 0, 1:].reshape([n3, h3, w3]).unsqueeze(1)   ###Semantic-aware Maps
        tscams2 = cams2 * feature_map2


        output_left=x_logits
        output_right=x_logits2
        _value1, preds_fianl1 = output_left.max(1)
        _value2, preds_fianl2 = output_right.max(1)
        # a=cams[0]
        CAM1=[]
        CAM2 = []

        for i in range(b):

            CAMs_l = returnCAM_att2(tscams[i],preds_fianl1[i])
            CAMs_r = returnCAM_att2(tscams2[i],preds_fianl2[i])
            CAM1.append(CAMs_l)
            CAM2.append(CAMs_r)
        return output_left,output_right,CAM1,CAM2





def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def make_layers_VGG_vit(cfg, batch_norm=False):
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


def make_layers_VGG_vit_FZFX(cfg, batch_norm=False):
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



def VGG_vit_model():
    return VGG_vit(make_layers_VGG_vit(cfg['D'], batch_norm=True))


def VGG_vit_model_FZFX():
    return VGG_vit_FZFX(make_layers_VGG_vit_FZFX(cfg['D'], batch_norm=True))
