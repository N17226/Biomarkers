
from medpy import metric

from models.vision_transformer import VGG_vit_model_FZFX
import os
import sys
import argparse
import time

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from models.vgg import *
from conf import settings
from utils import WarmUpLR
from dataset import CESM_cls, CESM_dice


def calculate_metric_percase(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    gt = torch.squeeze(gt).numpy()
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        return dice
    else:
        return 0


def calculate_metric_TPR(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    gt = torch.squeeze(gt).numpy()
    if pred.sum() > 0:
        TPR = metric.binary.true_positive_rate(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        return TPR
    else:
        return 0


def calculate_metric_TNR(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    gt = torch.squeeze(gt).numpy()
    if pred.sum() > 0:
        TNR = metric.binary.true_negative_rate(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        return TNR
    else:
        return 0


def calculate_metric_PPV(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    gt = torch.squeeze(gt).numpy()
    if pred.sum() > 0:
        PPV = metric.binary.positive_predictive_value(pred, gt)
        # hd95 = metric.binary.hd95(pred, gt)
        return PPV
    else:
        return 0



def image_morphology(thresh):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=5)
    closed = cv2.dilate(closed, None, iterations=5)
    return closed


@torch.no_grad()
def evaluate():

    start = time.time()
    net.eval()
    # net2.eval()
    # net3.eval()
    test_loss = 0.0  # cost function error
    correct = 0.0
    correct2=0.0
    correct3 = 0.0
    pro=[]
    label=[]

    for i, x in enumerate(CESM_dice_lr):


        YY1 = x['LOW_ENERGY_CCL']
        YY2 = x['RECOMBINED_CCL']
        YY3 = x['LOW_ENERGY_CCR']
        YY4 = x['RECOMBINED_CCR']
        YY5 = x['LOW_ENERGY_MLOL']
        YY6 = x['RECOMBINED_MLOL']
        YY7 = x['LOW_ENERGY_MLOR']
        YY8 = x['RECOMBINED_MLOR']
        labels1 = x['label1']
        labels2 = x['label2']
        labels3 = x['label3']
        labels4 = x['label4']

        YY_CCL = torch.cat((YY1, YY2), dim=1)
        YY_CCR = torch.cat((YY3, YY4), dim=1)
        YY_MLOL = torch.cat((YY5, YY6), dim=1)
        YY_MLOR = torch.cat((YY7, YY8), dim=1)


        labels1 = torch.IntTensor(labels1).to(torch.long)
        labels2 = torch.IntTensor(labels2).to(torch.long)

        if args.gpu:
            # labels = labels.cuda()
            # images = images.cuda()
            YY_CCL = YY_CCL.cuda()
            YY_CCR = YY_CCR.cuda()
            YY_MLOL = YY_MLOL.cuda()
            YY_MLOR = YY_MLOR.cuda()
            labels1=labels1.cuda()
            labels2 = labels2.cuda()
            labels3 = labels3.cuda()
            labels4 = labels4.cuda()
            # labels5 = labels5.cuda()

        ou_CCL,ou_CCR,CAM_CCL,CAM_CCR = net(YY_CCL,YY_CCR,2)
        ou_MLOL,ou_MLOR,CAM_MLOL,CAM_MLOR = net(YY_MLOL,YY_MLOR,2)

###########malignant lesions in one side  (in prepocess_seg.py)
        map1_CCL, map2_CCL = CAM_CCL[0],CAM_CCL[1]
        map1_MLOL, map2_MLOL = CAM_MLOL[0],CAM_MLOL[1]


        CC_dice1 = calculate_metric_percase(map1_CCL.cpu().numpy(),labels3[0].cpu())
        CC_dice2= calculate_metric_percase(map2_CCL.cpu().numpy(), labels3[1].cpu())
        MLO_dice1 = calculate_metric_percase(map1_MLOL.cpu().numpy(), labels4[0].cpu())
        MLO_dice2 = calculate_metric_percase(map2_MLOL.cpu().numpy(), labels4[1].cpu())

        CC_TPR1 = calculate_metric_TPR(map1_CCL.cpu().numpy(),labels3[0].cpu())
        CC_TPR2= calculate_metric_TPR(map2_CCL.cpu().numpy(), labels3[1].cpu())
        MLO_TPR1 = calculate_metric_TPR(map1_MLOL.cpu().numpy(), labels4[0].cpu())
        MLO_TPR2 = calculate_metric_TPR(map2_MLOL.cpu().numpy(), labels4[1].cpu())

        CC_PPV1 = calculate_metric_PPV(map1_CCL.cpu().numpy(),labels3[0].cpu())
        CC_PPV2= calculate_metric_PPV(map2_CCL.cpu().numpy(), labels3[1].cpu())
        MLO_PPV1 = calculate_metric_PPV(map1_MLOL.cpu().numpy(), labels4[0].cpu())
        MLO_PPV2 = calculate_metric_PPV(map2_MLOL.cpu().numpy(), labels4[1].cpu())

        correct += CC_dice1
        correct += CC_dice2
        correct += MLO_dice1
        correct += MLO_dice2

        correct2 += CC_TPR1
        correct2 += CC_TPR2
        correct2 += MLO_TPR1
        correct2 += MLO_TPR2


        correct3 += CC_PPV1
        correct3 += CC_PPV2
        correct3 += MLO_PPV1
        correct3 += MLO_PPV2

    print(correct / (2* len(CESMdata)),'DICE')
    print(correct2 / (2 * len(CESMdata)),'TPR')
    print(correct3 / (2 * len(CESMdata)),'PPV')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=2, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()


    net = VGG_vit_model_FZFX()
    path1=r''
    net.load_state_dict(torch.load(path1), strict=False)
    net = net.cuda()
    for param in net.parameters():
        param.requires_grad_(False)
    CESMdata = CESM_dice(base_dir=r'',transform=transforms.Compose([
                       # Random表示有可能做，所以也可能不做
                       # transforms.RandomHorizontalFlip(p=0.5),# 水平翻转
                       # transforms.RandomVerticalFlip(p=0.5), # 上下翻转
                       # transforms.RandomRotation(10), # 随机旋转-15°~15°
                       # transforms.RandomRotation([90, 180]), # 随机在90°、180°中选一个度数来旋转，如果想有一定概率不旋转，可以加一个0进去
                       # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3)
                       # transforms.Normalize(0, 1)
                       transforms.ToTensor(),
                       # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                   ]))  # , transform=ToTensor())
    CESM_dice_lr = DataLoader(CESMdata, batch_size=args.b, shuffle=False, drop_last=False,
                                 pin_memory=torch.cuda.is_available())

    evaluate()
