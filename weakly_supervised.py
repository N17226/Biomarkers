
import os
import sys
import argparse
import time

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from medpy import metric
from torch.nn import CosineEmbeddingLoss

from torch.utils.data import DataLoader
from models.vgg import *
from conf import settings
from models.vision_transformer import VGG_vit_model
from utils import  WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights,ContrastiveLoss
from dataset import CESM_dice, CESM_cls


def Thresh_and_blur(img):  #设定阈值
    # blurred = cv2.GaussianBlur(gradient, (13, 13),0)
    # (_, thresh) = cv2.threshold(blurred, gradient.max()-30, 1, cv2.THRESH_BINARY)
    # athdMEAN = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 1)
    ret, otsu = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    ret2, dst = cv2.threshold(img, 0.6 * ret, 255, cv2.THRESH_BINARY)
    # print(ret)
    return dst


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

class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        # pt = torch.sigmoid(_input)
        pt = torch.softmax(_input,dim=1)
        eps = 1e-7
        # print(pt)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt+eps) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt+eps)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

def train(epoch):
    start = time.time()
    train_loss = 0.0 # cost function error
    correct = 0.0
    net.train()
    p=0
    for i, x in enumerate(CESM_10_train_l):

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


        seed = np.random.randint(low=0,high=500000)
        transforms1=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 上下翻
        transforms.RandomRotation(10)
                                        ])# 随机旋转-10°~10°

        torch.manual_seed(seed)
            # print(seed)
        YY1_trans = transforms1(YY1)
        torch.manual_seed(seed)
        YY2_trans = transforms1(YY2)
        torch.manual_seed(seed)
        YY3_trans = transforms1(YY3)
        torch.manual_seed(seed)
        YY4_trans = transforms1(YY4)
        torch.manual_seed(seed)
        YY5_trans = transforms1(YY5)
        torch.manual_seed(seed)
        YY6_trans = transforms1(YY6)
        torch.manual_seed(seed)
        YY7_trans = transforms1(YY7)
        torch.manual_seed(seed)
        YY8_trans = transforms1(YY8)


        YY_CCL = torch.cat((YY1, YY2), dim=1)
        YY_CCR = torch.cat((YY3, YY4), dim=1)
        YY_MLOL = torch.cat((YY5, YY6), dim=1)
        YY_MLOR = torch.cat((YY7, YY8), dim=1)


        YY_CCL_trans = torch.cat((YY1_trans, YY2_trans), dim=1)
        YY_CCR_trans = torch.cat((YY3_trans, YY4_trans), dim=1)
        YY_MLOL_trans = torch.cat((YY5_trans, YY6_trans), dim=1)
        YY_MLOR_trans = torch.cat((YY7_trans, YY8_trans), dim=1)


        labels1 = torch.IntTensor(labels1).to(torch.long)
        labels2 = torch.IntTensor(labels2).to(torch.long)


        if args.gpu:

            YY_CCL = YY_CCL.cuda()
            YY_CCR = YY_CCR.cuda()
            YY_MLOL = YY_MLOL.cuda()
            YY_MLOR = YY_MLOR.cuda()
            YY_CCL_trans = YY_CCL_trans.cuda()
            YY_CCR_trans = YY_CCR_trans.cuda()
            YY_MLOL_trans = YY_MLOL_trans.cuda()
            YY_MLOR_trans = YY_MLOR_trans.cuda()
            labels1=labels1.cuda()
            labels2 = labels2.cuda()
            # labels3 = labels3.cuda()
            # labels4 = labels4.cuda()
        optimizer.zero_grad()

        ou_CCL,ou_CCR,CAM_CCL,CAM_CCR= net(YY_CCL,YY_CCR,2)
        ou_MLOL,ou_MLOR,CAM_MLOL,CAM_MLOR  = net(YY_MLOL,YY_MLOR,2)

        map1_CCL, map2_CCL = CAM_CCL[0],CAM_CCL[1]
        map1_CCR, map2_CCR = CAM_CCR[0],CAM_CCR[1]
        map1_MLOL, map2_MLOL = CAM_MLOL[0],CAM_MLOL[1]
        map1_MLOR, map2_MLOR = CAM_MLOR[0],CAM_MLOR[1]

        ou_CCL_trans,ou_CCR_trans,CAM_CCL_trans,CAM_CCR_trans = net(YY_CCL_trans,YY_CCR_trans,2)
        # ou_CCR_trans,CAM_CCR_trans = net(YY_CCR_trans,YY_CCL_trans,2)
        ou_MLOL_trans,ou_MLOR_trans,CAM_MLOL_trans,CAM_MLOR_trans = net(YY_MLOL_trans,YY_MLOR_trans,2)
        # ou_MLOR_trans,CAM_MLOR_trans = net(YY_MLOR_trans,YY_MLOL_trans,2)

        map1_CCL_trans, map2_CCL_trans = CAM_CCL_trans[0],CAM_CCL_trans[1]
        map1_CCR_trans, map2_CCR_trans = CAM_CCR_trans[0], CAM_CCR_trans[1]
        map1_MLOL_trans, map2_MLOL_trans = CAM_MLOL_trans[0] ,CAM_MLOL_trans[1]
        map1_MLOR_trans, map2_MLOR_trans = CAM_MLOR_trans[0],CAM_MLOR_trans[1]


        map_CCL = torch.cat((map1_CCL.unsqueeze(0), map2_CCL.unsqueeze(0)), dim=0)
        map_CCR = torch.cat((map1_CCR.unsqueeze(0), map2_CCR.unsqueeze(0)), dim=0)
        map_MLOL = torch.cat((map1_MLOL.unsqueeze(0), map2_MLOL.unsqueeze(0)), dim=0)
        map_MLOR = torch.cat((map1_MLOR.unsqueeze(0), map2_MLOR.unsqueeze(0)), dim=0)


        map_CCL_trans = torch.cat((map1_CCL_trans.unsqueeze(0), map2_CCL_trans.unsqueeze(0)), dim=0)
        map_CCR_trans = torch.cat((map1_CCR_trans.unsqueeze(0), map2_CCR_trans.unsqueeze(0)), dim=0)
        map_MLOL_trans = torch.cat((map1_MLOL_trans.unsqueeze(0), map2_MLOL_trans.unsqueeze(0)), dim=0)
        map_MLOR_trans = torch.cat((map1_MLOR_trans.unsqueeze(0), map2_MLOR_trans.unsqueeze(0)), dim=0)

        resize_all=transforms.Resize((512,256))

        map_CCL_trans = resize_all(map_CCL_trans)
        map_CCR_trans= resize_all(map_CCR_trans)
        map_MLOL_trans= resize_all(map_MLOL_trans)
        map_MLOR_trans= resize_all(map_MLOR_trans)

        torch.manual_seed(seed)
        map_CCL = transforms1(map_CCL)
        map_CCL = resize_all(map_CCL)
        torch.manual_seed(seed)
        map_CCR = transforms1(map_CCR)
        map_CCR = resize_all(map_CCR)
        torch.manual_seed(seed)
        map_MLOL = transforms1(map_MLOL)
        map_MLOL = resize_all(map_MLOL)
        torch.manual_seed(seed)
        map_MLOR= transforms1(map_MLOR)
        map_MLOR = resize_all(map_MLOR)
        torch.manual_seed(seed)


        loss1 = loss_function(ou_CCL_trans, labels1)
        loss3 = loss_function(ou_MLOL_trans, labels1)
        loss2 = loss_function(ou_CCR_trans, labels2)
        loss4 = loss_function(ou_MLOR_trans, labels2)

        loss11 = loss_function(ou_CCL, labels1)
        loss31 = loss_function(ou_MLOL, labels1)
        loss21 = loss_function(ou_CCR, labels2)
        loss41 = loss_function(ou_MLOR, labels2)



        loss1_mse = loss_function2(map_CCL, map_CCL_trans)
        loss3_mse = loss_function2(map_MLOL, map_MLOL_trans)
        loss2_mse = loss_function2(map_CCR, map_CCR_trans)
        loss4_mse = loss_function2(map_MLOR, map_MLOR_trans)

        if epoch <10:
            alpha = 1
        elif epoch >=10 and epoch <30:
                alpha = 5
        else:
            alpha = 10
        loss_ce = 0.5*(loss1 + loss2 + loss3 + loss4 +loss11+loss21 + loss31 + loss41)
        # loss_ce = loss1 + loss2 + loss3 + loss4
        loss_mse = alpha*(loss1_mse + loss2_mse + loss3_mse + loss4_mse)
        loss = loss_ce + loss_mse

        print('loss_ce:{},loss_mse:{}'.format(
            loss_ce.item(),
            loss_mse.item()
        )
        )


        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = ou_CCL_trans.max(1)
        _2, preds3 = ou_MLOL_trans.max(1)
        _1, preds2 = ou_CCR_trans.max(1)
        _3, preds4 = ou_MLOR_trans.max(1)




        correct += preds.eq(labels1).sum()
        correct += preds3.eq(labels1).sum()
        correct += preds2.eq(labels2).sum()
        correct += preds4.eq(labels2).sum()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            # correct.float() / len(CESMdata),
            epoch=epoch,
            trained_samples=i * args.b + len(YY_CCL),
            total_samples=(len(CESMdata))
        ))


        # writer.add_scalar('Test/Average loss', correct.float() , n_iter)
        if epoch <= args.warm:
            warmup_scheduler.step()


    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    print('Average loss: {:.4f}\tAccuarcy: {:0.6f}'.format(
        train_loss / (4.0*len(CESMdata)),
        correct.float() / (4*len(CESMdata))
        ))


@torch.no_grad()
def eval_cls(epoch=0):

    start = time.time()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    correct2=0.0
    pro=[]
    label=[]
    # class_correct = list(0. for i in range(2))
    # class_total = list(0. for i in range(2))
    for i, x in enumerate(CESM_10_valid_l):


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
            # labels3 = labels3.cuda()
            # labels4 = labels4.cuda()
            # labels5 = labels5.cuda()
        optimizer.zero_grad()
        ou_CCL,ou_CCR,CAM_CCL,CAM_CCR = net(YY_CCL,YY_CCR,2)
        # ou_CCR,CAM_CCR= net(YY_CCR,YY_CCL,2)
        ou_MLOL,ou_MLOR,CAM_MLOL,CAM_MLOR = net(YY_MLOL,YY_MLOR,2)
        # ou_MLOR,CAM_MLOR = net(YY_MLOR,YY_MLOL,2)


        _, preds = ou_CCL.max(1)
        _2, preds3 = ou_MLOL.max(1)
        _1, preds2 = ou_CCR.max(1)
        _3, preds4 = ou_MLOR.max(1)

        correct += preds.eq(labels1).sum()
        correct += preds3.eq(labels1).sum()
        correct += preds2.eq(labels2).sum()
        correct += preds4.eq(labels2).sum()


    finish = time.time()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Accuracy: {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        correct.float() / (4 * len(CESMdata2)),
        finish - start
    ))


@torch.no_grad()
def eval_seg(epoch=0):

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

    for i, x in enumerate(CESM_10_dice_l):


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
        optimizer.zero_grad()
        ou_CCL,ou_CCR,CAM_CCL,CAM_CCR = net(YY_CCL,YY_CCR,2)
        ou_MLOL,ou_MLOR,CAM_MLOL,CAM_MLOR = net(YY_MLOL,YY_MLOR,2)

###########malignant lesions in one side  (in prepocess_seg.py)
        map1_CCL, map2_CCL = CAM_CCL[0],CAM_CCL[1]
        map1_MLOL, map2_MLOL = CAM_MLOL[0],CAM_MLOL[1]

        map_CCL = torch.cat((map1_CCL.unsqueeze(0), map2_CCL.unsqueeze(0)), dim=0)
        map_MLOL = torch.cat((map1_MLOL.unsqueeze(0), map2_MLOL.unsqueeze(0)), dim=0)

        resize_all=transforms.Resize((512,256))

        map_CCL = resize_all(map_CCL)
        map_MLOL= resize_all(map_MLOL)

        map_CCL1= Thresh_and_blur(np.uint8(255*map_CCL[0].cpu()))
        map_CCL1 = (map_CCL1 - map_CCL1.min()) / max((map_CCL1.max() - map_CCL1.min()),1)
        map_CCL1= torch.tensor(map_CCL1)

        map_CCL2= Thresh_and_blur(np.uint8(255*map_CCL[1].cpu()))
        map_CCL2 = (map_CCL2 - map_CCL2.min()) / max((map_CCL2.max() - map_CCL2.min()),1)
        map_CCL2= torch.tensor(map_CCL2)

        map_MLOL1= Thresh_and_blur(np.uint8(255*map_MLOL[0].cpu()))
        map_MLOL1 = (map_MLOL1 - map_MLOL1.min()) / max((map_MLOL1.max() - map_MLOL1.min()),1)
        map_MLOL1= torch.tensor(map_MLOL1)

        map_MLOL2= Thresh_and_blur(np.uint8(255*map_MLOL[1].cpu()))
        map_MLOL2 = (map_MLOL2 - map_MLOL2.min()) / max((map_MLOL2.max() - map_MLOL2.min()),1)
        map_MLOL2= torch.tensor(map_MLOL2)

        CC_dice1 = calculate_metric_percase(map_CCL1.cpu().numpy(),labels3[0].cpu())
        CC_dice2= calculate_metric_percase(map_CCL2.cpu().numpy(), labels3[1].cpu())
        MLO_dice1 = calculate_metric_percase(map_MLOL1.cpu().numpy(), labels4[0].cpu())
        MLO_dice2 = calculate_metric_percase(map_MLOL2.cpu().numpy(), labels4[1].cpu())

        CC_TPR1 = calculate_metric_TPR(map_CCL1.cpu().numpy(),labels3[0].cpu())
        CC_TPR2= calculate_metric_TPR(map_CCL2.cpu().numpy(), labels3[1].cpu())
        MLO_TPR1 = calculate_metric_TPR(map_MLOL1.cpu().numpy(), labels4[0].cpu())
        MLO_TPR2 = calculate_metric_TPR(map_MLOL2.cpu().numpy(), labels4[1].cpu())

        CC_PPV1 = calculate_metric_PPV(map_CCL1.cpu().numpy(),labels3[0].cpu())
        CC_PPV2= calculate_metric_PPV(map_CCL2.cpu().numpy(), labels3[1].cpu())
        MLO_PPV1 = calculate_metric_PPV(map_MLOL1.cpu().numpy(), labels4[0].cpu())
        MLO_PPV2 = calculate_metric_PPV(map_MLOL2.cpu().numpy(), labels4[1].cpu())

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

    print(correct / (2* len(CESMdata3)),'DICE')
    print(correct2 / (2 * len(CESMdata3)),'TPR')
    print(correct3 / (2 * len(CESMdata3)),'PPV')

    return correct / (2* len(CESMdata3))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=2, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = VGG_vit_model()
    net = net.cuda()

    CESMdata = CESM_cls(base_dir=r'path to training dataset with image-level annonations:',transform=transforms.Compose([
                       # Random表示有可能做，所以也可能不做
                       # transforms.RandomHorizontalFlip(p=0.5),# 水平翻转
                       # transforms.RandomVerticalFlip(p=0.5), # 上下翻转
                       # transforms.RandomRotation(10), # 随机旋转-15°~15°
                       # transforms.RandomRotation([90, 180]), # 随机在90°、180°中选一个度数来旋转，如果想有一定概率不旋转，可以加一个0进去
                       # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3)
                       # transforms.Normalize(0, 1)
                       transforms.ToTensor(),
                       # transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
                   ]))  # , transform=ToTensor())

    CESM_10_train_l = DataLoader(CESMdata, batch_size=args.b, shuffle=True, drop_last=True,
                                 pin_memory=torch.cuda.is_available())

######training dataset with image-level annonations:cancer or non-cancer


    CESMdata2 = CESM_cls(base_dir=r'path to valid dataset with image-level annonations',transform=transforms.Compose([
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
    CESM_10_valid_l = DataLoader(CESMdata2, batch_size=args.b, shuffle=False, drop_last=True,
                                 pin_memory=torch.cuda.is_available())



    CESMdata3 = CESM_dice(base_dir=r'path to dataset with pixel-level annonations',transform=transforms.Compose([
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
    CESM_10_dice_l = DataLoader(CESMdata3, batch_size=args.b, shuffle=False, drop_last=False,
                                 pin_memory=torch.cuda.is_available())


    loss_function = nn.CrossEntropyLoss()
    loss_function.cuda()
    loss_function2 = nn.MSELoss()
    # loss_function2 = CosineEmbeddingLoss()
    loss_function2.cuda()



    optimizer = optim.SGD(net.parameters(),lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.5) #learning rate decay
    iter_per_epoch = len(CESM_10_train_l)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'VIT', recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, 'VIT', settings.TIME_NOW)


    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_dice = 0.0

    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, 'VIT', recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_cls()
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, 'VIT', recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, 'VIT', recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue


        train(epoch)
        acc= eval_cls(epoch)
        dice = eval_seg(epoch)


        if epoch < settings.MILESTONES[3] and best_dice < dice:
            weights_path = checkpoint_path.format(net='VIT', epoch=epoch, type='best_dice')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

            best_dice = dice


        if  epoch % settings.SAVE_EPOCH ==0:
            weights_path = checkpoint_path.format(net='VIT', epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)









