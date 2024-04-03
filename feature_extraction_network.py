

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.nn import CosineEmbeddingLoss

from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from models.vgg import *
from conf import settings
from models.vision_transformer import VGG_vit_model_FZFX, VGG_vit_model
from models.xception import xception
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights,ContrastiveLoss
from dataset import CESM, CESM_FZFX


def train(epoch):
    start = time.time()
    train_loss = 0.0 # cost function error
    correct = 0.0

    pro_train=[]
    lab_train=[]
    # correct2 = 0.0
    # correct3 = 0.0
    # correct4 = 0.0
    net.train()
    net2.eval()
    # net3.train()
    for i, x in enumerate(CESM_10_train_l):

        YY1 = x['LOW_ENERGY_CCL']
        YY2 = x['RECOMBINED_CCL']
        YY3 = x['LOW_ENERGY_CCR']
        YY4 = x['RECOMBINED_CCR']
        YY5 = x['LOW_ENERGY_MLOL']
        YY6 = x['RECOMBINED_MLOL']
        YY7 = x['LOW_ENERGY_MLOR']
        YY8 = x['RECOMBINED_MLOR']
        labels1 = x['label1']#ki67 status
        labels2 = x['label2']#er status
        labels3 = x['label3']#pr status
        labels4 = x['label4']#her2 status
        labels5 = x['label5']#TNBC

        YY_CCL = torch.cat((YY1, YY2), dim=1)
        # YY_CCL = torch.cat((YY_CCL, YY2), dim=1)
        YY_CCR = torch.cat((YY3, YY4), dim=1)
        YY_MLOL = torch.cat((YY5, YY6), dim=1)
        # YY_MLOL = torch.cat((YY_MLOL, YY6), dim=1)
        YY_MLOR = torch.cat((YY7, YY8), dim=1)

        labels1 = torch.IntTensor(labels1).to(torch.long)
        labels2 = torch.IntTensor(labels2).to(torch.long)
        labels3 = torch.IntTensor(labels3).to(torch.long)
        labels4 = torch.IntTensor(labels4).to(torch.long)
        labels5 = torch.IntTensor(labels5).to(torch.long)

        if args.gpu:
            # labels = labels.cuda()
            # images = images.cuda()
            YY_CCL = YY_CCL.cuda()
            YY_CCR = YY_CCR.cuda()
            YY_MLOL = YY_MLOL.cuda()
            YY_MLOR = YY_MLOR.cuda()
            labels1 = labels1.cuda()
            labels2 = labels2.cuda()
            labels3 = labels3.cuda()
            labels4 = labels4.cuda()
            labels5 = labels5.cuda()
        optimizer.zero_grad()
        out, outputs_CCL, outputs_MLOL= net(YY_CCL, YY_MLOL, 2)
        net2.eval()
        ou_CCL,ou_CCR,outputs_CCL2,CAM_CCR= net2(YY_CCL,YY_CCR,2)
        ou_MLOL,ou_MLOR,outputs_MLOL2,CAM_MLOR  = net2(YY_MLOL,YY_MLOR,2)

        outputs_CCL = torch.cat((outputs_CCL[0].unsqueeze(0), outputs_CCL[1].unsqueeze(0)), dim=0)
        outputs_MLOL  = torch.cat((outputs_MLOL[0].unsqueeze(0), outputs_MLOL[1].unsqueeze(0)), dim=0)
        outputs_CCL2 = torch.cat((outputs_CCL2[0].unsqueeze(0), outputs_CCL2[1].unsqueeze(0)), dim=0).cuda()
        outputs_MLOL2  = torch.cat((outputs_MLOL2[0].unsqueeze(0), outputs_MLOL2[1].unsqueeze(0)), dim=0).cuda()

        resize_all=transforms.Resize((512,256))
        outputs_CCL=resize_all(outputs_CCL)
        outputs_MLOL = resize_all(outputs_MLOL)

        loss_mse1= loss_function2(outputs_CCL,outputs_CCL2)
        loss_mse2 = loss_function2(outputs_MLOL, outputs_MLOL2)

        loss_mse =2*(loss_mse1 + loss_mse2)

        loss_ce = loss_function(out, labels1)

        loss = loss_ce + loss_mse

        print('loss_ce:{},loss_mse:{}'.format(
            loss_ce.item(),loss_mse.item()
        )

        )


        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = out.max(1)
        correct += preds.eq(labels1).sum()
        sot1 = torch.softmax(out, dim=1)
        lab_train.append(labels1[0].cpu().item())
        lab_train.append(labels1[1].cpu().item())
        pro1 = torch.index_select(sot1.cpu(), dim=1, index=torch.tensor(1))
        pro_train.append(pro1[0].cpu().item())
        pro_train.append(pro1[1].cpu().item())

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            # correct.float() / len(CESMdata),
            epoch=epoch,
            trained_samples=i * args.b + len(YY_CCL),
            # total_samples=len(CESMdata)
            total_samples=len(train_iter)
            # total_samples=600
        ))

        # writer.add_scalar('Test/Average loss', correct.float() , n_iter)
        if epoch <= args.warm:
            warmup_scheduler.step()
    auc_score = roc_auc_score(lab_train, pro_train)
    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
    print('Average loss: {:.4f}\tAccuarcy: {:0.6f}\tAUC: {:0.6f}'.format(
        train_loss / len(train_iter),
        # train_loss / 600,
        # correct.float() / (2*len(CESMdata))
        correct.float() /len(train_iter),
        auc_score
        ))



@torch.no_grad()
def eval_VALID(epoch=0, tb=True):

    start = time.time()
    net.eval()
    # net2.eval()
    # net3.eval()
    valid_loss = 0.0  # cost function error
    correct = 0.0
    correct2=0.0
    pro=[]
    label=[]
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
        labels3 = x['label3']
        labels4 = x['label4']
        labels5 = x['label5']

        YY_CCL = torch.cat((YY1, YY2), dim=1)
        YY_CCR = torch.cat((YY3, YY4), dim=1)
        YY_MLOL = torch.cat((YY5, YY6), dim=1)
        YY_MLOR = torch.cat((YY7, YY8), dim=1)

        labels1 = torch.IntTensor(labels1).to(torch.long)
        labels2 = torch.IntTensor(labels2).to(torch.long)
        labels3 = torch.IntTensor(labels3).to(torch.long)
        labels4 = torch.IntTensor(labels4).to(torch.long)
        labels5 = torch.IntTensor(labels5).to(torch.long)

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
            labels5 = labels5.cuda()
        optimizer.zero_grad()

        out, outputs_CCL, outputs_MLOL = net(YY_CCL, YY_MLOL, 2)
        loss1 = loss_function(out, labels1)
        loss = loss1
        valid_loss += loss.item()
        _, preds = out.max(1)
        correct += preds.eq(labels1).sum()
        sot1 = torch.softmax(out, dim=1)
        label.append(labels1[0].cpu().item())
        label.append(labels1[1].cpu().item())
        pro1 = torch.index_select(sot1.cpu(), dim=1, index=torch.tensor(1))
        pro.append(pro1[0].cpu().item())
        pro.append(pro1[1].cpu().item())
    finish = time.time()
    auc_score = roc_auc_score(label, pro)
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('VALID set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f},AUC: {:.4f},Time consumed:{:.2f}s'.format(
        epoch,
        valid_loss / len(valid_iter),
        correct.float() / (len(valid_iter)),
        auc_score,
        finish - start
    ))
    print()

    return valid_loss

if __name__ == '__main__':


    p = 1

    CESMdata = CESM_FZFX(base_dir=r'path to ki67\er\pr\her2 developing dataset',transform=transforms.Compose([
                       # Random表示有可能做，所以也可能不做
                       transforms.RandomHorizontalFlip(p=0.5),# 水平翻转
                       transforms.RandomVerticalFlip(p=0.5), # 上下翻转
                       transforms.RandomRotation(10), # 随机旋转-15°~15°
                       # transforms.RandomRotation([90, 180]), # 随机在90°、180°中选一个度数来旋转，如果想有一定概率不旋转，可以加一个0进去
                       # transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.3)
                       # transforms.Normalize(0, 1)
                       transforms.ToTensor(),
                       # transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
                   ]))  # , transform=ToTensor())

    labe=[]
    for Y in range(len(CESMdata)):
        CESM_label = CESMdata[Y]['label1']
        labe.append(CESM_label)
        #KI67
    kf = StratifiedKFold(n_splits=5, shuffle=True)

    #
    iterara=1

    for train_idxs, valid_idxs in kf.split(CESMdata, labe):
        num_T = 0
        num_V = 0
        # train_iter, valid_iter = [], []
        train_iter = torch.utils.data.Subset(CESMdata, train_idxs)
        valid_iter = torch.utils.data.Subset(CESMdata, valid_idxs)







        parser = argparse.ArgumentParser()
        # parser.add_argument('-net', type=str, required=True, help='net type')
        parser.add_argument('-net', type=str, required=False,default='VGG16', help='net type')
        parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
        parser.add_argument('-b', type=int, default=2, help='batch size for dataloader')
        parser.add_argument('-warm', type=int, default=0, help='warm up training phase')
        parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate')
        parser.add_argument('-resume', action='store_true', default=False, help='resume training')
        args = parser.parse_args()


        CESM_10_train_l = DataLoader(train_iter,batch_size=args.b,shuffle=True,drop_last=False)
        CESM_10_valid_l = DataLoader(valid_iter, batch_size=args.b,shuffle=False,drop_last=False)
        # print(CESM_10_train_l.dataset)
        # weight1 = [1 / 161 if train_iter[i]['label1'] == 0 else 1 / 795 for i in range(len(train_iter))]
        # sampler = WeightedRandomSampler(weights=weight1,num_samples=340)###weighted sampler is used
        # CESM_10_train_l = DataLoader(train_iter,batch_size=args.b,sampler=sampler,shuffle=False,drop_last=False)



        net = vgg16_bn()
        net = net.cuda()
        net2 = VGG_vit_model_FZFX()

        path2 = r''
###########models from weakly supervised network
        net2.load_state_dict(torch.load(path2), strict=True)
        net2 = net2.cuda()

        for param2 in net2.parameters():
            param2.requires_grad_(False)

        loss_function = nn.CrossEntropyLoss()
        loss_function.cuda()
        loss_function2 = nn.SmoothL1Loss()
        loss_function2.cuda()
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.5) #learning rate decay
        iter_per_epoch = len(CESM_10_train_l)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

        if args.resume:
            recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
            if not recent_folder:
                raise Exception('no recent folder were found')

            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

        else:
            # checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

            checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW,str(p))
            # print(settings.TIME_NOW)
            p = p + 1


        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

        best_model = 100000.0
        if args.resume:
            best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
            if best_weights:
                weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
                print('found best acc weights file:{}'.format(weights_path))
                print('load best training file to test acc...')
                net.load_state_dict(torch.load(weights_path))
                best_acc = eval_VALID(tb=False)
                print('best acc is {:0.2f}'.format(best_acc))

            recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
            if not recent_weights_file:
                raise Exception('no recent weights file were found')
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
            print('loading weights file {} to resume training.....'.format(weights_path))
            net.load_state_dict(torch.load(weights_path))

            resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


        for epoch in range(1, settings.EPOCH1 + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)

            if args.resume:
                if epoch <= resume_epoch:
                    continue


            train(epoch)
            VLAID_LOSS= eval_VALID(epoch)



            if epoch < settings.MILESTONES[3] and best_model > VLAID_LOSS:
            # if epoch < settings.MILESTONES[3] and best_acc2 > acc2 and epoch >= 1:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best_loss')

                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)

                best_model = VLAID_LOSS
                set = epoch

            if  epoch % settings.SAVE_EPOCH ==0:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                # weights_path2 = checkpoint_path.format(net='net2', epoch=epoch, type='regular')
                # weights_path3 = checkpoint_path.format(net='net3', epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.state_dict(), weights_path)            # torch.save(net2.state_dict(), weights_path2)
                # torch.save(net3.state_dict(), weights_path3)

        print(set)

        iterara= iterara + 1
        print(iterara)
        if iterara==2 or iterara==3 or iterara==4 or iterara==5:
            epoch=set
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best_loss')
            # print(checkpoint_path)
            pa=os.getcwd()
            path=os.path.join(pa,weights_path)
            # print(weights_path)
            print(path)
            net.load_state_dict(torch.load(path),strict=True)


