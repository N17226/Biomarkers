
import os
from PIL import Image
# from data.transform import get_test_transform
from tqdm import tqdm
import torch.nn as nn
import pickle
import numpy as np
from models.vgg import *
import cfg
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # print(checkpoint)
    net= vgg16_bn()
    # model = checkpoint['model']  # 提取网络结构
    # print(net)
    net.load_state_dict(checkpoint)  # 加载网络权重参数
    for parameter in net.parameters():
        parameter.requires_grad = False
    net.eval()
    return net


n_features =1024
def save_feature(model, feature_path, label_path):

    net = load_checkpoint(model)

    # print(model)
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        net.cuda()
    ## 特征的维度需要自己根据特定的模型调整，我这里采用的是哪一个我也忘了

    features = np.empty((len(imgs), n_features))
    # print(features)
    labels = []
    feature_all=[]
    fea=[]
    mm=0
    for i in tqdm(range(len(imgs))):
        mm=mm+1
        img_path = imgs[i].strip()
        case=img_path
        file_path=case
        (filepath, tempfilename) = os.path.split(file_path)
        (filename, extension) = os.path.splitext(tempfilename)

        h5f = h5py.File(img_path, 'r')
        # print(h5f)
        LOW_ENERGY_CCL = h5f['LOW_ENERGY_CCL'][:].astype(np.float32)
        RECOMBINED_CCL = h5f['RECOMBINED_CCL'][:].astype(np.float32)
        LOW_ENERGY_CCR = h5f['LOW_ENERGY_CCR'][:].astype(np.float32)
        RECOMBINED_CCR = h5f['RECOMBINED_CCR'][:].astype(np.float32)
        LOW_ENERGY_MLOL = h5f['LOW_ENERGY_MLOL'][:].astype(np.float32)
        RECOMBINED_MLOL = h5f['RECOMBINED_MLOL'][:].astype(np.float32)
        LOW_ENERGY_MLOR = h5f['LOW_ENERGY_MLOR'][:].astype(np.float32)
        RECOMBINED_MLOR = h5f['RECOMBINED_MLOR'][:].astype(np.float32)
        label1 = h5f['label1'][()]#ki67
        label2 = h5f['label2'][()]#er
        label3 = h5f['label3'][()]#pr
        label4 = h5f['label4'][()]#her2

        LOW_ENERGY_CCL = cv2.resize(LOW_ENERGY_CCL, [256, 512])
        # print(LOW_ENERGY)
        RECOMBINED_CCL = cv2.resize(RECOMBINED_CCL, [256, 512])
        LOW_ENERGY_CCR = cv2.resize(LOW_ENERGY_CCR, [256, 512])
        # print(LOW_ENERGY)
        RECOMBINED_CCR = cv2.resize(RECOMBINED_CCR, [256, 512])
        LOW_ENERGY_MLOL = cv2.resize(LOW_ENERGY_MLOL, [256, 512])
        # print(LOW_ENERGY)
        RECOMBINED_MLOL = cv2.resize(RECOMBINED_MLOL, [256, 512])
        LOW_ENERGY_MLOR = cv2.resize(LOW_ENERGY_MLOR, [256, 512])
        # print(LOW_ENERGY)
        RECOMBINED_MLOR = cv2.resize(RECOMBINED_MLOR, [256, 512])
        # LOW_ENERGY_CCL=get_test_transform(size=cfg.INPUT_SIZE)(LOW_ENERGY_CCL).unsqueeze(0)
        LOW_ENERGY_CCL=torch.from_numpy(LOW_ENERGY_CCL).unsqueeze(0)
        RECOMBINED_CCL=torch.from_numpy(RECOMBINED_CCL).unsqueeze(0)
        LOW_ENERGY_CCR=torch.from_numpy(LOW_ENERGY_CCR).unsqueeze(0)
        RECOMBINED_CCR=torch.from_numpy(RECOMBINED_CCR).unsqueeze(0)
        LOW_ENERGY_MLOL=torch.from_numpy(LOW_ENERGY_MLOL).unsqueeze(0)
        RECOMBINED_MLOL=torch.from_numpy(RECOMBINED_MLOL).unsqueeze(0)
        RECOMBINED_MLOR=torch.from_numpy(RECOMBINED_MLOR).unsqueeze(0)
        LOW_ENERGY_MLOR=torch.from_numpy(LOW_ENERGY_MLOR).unsqueeze(0)

        YY_CCL = torch.cat((LOW_ENERGY_CCL, RECOMBINED_CCL), dim=0).unsqueeze(0)
        YY_CCR = torch.cat((LOW_ENERGY_CCR, RECOMBINED_CCR), dim=0).unsqueeze(0)
        YY_MLOL = torch.cat((LOW_ENERGY_MLOL, RECOMBINED_MLOL), dim=0).unsqueeze(0)
        YY_MLOR = torch.cat((LOW_ENERGY_MLOR, RECOMBINED_MLOR), dim=0).unsqueeze(0)


        YY_CCL = YY_CCL.cuda()
        YY_CCR = YY_CCR.cuda()
        YY_MLOL = YY_MLOL.cuda()
        YY_MLOR = YY_MLOR.cuda()
        # label1=label1.cuda()

        with torch.no_grad():

            output_cc= net.features(YY_CCL)
            output_mlo=net.features(YY_MLOL)
            output_cc=net.avgpool(output_cc)
            output_mlo = net.avgpool(output_mlo)
            output_cc=output_cc.view(output_cc.size(0), -1)
            output_mlo = output_mlo.view(output_mlo.size(0), -1)



            feature_m = torch.cat((output_cc,output_mlo), dim=1).squeeze(0)

        features[i, :] = feature_m.cpu().numpy()
        fea=features
        labels.append(label1)

    pickle.dump(fea, open(feature_path, 'wb'))
    pickle.dump(labels, open(label_path, 'wb'))
    print('CNN features obtained and saved.')




if __name__ == "__main__":
    # #构建保存特征的文件夹
    feature_path = './features_1024/'
    os.makedirs(feature_path, exist_ok=True)


#############################################################
    #### save training feature
    with open(cfg.TRAIN_LABEL_DIR, 'r')as f:
        imgs = f.readlines()
        # print(imgs)
    train_feature_path = feature_path + 'psdufeature.pkl'
    train_label_path = feature_path + 'psdulabel.pkl'
    cnn_model = cfg.TRAINED_MODEL
    save_feature(cnn_model, train_feature_path, train_label_path)
    #

    with open(cfg.VAL_LABEL_DIR, 'r')as f:
        imgs = f.readlines()
    test_feature_path = feature_path + 'validfeature.pkl'
    test_id_path = feature_path + 'validid.pkl'
    cnn_model = cfg.TRAINED_MODEL
    save_feature(cnn_model, test_feature_path, test_id_path)


    with open(cfg.TEST_LABEL_DIR, 'r')as f:
        imgs = f.readlines()
    test_feature_path = feature_path + 'testfeature.pkl'
    test_id_path = feature_path + 'testid.pkl'
    cnn_model = cfg.TRAINED_MODEL
    save_feature(cnn_model, test_feature_path, test_id_path)
