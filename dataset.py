
import os
import sys
import pickle
from natsort import ns, natsorted
import matplotlib.pyplot
import pandas as pd
import xlrd
from skimage import io
import matplotlib.pyplot as plt
import numpy
import torch
from torch.utils.data import Dataset
import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image


class CESM_cls(Dataset):
    def __init__(self, base_dir=None, labeled_num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.targets = []
        # self.split = split
        self.transform = transform
        dir = os.listdir(self._base_dir)
        dir=natsorted(dir,alg=ns.PATH)
        # dir.sort()
        # print(dir)
        for name in dir:
            image = os.path.join(self._base_dir, name)
            # print(image)
            self.sample_list.append(image)


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')

        LOW_ENERGY_CCL = h5f['LOW_ENERGY_CCL'][:].astype(np.float32)
        RECOMBINED_CCL = h5f['RECOMBINED_CCL'][:].astype(np.float32)
        LOW_ENERGY_CCR = h5f['LOW_ENERGY_CCR'][:].astype(np.float32)
        RECOMBINED_CCR = h5f['RECOMBINED_CCR'][:].astype(np.float32)
        LOW_ENERGY_MLOL = h5f['LOW_ENERGY_MLOL'][:].astype(np.float32)
        RECOMBINED_MLOL = h5f['RECOMBINED_MLOL'][:].astype(np.float32)
        LOW_ENERGY_MLOR = h5f['LOW_ENERGY_MLOR'][:].astype(np.float32)
        RECOMBINED_MLOR = h5f['RECOMBINED_MLOR'][:].astype(np.float32)
        label1 = h5f['label1'][()]##left breast cancer or non-cancer
        label2 = h5f['label2'][()]##right breast cancer or non-cancer




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
        # print(type(RECOMBINED))

        seed = np.random.randint(500000)
        if self.transform:
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_CCL = Image.fromarray(np.uint8(LOW_ENERGY_CCL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_CCR= Image.fromarray(np.uint8(LOW_ENERGY_CCR * 255))
            torch.manual_seed(seed)
            RECOMBINED_CCL = Image.fromarray(np.uint8(255 * RECOMBINED_CCL))
            torch.manual_seed(seed)
            RECOMBINED_CCR = Image.fromarray(np.uint8(255*RECOMBINED_CCR))
            torch.manual_seed(seed)
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_MLOL = Image.fromarray(np.uint8(LOW_ENERGY_MLOL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR= Image.fromarray(np.uint8(LOW_ENERGY_MLOR * 255))
            torch.manual_seed(seed)
            RECOMBINED_MLOL = Image.fromarray(np.uint8(255 * RECOMBINED_MLOL))
            torch.manual_seed(seed)
            RECOMBINED_MLOR = Image.fromarray(np.uint8(255*RECOMBINED_MLOR))
            torch.manual_seed(seed)
            # print(seed)
            LOW_ENERGY_CCL = self.transform(LOW_ENERGY_CCL)

            torch.manual_seed(seed)
            RECOMBINED_CCL=self.transform(RECOMBINED_CCL)
            # RECOMBINED_L.show()
            torch.manual_seed(seed)
            LOW_ENERGY_CCR = self.transform(LOW_ENERGY_CCR)
            # LOW_ENERGY.show()
            # LOW_ENERGY_R.show()
            torch.manual_seed(seed)
            RECOMBINED_CCR = self.transform(RECOMBINED_CCR)
            # RECOMBINED_R.show()
            # seed = np.random.randint(2220)
            torch.manual_seed(seed)
            LOW_ENERGY_MLOL = self.transform(LOW_ENERGY_MLOL)
            # LOW_ENERGY_L.show()
            torch.manual_seed(seed)
            RECOMBINED_MLOL=self.transform(RECOMBINED_MLOL)
            # RECOMBINED_L.show()
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR = self.transform(LOW_ENERGY_MLOR)
            # LOW_ENERGY.show()
            # LOW_ENERGY_R.show()
            torch.manual_seed(seed)
            RECOMBINED_MLOR = self.transform(RECOMBINED_MLOR)
            # RECOMBINED_R.show()
            # seed = np.random.randint(2220)
            torch.manual_seed(seed)


        sample = {'LOW_ENERGY_CCL': LOW_ENERGY_CCL, 'RECOMBINED_CCL': RECOMBINED_CCL,'LOW_ENERGY_CCR': LOW_ENERGY_CCR,
                  'RECOMBINED_CCR': RECOMBINED_CCR,'LOW_ENERGY_MLOL': LOW_ENERGY_MLOL, 'RECOMBINED_MLOL': RECOMBINED_MLOL,
                  'LOW_ENERGY_MLOR': LOW_ENERGY_MLOR, 'RECOMBINED_MLOR': RECOMBINED_MLOR,
                  'label1': label1,'label2': label2,'case':case}

        return sample



class CESM_dice(Dataset):
    def __init__(self, base_dir=None, labeled_num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.targets = []
        # self.split = split
        self.transform = transform
        dir = os.listdir(self._base_dir)
        dir=natsorted(dir,alg=ns.PATH)
        # dir.sort()
        # print(dir)
        for name in dir:
            image = os.path.join(self._base_dir, name)
            # print(image)
            self.sample_list.append(image)


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')

        LOW_ENERGY_CCL = h5f['LOW_ENERGY_CCL'][:].astype(np.float32)
        RECOMBINED_CCL = h5f['RECOMBINED_CCL'][:].astype(np.float32)
        LOW_ENERGY_CCR = h5f['LOW_ENERGY_CCR'][:].astype(np.float32)
        RECOMBINED_CCR = h5f['RECOMBINED_CCR'][:].astype(np.float32)
        LOW_ENERGY_MLOL = h5f['LOW_ENERGY_MLOL'][:].astype(np.float32)
        RECOMBINED_MLOL = h5f['RECOMBINED_MLOL'][:].astype(np.float32)
        LOW_ENERGY_MLOR = h5f['LOW_ENERGY_MLOR'][:].astype(np.float32)
        RECOMBINED_MLOR = h5f['RECOMBINED_MLOR'][:].astype(np.float32)
        label1 = h5f['label1'][()]#
        label2 = h5f['label2'][()]#
        label3 = h5f['label3'][()].astype(np.uint8)#####CC manual segementation label
        label4 = h5f['label4'][()].astype(np.uint8)#####mlo manual segementation label




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
        # print(type(RECOMBINED))

        label3 = cv2.resize(label3, [256, 512])
        # print(LOW_ENERGY)
        label4 = cv2.resize(label4, [256, 512])


        seed = np.random.randint(500000)
        if self.transform:
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_CCL = Image.fromarray(np.uint8(LOW_ENERGY_CCL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_CCR= Image.fromarray(np.uint8(LOW_ENERGY_CCR * 255))
            torch.manual_seed(seed)
            RECOMBINED_CCL = Image.fromarray(np.uint8(255 * RECOMBINED_CCL))
            torch.manual_seed(seed)
            RECOMBINED_CCR = Image.fromarray(np.uint8(255*RECOMBINED_CCR))
            torch.manual_seed(seed)
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_MLOL = Image.fromarray(np.uint8(LOW_ENERGY_MLOL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR= Image.fromarray(np.uint8(LOW_ENERGY_MLOR * 255))
            torch.manual_seed(seed)
            RECOMBINED_MLOL = Image.fromarray(np.uint8(255 * RECOMBINED_MLOL))
            torch.manual_seed(seed)
            RECOMBINED_MLOR = Image.fromarray(np.uint8(255*RECOMBINED_MLOR))
            torch.manual_seed(seed)
            # print(seed)
            LOW_ENERGY_CCL = self.transform(LOW_ENERGY_CCL)

            torch.manual_seed(seed)
            RECOMBINED_CCL=self.transform(RECOMBINED_CCL)
            # RECOMBINED_L.show()
            torch.manual_seed(seed)
            LOW_ENERGY_CCR = self.transform(LOW_ENERGY_CCR)
            # LOW_ENERGY.show()
            # LOW_ENERGY_R.show()
            torch.manual_seed(seed)
            RECOMBINED_CCR = self.transform(RECOMBINED_CCR)
            # RECOMBINED_R.show()
            # seed = np.random.randint(2220)
            torch.manual_seed(seed)
            LOW_ENERGY_MLOL = self.transform(LOW_ENERGY_MLOL)
            # LOW_ENERGY_L.show()
            torch.manual_seed(seed)
            RECOMBINED_MLOL=self.transform(RECOMBINED_MLOL)
            # RECOMBINED_L.show()
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR = self.transform(LOW_ENERGY_MLOR)
            # LOW_ENERGY.show()
            # LOW_ENERGY_R.show()
            torch.manual_seed(seed)
            RECOMBINED_MLOR = self.transform(RECOMBINED_MLOR)
            # RECOMBINED_R.show()
            # seed = np.random.randint(2220)
            torch.manual_seed(seed)

        sample = {'LOW_ENERGY_CCL': LOW_ENERGY_CCL, 'RECOMBINED_CCL': RECOMBINED_CCL,'LOW_ENERGY_CCR': LOW_ENERGY_CCR,
                  'RECOMBINED_CCR': RECOMBINED_CCR,'LOW_ENERGY_MLOL': LOW_ENERGY_MLOL, 'RECOMBINED_MLOL': RECOMBINED_MLOL,
                  'LOW_ENERGY_MLOR': LOW_ENERGY_MLOR, 'RECOMBINED_MLOR': RECOMBINED_MLOR,
                  'label1': label1,'label2': label2,'label3': label3,'label4': label4,'case':case}

        return sample
class CESM_FZFX(Dataset):
    def __init__(self, base_dir=None, labeled_num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.targets = []
        # self.split = split
        self.transform = transform
        # dir = os.listdir(self._base_dir)
        # dir.sort()
        dir = os.listdir(self._base_dir)
        dir=natsorted(dir,alg=ns.PATH)
        for name in dir:
            image = os.path.join(self._base_dir, name)
            # print(image)
            self.sample_list.append(image)


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]

        h5f = h5py.File(case, 'r')

        LOW_ENERGY_CCL = h5f['LOW_ENERGY_CCL'][:].astype(np.float32)
        RECOMBINED_CCL = h5f['RECOMBINED_CCL'][:].astype(np.float32)
        LOW_ENERGY_CCR = h5f['LOW_ENERGY_CCR'][:].astype(np.float32)
        RECOMBINED_CCR = h5f['RECOMBINED_CCR'][:].astype(np.float32)
        LOW_ENERGY_MLOL = h5f['LOW_ENERGY_MLOL'][:].astype(np.float32)
        RECOMBINED_MLOL = h5f['RECOMBINED_MLOL'][:].astype(np.float32)
        LOW_ENERGY_MLOR = h5f['LOW_ENERGY_MLOR'][:].astype(np.float32)
        RECOMBINED_MLOR = h5f['RECOMBINED_MLOR'][:].astype(np.float32)
        label1 = h5f['label1'][()]##ki67 status
        label2 = h5f['label2'][()]##ER

        label3 = h5f['label3'][()]##PR
        label4 = h5f['label4'][()]##HER2
        label5 = h5f['label5'][()]##triple-negative breast cancer

        label6 = h5f['label_L'][()]
        label7 = h5f['label_R'][()]
        # print(idx)

        seed = np.random.randint(500000)
        if self.transform:
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_CCL = Image.fromarray(np.uint8(LOW_ENERGY_CCL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_CCR= Image.fromarray(np.uint8(LOW_ENERGY_CCR * 255))
            torch.manual_seed(seed)
            RECOMBINED_CCL = Image.fromarray(np.uint8(255 * RECOMBINED_CCL))
            torch.manual_seed(seed)
            RECOMBINED_CCR = Image.fromarray(np.uint8(255*RECOMBINED_CCR))
            torch.manual_seed(seed)
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_MLOL = Image.fromarray(np.uint8(LOW_ENERGY_MLOL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR= Image.fromarray(np.uint8(LOW_ENERGY_MLOR * 255))
            torch.manual_seed(seed)
            RECOMBINED_MLOL = Image.fromarray(np.uint8(255 * RECOMBINED_MLOL))
            torch.manual_seed(seed)
            RECOMBINED_MLOR = Image.fromarray(np.uint8(255*RECOMBINED_MLOR))
            torch.manual_seed(seed)
            # print(seed)
            LOW_ENERGY_CCL = self.transform(LOW_ENERGY_CCL)

            torch.manual_seed(seed)
            RECOMBINED_CCL=self.transform(RECOMBINED_CCL)
            # RECOMBINED_L.show()
            torch.manual_seed(seed)
            LOW_ENERGY_CCR = self.transform(LOW_ENERGY_CCR)
            # LOW_ENERGY.show()
            # LOW_ENERGY_R.show()
            torch.manual_seed(seed)
            RECOMBINED_CCR = self.transform(RECOMBINED_CCR)
            # RECOMBINED_R.show()
            # seed = np.random.randint(2220)
            torch.manual_seed(seed)
            LOW_ENERGY_MLOL = self.transform(LOW_ENERGY_MLOL)
            # LOW_ENERGY_L.show()
            torch.manual_seed(seed)
            RECOMBINED_MLOL=self.transform(RECOMBINED_MLOL)
            # RECOMBINED_L.show()
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR = self.transform(LOW_ENERGY_MLOR)
            # LOW_ENERGY.show()
            # LOW_ENERGY_R.show()
            torch.manual_seed(seed)
            RECOMBINED_MLOR = self.transform(RECOMBINED_MLOR)
            # RECOMBINED_R.show()
            # seed = np.random.randint(2220)
            torch.manual_seed(seed)


        sample = {'LOW_ENERGY_CCL': LOW_ENERGY_CCL, 'RECOMBINED_CCL': RECOMBINED_CCL,'LOW_ENERGY_CCR': LOW_ENERGY_CCR,
                  'RECOMBINED_CCR': RECOMBINED_CCR,'LOW_ENERGY_MLOL': LOW_ENERGY_MLOL, 'RECOMBINED_MLOL': RECOMBINED_MLOL,
                  'LOW_ENERGY_MLOR': LOW_ENERGY_MLOR, 'RECOMBINED_MLOR': RECOMBINED_MLOR,
                  'label1': label1,'label2': label2,'label3': label3,'label4': label4,'label5': label5,'label6': label6,'label7': label7,'case': case}
        # if self.split == 'train':
        #     sample = self.transform(sample)
        # sample["idx"] = idx
        return sample
class CESM(Dataset):
    def __init__(self, base_dir=None, labeled_num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.targets = []
        # self.split = split
        self.transform = transform
        # dir = os.listdir(self._base_dir)
        # dir.sort()

        dir = os.listdir(self._base_dir)
        dir=natsorted(dir,alg=ns.PATH)
        for name in dir:
            image = os.path.join(self._base_dir, name)
            # print(image)
            self.sample_list.append(image)
        print(self.sample_list)


    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        file_path=case
        print(case)
        (filepath, tempfilename) = os.path.split(file_path)
        (filename, extension) = os.path.splitext(tempfilename)
        dst_path=r'.\clinical.pkl'####add clinical information
        x1 = pd.read_pickle(dst_path)
        nummber =(x1['num']).values

        age = x1['age']#####patient age
        ex1= x1['xy1']#####DCIS
        ex2= x1['xy2']#IDC
        ex3= x1['xy3']
        ex4= x1['xy4']
        ex5= x1['xy5']

        sh=filename.split('_')[0]

        xy1=0
        xy2 = 0
        xy3 = 0
        xy4 = 0
        xy5 = 0
        xy6 = 0
        xy7 = 0
        if sh in nummber:
            # print('yes')
            xy1 = int(age[nummber.tolist().index(sh)])
            #
            xy2 = int(ex1[nummber.tolist().index(sh)])
            xy3 = int(ex2[nummber.tolist().index(sh)])
            xy4 = int(ex3[nummber.tolist().index(sh)])
            xy5 = int(ex4[nummber.tolist().index(sh)])
            xy6 = int(ex5[nummber.tolist().index(sh)])



        h5f = h5py.File(case, 'r')

        LOW_ENERGY_CCL = h5f['LOW_ENERGY_CCL'][:].astype(np.float32)
        RECOMBINED_CCL = h5f['RECOMBINED_CCL'][:].astype(np.float32)
        LOW_ENERGY_CCR = h5f['LOW_ENERGY_CCR'][:].astype(np.float32)
        RECOMBINED_CCR = h5f['RECOMBINED_CCR'][:].astype(np.float32)
        LOW_ENERGY_MLOL = h5f['LOW_ENERGY_MLOL'][:].astype(np.float32)
        RECOMBINED_MLOL = h5f['RECOMBINED_MLOL'][:].astype(np.float32)
        LOW_ENERGY_MLOR = h5f['LOW_ENERGY_MLOR'][:].astype(np.float32)
        RECOMBINED_MLOR = h5f['RECOMBINED_MLOR'][:].astype(np.float32)
        label1 = h5f['label1'][()]
        label2 = h5f['label2'][()]
        # print(label2)
        label3 = h5f['label3'][()]
        label4 = h5f['label4'][()]
        label5 = h5f['label5'][()]

        label6 = h5f['label_L'][()]
        label7 = h5f['label_R'][()]

        seed = np.random.randint(500000)
        if self.transform:
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_CCL = Image.fromarray(np.uint8(LOW_ENERGY_CCL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_CCR= Image.fromarray(np.uint8(LOW_ENERGY_CCR * 255))
            torch.manual_seed(seed)
            RECOMBINED_CCL = Image.fromarray(np.uint8(255 * RECOMBINED_CCL))
            torch.manual_seed(seed)
            RECOMBINED_CCR = Image.fromarray(np.uint8(255*RECOMBINED_CCR))
            torch.manual_seed(seed)
            torch.manual_seed(seed)
            # print(seed )
            LOW_ENERGY_MLOL = Image.fromarray(np.uint8(LOW_ENERGY_MLOL*255))
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR= Image.fromarray(np.uint8(LOW_ENERGY_MLOR * 255))
            torch.manual_seed(seed)
            RECOMBINED_MLOL = Image.fromarray(np.uint8(255 * RECOMBINED_MLOL))
            torch.manual_seed(seed)
            RECOMBINED_MLOR = Image.fromarray(np.uint8(255*RECOMBINED_MLOR))
            torch.manual_seed(seed)
            # print(seed)
            LOW_ENERGY_CCL = self.transform(LOW_ENERGY_CCL)

            torch.manual_seed(seed)
            RECOMBINED_CCL=self.transform(RECOMBINED_CCL)
            # RECOMBINED_L.show()
            torch.manual_seed(seed)
            LOW_ENERGY_CCR = self.transform(LOW_ENERGY_CCR)
            # LOW_ENERGY.show()
            # LOW_ENERGY_R.show()
            torch.manual_seed(seed)
            RECOMBINED_CCR = self.transform(RECOMBINED_CCR)
            # RECOMBINED_R.show()
            # seed = np.random.randint(2220)
            torch.manual_seed(seed)
            LOW_ENERGY_MLOL = self.transform(LOW_ENERGY_MLOL)
            # LOW_ENERGY_L.show()
            torch.manual_seed(seed)
            RECOMBINED_MLOL=self.transform(RECOMBINED_MLOL)
            # RECOMBINED_L.show()
            torch.manual_seed(seed)
            LOW_ENERGY_MLOR = self.transform(LOW_ENERGY_MLOR)
            # LOW_ENERGY.show()
            # LOW_ENERGY_R.show()
            torch.manual_seed(seed)
            RECOMBINED_MLOR = self.transform(RECOMBINED_MLOR)
            # RECOMBINED_R.show()
            # seed = np.random.randint(2220)
            torch.manual_seed(seed)

        sample = {'LOW_ENERGY_CCL': LOW_ENERGY_CCL, 'RECOMBINED_CCL': RECOMBINED_CCL,'LOW_ENERGY_CCR': LOW_ENERGY_CCR,
                  'RECOMBINED_CCR': RECOMBINED_CCR,'LOW_ENERGY_MLOL': LOW_ENERGY_MLOL, 'RECOMBINED_MLOL': RECOMBINED_MLOL,
                  'LOW_ENERGY_MLOR': LOW_ENERGY_MLOR, 'RECOMBINED_MLOR': RECOMBINED_MLOR,
                  'label1': label1,'label2': label2,'label3': label3,'label4': label4,'label5': label5,'label6': label6,'label7': label7,'case': case,
                  'xy1': xy1,'xy2':xy2,'xy3':xy3,'xy4':xy4,'xy5':xy5,'xy6':xy6,'xy7':xy7}

        return sample





