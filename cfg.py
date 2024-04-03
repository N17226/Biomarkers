
import os
home = os.path.expanduser('~')

#网络默认输入图像的大小
INPUT_SIZE = 512*256


BASE = r''
#数据集的存放位置
TRAIN_LABEL_DIR =BASE + r'\train.txt'
VAL_LABEL_DIR = BASE + r'\valid.txt'
TEST_LABEL_DIR = BASE + r'\test.txt'

####Pretrained model from feature extraction network
#ki67/er/pr/her2
TRAINED_MODEL = r''
