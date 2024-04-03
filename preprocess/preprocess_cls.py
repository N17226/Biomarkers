import random
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import math
import glob
import os
import h5py
import pydicom
import cv2
from skimage.measure import label
import gdcm
import xlrd


def largestConnectComponent(bw_img, ):
    labeled_img, num = label(bw_img, background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    return lcc



def location_point(img_CC_R):
    im = np.where(img_CC_R != 0, 1, 0)
    im = largestConnectComponent(im).astype(np.uint8)
    upper_point = np.where(im[100:3200, 0:3000] != 0)
    y_start = np.min(np.where(im != 0)[0])
    y_end = np.max(upper_point[0])+100
    x_start = np.min(upper_point[1])
    x_end = np.max(upper_point[1])

    x_point = np.sum(im, axis=1)
    if x_point[y_end] <= 10:
        y_end = np.max(np.where(x_point >= 12 * x_point[y_end]))
    if 40 >= x_point[y_end] > 10:
        y_end = np.max(np.where(x_point >= 5 * x_point[y_end]))
    if 70 >= x_point[y_end] > 40:
        y_end = np.max(np.where(x_point >= 3 * x_point[y_end]))
    if 100 >= x_point[y_end] > 70:
        y_end = np.max(np.where(x_point >= 2 * x_point[y_end]))

    if x_point[y_start] <= 10:
        y_start = np.min(np.where(x_point >= 12 * x_point[y_start]))
    if 40 >= x_point[y_start] > 10:
        y_start = np.min(np.where(x_point >= 6 * x_point[y_start]))
    if 70 >= x_point[y_start] > 40:
        y_start = np.min(np.where(x_point >= 4 * x_point[y_start]))
    if 100 >= x_point[y_start] > 70:
        y_start = np.min(np.where(x_point >= 2 * x_point[y_start]))

    return y_start, y_end, x_start, x_end

class MedicalImageDeal(object):
    def __init__(self, img, path, percent=1):
        self.img = img
        self.percent = percent
        self.p = path

    @property
    def valid_img(self):
        from skimage import exposure
        cdf = exposure.cumulative_distribution(self.img)
        watershed = cdf[1][cdf[0] >= self.percent][0]
        return np.clip(self.img, self.img.min(), watershed)

    @property
    def norm_img(self):
        return ((self.img - self.img.min()) / (self.img.max() - self.img.min())).astype(np.float32)

    @property
    def width_center1(self):
        from skimage import exposure
        image = pydicom.read_file(self.p).pixel_array
        cdf = exposure.cumulative_distribution(image)
        cdf_min = np.min(cdf[1])
        a = np.diff(cdf)
        aa = a[0][1500 - cdf_min:3300]
        aa_max = np.max(aa)
        xx = []
        for i in range(aa.shape[0]):
            if aa[i] > aa_max * 0.005:
                xx.append(i + 1500)
        ds = np.clip(image, xx[0], xx[len(xx) - 3])
        max = np.max(ds)
        min = np.min(ds)
        ds = (ds-min)/(max-min)
        return ds


datapath = r'./cls_label' ######data with image0level labels
x1 = xlrd.open_workbook(datapath)
sheet1 = x1.sheet_by_name("Sheet1")
idlist = sheet1.col_values(3)
xylist = sheet1.col_values(4)
root_dir = r'./data'
save_dir = r'./h5py'


saveflie_all = save_dir
patient_name = os.listdir(root_dir)

image = []
lab = []
n = 0
for i in range(len(patient_name)):  # len(patient_name)

    a = glob.glob(root_dir + '/' + patient_name[i] + '/'+'*LOW_ENERGY_CC_R.dcm')[0]

    b = glob.glob(root_dir + '/' + patient_name[i]  + '/*RECOMBINED_CC_R.dcm')[0]
    c = glob.glob(root_dir + '/' + patient_name[i]  + '/*LOW_ENERGY_MLO_R.dcm')[0]
    d = glob.glob(root_dir + '/' + patient_name[i]  + '/*RECOMBINED_MLO_R.dcm')[0]
    (flepath, flename) = os.path.split(b)
    (rename, suffix) = os.path.splitext(flename)
    llabel=rename
    if llabel in idlist:
        xy = xylist[idlist.index(llabel)]

    e = glob.glob(root_dir + '/' + patient_name[i] +  '/*LOW_ENERGY_CC_L.dcm')[0]
    f = glob.glob(root_dir + '/' + patient_name[i] +  '/*RECOMBINED_CC_L.dcm')[0]
    g = glob.glob(root_dir + '/' + patient_name[i] +  '/*LOW_ENERGY_MLO_L.dcm')[0]
    h = glob.glob(root_dir + '/' + patient_name[i] +  '/*RECOMBINED_MLO_L.dcm')[0]
    (flepath2, flename2) = os.path.split(f)
    (rename2, suffix2) = os.path.splitext(flename2)
    llabel2=rename2
    if llabel2 in idlist:
        xy2 = xylist[idlist.index(llabel2)]

    patient_image = {patient_name[i]: [a, b, c, d,e,f,g,h]}
    patient_label = {patient_name[i]: [xy, xy2]}

    lab.append(patient_label)

    image.append(patient_image)
    n += 8




path = saveflie_all

if os.path.exists(path):
    print("directory already exists")
else:
    os.makedirs(path)

patch_size = [640, 640]
ww, hh = 3062, 2394
stride_x = 128
stride_y = 128
sx = math.ceil((ww - patch_size[0]) / stride_x) + 1  # math.ceil:小数部分直接舍去，并向正数部分进1; sx=20
sy = math.ceil((hh - patch_size[1]) / stride_y) + 1  # sy=15
rr_num = 0
r_num = 0
l_num = 0
ll_num = 0
for i in range(len(image)):

    uid = patient_name[i]
    LOW_ENERGY_CC_R = MedicalImageDeal(pydicom.dcmread(image[i][patient_name[i]][0]),
                                (image[i][patient_name[i]][0])).width_center1  # 最值归一化
    RECOMBINED_CC_R = MedicalImageDeal(pydicom.dcmread(image[i][patient_name[i]][1]),
                                (image[i][patient_name[i]][1])).width_center1
    LOW_ENERGY_MLO_R = MedicalImageDeal(pydicom.dcmread(image[i][patient_name[i]][2]),
                                 (image[i][patient_name[i]][2])).width_center1
    RECOMBINED_MLO_R = MedicalImageDeal(pydicom.dcmread(image[i][patient_name[i]][3]),
                                 (image[i][patient_name[i]][3])).width_center1
    LOW_ENERGY_CC_L = MedicalImageDeal(pydicom.dcmread(image[i][patient_name[i]][4]),
                         (image[i][patient_name[i]][4])).width_center1
    RECOMBINED_CC_L = MedicalImageDeal(pydicom.dcmread(image[i][patient_name[i]][5]),
                         (image[i][patient_name[i]][5])).width_center1
    LOW_ENERGY_MLO_L = MedicalImageDeal(pydicom.dcmread(image[i][patient_name[i]][6]),
                         (image[i][patient_name[i]][6])).width_center1
    RECOMBINED_MLO_L = MedicalImageDeal(pydicom.dcmread(image[i][patient_name[i]][7]),
                         (image[i][patient_name[i]][7])).width_center1


    label_R = int((lab[i][patient_name[i]][0]))
    label_L = int((lab[i][patient_name[i]][1]))
    label_3 = label_R | label_L
    label_4 = label_R ^ label_L



    [y_start, y_end, x_start, x_end] = location_point(LOW_ENERGY_CC_R)
    [y1_start, y1_end, x1_start, x1_end] = location_point(LOW_ENERGY_MLO_R)
    [y2_start, y2_end, x2_start, x2_end] = location_point(LOW_ENERGY_CC_L)
    [y3_start, y3_end, x3_start, x3_end] = location_point(LOW_ENERGY_MLO_L)


    LOW_ENERGY_CCR = LOW_ENERGY_CC_R[y_start:y_end, x_start:x_end]
    RECOMBINED_CCR = RECOMBINED_CC_R[y_start:y_end, x_start:x_end]
    LOW_ENERGY_MLOR = LOW_ENERGY_MLO_R[0:y1_end , x1_start :x1_end]
    RECOMBINED_MLOR = RECOMBINED_MLO_R[0:y1_end, x1_start:x1_end]
    LOW_ENERGY_CCL = LOW_ENERGY_CC_L[y2_start:y2_end, x2_start :x2_end]
    RECOMBINED_CCL = RECOMBINED_CC_L[y2_start:y2_end, x2_start :x2_end]
    LOW_ENERGY_MLOL = LOW_ENERGY_MLO_L[0:y3_end, x3_start :x3_end]
    RECOMBINED_MLOL = RECOMBINED_MLO_L[0:y3_end, x3_start :x3_end]



    LOW_ENERGY_CCR = cv2.resize(LOW_ENERGY_CCR, (512, 1024))
    RECOMBINED_CCR = cv2.resize(RECOMBINED_CCR, (512, 1024))
    LOW_ENERGY_MLOR = cv2.resize(LOW_ENERGY_MLOR, (512, 1024))
    RECOMBINED_MLOR = cv2.resize(RECOMBINED_MLOR, (512, 1024))
    LOW_ENERGY_CCL = cv2.resize(LOW_ENERGY_CCL, (512, 1024))
    RECOMBINED_CCL = cv2.resize(RECOMBINED_CCL, (512, 1024))
    LOW_ENERGY_MLOL = cv2.resize(LOW_ENERGY_MLOL, (512, 1024))
    RECOMBINED_MLOL = cv2.resize(RECOMBINED_MLOL, (512, 1024))


    LOW_ENERGY_CCR2=np.flip(LOW_ENERGY_CCR,axis=1)#flip R-L
    RECOMBINED_CCR2=np.flip(RECOMBINED_CCR,axis=1)#flip R-L
    LOW_ENERGY_MLOR2=np.flip(LOW_ENERGY_MLOR,axis=1)#flip R-L
    RECOMBINED_MLOR2=np.flip(RECOMBINED_MLOR,axis=1)#flip R-L



    f = h5py.File(path + '/{}_{}_{}.h5'.format(uid, 'CC_MLO_L', r_num), 'w')
    f.create_dataset('LOW_ENERGY_CCL', data=LOW_ENERGY_CCL, compression="gzip")
    f.create_dataset('RECOMBINED_CCL', data=RECOMBINED_CCL, compression="gzip")
    f.create_dataset('LOW_ENERGY_MLOL', data=LOW_ENERGY_MLOL, compression="gzip")
    f.create_dataset('RECOMBINED_MLOL', data=RECOMBINED_MLOL, compression="gzip")
    f.create_dataset('LOW_ENERGY_CCR', data=LOW_ENERGY_CCR2, compression="gzip")
    f.create_dataset('RECOMBINED_CCR', data=RECOMBINED_CCR2, compression="gzip")
    f.create_dataset('LOW_ENERGY_MLOR', data=LOW_ENERGY_MLOR2, compression="gzip")
    f.create_dataset('RECOMBINED_MLOR', data=RECOMBINED_MLOR2, compression="gzip")
    f.create_dataset('label1', data=label_L)
    f.create_dataset('label2', data=label_R)
    f.create_dataset('label3', data=label_3)
    f.create_dataset('label4', data=label_4)
    f.close()
    r_num += 1

