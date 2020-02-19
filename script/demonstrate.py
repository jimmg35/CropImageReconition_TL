import os
import cv2
import logging
import numpy as np
import pandas as pd 
from os import listdir
import matplotlib.pyplot as plt
from numpy import expand_dims

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from hypepara.Configuration import GeneralCon, VGGCon, ResNetCon, MobileNetCon

images = []
images_origin = []
dict_hot = {0:'Rice',
            1:'SugarCane',
            2:'BananaTree',
            3:'GreenOnion'}
dict_hot_Ch = {'Rice':"稻作",
            'SugarCane':"甘蔗",
            'BananaTree':"香蕉樹",
            'GreenOnion':"蔥"}

DATA_PATH = r'C:\Users\iadc\Jim_CropImage\D2'

for i in listdir(DATA_PATH):
    img = cv2.imread(os.path.join(DATA_PATH,i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images_origin.append(img)
    img = cv2.resize(img, GeneralCon.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    images.append(img)


def quick_sort(list):
    smaller = []
    bigger = []
    keylist = []

    if len(list) <= 1:
        return list

    else:
        key = list[0]
        for i in list:
            if i > key:
                smaller.append(i)
            elif i < key:
                bigger.append(i)
            else:
                keylist.append(i)

    smaller = quick_sort(smaller)
    bigger = quick_sort(bigger)
    return smaller + keylist + bigger


ResNet = load_model(r'C:\Users\iadc\Jim_CropImage\result\attempt_06\models\ResNet_10epochs.h5')
VGG = load_model(r'C:\Users\iadc\Jim_CropImage\result\attempt_06\models\VGG_10epochs.h5')
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    MobileNet = load_model(r'C:\Users\iadc\Jim_CropImage\result\attempt_06\models\MoblieNet_10epochs.h5')


images = np.array(images) 

images = images / 255.0
result_ResNet = ResNet.predict(images)
result_VGG = VGG.predict(images)
result_MobileNet = MobileNet.predict(images)



for i in range(0, images.shape[0]):
    load_1 = [result_ResNet[i][0],result_ResNet[i][1],result_ResNet[i][2],result_ResNet[i][3]]
    load_2 = [result_VGG[i][0],result_VGG[i][1],result_VGG[i][2],result_VGG[i][3]]
    load_3 = [result_MobileNet[i][0],result_MobileNet[i][1],result_MobileNet[i][2],result_MobileNet[i][3]]
    
    print(load_1)
    print(load_2)
    print(load_3)
    print('=============================')

    load_1 = quick_sort(load_1)
    load_2 = quick_sort(load_2)
    load_3 = quick_sort(load_3)

    ResNet_Predict = dict_hot[list(result_ResNet[i]).index(load_1[0])]
    VGG_Predict = dict_hot[list(result_VGG[i]).index(load_2[0])]
    MobileNet_Predict = dict_hot[list(result_MobileNet[i]).index(load_3[0])]

    plt.imshow(images_origin[i])
    plt.title(f"ResNet:{ResNet_Predict}/{load_1[0]}   VGG:{VGG_Predict}/{load_2[0]}   MobileNet:{MobileNet_Predict}/{load_3[0]}")
    plt.show()


#print(f"ResNet:{}")
#print(f"VGG:{}")
#print(f"MobileNet:{}")


#print('===========================================')
#print(f"ResNet:{result_ResNet}")
#print(f"VGG:{result_VGG}")
#print(f"MobileNet:{result_MobileNet}")



