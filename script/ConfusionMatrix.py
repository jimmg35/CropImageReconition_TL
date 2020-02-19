import os
import cv2
import logging
import numpy as np
import pandas as pd 
import seaborn as sns
from os import listdir
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope

from hypepara.Configuration import GeneralCon, VGGCon, ResNetCon, MobileNetCon
import tensorflow as tf


DATA_PATH = r'C:\Users\iadc\Jim_CropImage\DatasetV1\Train'
dict_hot = {'Rice':0,
            'SugarCane':1,
            'BananaTree':2,
            'GreenOnion':3}
label = []
label_hot = []
images = []
test_class_count = [0,0,0,0]
classes = ['Rice', 'SugarCane', 'BananaTree', 'GreenOnion']

def Kappa(Series):
    row, col = Series.shape
    po_A = 0
    po_B = 0

    for i in range(0,row):
        po_A += Series[i][i]

    for i in range(0,row):
        for j in range(0,col):
            po_B += Series[i][j]
            
    po = po_A/po_B
    
    pro = 0
    for i in range(0,row):
        row_N = 0
        col_N = 0
        for j in range(0,row):
            row_N += Series[i][j]
            col_N += Series[j][i]
        pro += row_N * col_N
    pe = pro / (po_B ** 2)

    kappa = (po - pe) / (1 - pe)
    return kappa

for i in listdir(DATA_PATH):
    #get the label of each image, and store it into label
    class_name = i[:i.index("_")]
    label.append(class_name)
    label_hot.append(dict_hot[class_name])
    
    #load the images
    img = cv2.imread(os.path.join(DATA_PATH, i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, GeneralCon.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    images.append(np.array(img))

print('Loading Complete')

images = np.array(images) 
label_hot = np.array(label_hot)

(trainData, testData, trainLabels, testLabels) = train_test_split(images, label_hot, test_size=0.2, random_state=42)

for i in testLabels:
    test_class_count[i] += 1

testData = testData / 255.0
testLabels_hot = np_utils.to_categorical(testLabels)


#Reloading models
ResNet = load_model(r'C:\Users\iadc\Jim_CropImage\result\attempt_06\models\ResNet_10epochs.h5')
VGG = load_model(r'C:\Users\iadc\Jim_CropImage\result\attempt_06\models\VGG_10epochs.h5')
with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    MobileNet = load_model(r'C:\Users\iadc\Jim_CropImage\result\attempt_06\models\MoblieNet_10epochs.h5')


score_ResNet = ResNet.evaluate(testData, testLabels_hot, verbose=0)
score_VGG = VGG.evaluate(testData, testLabels_hot, verbose=0)
score_MobileNet = MobileNet.evaluate(testData, testLabels_hot, verbose=0)

Index= ['Rice', 'SugarCane', 'BananaTree', 'GreenOnion']

predictions_ResNet = ResNet.predict_classes(testData) 
predictions_VGG = VGG.predict_classes(testData) 
predictions_MobileNet = MobileNet.predict_classes(testData) 


C_ResNet = confusion_matrix(testLabels, predictions_ResNet)
print('===================================================')
print(C_ResNet)
kappa_ResNet = Kappa(C_ResNet)
C_ResNet = C_ResNet / C_ResNet.astype(np.float).sum(axis=0)
print(C_ResNet)
df_cm_1 = pd.DataFrame(C_ResNet, index = Index, columns = Index)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm_1, annot=True, cmap="YlGnBu",annot_kws={"size": 20})
plt.title('ResNet confusion matrix')
plt.xlabel('Predicted',fontsize=16)
plt.ylabel('Truth',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=12)
plt.show()

C_VGG = confusion_matrix(testLabels, predictions_VGG)
print(C_VGG)
kappa_VGG = Kappa(C_VGG)
C_VGG = C_VGG / C_VGG.astype(np.float).sum(axis=0)
print(C_VGG)
df_cm_2 = pd.DataFrame(C_VGG, index = Index, columns = Index)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm_2, annot=True, cmap="YlGnBu",annot_kws={"size": 20})
plt.title('VGG confusion matrix')
plt.xlabel('Predicted',fontsize=16)
plt.ylabel('Truth',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=12)
plt.show()

C_MobileNet = confusion_matrix(testLabels, predictions_MobileNet)
print(C_MobileNet)
kappa_MobileNet = Kappa(C_MobileNet)
C_MobileNet = C_MobileNet / C_MobileNet.astype(np.float).sum(axis=0)
print(C_MobileNet)
df_cm_3 = pd.DataFrame(C_MobileNet, index = Index, columns = Index)
plt.figure(figsize = (10,7))
sns.heatmap(df_cm_3, annot=True, cmap="YlGnBu",annot_kws={"size": 20})
plt.title('MobileNet confusion matrix')
plt.xlabel('Predicted',fontsize=16)
plt.ylabel('Truth',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=12)
plt.show()

print('=====================================')
print(' ResNet:')
print(f"     loss : {score_ResNet[0]}")
print(f" accuracy : {score_ResNet[1]}")
print(f"    Kappa : {kappa_ResNet}\n")

print(' VGG:')
print(f"     loss : {score_VGG[0]}")
print(f" accuracy : {score_VGG[1]}")
print(f"    Kappa : {kappa_VGG}\n")

print(' MobileNet:')
print(f"     loss : {score_MobileNet[0]}")
print(f" accuracy : {score_MobileNet[1]}")
print(f"    Kappa : {kappa_MobileNet}\n")



