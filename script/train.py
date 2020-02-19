
#import essential module
import os
import cv2
import logging
import numpy as np
import pandas as pd 
from os import listdir
import matplotlib.pyplot as plt

#import keras tool
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D ,MaxPooling2D,Dropout
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Model
from keras import optimizers

#import pre-trained models
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.mobilenet import MobileNet

#import model hyperparameter
from hypepara.Configuration import GeneralCon, VGGCon, ResNetCon, MobileNetCon

#==============================================================================


#Main variables
DATA_PATH = r'C:\Users\iadc\Jim_CropImage\DatasetV1\Train'
dict_hot = {'Rice':0,
            'SugarCane':1,
            'BananaTree':2,
            'GreenOnion':3}
label = []
label_hot = []
images = []
train_class_count = [0,0,0,0]
vali_class_count = [0,0,0,0]
test_class_count = [0,0,0,0]


#Experiment variables
result_saving_path = r'C:\Users\iadc\Jim_CropImage\result\attempt_{}'.format(GeneralCon.ATTEMPT)

#Create result folder
if not os.path.exists(result_saving_path):
    os.makedirs(result_saving_path)
    os.makedirs(os.path.join(result_saving_path, 'chart'))
    os.makedirs(os.path.join(result_saving_path, 'log'))
    os.makedirs(os.path.join(result_saving_path, 'models'))


#logging setting
logging.basicConfig(level = 20,
                    format = '%(asctime)s %(levelname)s %(message)s',
                    datefmt = '%Y-%m-%d %H:%M',
                    handlers = [logging.FileHandler(os.path.join(result_saving_path, 'log','modelHistory.log'), 'w', 'utf-8'), ])



logging.info('Models will be stored in {}'.format(os.path.join(result_saving_path, 'models')))
logging.info('Charts will be stored in {}'.format(os.path.join(result_saving_path, 'chart')))
logging.info('Log will be stored in {}'.format(os.path.join(result_saving_path, 'log')))
logging.info('============================================================================')



#Loading images and labels(it needs to take a while)
logging.info('Loading images...')
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
logging.info('Data loading complete')
logging.info('============================================================================')

#Convert list to numpy array
images = np.array(images) 
label_hot = np.array(label_hot)

#Counting the size of entire dataset
logging.info('images_array_size = {} , total number of data = {}\n'.format(images.shape, label_hot.shape))
#print("images.shape={} , label_hot.shape=={}\n".format(images.shape, label_hot.shape))



#Spliting training, validadtion and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(images, label_hot, test_size=0.2, random_state=42)
(trainData, valiData, trainLabels, valiLabels) = train_test_split(trainData, trainLabels, test_size=0.2, random_state=43)


#Counting the number of each class
for i in trainLabels:
    train_class_count[i] += 1
for i in valiLabels:
    vali_class_count[i] += 1
for i in testLabels:
    test_class_count[i] += 1


logging.info('number of training data of each class:')
logging.info('Rice:{} , SugarCane:{} , BananaTree:{} , GreenOnion:{}\n'.format(train_class_count[0],train_class_count[1],train_class_count[2],train_class_count[3]))

logging.info('number of validation data of each class:')
logging.info('Rice:{} , SugarCane:{} , BananaTree:{} , GreenOnion:{}\n'.format(vali_class_count[0],vali_class_count[1],vali_class_count[2],vali_class_count[3]))

logging.info('number of testing data of each class:')
logging.info('Rice:{} , SugarCane:{} , BananaTree:{} , GreenOnion:{}\n'.format(test_class_count[0],test_class_count[1],test_class_count[2],test_class_count[3]))


#One hot encoding
trainLabels_hot = np_utils.to_categorical(trainLabels)
valiLabels_hot = np_utils.to_categorical(valiLabels)
testLabels_hot = np_utils.to_categorical(testLabels)
logging.info('One hot encoding matrix shape (Train):{}'.format(trainLabels_hot.shape))
logging.info('One hot encoding matrix shape (Val):{}'.format(valiLabels_hot.shape))
logging.info('One hot encoding matrix shape (Test):{}\n'.format(testLabels_hot.shape))
logging.info('ImageDataGenerator times(Train):{}'.format(GeneralCon.DATA_FAKING_TRAIN))
logging.info('ImageDataGenerator times(Val):{}'.format(GeneralCon.DATA_FAKING_VAL))
logging.info('============================================================================\n')
    


#ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0/255.0,)#featurewise_center=True,#featurewise_std_normalization=True,

datagen.fit(trainData)
train_generator = datagen.flow(trainData, trainLabels_hot, shuffle=True, batch_size = GeneralCon.BATCH_SIZE)
val_generator = datagen.flow(valiData, valiLabels_hot, shuffle=True, batch_size = GeneralCon.BATCH_SIZE)



#Get pre-trained models, and exclude their output layer.
ResNet_Pre = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(GeneralCon.IMAGE_SIZE[0], GeneralCon.IMAGE_SIZE[1] , 3))
VGG_Pre = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(GeneralCon.IMAGE_SIZE[0], GeneralCon.IMAGE_SIZE[1] , 3))
MobileNet_Pre = MobileNet(include_top=False, weights='imagenet', input_tensor=None, input_shape=(GeneralCon.IMAGE_SIZE[0], GeneralCon.IMAGE_SIZE[1] , 3)) 


#Flatten the last layer of each pre-trained model
x = ResNet_Pre.layers[-1].output
x = Flatten()(x)
y = VGG_Pre.layers[-1].output
y = Flatten()(y)
z = MobileNet_Pre.layers[-1].output
z = Flatten()(z)

logging.info('============================================================================\n')
ResNet_final = Model(inputs=ResNet_Pre.input, outputs=x)
logging.info('Flatten complete --ResNet')
VGG_final = Model(inputs=VGG_Pre.input, outputs=y)
logging.info('Flatten complete --VGG')
MobileNet_final = Model(inputs=MobileNet_Pre.input, outputs=z)
logging.info('Flatten complete --MobileNet\n')
logging.info('Freeze mode:{}\n'.format(GeneralCon.FREEZE))



#Freeze mode on
if GeneralCon.FREEZE: 
    for layer in ResNet_final.layers:
        layer.trainable = False

    for layer in VGG_final.layers:
        layer.trainable = False

    for layer in MobileNet_final.layers:
        layer.trainable = False
else:
    #ResNet
    for layer in ResNet_final.layers[:ResNetCon.FREEZE_LAYER]:
        layer.trainable = False
    for layer in ResNet_final.layers[ResNetCon.FREEZE_LAYER:]:
        layer.trainable = True

    #VGG
    for layer in VGG_final.layers[:VGGCon.FREEZE_LAYER]:
        layer.trainable = False
    for layer in VGG_final.layers[VGGCon.FREEZE_LAYER:]:
        layer.trainable = True

    #MobileNet
    for layer in MobileNet_final.layers[:MobileNetCon.FREEZE_LAYER]:
        layer.trainable = False
    for layer in MobileNet_final.layers[MobileNetCon.FREEZE_LAYER:]:
        layer.trainable = True
    

logging.info('Universal Classifier mode:{}\n'.format(GeneralCon.UNIVERSE))

#Construct the model
full_model_ResNet = Sequential()
full_model_ResNet.add(ResNet_final)

full_model_VGG = Sequential()
full_model_VGG.add(VGG_final)

full_model_MobileNet = Sequential()
full_model_MobileNet.add(MobileNet_final)


#Using the same classifier or deploy compatible one
if GeneralCon.UNIVERSE:

    full_model_ResNet.add(Dense(100, activation = 'relu'))
    full_model_ResNet.add(Dropout(0.5))
    full_model_ResNet.add(Dense(100, activation = 'relu'))
    full_model_ResNet.add(Dropout(0.5))
    full_model_ResNet.add(Dense(GeneralCon.NUM_CLASSES, activation='softmax'))

    full_model_VGG.add(Dense(100, activation = 'relu'))
    full_model_VGG.add(Dropout(0.5))
    full_model_VGG.add(Dense(100, activation = 'relu'))
    full_model_VGG.add(Dropout(0.5))
    full_model_VGG.add(Dense(GeneralCon.NUM_CLASSES, activation='softmax'))

    full_model_MobileNet.add(Dense(100, activation = 'relu'))
    full_model_MobileNet.add(Dropout(0.5))
    full_model_MobileNet.add(Dense(100, activation = 'relu'))
    full_model_MobileNet.add(Dropout(0.5))
    full_model_MobileNet.add(Dense(GeneralCon.NUM_CLASSES, activation='softmax'))
else:

    full_model_ResNet.add(Dense(100, activation = 'relu'))
    full_model_ResNet.add(Dropout(0.5))
    full_model_ResNet.add(Dense(100, activation = 'relu'))
    full_model_ResNet.add(Dropout(0.5))
    full_model_ResNet.add(Dense(GeneralCon.NUM_CLASSES, activation='softmax'))

    full_model_VGG.add(Dense(80, activation = 'relu'))
    full_model_VGG.add(Dropout(0.5))
    full_model_VGG.add(Dense(80, activation = 'relu'))
    full_model_VGG.add(Dropout(0.5))
    full_model_VGG.add(Dense(80, activation = 'relu'))
    full_model_VGG.add(Dropout(0.5))
    full_model_VGG.add(Dense(GeneralCon.NUM_CLASSES, activation='softmax'))

    full_model_MobileNet.add(Dense(90, activation = 'relu'))
    full_model_MobileNet.add(Dropout(0.5))
    full_model_MobileNet.add(Dense(90, activation = 'relu'))
    full_model_MobileNet.add(Dropout(0.5))
    full_model_MobileNet.add(Dense(GeneralCon.NUM_CLASSES, activation='softmax'))
    

    


#Compiling models
full_model_ResNet.compile(optimizer = optimizers.Adam(lr = ResNetCon.LEARNING_RATE), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])
            
full_model_VGG.compile(optimizer = optimizers.Adam(lr = VGGCon.LEARNING_RATE), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

full_model_MobileNet.compile(optimizer = optimizers.Adam(lr = MobileNetCon.LEARNING_RATE), 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

logging.info('Construction complete\n')
logging.info('============================Model Hyperparameters=============================\n')
logging.info('ResNet:')
logging.info('Learning_rate = {}'.format(ResNetCon.LEARNING_RATE))
logging.info('Epochs = {}\n'.format(ResNetCon.EPOCH))

logging.info('VGG:')
logging.info('Learning_rate = {}'.format(VGGCon.LEARNING_RATE))
logging.info('Epochs = {}\n'.format(VGGCon.EPOCH))

logging.info('MobileNet:')
logging.info('Learning_rate = {}'.format(MobileNetCon.LEARNING_RATE))
logging.info('Epochs = {}\n'.format(MobileNetCon.EPOCH))

print(full_model_ResNet.summary())
print(full_model_VGG.summary())
print(full_model_MobileNet.summary())


#Faking training data
STEPS_PER_EPOCH = int( (len(trainData) * GeneralCon.DATA_FAKING_TRAIN) / GeneralCon.BATCH_SIZE)
VALIDATION_STEPS = int( (len(trainData)* GeneralCon.DATA_FAKING_VAL)  / GeneralCon.BATCH_SIZE)


#Training
logging.info('Start training ResNet based network')
train_history_ResNet = full_model_ResNet.fit_generator(train_generator,
                        steps_per_epoch = STEPS_PER_EPOCH, epochs = ResNetCon.EPOCH,
                        validation_data = val_generator, validation_steps = VALIDATION_STEPS)
logging.info('Training completes\n')


logging.info('Start training VGG based network')
train_history_VGG = full_model_VGG.fit_generator(train_generator,
                        steps_per_epoch = STEPS_PER_EPOCH, epochs = VGGCon.EPOCH,
                        validation_data = val_generator, validation_steps = VALIDATION_STEPS)
logging.info('Training completes\n')


logging.info('Start training MobileNet based network')
train_history_MobileNet = full_model_MobileNet.fit_generator(train_generator,
                        steps_per_epoch = STEPS_PER_EPOCH, epochs = MobileNetCon.EPOCH,
                        validation_data = val_generator, validation_steps = VALIDATION_STEPS)
logging.info('Training completes\n')


ResNet_SAVE_PATH = os.path.join(result_saving_path ,r'models\ResNet_{}epochs.h5'.format(ResNetCon.EPOCH))
VGG_SAVE_PATH = os.path.join(result_saving_path ,r'models\VGG_{}epochs.h5'.format(VGGCon.EPOCH))
MobileNet_SAVE_PATH = os.path.join(result_saving_path ,r'models\MoblieNet_{}epochs.h5'.format(MobileNetCon.EPOCH))


#Saving models
full_model_ResNet.save(ResNet_SAVE_PATH)
full_model_VGG.save(VGG_SAVE_PATH)
full_model_MobileNet.save(MobileNet_SAVE_PATH)


epo_list_ResNet = [i for i in range(0,ResNetCon.EPOCH)]
epo_list_VGG = [i for i in range(0,VGGCon.EPOCH)]
epo_list_MobileNet = [i for i in range(0,MobileNetCon.EPOCH)]


#Visualize the loss and accuracy
plt.figure(1)
plt.title("ResNet Loss")
plt.plot(epo_list_ResNet,train_history_ResNet.history['loss'],label = 'loss')
plt.plot(epo_list_ResNet,train_history_ResNet.history['val_loss'],label = 'val loss')
plt.ylim(0,1)
plt.legend()
plt.xlabel('epochs')
plt.savefig(os.path.join(result_saving_path, 'chart', 'ResNet_Loss_{}epochs.png'.format(ResNetCon.EPOCH)))

plt.figure(2)
plt.title("ResNet Accuracy")
plt.plot(epo_list_ResNet,train_history_ResNet.history['acc'],label = 'accuracy')
plt.plot(epo_list_ResNet,train_history_ResNet.history['val_acc'],label = 'val accuracy')
plt.ylim(0,1)
plt.legend()
plt.xlabel('epochs')
plt.savefig(os.path.join(result_saving_path, 'chart', 'ResNet_Acc_{}epochs.png'.format(ResNetCon.EPOCH)))



plt.figure(3)
plt.title("VGG Loss")
plt.plot(epo_list_VGG,train_history_VGG.history['loss'],label = 'loss')
plt.plot(epo_list_VGG,train_history_VGG.history['val_loss'],label = 'val loss')
plt.ylim(0,1)
plt.legend()
plt.xlabel('epochs')
plt.savefig(os.path.join(result_saving_path, 'chart', 'VGG_Loss_{}epochs.png'.format(VGGCon.EPOCH)))

plt.figure(4)
plt.title("VGG Accuracy")
plt.plot(epo_list_VGG,train_history_VGG.history['acc'],label = 'accuracy')
plt.plot(epo_list_VGG,train_history_VGG.history['val_acc'],label = 'val accuracy')
plt.ylim(0,1)
plt.legend()
plt.xlabel('epochs')
plt.savefig(os.path.join(result_saving_path, 'chart', 'VGG_Acc_{}epochs.png'.format(VGGCon.EPOCH)))



plt.figure(5)
plt.title("MobileNet Loss")
plt.plot(epo_list_MobileNet,train_history_MobileNet.history['loss'],label = 'loss')
plt.plot(epo_list_MobileNet,train_history_MobileNet.history['val_loss'],label = 'val loss')
plt.ylim(0,1)
plt.legend()
plt.xlabel('epochs')
plt.savefig(os.path.join(result_saving_path, 'chart', 'MobileNet_Loss_{}epochs.png'.format(MobileNetCon.EPOCH)))

plt.figure(6)
plt.title("MobileNet Accuracy")
plt.plot(epo_list_MobileNet,train_history_MobileNet.history['acc'],label = 'accuracy')
plt.plot(epo_list_MobileNet,train_history_MobileNet.history['val_acc'],label = 'val accuracy')
plt.ylim(0,1)
plt.legend()
plt.xlabel('epochs')
plt.savefig(os.path.join(result_saving_path, 'chart', 'MobileNet_Acc_{}epochs.png'.format(MobileNetCon.EPOCH)))
plt.show()

