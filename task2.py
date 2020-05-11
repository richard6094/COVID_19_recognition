import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
import PIL.Image

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.__version__

DATA_LIST = os.listdir('all/train')
DATASET_PATH  = 'all/train'
TEST_DIR =  'all/test'
IMAGE_SIZE    = (224, 224)
NUM_CLASSES   = len(DATA_LIST)
BATCH_SIZE    = 10  # try reducing batch size or freeze more layers if your GPU runs out of memory
NUM_EPOCHS    = 100
LEARNING_RATE = 0.0001 # start off with high rate first 0.001 and experiment with reducing it gradually

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=50,featurewise_center = True,
                                   featurewise_std_normalization = True,width_shift_range=0.2,
                                   height_shift_range=0.2,shear_range=0.25,zoom_range=0.1,
                                   zca_whitening = True,channel_shift_range = 20,
                                   horizontal_flip = True,vertical_flip = True,
                                   validation_split = 0.2,fill_mode='constant')


train_batches = train_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE,
                                                  shuffle=True,batch_size=BATCH_SIZE,
                                                  subset = "training",seed=42,
                                                  class_mode="categorical")

valid_batches = train_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE,
                                                  shuffle=True,batch_size=BATCH_SIZE,
                                                  subset = "validation",
                                                  seed=42,class_mode="categorical")

model_vgg = VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))
for layer in model_vgg.layers:
    layer.trainable = False
model = Flatten(name='flatten')(model_vgg.output)
model = Dense(256, activation='relu', name='feature_dense')(model)
model = Dense(4, activation='softmax', name='dense')(model)

model_COVID_19 = Model(inputs=model_vgg.input, outputs=model, name='vgg16')

model_COVID_19.summary()

model_COVID_19.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print(len(train_batches))
print(len(valid_batches))

STEP_SIZE_TRAIN=train_batches.n//train_batches.batch_size
STEP_SIZE_VALID=valid_batches.n//valid_batches.batch_size

# raise NotImplementedError("Use the model.fit function to train your network")
history = model_COVID_19.fit(train_batches, steps_per_epoch=STEP_SIZE_TRAIN, epochs=100, validation_data=valid_batches, validation_steps=STEP_SIZE_VALID)

import matplotlib.pyplot as plt

# raise NotImplementedError("Plot the accuracy and the loss during training")
plt.figure()
plt.plot(history.history['accuracy'],label='Train_acc')
plt.plot(history.history['val_accuracy'],label='Test_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.25,0.9])
plt.grid(True)
plt.suptitle('Accuracy over 100 Epochs')
plt.legend(loc = 'lower right')

plt.figure()
plt.plot(history.history['loss'],label='Train_loss')
plt.plot(history.history['val_loss'],label='Test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0,4])
plt.grid(True)
plt.suptitle('Loss over 100 Epochs')
plt.legend(loc = 'upper right')

test_datagen = ImageDataGenerator(rescale=1. / 255)

eval_generator = test_datagen.flow_from_directory(TEST_DIR,target_size=IMAGE_SIZE,
                                                  batch_size=1,shuffle=True,seed=42,class_mode="categorical")
eval_generator.reset()
print(len(eval_generator))
x = model_COVID_19.evaluate_generator(eval_generator,steps = np.ceil(len(eval_generator)),
                           use_multiprocessing = False,verbose = 1,workers=1)
print('Test loss:' , x[0])
print('Test accuracy:',x[1])

from sklearn.manifold import TSNE
from keras import models

intermediate_layer_model = models.Model(inputs=model_COVID_19.input,
                                        outputs=model_COVID_19.get_layer('dense').output)

tsne_eval_generator = test_datagen.flow_from_directory(DATASET_PATH,target_size=IMAGE_SIZE,
                                                  batch_size=1,shuffle=True,seed=42,class_mode="categorical")

tsne_data = intermediate_layer_model.predict(tsne_eval_generator)

type_0_0=[]
type_0_1=[]
type_1_0=[]
type_1_1=[]
type_2_0=[]
type_2_1=[]
type_3_0=[]
type_3_1=[]

features = np.zeros(270)
for i in range(tsne_data.shape[0]):
    max = tsne_data[i][0]
    max_index = 0
    for j in range(tsne_data[i].shape[0]):
        if tsne_data[i][j]>max:
            max = tsne_data[i][j]
            max_index = j
    features[i] = max_index

# print(tsne_data)

tsne = TSNE(n_components=2)
result = tsne.fit_transform(tsne_data)

for i in range(result.shape[0]):
    if features[i] == 0:
        type_0_0.append(result[i][0])
        type_0_1.append(result[i][1])
    elif features[i] == 1:
        type_1_0.append(result[i][0])
        type_1_1.append(result[i][1])
    elif features[i] == 2:
        type_2_0.append(result[i][0])
        type_2_1.append(result[i][1])
    elif features[i] == 3:
        type_3_0.append(result[i][0])
        type_3_1.append(result[i][1])

fig = plt.figure()
nom = plt.scatter(type_0_0, type_0_1, c='r')
cov = plt.scatter(type_1_0, type_1_1, c='b')
p_b = plt.scatter(type_2_0, type_2_1, c='g')
p_v = plt.scatter(type_3_0, type_3_1, c='y')
plt.legend((nom,cov,p_b,p_v),('Normal','COVID-19','Pneumonia-Bacterial','Pneumonia-Viral'),loc='lower right')
plt.show()
