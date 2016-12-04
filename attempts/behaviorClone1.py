## Training model #1
##
## Keras model from Keras lab
##
## import some useful python modules
import os
import numpy as np
from numpy.random import random
import cv2
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

## for reproducibility
np.random.seed(178951)

## model input files for initial model save and retrain
fileDataPath = './drivingDataTrack1'
fileDataCSV = '/driving_log.csv'
fileModelJSON = 'model1.json'
fileWeights = 'model1.h5'

## model parameters defined here
batch_size = 512
nb_epoch = 10
img_rows, img_cols = 160, 320
nb_classes = 1
sgd = SGD(lr=0.01, decay=0.1, momentum=0.5, nesterov=True)

## get our training and validation data
## features: center,left,right,steering,throttle,brake,speed
## We only want center images that are not braking, have throttle and speed.
print("\n\ntraining data from: ", fileDataPath+fileDataCSV)
data = pd.read_csv(fileDataPath+fileDataCSV)
print( len(data), "read from: ", fileDataPath+fileDataCSV)
print(data.describe())
print("\n\ntypes:")
print(data.dtypes)

data = data[(data.throttle>0.0)&(data.brake==0.0)&(data.speed>0.0)]
X_train = np.copy(data['center'])
Y_train = np.copy(data['steering'])
Y_train = Y_train.astype(np.float32)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=10)

def gaussian_blur(img, kernel_size):
    # Applies a Gaussian Noise kernel
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def load_image(imagepath):
    image = cv2.imread(imagepath, 1)
    #print("image: ", type(image))
    shape = image.shape
    #print("image: ", shape)
    if not shape[0] == img_rows or not shape[1] == img_cols:
        image = cv2.resize(image, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    # kernel_size = int(random()*3)*2+1
    # image = gaussian_blur(image, kernel_size)
    image = np.array(image).astype(np.float32)
    image /= 127.5
    image -= 1.0
    return image

def batchgen(X, Y):
    while 1:
        for i in range(len(X)):
            imagepath, y = X[i], Y[i]
            # print("imagepath: ", '"'+imagepath+'"', "steering: ", y)
            image = load_image(imagepath)
            # print("image: ", image.shape, "steering: ", y)
            y = np.array([[y]])
            image = image.reshape(1, img_rows, img_cols, 3)
            # print("image: ", image.shape, "y: ", y.shape)
            yield (image, y)

# number of convolutional filters to use
nb1_filters = 32
nb2_filters = 64
nb3_filters = 128
nb4_filters = 256
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Convolution2D(nb1_filters, kernel_size[0], kernel_size[1],
                        border_mode='same',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb2_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(nb3_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(nb4_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Flatten())
model.add(Dense(128))
#model.add(Dropout(0.5))
model.add(Dense(nb_classes, name='output'))
model.summary()

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit_generator(batchgen(X_train, Y_train),
                    samples_per_epoch=batch_size, nb_epoch=nb_epoch,
                    validation_data=batchgen(X_val, Y_val), 
                    nb_val_samples=X_val.shape[0],
                    verbose=1)

json_string = model.to_json()
with open(fileModelJSON,'w' ) as f:
   json.dump(json_string, f)
model.save_weights(fileWeights)

