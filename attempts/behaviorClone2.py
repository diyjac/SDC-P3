## Training model #2
##
## Comma.ai model from challenge 2
## with preprocessing outside the model
##
## import some useful python modules
import os
import numpy as np
from numpy.random import random
import cv2
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

## for reproducibility
# np.random.seed(178951)

## model input files for initial model save and retrain
fileDataPath = './drivingDataTrack1'
fileDataCSV = '/driving_log.csv'
fileModelJSON = 'model2.json'
fileWeights = 'model2.h5'

## model parameters defined here
ch, img_rows, img_cols = 3, 160, 320  # camera format
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

# data = data[(data.steering!=0.0)&(data.throttle>0.0)&(data.brake==0.0)&(data.speed>0.0)]
data = data[(data.throttle>0.0)&(data.brake==0.0)&(data.speed>0.0)]

X_train = np.copy(data['center'])
Y_train = np.copy(data['steering'])
Y_train = Y_train.astype(np.float32)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=10)

batch_size = 20
samples_per_epoch = len(X_train)/batch_size
val_size = int(samples_per_epoch/10.0)
nb_epoch = 100

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
    kernel_size = int(random()*3)*2+1
    image = gaussian_blur(image, kernel_size)
    return image

def batchgen(X, Y):
    while 1:
        for i in range(len(X)):
            imagepath, y = X[i], Y[i]
            # print("imagepath: ", '"'+imagepath+'"', "steering: ", y)
            image = load_image(imagepath)
            # print("image: ", image.shape, "steering: ", y)
            y = np.array([[y]])
            image = image.reshape(1, img_rows, img_cols, ch)
            # print("image: ", image.shape, "y: ", y.shape)
            yield image, y

input_shape = (img_rows, img_cols, ch)

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
          input_shape=(img_rows, img_cols, ch),
          output_shape=(img_rows, img_cols, ch)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
model.summary()

model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

history = model.fit_generator(batchgen(X_train, Y_train),
                    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                    verbose=1)
#                    validation_data=batchgen(X_val, Y_val),
#                    nb_val_samples=val_size,

json_string = model.to_json()
with open(fileModelJSON,'w' ) as f:
   json.dump(json_string, f)
model.save_weights(fileWeights)

