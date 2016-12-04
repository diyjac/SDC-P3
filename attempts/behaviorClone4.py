## Training model #4
##
## Nvidia's Model from https://arxiv.org/pdf/1604.07316v1.pdf
##
## import some useful python modules
import os
from pathlib import Path
import numpy as np
from numpy.random import random
import cv2
import pandas as pd
import json
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

## for reproducibility
# np.random.seed(178951)

## model input files for initial model save and retrain
fileDataPath = './drivingDataTrack1'
fileDataCSV = '/driving_log.csv'
fileModelJSON = 'model4.json'
fileWeights = 'model4.h5'

## model parameters defined here
# ch, img_rows, img_cols = 3, 160, 320  # camera format
ch, img_rows, img_cols = 3, 66, 200  # Nvidia's camera format
nb_classes = 1

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
# data = data[(data.throttle>0.0)&(data.brake==0.0)&(data.speed>0.0)]

# centered
#data = data[((data.steering>-0.01)|(data.steering<0.01))&(data.throttle>0.0)&(data.brake==0.0)&(data.speed>0.0)]
#X_train = np.copy(data['center'])

# left
#data = data[(data.steering<-0.05)&(data.throttle>0.0)&(data.brake==0.0)&(data.speed>0.0)]
#X_train = np.copy(data['left'])

# right
data = data[(data.steering>0.05)&(data.throttle>0.0)&(data.brake==0.0)&(data.speed>0.0)]
X_train = np.copy(data['right'])

Y_train = np.copy(data['steering'])
Y_train = Y_train.astype(np.float32)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=10)

batch_size = 20
samples_per_epoch = len(X_train)/batch_size
val_size = int(samples_per_epoch/10.0)
nb_epoch = 100


def load_image(imagepath):
    imagepath = imagepath.replace(' ', '')
    image = cv2.imread(imagepath, 1)
    image = cv2.resize(image, (img_rows, img_cols), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image

def batchgen(X, Y):
    while 1:
        for i in range(len(X)):
            imagepath, y = X[i], Y[i]
            image = load_image(imagepath)
            y = np.array([[y]])
            image = image.reshape(1, img_rows, img_cols, ch)
            yield image, y
input_shape = (img_rows, img_cols, ch)

# load previous session model and retrain
if Path(fileModelJSON).is_file():
    with open(fileModelJSON, 'r') as jfile:
       model = model_from_json(json.load(jfile))
    # load weights into new model
    # centered
    # adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    # left
    # adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    # right
    adam = Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
    model.load_weights(fileWeights)
    print("Loaded model from disk:")
    model.summary()
    
else:
    pool_size = (2, 3)
    model = Sequential()
    model.add(MaxPooling2D(pool_size=pool_size,input_shape=input_shape))
    model.add(Lambda(lambda x: x/127.5 - 1.))
    model.add(Convolution2D(5, 5, 24, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(5, 5, 36, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(5, 5, 48, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    #model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(1164))
    #model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(100))
    #model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(50))
    #model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(10))
    #model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
    model.summary()


history = model.fit_generator(batchgen(X_train, Y_train),
                    samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch,
                    validation_data=batchgen(X_val, Y_val),
                    nb_val_samples=val_size,
                    verbose=1)

print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
os.remove(fileModelJSON)
json_string = model.to_json()
with open(fileModelJSON,'w' ) as f:
   json.dump(json_string, f)
os.remove(fileWeights)
model.save_weights(fileWeights)

