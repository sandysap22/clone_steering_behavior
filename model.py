import os
import csv
import sklearn
import matplotlib.pyplot as plt

import cv2
import numpy as np
import sklearn


from keras.models import Sequential
from keras.layers import Cropping2D,Lambda, ELU
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

print("Started")

"""
Read driving_log to get samples 
file has following details : center camera image path,left camera image path ,
right camera image path,steering, throttle, brake and	speed
"""

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader,None) # skip header
    for line in reader:
        samples.append(line)

samples=samples[:8000] # divisible by 32
print('number of samples :', len(samples))
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

BATCH_SIZE=32
    
# Read images and steering angles
# Adjust steering angle for left and right angle (treat it as vehicle is at left and streer to right by 0.15) 
# Add reverse lap (if image is for right turning then it will converted to left turning and ajust steering value by multiplying by -1
def get_images_and_steering_angles(batch_sample):
    
    # create adjusted steering measurements for the side camera images
    steering_center = float(batch_sample[3])
    correction=0.15 # this is a parameter to tune
    
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # read in images from center, left and right cameras
    path = './data/IMG/' 
    
    img_center = cv2.imread(path + batch_sample[0].split('/')[-1])
    img_center = cv2.cvtColor(img_center,cv2.COLOR_BGR2RGB)
    img_left =   cv2.imread(path + batch_sample[1].split('/')[-1])
    img_left = cv2.cvtColor(img_left,cv2.COLOR_BGR2RGB)
    img_right =  cv2.imread(path + batch_sample[2].split('/')[-1])
    img_right = cv2.cvtColor(img_right,cv2.COLOR_BGR2RGB)

    # add flipped images 
    img_center_fliped = cv2.flip(img_center,1) # flip along virtical axis
    img_left_fliped = cv2.flip(img_left,1)
    img_right_fliped = cv2.flip(img_right,1)
    
    car_images = [img_center, img_left, img_right,img_center_fliped,img_left_fliped,img_right_fliped]
    steering_angles = [steering_center, steering_left, steering_right,-1.0 * steering_center, -1.0 * steering_left, -1.0 *steering_right] 
    
    return car_images, steering_angles   

"""
Generate to provide images from disk in batches. 
model.fit_generator will indirectly call it.
"""
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                car_images,steering_angles=get_images_and_steering_angles(batch_sample) 
                car_images,steering_angles = sklearn.utils.shuffle(car_images,steering_angles)
                images.extend(car_images)
                angles.extend(steering_angles)
              
            X_train = np.array(images)
            y_train = np.array(angles)            
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=BATCH_SIZE) # will get 16*3*2 = 96 images per batch
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


"""
Nvidia model with chnaged input shape
Architecture
Cropping 
Image normalization to avoid saturation
Convolution : 5x5 filters=24, strides=2x2, activation=ELU
Convolution : 5x5 filters=36, strides=2x2, activation=ELU
Convolution : 5x5 filters=48, strides=2x2, activation=ELU
Convolution : 3x3 filters=64, strides=1x1, activation=ELU
Convolution : 3x3 filters=64, strides=1x1, activation=ELU

ELU : Expontential linear unit : this activation takes care of vanishing gradient problem

"""

def get_nvidia_model():
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    # crop images 50 pixels from top and 20 pixels from bottom
    # normalization
    model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((65,20), (0,0)))) #  output = 3@90x320
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2),name='convo1')) # border_mode==padding # subsample == strides = out 24@36x158
    model.add(Activation('elu')) # exponetial linera units 
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2),name='convo2')) # output 36@16x77
    model.add(Activation('elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2),name='convo3')) # output 48@6x37
    model.add(Activation('elu'))
    model.add(Convolution2D(64, 3, 3, name='convo4')) # output 64@4x35
    model.add(Activation('elu'))
    model.add(Convolution2D(64, 3, 3, name='convo5')) # output 64@2x33
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dropout(0.50))
    model.add(Dense(100)) # output 100
    model.add(Activation('elu'))
    model.add(Dropout(0.50))
    model.add(Dense(50))
    model.add(Activation('elu'))
    model.add(Dropout(0.20))
    model.add(Dense(10))
    model.add(Activation('elu'))
    model.add(Dense(1))
    
    return model



print('Building model')

# get model defination
model=get_nvidia_model()

#Compile and train the model using the generator function
print('Complining model')
model.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
print(model.summary())

samples_per_epoch = len(samples) * 6 # each pass in samples produce 6 times images : 3(centre,left,right) + 3 filpped.

print('Training model')
history_object= model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch , 
			validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=5)

model_file='model.h5'
# Save model to disk that would be used by drive.py to give commands to simulator
print('Saving model',model_file)
model.save(model_file)

# Save loss and accuracy details
print('saving history')
import pickle
with open(model_file.replace('h5','pkl'), 'wb') as pickle_file:
    pickle.dump(history_object.history, pickle_file)