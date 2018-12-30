# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 23:38:23 2018

@author: Partho
"""

####################### Library Imports ################################

import numpy as np
import cv2
from keras.applications import VGG19
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

################# Environment and Variables #############################

from keras import backend as K
K.set_image_data_format('channels_last')

seed = 7
np.random.seed(seed)
num_classes = 10
imageSize = 224

weightFile = 'G:/Project/saved_weights_8160.hdf5'
imagePath = 'G:/Project/Test_objects/test1234.jpg'
foodValueImg = 'G:/Project/Food Value'
captureImg = 'G:/Project/captured_image/captureImg.JPG'

####################### Taking the input image via webcam ###############################

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    cv2.imshow('Capturing', frame)
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break
cap.release()
cv2.destroyAllWindows()
img = frame 
cv2.imshow('Catured Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(captureImg, img)
del(img)
del(frame)



###################### Taking image input ###############################

img = image.load_img(imagePath, target_size=(imageSize, imageSize))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
X = np.asarray(x)
del(x)


################## Defining & Loading Model #############################
#definig the model

image_input = Input(shape=(224, 224, 3))

model = VGG19(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()

# freeze all the layers except the output layers
for layer in custom_vgg_model.layers[:-1]:
	layer.trainable = False

custom_vgg_model.summary()
del(model)

custom_vgg_model.load_weights(weightFile)

################### Predicting the Class ###############################

y = custom_vgg_model.predict(X)
print(y)
classno = np.argmax(y)
print(classno)
dict = {0 : 'Burger', 1 : 'chicken_curry', 2 : 'Chicken_wings', 3 : 'club_sandwich', 4 : 'donuts',
        5 : 'French_fries', 6 : 'omelette', 7 : 'pizza',8 : 'samosa',9 : 'spring_rolls'}
objectClass = dict[classno]
print(objectClass)

################### Previewing Prediction ##############################

for i in range(10):
    if i==classno:
        value_img = cv2.imread(foodValueImg + '/' + dict[i] + '.JPG')
        value_img = cv2.resize(value_img,(480,720))
        cv2.imshow('Food Value',value_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
