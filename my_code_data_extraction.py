# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:06:23 2018

@author: Partho
"""

#Necessary Libraries
import numpy as np
import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


######################          Parameters      ################################

datafile = 'G:/Project/datafile_vgg.npz'
PATH = 'G:/Project'


##########################  Creating datasets     ################################
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)
img_data_list=[]
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		img_data_list.append(x)

#######################     Make data as numpy array ###########################

img_data = np.asarray(img_data_list)
print (img_data.shape)

###############         Reshape the data for the model ###########################

img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

###############     Create Y for correspnding image data    #####################
num_classes = 10
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:1000]=0
labels[1000:2000]=1
labels[2000:3000]=2
labels[3000:4000]=3
labels[4000:5000]=4
labels[5000:6000]=5
labels[6000:7000]=6
labels[7000:8000]=7
labels[8000:9000]=8
labels[9000:]=9

names = ['Burger','chicken_curry','Chicken_wings','club_sandwich','donuts','French_fries','omelette','pizza','samosa','spring_rolls']
Y = np_utils.to_categorical(labels, num_classes)

#deleting variable to free up meomory
del(img_data_list)

#shuffle the data
x,y = shuffle(img_data,Y, random_state=2)

#################   delete variale to free up meomory   #########################
del(img_data)

#train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

#delete variale to free up meomory
del(x)
del(y)
del(Y)

###############     Save the datasets in a single file ##########################
np.savez(datafile, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test)