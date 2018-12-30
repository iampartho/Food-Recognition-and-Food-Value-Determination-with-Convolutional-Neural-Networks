# Food Recognition and Food Value Determinaton Using Convolutional Neural Networks

This repository contains the code to extract data, train the data and classify data.
To train the data I have used the **google collaboratory** to use the powerful google's GPU.
I have shown the dataset structure(or directory) in my hard-drive in below.

# About my_code_data_extraction.py

In this code I have extracted the images from the dataset. Only 10 food classes are selected for this project
After extraction the image are preprocessed for **VGG19** model stantard (Or ImageNet challenge standard).
Then we **train_test_split** the dataset and save them in a file named **datafile_vgg.npz**

# About Project.ipynb

This is the google colaboratory file that we used to train our model. 
Since we used transfer learning we just train the last layer and freeze all other layers.
Here we import the file **datafile_vgg.npz** and use our preprocess image to train our model.
At running 35 epochs I received a 81.6% validation accuracy and saved the weights to the file **saved_weights_8160.hdf5**


# About my_cassifier.py

This is our classifier where we classify our input image against our 10 classes of food.
The input image is taken from the webcam (you can also give input pre-taken images if you tweak the code a little)
After image is taken and processed, then the **VGG19** model is defined and weights are loaded. Then the code predict
the image against 10 classes. After evaluating the class the food-value image of the corresponding class is displayed.

# Datas are in the following directory in the hard-drive:
*	'G:\Project\data\Burger\' [1000 pictures of hamburgers]
*	'G:\Project\data\chicken_curry\' [1000 pictures of chicken_curry]
*	'G:\Project\data\Chicken_wings\' [1000 pictures of Chicken_wings]
*	'G:\Project\data\club_sandwich\' [1000 pictures of club_sandwich]
*	'G:\Project\data\donuts\' [1000 pictures of donuts]
*	'G:\Project\data\French_fries\' [1000 pictures of French_fries]
*	'G:\Project\data\omelette\' [1000 pictures of omelette]
*	'G:\Project\data\pizza\' [1000 pictures of pizza]
*	'G:\Project\data\samosa\' [1000 pictures of samosa]
*	'G:\Project\data\spring_rolls\' [1000 pictures of spring_rolls] <br />
**Datasets are colleced from [Food-101](https://www.kaggle.com/kmader/food41)**

**Food value image are in the following directory in the hard-drive**
*		'G:\Project\Food Value\*

# Download link for the weight file of this project
[saved_weights_8160.hdf5](https://drive.google.com/open?id=1XOx4U1PtAE6JY2mEwBg6EoHQsaHCpEGS)
