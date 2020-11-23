# Pytorch NeuralNet Tutorial Code Learning Tool

Sample Code of a Linear Neural Network and a Convolutional Neural Network created in Pytorch running on CPU with Windows 10 with comments throughout explaining the basics of deep-learning within Pytorch.  

NOTE: it is assumed python and pytorch is installed.

The Linear Neural Network is 4 layer fully connected neural network that inputs flattened 28*28 images from the MNEST dataset
and predicts the value of hand drawn integers from 0-9. In this example the MNEST dataset is acquired within the code and does not need to be sourced elsewhere.

The Convolutional Neural Network consists of 3 convolutional layers followed by 2 linear/fully-connected layers. The first convolutional layer inputs 50 by 50 images and uses a 5*5 convolutional kernel to output 32 convolutonal features, the follower layer outputs 64 features and the third layer outputs 128. Random sample data is thenused to determine the shape of this layers output so that it can be flattened and passed into the subsequent fully-connected layers.

The network uses the kaggle "cats and dogs" dataset from Microsoft https://www.microsoft.com/en-us/download/details.aspx?id=54765 which needs to be downloaded and the directory within the code needs to be changed to the folder location. The network outputs a tensor value of 1 or 0 where 1 is a "dog" prediction and 0 is a "cat" prdiction. 

These files were created by following the "sentdex" video tutorial pytorch series on YouTube. ---> https://www.youtube.com/channel/UCfzlCWGWYyIQ0aLC5w48gBQ
