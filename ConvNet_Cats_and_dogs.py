# Convolutional Neural Network(CNN)
# CNN's are capable of taking both 2D and 3D inputs
# this is different to the fully connected layer NN previously which needs
# flattened 1 dimensional inputs

# Example --> we input an image (2D array of pixels and we use convolutions which are a 3*3 convolutional kernel
# this does the dot product with each pixel in the image, this "condenses the image" into an image of features
# we then do "pooling" which uses the same 3*3 window and finds the max value within each window
# the first layer finds simple features (edges, corners etc) this is then fed into subsequent layers to combine
# edges, corners, curves so that things like shapes can be found which is then passed to the next layer and so on

# using the kaggle dataset of cats and dogs from microsoft

import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

REBUILD_DATA = False   # set to true if you its the first time running the dataset

# reshape/normalise the dataset
# you can resize and change aspect ratio or you can just pad white pixels in to make it square
# you can also flip/rotate these images to increase the size of your dataset
class DogsVSCats():
    IMG_SIZE = 50
    CATS = "C:\\Users\\Eamon\\Desktop\\PetImages\\PetImages\\Cat"   # directory to each dataset
    DOGS = "C:\\Users\\Eamon\\Desktop\\PetImages\\PetImages\\Dog"
    LABELS = {CATS: 0, DOGS: 1}
    training_data = []
    catcount = 0    # so we can ensure balance in dataset
    dogcount = 0    # one-hot [0,0] neither, [1,0] cat, [0,1] dog, [1,1] cat-dog?

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):    # tqdm is a progress bar
                try:   # handle errors if db contains bad images/fails to load
                    path = os.path.join(label, f)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # convert image to grayscale since we dont need colour
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) # resize the images to 50*50
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]]) # append the numpy array of the image + the class (one-hot)

                    # to help us determine database balance
                    if label == self.CATS:
                        self.catcount += 1
                    elif label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass
                    #print(str(e))
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Cats: ", self.catcount)
        print("Dogs: ", self.dogcount)
if REBUILD_DATA:
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()


# we're using this trick to create the one-hot vectors
# np.eye(2)[0] # [1 0]
# np.eye(2)[1] # [0 1]
# prevents us from needing to load data all the time
training_data = np.load("training_data.npy", allow_pickle = True)

print(training_data[0])

import matplotlib.pyplot as plt
# plt.imshow(training_data[5][0], cmap="gray")
# plt.show()

# Build the model 3 layer 2d convolutional network

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # initialise neural network conv layers and find output shape to flatten for forward linear layers
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 5*5 kernel 1 image input 32 convolutional features
        self.conv2 = nn.Conv2d(32, 64, 5) # input/output = 2 per layer
        self.conv3 = nn.Conv2d(64, 128, 5)

        # now we need to flatten it to create linear/fully-connected layers
        # torch doesnt have an inbuilt way to go from conv-linear layers
        # we need to find the input to the Linear layer

        # specify random 50*50 image (the input) which we reshape to 1*50*50 tensor of any size (-1)
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)
        # we need to know what to pass to the first linear layer for the input
        # to do this we run it (on random data) first check the shape and multiply the dimensions together

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512,2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        print(x[0].shape) # size of [128, 2, 2]
        # x is a batch of data we're taking the first element and taking the shape
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)    # pass through all conv layers
        x = x.view(-1, self._to_linear)    # flatten it
        x = F.relu(self.fc1(x))     # pass through first linear layer
        x = self.fc2(x)      # pass through 2nd linear layer
        return F.softmax(x, dim=1)

net = Net()

import torch.optim as optim

optimiser = optim.Adam(net.parameters(), lr=0.0005)
loss_function = nn.MSELoss() # mean squared error due to on-hot vectors

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0   # scale pixels to between 0 and 1
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size) #2494 values we're gonna train with

train_X = X[:-val_size]  # train on data up to sample size
train_y = y[:-val_size]

test_X = X[-val_size:] # test on data beyond sample size
test_y = y[-val_size:]

# print(len(train_X))
# print(len(test_X))

BATCHSIZE = 100
EPOCHS = 3

def train(net):
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCHSIZE)):
            #print(i, i+BATCHSIZE)
            batch_X = train_X[i:i+BATCHSIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i + BATCHSIZE]
            # we could use model.zero_grad or optimiser.zero_grad most of the time optimiser.zero_grad
            # in our case no real difference but sometimes there will be 2 different optimisers or NN's
            net.zero_grad()    # we can use this in most cases
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimiser.step()
            print(f" for EPOCH: {epoch}. loss is: {loss} ")



def test(net):
    correct = 0
    total = 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
            predicted_class = torch.argmax(net_out)
            total += 1
            if predicted_class == real_class:
                correct += 1
        print("Accuracy:", correct/total)


train(net)
test(net)


# Lets test the network on some images we can see!
#tensor(1) --> dog, tensor(0) --> cat
import matplotlib.pyplot as plt

print("image: 12")
# test on dog image 12
plt.imshow(test_X[12].view(50, 50), cmap= 'gray')
plt.show()
print (torch.argmax((net(test_X[12].view(-1, 1, 50, 50))[0])))


print("image: 40")
# test on image 40
plt.imshow(test_X[40].view(50, 50), cmap= 'gray')
plt.show()
print (torch.argmax((net(test_X[40].view(-1, 1, 50, 50))[0])))


print("image: 2000")
# test on image 2000
plt.imshow(test_X[2000].view(50, 50), cmap= 'gray')
plt.show()
print (torch.argmax((net(test_X[2000].view(-1, 1, 50, 50))[0])))

print("image: 2222")
plt.imshow(test_X[2222].view(50, 50), cmap= 'gray')
plt.show()
print (torch.argmax((net(test_X[2200].view(-1, 1, 50, 50))[0])))

print("image: 2250")
plt.imshow(test_X[2250].view(50, 50), cmap= 'gray')
plt.show()
print (torch.argmax((net(test_X[2250].view(-1, 1, 50, 50))[0])))


