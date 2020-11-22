
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


# in order to validate your model you want out-of-sample testing data
# if you use in-sample data it may overfit where it basically only works with the training data
# when training it is important not to overtrain

train = datasets.MNIST("", train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))


# we could fit this small dataset through our NN in one go but its not practical to do this in most cases
# so instead we use batch sizes so we feed through 10 items at a time through our model and the model is optimised
# in tiny increments per batch
# in terms of how many neurons per layer its almost always trial and error its a gradient descent operation
# batches help precent overfitting and improves efficieny you dont want very big batch sizes often between 8 and 64
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

# we are using an MNIST dataset which is hand drawn number dataset handrawn 28*28images of handdrawn numbers
# we want it to learn general priniciples not to learn tricks shuffle reorders the dataset and helps with overfitting


# # iterate over data
# for data in trainset:
#     print(data)
#     break       # single iteration
#
# # we are trying to get the zeroeth object
# # data is tensor object 1st containing a tensor of tensors which is the images and 2nd containing a tensor of tensors
# # which is the labels
#
#
# x, y = data[0][0], data[1][0]
#
#
# # outputs torch.Size([1, 28, 28]) this is not a typical image
# # if you were to load in a 28*28 grayscale image and convert it to a tensor you'd get a [28],[28]
# # but pytorch wants it as [1][x][y]
#
# # print(data[0][0].shape)
#
# plt.imshow(data[0][0].view(28, 28))     # reshapes it to a 28*28
# plt.show()
#
# # if the data is 3% 1's 10% 5's and 60% 2's etc it will optimise to find 2's as quickly as possible and it'll get stuck
# # in that hole so you wont be able to train out of it
# # typically you want your dataset to be as balanced as possible
#
# # Lets look at dataset to see how balanced it is
# total = 0
# counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
#
# for data in trainset:
#     xs, ys = data # x's and y's stored in variables
#     for y in ys:
#         counter_dict[int(y)] += 1
#         total += 1
# print(counter_dict)
#
# for i in counter_dict:
#     print(f"{i}: {counter_dict[i]/total*100}")

# tutorial 3 -- making the neural network

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module): # learn about inheritance to understand this better
    def __init__(self):
        super().__init__() # initialise nn.Module

    #  define layer inputs and outputs
        # the input is our 28*28 images which is flattened giving you 784
        # output can be what ever we want
        self.fc1 = nn.Linear(28 * 28, 64)  # fc - fully connnected 1 is first layer
        self.fc2 = nn.Linear(64, 64)      # has to input 64 output could be what we want
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)      # 10 outputs for 10 numbers

    # Define feed forward NN layers
    def forward(self, x):
        x = F.relu(self.fc1(x))  # relu - rectified linear which is our activation function
    #  activation function determines whether or not neuron is firing or not firing
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # output layer doesnt use F.relu
        # we want a probability distribution on output so we use log soft max
        x = self.fc4(x)
        # you can put logic into this foward method with pytorch and isnt hard

        return F.log_softmax(x, dim=1)    # dim means 1 dimension we want this distribution to sum to 1





net = Net()
print(net)

# lets pass data through the network
# must be reshaped correctly
# X = torch.rand((28,28))
# X = X.view(-1, 28*28) # -1 specifies that its of an unknown shape
# output = net(X)
# print(output)

# now we need to calculate how far off we are
# i.e. we need to train the network

# two things we need to look at Loss and an Optimiser
# loss is a measure of how wrong the model is we want it to decrease during training
# the optimiser uses loss to adjust the possible weights that it can adjust so it can lower the loss


import torch.optim as optim

# parameters are the layers/nodes we want to be changed
# for example with transfer learning the first layers in the model will be very good at small/general image recognition
# whereas the later layers will be more specific to the layers its been trained on
# so in this case we'd want to freeze the first few layers and only do transfer learning on the later layers
optimiser = optim.Adam(net.parameters(), lr = 0.0001) # lr --> learning rate

# dictates size of the steps the optimiser takes if we make it learn to fast it wont be able to become optimal
# we want a big enough learning rate to avoid smaller learning "holes" without being so big so it never becomes optimal
# to address this you can have the learning rate descend/decay but we wont do it in this case

# an EPOCH is a full pass throguh the data
EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        # data is a batch of featruesets and labels
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        # get the loss because our data is a scalar we use nnl_loss
        # if it was a one hot vector (0,1,0,0) we'd use mean squared error
        loss = F.nll_loss(output, y)
        # back propagation handled by pytorch
        loss.backward()
        # adjust weights for us
        optimiser.step()
    print(loss)


correct = 0
total = 0

with torch.no_grad():
    for data in testset:
        x, y = data
        output = net(x.view(-1, 28*28))
        for idx, i in enumerate (output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

# accuracy seems too high (95%) but because of data set and set up is probably correct
# you have to be careful it isn't cheating and over-fitting
# typically this accuracy would be way too high
print("Accuracy: ", round(correct/total, 3))

#print(X)

import matplotlib.pyplot as plt
plt.imshow(x[0].view(28, 28))
plt.show()

#test with an image
print(torch.argmax(net(x[0].view(-1, 784))[0]))




