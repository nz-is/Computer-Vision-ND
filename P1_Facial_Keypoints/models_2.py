## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import logging
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

#Implementation of NaimishNet
class NaimishNet(nn.Module):
    def __init__(self):
        super(NaimishNet, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature map s, 5x5 square convolution kernel

        #Architecture summary
        #Activation Layer 1 to 5 is ELU
        #Activation Layer 6: Linear activation
        #Dropout is increased by stepsize .1  from .1 to .6from layer 1 to 6
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #LAYER 1
        #IN (1, 224, 224)
        #OUT conv1: ( 32, 221, 221)
        #maxpool1: (32, 110, 110)
        #Layer 1 out: (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.1)

        #LAYER 2
        #IN ( 32, 110, 110)
        #conv2 ( 64, 108, 108)
        #maxpool2: (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)

        self.dropout2 = nn.Dropout2d(0.2)

        #LAYER 3
        #IN (64, 53, 53)
        #Conv3: (128, 52, 52)
        #maxpool: (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout2d(0.3)

        #Layer 4
        #IN (128, 26, 26)
        #conv4: (256, 26, 26)
        #maxpool4: ( 256, 13, 13)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout4 = nn.Dropout2d(0.4)

        #IN( 256, 5, 5)
        #Flatten (256 * 13* 13)
        self.fc1 = nn.Linear(in_features=13*13*256, out_features=1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.dropout5 = nn.Dropout2d(0.5)

        #Layer 6
        self.fc2 = nn.Linear(in_features=1000, out_features=500)
        self.bn6 = nn.BatchNorm1d(500)
        self.dropout6 = nn.Dropout2d(0.6)

        #Layer 7
        #OUT FKP: (X, Y)
        self.fc3 = nn.Linear(in_features=500, out_features=136)

        #Custom weights initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = I.uniform_(m.weight, a = 0.0, b = 1.0)
            elif isinstance(m, nn.Linear):
                m.weight= I.xavier_uniform_(m.weight, gain=1)


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        #LAYER 1
        #IN (1, 224, 224)
        #OUT conv1: ( 32, 221, 221)
        #maxpool1: ( 32, 110, 110)
        #Layer 1 out: ( 32, 110, 110)
        x = self.pool(F.elu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        #print(f"CONV1: {x.size()}")

        #LAYER 2
        #IN ( 32, 110, 110)
        #conv2 ( 64, 44, 44)
        #maxpool2: (64, 22, 22)
        x = self.pool(F.elu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        #print(f"CONV2: {x.size()}")

        #LAYER 3
        #IN (64, 22, 22)
        #Conv3: (128, 21, 21)
        #maxpool: (128, 10, 10)
        x = self.pool(F.elu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        #print(f"CONV3: {x.size()}")

        #Layer 4
        #IN (128, 10, 10)
        #conv4: (256, 10, 10)
        #maxpool4: ( 256, 5, 5)
        x = self.pool(F.elu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)

        #Layer 5
        #print(f"CONV4: {x.size()}")
        #Flatten before fc1
        x = x.view(x.size(0), -1)

        #Layer 5
        #IN: ( 256, 5, 5)
        #Flatten (256 * 5 * 5) = 6400
        #OUT: (1000)
        x = F.elu(self.bn5(self.fc1(x)))
        x = self.dropout5(x)

        #Layer 6
        #IN: (1000)
        #out: (500)
        x = torch.tanh(self.bn6(self.fc2(x)))
        x = self.dropout6(x)

        #Final Layer 7
        #OUT FKP: (X, Y)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
