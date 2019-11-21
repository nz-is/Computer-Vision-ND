import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import logging
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models
from collections import OrderedDict

#Implementation of NaimishNet
class NaimishNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=[32, 64, 128, 256],
                 dropout_prob=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]):
        super(NaimishNet, self).__init__()

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
        self.conv1 = nn.Sequential(
            OrderedDict([
            ('conv1',  nn.Conv2d(in_channel, out_channel[0], 4)),
            ('elu_1', nn.ELU()),
            ('bn1', nn.BatchNorm2d(out_channel[0])),
            ('dropout_1', nn.Dropout2d(dropout_prob[0])) ]
            )
        )

        #LAYER 2
        #IN ( 32, 110, 110)
        #conv2 ( 64, 108, 108)
        #maxpool2: (64, 53, 53)
        self.conv2 = nn.Sequential(
            OrderedDict([
            ('conv2',  nn.Conv2d(out_channel[0], out_channel[1], 3)),
            ('elu_2', nn.ELU()),
            ('bn2', nn.BatchNorm2d(out_channel[1])),
            ('dropout_2', nn.Dropout2d(dropout_prob[1])) ]
            )
        )

        #LAYER 3
        #IN (64, 53, 53)
        #Conv3: (128, 52, 52)
        #maxpool: (128, 26, 26)
        self.conv3 = nn.Sequential(
            OrderedDict([
            ('conv3',  nn.Conv2d(out_channel[1], out_channel[2], 2)),
            ('elu_3', nn.ELU()),
            ('bn3', nn.BatchNorm2d(out_channel[2])),
            ('dropout_3', nn.Dropout2d(dropout_prob[2])) ]
            )
        )

        #Layer 4
        #IN (128, 26, 26)
        #conv4: (256, 26, 26)
        #maxpool4: ( 256, 13, 13)
        self.conv4 = nn.Sequential(
            OrderedDict([
            ('conv4',  nn.Conv2d(out_channel[2], out_channel[3], 1)),
            ('elu_4', nn.ELU()),
            ('bn4' , nn.BatchNorm2d(out_channel[3])),
            ('dropout_4', nn.Dropout2d(dropout_prob[3])) ]
            )
        )

        #IN( 256, 5, 5)
        #Flatten (256 * 13* 13)
        self.fc1 = nn.Sequential(
            OrderedDict([
                         ('fc1', nn.Linear(in_features=13*13*256, out_features=1000)),
                          ('elu_5' , nn.ELU()),
                          ('bn5' , nn.BatchNorm1d(1000)),
                         ('dropout_5' , nn.Dropout2d(dropout_prob[4]))
            ])
        )

        self.fc2 = nn.Sequential(
            OrderedDict([
                         ('fc2' , nn.Linear(in_features=1000, out_features=500)),
                          ('tanh_6' , nn.Tanh()),
                          ('bn6' , nn.BatchNorm1d(500)),
                         ('dropout_6' , nn.Dropout2d(dropout_prob[5]))
            ])
        )

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
        #LAYER 1
        #IN (1, 224, 224)
        #OUT conv1: ( 32, 221, 221)
        #maxpool1: ( 32, 110, 110)
        #Layer 1 out: ( 32, 110, 110)
        x = self.pool(self.conv1(x))
        #print(f"CONV1: {x.size()}")

        #LAYER 2
        #IN ( 32, 110, 110)
        #conv2 ( 64, 44, 44)
        #maxpool2: (64, 22, 22)
        x = self.pool(self.conv2(x))
        #print(f"CONV2: {x.size()}")

        #LAYER 3
        #IN (64, 22, 22)
        #Conv3: (128, 21, 21)
        #maxpool: (128, 10, 10)
        x = self.pool(self.conv3(x))
        #print(f"CONV3: {x.size()}")

        #Layer 4
        #IN (128, 10, 10)
        #conv4: (256, 10, 10)
        #maxpool4: ( 256, 5, 5)
        x = self.pool(self.conv4(x))

        #Layer 5
        #print(f"CONV4: {x.size()}")
        #Flatten before fc1
        x = x.view(x.size(0), -1)

        #Layer 5
        #IN: ( 256, 5, 5)
        #Flatten (256 * 5 * 5) = 6400
        #OUT: (1000)
        x = self.fc1(x)

        #Layer 6
        #IN: (1000)
        #out: (500)
        x = self.fc2(x)

        #Final Layer 7
        #OUT FKP: (X, Y)
        x = self.fc3(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2, 2), padding=(3,3), bias=False)
        n_inputs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(n_inputs, 136)

    def forward(self, x):
        x = self.resnet18(x)
        return x
