import numpy as np
import torch.nn as nn


class DilatedConv(nn.Module):
    '''Custom class that implements Dilated convolution layer that can be used inplace of MaxPool2D layer'''
    def __init__(self, in_channels, out_channels, kernel_size, dilation_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=dilation_factor, dilation=dilation_factor)

    def forward(self, x):
        x = self.conv(x)
        return x



class CIFAR10CNN3(nn.Module):
    '''This module uses DilatedConv module'''
    def __init__(self, num_classes=10):
        super(CIFAR10CNN3, self).__init__()

        # Convolution block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, groups=32),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )


        # Convolution block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Dilation convolution instead of MaxPooling Layer; has the same RF impact that of MaxPooling.
            DilatedConv(in_channels=64, out_channels=64, kernel_size=3, dilation_factor=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) 


        # Dilated Convolution block - convolution block 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Depth-wise separable Convolution - block 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, groups=128),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        # Global Average Pooling Layer.
        self.gap = nn.AvgPool2d(kernel_size=22)

        # A single FC Layer to get the number of classes.
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x



class CIFAR10CNN2(nn.Module):
    '''This module uses MaxPool2D layer'''
    def __init__(self, num_classes=10):
        super(CIFAR10CNN2, self).__init__()

        # Convolution block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, groups=32),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
   
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Convolution block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Contains MaxPooling layer instead of Dilated Convolution layer.
            nn.MaxPool2d((2,2)),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Convolution block 3 - Dilated convolution block
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ) 

        # Convlution block 4 - Depth-wise seperable convolution
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, groups=128),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64)
        )

        # Global Average Pooling Layer.
        self.gap = nn.AvgPool2d(kernel_size=5)
        # Final FC Layer.
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        return x
