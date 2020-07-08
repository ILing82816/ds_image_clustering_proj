# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:02:28 2020

@author: USER
"""
from torch import nn
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # input is [3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),     #[64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),                              #[64, 16, 16]
            
            nn.Conv2d(64, 128, 3, stride=1, padding=1),   #[128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2),
                             #[128, 8, 8]
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  #[256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2)                               #[256, 4, 4]
        )
 
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 9, stride=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 17, stride=1),
            nn.Tanh()    
        )
        
    def forward(self, inputs):
        x1 = self.encoder(inputs)
        x = self.decoder(x1)
        return x1, x
 

