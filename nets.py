# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:12:54 2022

@author: menghao.zhang
"""

import torch
import torch.nn as nn
from torchvision import models

class DOT_classification_Reconimages_withUS(nn.Module):
    def __init__(self):
        super(DOT_classification_Reconimages_withUS, self).__init__()
        # 3 Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))
        # Fully connected layers
        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 1)
        self.ReLu = torch.nn.ReLU()
    # Forward model
    def forward(self, x, us_features):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x1 = x
        x = torch.cat((x, us_features), 1)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.ReLu(self.linear1(x))
        outputs = self.linear2(x)

        return outputs, x1


class DOT_classification_Reconimages_only(nn.Module):
    def __init__(self):
        super(DOT_classification_Reconimages_only, self).__init__()
        # 3 Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))
        # Fully connected layers
        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 1)
        self.ReLu = torch.nn.ReLU()
    # Forward model
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x1 = x
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.ReLu(self.linear1(x))
        outputs = self.linear2(x)
        return outputs, x1
        
def get_vgg11():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    vgg = models.vgg11_bn(pretrained=True)
    # print(vgg16.classifier[6].out_features) # 1000 
    idx_layer = 0
    for param in vgg.features.parameters():
        idx_layer = idx_layer + 1
        if idx_layer > 9:
            break
        param.requires_grad = False
    # Newly created modules have require_grad=True by default
    feature_features = list(vgg.features.children())[:-4]
    # feature_features[18:21]=[]
    # the following's output is 8*8*64
    feature_features.extend([nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                              nn.ZeroPad2d((1,0,1,0)), 
                              nn.Conv2d(512, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                              nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 
                              nn.ReLU(inplace=True)])
    vgg.features = nn.Sequential(*feature_features)
    num_features = vgg.classifier[6].in_features
    classfy_features = list(vgg.classifier.children())[3:-1]
    classfy_features.extend([nn.Linear(num_features, 512), nn.ReLU(),
                              nn.Dropout(p=0.5, inplace=False),
                              nn.Linear(512, 1)
                              ]) 
    # classfy_features=[nn.Linear(256, 128), nn.ReLU(),nn.Dropout(p=0.5, inplace=False),nn.Linear(128, 2)]
    vgg.classifier = nn.Sequential(*classfy_features) # Replace the model classifier
    vgg.avgpool = nn.AdaptiveAvgPool2d(output_size=(8,8))
    return vgg

class DOT_classification_combine_imgandUS(nn.Module):
    def __init__(self):
        super(DOT_classification_combine_imgandUS, self).__init__()
        self.conv = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 1)
        self.ReLu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.ReLu(self.linear1(x))
        outputs = self.linear2(x)
        return outputs, x

class DOT_classification_combine_imgandUS_V2(nn.Module):
    def __init__(self):
        super(DOT_classification_combine_imgandUS, self).__init__()
        self.us = get_vgg11()
        self.dot = DOT_classification_Reconimages_only()
        self.conv = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.linear1 = nn.Linear(512, 64)
        self.linear2 = nn.Linear(64, 1)
        self.ReLu = torch.nn.ReLU()
        
    def forward(self, dot_input, us_input):
        # Using the trained vgg model to extracted features from the US images and histogram
        features = {}
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook
        self.us.features[-1].register_forward_hook(get_features('vgg11'))
        self.us.eval()
        us_results = self.us(us_input)
        us_features = features['vgg11'] 
        dot_pred, dot_features = self.dot(dot_input)
        x = torch.cat((dot_features, us_features), 1)
        x = self.conv(x)
        x = self.batch(x)
        x = self.ReLu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.ReLu(self.linear1(x))
        outputs = self.linear2(x)
        return outputs, x