#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:31:41 2025

@author: nmit
"""

# -*- coding: utf-8 -*-
"""
Model definitions for PResUP
Simple fully connected neural network
"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)   # Output is 8 features
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x   # returns an 8-dimensional feature vector



class FinalClassifier(nn.Module):
    def __init__(self, in_dim):
        super(FinalClassifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, 8)
        self.relu = nn.ReLU()
        self.out = nn.Linear(8, 2)  # binary classification (Valence or Arousal)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.out(x)
        return x
