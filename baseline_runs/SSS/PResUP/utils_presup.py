#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:32:00 2025

@author: nmit
"""

# -*- coding: utf-8 -*-
"""
Utility functions for PResUP model training
"""

import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data_loaders(X_train, y_train, X_test, y_test, device, batch_size=64):
    """Converts numpy arrays to PyTorch dataloaders"""
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
