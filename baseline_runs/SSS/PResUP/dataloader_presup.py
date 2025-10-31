#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:30:55 2025

@author: nmit
"""

# -*- coding: utf-8 -*-
"""
Custom dataloader for PResUP.csv
Performs 20:80 train-test split and MinMax normalization
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_presup_dataset(path, target_col):
    """
    Loads the PResUP dataset and returns train-test splits for given target.
    target_col: 'Valence' or 'Arousal'
    """
    df = pd.read_csv(path)

    # Drop non-feature columns
    feature_cols = [c for c in df.columns if c not in ["P_id", "Valence", "Arousal"]]
    X = df[feature_cols].values
    y = df[target_col].astype(int).values

    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 20:80 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
