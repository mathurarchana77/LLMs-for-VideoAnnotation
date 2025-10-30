#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 16:27:49 2025

@author: nmit
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

def load_presup_dataset(dataset_path, target_column, label_data_rate=0.1, save_scaler=True , feature_cols=None, test_size=0.8):
    """
    Loads and preprocesses a physiological dataset for emotion classification.

    Args:
        dataset_path (str): Path to the CSV dataset file.
        target_column (str): Column name of the target variable (e.g., 'class1' or 'class2').
        feature_cols (list): List of feature column names to use. If None, defaults to CASE dataset features.
        label_data_rate (float): Fraction of training data to treat as labelled (for semi-supervised setups).
        test_size (float): Proportion of data reserved for testing.
        save_scaler (bool): Whether to fit and save a new scaler or load an existing one.

    Returns:
        x_train (np.ndarray): Labelled training features
        y_train (np.ndarray): Labelled training labels
        x_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        x_unlab (np.ndarray): Unlabelled training features
    """
    # load case dataset
    df = pd.read_csv(dataset_path)

    if feature_cols is None:
        feature_cols = ['mean_GSR', 'std_GSR', 'mean_HR', 'std_HR', 'change_score']

    # one-hot encode class column
    ohe = OneHotEncoder()

    df_ohe = pd.DataFrame(ohe.fit_transform(df[[target_column]]).toarray())
    df_ohe.columns = [f"{target_column}_{i}" for i in range(df_ohe.shape[1])]
    df = pd.concat([df, df_ohe], axis=1)

    label_cols = list(df_ohe.columns)
    df = df[feature_cols + label_cols]

    X = df[feature_cols]
    Y = df[label_cols]

    scaler_path = f"{target_column}_scaler.pkl"

    # feature scaling
    if save_scaler:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X[feature_cols])
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler for '{target_column}' fitted and saved as '{scaler_path}'.")
    else:
        # 🆕 Load the existing scaler and use transform instead of fit_transform
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X[feature_cols])
        print(f"✅ Loaded existing scaler: {scaler_path}")

    """
    Train-test split
    20% :- Test 
    10% of 80% = 8% :- Labelled dataset
    90% of 80% = 72% :- Unlabelled dataset
    """
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=42, stratify=Y
    )

    # converting to numpy arrays
    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values

    # Divide labeled and unlabeled data
    idx = np.random.permutation(len(y_train))

    # Label data : Unlabeled data = label_data_rate:(1-label_data_rate)
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    # Unlabeled data
    x_unlab = x_train[unlab_idx, :]

    # Labeled data
    x_train = x_train[label_idx, :]
    y_train = y_train[label_idx, :]

    return x_train, y_train, x_test, y_test, x_unlab


# Quick test
if __name__ == "__main__":
    x_train, y_train, x_test, y_test, x_unlab = load_presup_dataset(
        dataset_path="PResUP.csv",
        target_column="Arousal",
        feature_cols=None,
        label_data_rate=0.1,
        test_size=0.8)
    print("✅ Preprocessing complete!")
    print("Train shape:", x_train.shape)
    print("Test shape:", x_test.shape)
    print("Unlabelled shape:", x_unlab.shape)