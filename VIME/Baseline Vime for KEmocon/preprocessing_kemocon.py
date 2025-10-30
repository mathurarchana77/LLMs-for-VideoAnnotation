#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 15:50:22 2025

@author: nmit
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib


def load_kemocon_dataset(csv_path, target_column, label_data_rate=0.1, test_size=0.2,
                         save_scaler=True, feature_cols=None):
    """
    Loads preprocessed K-EmoCon dataset from CSV and prepares it for modeling.

    Args:
        csv_path (str): Path to the preprocessed K-EmoCon CSV file.
        target_column (str): Column name for the target label (e.g., 'valence_class' or 'arousal_class').
        label_data_rate (float): Fraction of training data to keep labeled (for semi-supervised setup).
        test_size (float): Fraction of data to reserve for testing.
        save_scaler (bool): Whether to save or load the MinMaxScaler.

    Returns:
        x_train, y_train, x_test, y_test, x_unlab : np.ndarray
    """
    # 1️⃣ Load CSV
    df = pd.read_csv(csv_path)

    # 2️⃣ Separate features and label
    if feature_cols is None:
        feature_cols = ['ecg_mean_hr', 'bvp_mean_hr', 'bvp_rmssd', 'eda_mean', 'eda_std', 'temp_mean']

    # one-hot encode class column
    ohe = OneHotEncoder()

    df_ohe = pd.DataFrame(ohe.fit_transform(df[[target_column]]).toarray())
    df_ohe.columns = [f"{target_column}_{i}" for i in range(df_ohe.shape[1])]
    df = pd.concat([df, df_ohe], axis=1)

    label_cols = list(df_ohe.columns)
    df = df[feature_cols + label_cols]

    X = df[feature_cols]
    Y = df[label_cols]

    target_name = target_column.replace("_class", "")
    scaler_path = f"kemocon_scaler_{target_name}.pkl"

    # feature scaling
    if save_scaler:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X[feature_cols])
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler for '{target_name}' fitted and saved as '{scaler_path}'.")
    else:
        # 🆕 Load the existing scaler and use transform instead of fit_transform
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X[feature_cols])
        print(f"✅ Loaded existing scaler: {scaler_path}")

    # 4️⃣ Split into train-test
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, Y, test_size=test_size, random_state=42, stratify=Y
    )

    # 5️⃣ Further divide training into labeled & unlabeled
    idx = np.random.permutation(len(y_train))
    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    x_lab, y_lab = x_train[label_idx], y_train.values[label_idx]
    x_unlab = x_train[unlab_idx]

    return x_lab, y_lab, x_test, y_test.values, x_unlab


if __name__ == "__main__":
    x_train, y_train, x_test, y_test, x_unlab = load_kemocon_dataset(
        csv_path="/home/user/Work/LLM/Datasets/kemocon.csv",
        target_column="self_arousal_binary",
        label_data_rate=0.1,
        test_size=0.2
    )
    print("✅ Preprocessing complete!")
    print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}, Unlabeled shape: {x_unlab.shape}")