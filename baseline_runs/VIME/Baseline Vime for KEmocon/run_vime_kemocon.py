#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 11:09:59 2025

@author: nmit
"""

"""
Runs VIME-Self (self-supervised learning) on the K-EmoCon dataset.
Minimal changes — reuses core logic from the VIME folder.
"""

import os, sys
import numpy as np

# Add path to VIME core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../VIME")))

# Imports from VIME
from vime_self import vime_self
from supervised_models import mlp
from vime_utils import perf_metric

# Dataset loader
from preprocessing_kemocon import load_kemocon_dataset

# Silence TensorFlow warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def run_vime_kemocon(csv_path, target_column="self_valence_binary"):
    """Train and evaluate VIME-Self on the K-EmoCon dataset."""
    # 1️⃣ Load dataset
    x_train, y_train, x_test, y_test, x_unlab = load_kemocon_dataset(
        csv_path=csv_path,
        target_column=target_column,
        label_data_rate=0.1,
        test_size=0.2,
        save_scaler=True if target_column == "self_valence_binary" else False
    )

    print(f"\n📊 Loaded K-EmoCon dataset: {target_column}")
    print("Train:", x_train.shape, "Test:", x_test.shape, "Unlab:", x_unlab.shape)

    # 2️⃣ Train VIME-Self encoder
    enc_params = {"epochs": 10, "batch_size": 128}
    encoder = vime_self(x_unlab, p_m=0.3, alpha=2.0, parameters=enc_params)

    # 3️⃣ Encode data and train supervised classifier (MLP)
    x_train_enc = encoder.predict(x_train)
    x_test_enc  = encoder.predict(x_test)

    mlp_params = {
        "hidden_dim": 100,
        "epochs": 100,       
        "activation": "relu",
        "batch_size": 128
    }
    y_pred = mlp(x_train_enc, y_train, x_test_enc, mlp_params)

    # 4️⃣ Evaluate & save metrics
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"vime_kemocon_{target_column}.csv")

    acc, auc, f1 = perf_metric(y_test, y_pred, save_path=save_path)
    print(f"\n✅ VIME-Self on {target_column}: "
          f"Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}")

    return acc, auc, f1


if __name__ == "__main__":
    csv_path = "/home/user/Work/LLM/Datasets/kemocon.csv"

    # 🔹 Valence (Self-Valence)
    run_vime_kemocon(csv_path, target_column="self_valence_binary")

    # 🔹 Arousal (Self-Arousal)
    run_vime_kemocon(csv_path, target_column="self_arousal_binary")