#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A single caller script to train the Self-SLAM (HEXR) encoder 
and run downstream supervised models on the CASE dataset.
"""

import os
import sys
import numpy as np
import pandas as pd

# Make sure Python can import the user's local modules
sys.path.insert(0, os.getcwd())

# Imported user modules
from Preprocessing_presup import load_presup_dataset
from hexr_self import hexr_self
import supervised_models as supm
import utils


def main(args):
    """
    Runs the SSLAM pipeline once — from preprocessing to evaluation.
    Returns a dict of results {'acc': value, 'f1': value}.
    """
    print(f"📂 Loading PresUP dataset from {args.dataset_path} ...")
    X_train, y_train, X_test, y_test, X_unlab = load_presup_dataset(
        dataset_path=args.dataset_path,
        target_column=args.target_column,
        label_data_rate=args.label_data_rate,
        test_size=args.test_size,
        save_scaler=True
    )

    print("\n✅ Dataset loaded successfully!")
    print("Shapes:")
    print("  X_unlab:", X_unlab.shape)
    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)
    print("  X_test:", X_test.shape)
    print("  y_test:", y_test.shape)

    # Encoder training parameters
    parameters = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'activation': args.activation
    }

    # ------------------------------
    # 1️⃣ SELF-SUPERVISED PRETRAINING
    # ------------------------------
    print("\n🚀 Training Self-SLAM encoder (HEXR)...")
    model, encoder = hexr_self(
        x_unlab=X_unlab,
        p_m=args.p_m,
        alpha=args.alpha,
        parameters=parameters
    )

    # ------------------------------
    # 2️⃣ REPRESENTATION EXTRACTION
    # ------------------------------
    print("\n🔍 Generating latent representations...")
    X_train_repr = encoder.predict(X_train)
    X_test_repr = encoder.predict(X_test)

    # ------------------------------
    # 3️⃣ SUPERVISED TRAINING
    # ------------------------------
    print("\n🎯 Training supervised model:", args.supervised)
    sup = args.supervised.lower()

    if sup == 'mlp':
        y_test_hat = supm.mlp(X_train_repr, y_train, X_test_repr, parameters)
    elif sup in ['logit', 'logistic']:
        y_test_hat = supm.logit(X_train_repr, y_train, X_test_repr)
    elif sup in ['xgb', 'xgboost']:
        y_test_hat = supm.xgb_model(X_train_repr, y_train, X_test_repr)
    else:
        raise ValueError(f"Unknown supervised model: {args.supervised}. Choose mlp/logit/xgb.")

    # ------------------------------
    # 4️⃣ EVALUATION
    # ------------------------------
    metrics = [m.lower() for m in args.metric]
    results = {}

    for m in metrics:
        perf = utils.perf_metric(m, y_test, y_test_hat)
        results[m] = perf
        print(f"📊 {m.upper()}: {perf:.6f}")

    return results


# ==========================================================
# 🔁 Multiple runs for stability + Excel summary
# ==========================================================
def run_multiple_times(args, num_runs=5):
    """Run the SSL model multiple times and aggregate results."""
    all_results = {m.lower(): [] for m in args.metric}

    for run in range(num_runs):
        print(f"\n==============================")
        print(f"   🚀 Run {run + 1}/{num_runs}")
        print(f"==============================")
        results = main(args)

        for k, v in results.items():
            all_results[k].append(v)

    # Compute Mean and Std
    summary = {}
    for m, vals in all_results.items():
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        summary[m] = {"mean": mean_val, "std": std_val}
        print(f"\n✅ {m.upper()} Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")

    # Save all results to Excel
    os.makedirs(args.output_dir, exist_ok=True)
    rows = [[m, vals["mean"], vals["std"]] for m, vals in summary.items()]
    results_df = pd.DataFrame(rows, columns=["Metric", "Mean", "StdDev"])
    excel_path = os.path.join(args.output_dir, "results_5runs.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"\n💾 Final averaged results saved to Excel: {excel_path}")

    return summary


# ==========================================================
# 🧠 Run Configuration
# ==========================================================
if __name__ == "__main__":
    class Args:
        dataset_path = "PResUP.csv"     # ✅ use exact file name or full path
        target_column = "Valence"
        label_data_rate = 0.1
        test_size = 0.8
        p_m = 0.3
        alpha = 1.0
        epochs = 50
        batch_size = 128
        hidden_dim = 128
        num_layers = 1
        activation = "relu"
        supervised = "mlp"
        metric = ["acc", "f1"]           # ✅ Multiple metrics supported
        output_dir = "./sslam_outputs"
        save_encoder = None

    args = Args()
    summary = run_multiple_times(args, num_runs=5)
