import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from model import Net, FinalClassifier
from utils_presup import get_data_loaders

# ----------------------------
# CONFIGURATION
# ----------------------------
data_path = "PResUP.csv"   # path to your dataset
target = "Arousal"         # change to "arousal" when needed
EPOCHS = 30
RUNS = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD DATA
# ----------------------------
data = pd.read_csv(data_path)
X = data.drop(columns=[target])
y = data[target].values

# Split 20% train, 80% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42, stratify=y
)

# ----------------------------
# STORAGE FOR RESULTS
# ----------------------------
all_acc, all_f1 = [], []

# ----------------------------
# 5 RUNS LOOP
# ----------------------------
for run in range(RUNS):
    print(f"\n========== Run {run+1}/{RUNS} ==========")

    # Prepare data loaders
    train_loader, test_loader = get_data_loaders(
        X_train.values, y_train, X_test.values, y_test, DEVICE
    )

    # Model setup
    model = Net(input_dim=X_train.shape[1]).to(DEVICE)
    classifier = FinalClassifier(in_dim=8).to(DEVICE)
    optimizer = optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    train_acc_list, test_acc_list = [], []
    train_loss_list, test_loss_list = [], []

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------
    for epoch in range(EPOCHS):
        model.train()
        classifier.train()
        total_loss = 0
        y_true_train, y_pred_train = [], []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            features = model(X_batch)
            outputs = classifier(features)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            y_true_train.extend(y_batch.cpu().numpy())
            y_pred_train.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_loss = total_loss / len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # ----- EVALUATION -----
        model.eval()
        classifier.eval()
        y_true_test, y_pred_test = [], []
        total_test_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                features = model(X_batch)
                outputs = classifier(features)
                loss = criterion(outputs, y_batch)
                total_test_loss += loss.item()
                y_true_test.extend(y_batch.cpu().numpy())
                y_pred_test.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_loss = total_test_loss / len(test_loader)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
        )

    # ----------------------------
    # FINAL EVALUATION
    # ----------------------------
    final_acc = test_acc_list[-1]
    final_f1 = f1_score(y_true_test, y_pred_test, average="macro")
    all_acc.append(final_acc)
    all_f1.append(final_f1)

    print(f"Run {run+1} Final Accuracy: {final_acc:.4f}, F1: {final_f1:.4f}")

    # ----------------------------
    # PLOTS FOR THIS RUN
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc_list, label="Train Accuracy", marker="o")
    plt.plot(test_acc_list, label="Test Accuracy", marker="o")
    plt.title(f"{target.capitalize()} - Accuracy (Run {run+1})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(train_loss_list, label="Train Loss", marker="o")
    plt.plot(test_loss_list, label="Test Loss", marker="o")
    plt.title(f"{target.capitalize()} - Loss (Run {run+1})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# ----------------------------
# SUMMARY ACROSS RUNS
# ----------------------------
mean_acc = np.mean(all_acc)
std_acc = np.std(all_acc)
mean_f1 = np.mean(all_f1)
std_f1 = np.std(all_f1)

print("\n========== Final Results ==========")
print(f"{target.capitalize()} - Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
print(f"{target.capitalize()} - F1 Score: {mean_f1:.3f} ± {std_f1:.3f}")
