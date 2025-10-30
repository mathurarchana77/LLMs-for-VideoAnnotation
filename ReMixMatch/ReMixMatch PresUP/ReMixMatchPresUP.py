# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:46:24 2025

@author: mathu
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
class Config:
    batch_size = 32
    lr = 1e-4
    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unlabeled_ratio = 2
    temperature = 0.5
    alpha = 0.75
    test_split = 0.2

cfg = Config()

# ========== DATA PREPARATION ==========
data = pd.read_csv("PResUP.csv")

features = ['mean_GSR', 'std_GSR', 'mean_HR', 'std_HR', 'change_score']
target = 'Arousal'

# Normalize features
X = data[features].values
X = (X - X.mean(axis=0)) / X.std(axis=0)
y = data[target].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split labeled/unlabeled
n_total = len(X)
n_labeled = int(0.6 * n_total)
indices = torch.randperm(n_total)
labeled_idx = indices[:n_labeled]
unlabeled_idx = indices[n_labeled:]

X_labeled, y_labeled = X[labeled_idx], y[labeled_idx]
X_unlabeled, _ = X[unlabeled_idx], y[unlabeled_idx]

# Split test set
n_test = int(cfg.test_split * len(X_labeled))
X_train, X_test = X_labeled[:-n_test], X_labeled[-n_test:]
y_train, y_test = y_labeled[:-n_test], y_labeled[-n_test:]

# Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self.has_labels = y is not None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.has_labels:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

labeled_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
unlabeled_loader = DataLoader(TabularDataset(X_unlabeled), batch_size=cfg.batch_size * cfg.unlabeled_ratio, shuffle=True)
test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)

# ========== MODEL ==========
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

model = MLPClassifier(len(features)).to(cfg.device)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
criterion = nn.CrossEntropyLoss()

# ========== REMIXMATCH FUNCTIONS ==========
def mixup(x1, y1, x2, y2, alpha=0.75):
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)
    x_mix = l * x1 + (1 - l) * x2
    y_mix = l * y1 + (1 - l) * y2
    return x_mix, y_mix

def sharpen(p, T=0.5):
    p = p ** (1 / T)
    return p / p.sum(dim=1, keepdim=True)

def distribution_alignment(p_model, mean_p_model):
    p_target = p_model / mean_p_model
    return p_target / p_target.sum(dim=1, keepdim=True)

mean_p_model = torch.ones(2).to(cfg.device) / 2

# ====== Trackers ======
epoch_losses, epoch_sup, epoch_unsup = [], [], []
epoch_acc, epoch_f1 = [], []

# ========== TRAINING LOOP ==========
for epoch in range(cfg.num_epochs):
    model.train()
    total_loss, total_supervised, total_unsupervised = 0, 0, 0
    unlabeled_iter = iter(unlabeled_loader)

    for x_l, y_l in labeled_loader:
        try:
            x_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_u = next(unlabeled_iter)

        x_l, y_l, x_u = x_l.to(cfg.device), y_l.to(cfg.device), x_u.to(cfg.device)

        # ====== Pseudo-labeling ======
        with torch.no_grad():
            logits_u = model(x_u)
            probs_u = F.softmax(logits_u, dim=1)
            mean_p_model = 0.999 * mean_p_model + 0.001 * probs_u.mean(dim=0)
            probs_u_aligned = distribution_alignment(probs_u, mean_p_model)
            probs_u_sharpened = sharpen(probs_u_aligned, cfg.temperature)

        # ====== MixUp ======
        y_l_onehot = F.one_hot(y_l, num_classes=2).float()
        all_x = torch.cat([x_l, x_u], dim=0)
        all_y = torch.cat([y_l_onehot, probs_u_sharpened.detach()], dim=0)
        indices = torch.randperm(all_x.size(0))
        x2, y2 = all_x[indices], all_y[indices]
        x_mix, y_mix = mixup(all_x, all_y, x2, y2, alpha=cfg.alpha)

        # ====== Forward ======
        logits = model(x_mix)
        loss_supervised = criterion(logits[:len(x_l)], y_l)
        loss_unsupervised = F.mse_loss(F.softmax(logits[len(x_l):], dim=1), y_mix[len(x_l):])
        loss = loss_supervised + 0.5 * loss_unsupervised

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_supervised += loss_supervised.item()
        total_unsupervised += loss_unsupervised.item()

    # ====== Evaluate on Test Set ======
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t, y_t = x_t.to(cfg.device), y_t.to(cfg.device)
            out = model(x_t)
            p = torch.argmax(out, dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y_t.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    epoch_losses.append(total_loss)
    epoch_sup.append(total_supervised)
    epoch_unsup.append(total_unsupervised)
    epoch_acc.append(acc)
    epoch_f1.append(f1)

    print(f"Epoch [{epoch+1}/{cfg.num_epochs}] "
          f"Total: {total_loss:.4f} | Sup: {total_supervised:.4f} | Unsup: {total_unsupervised:.4f} | "
          f"Acc: {acc:.4f} | F1: {f1:.4f}")

# ========== FINAL EVALUATION ==========
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x_t, y_t in test_loader:
        x_t, y_t = x_t.to(cfg.device), y_t.to(cfg.device)
        outputs = model(x_t)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_t.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("\n📊 === Final Evaluation ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print("Confusion Matrix:\n", cm)

# ========== PLOTS ==========
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_losses, label="Total Loss")
plt.plot(epoch_sup, label="Supervised Loss")
plt.plot(epoch_unsup, label="Unsupervised Loss")
plt.title("Training Losses per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epoch_acc, label="Accuracy", marker='o')
plt.plot(epoch_f1, label="F1-score", marker='s')
plt.title("Test Metrics per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.show()
