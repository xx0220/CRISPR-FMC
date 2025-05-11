import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from scipy.stats import spearmanr, pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from CRISPR_FMC_model import CRISPR_FMC

def set_seed(seed=2024):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CrisprDataset(Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = torch.tensor(x1, dtype=torch.float32)  # [N, 23, 4]
        self.x2 = torch.tensor(x2, dtype=torch.float32)  # [N, 640]
        self.y = torch.tensor(y, dtype=torch.float32)    # [N]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        if x1.ndim == 3 and x1.shape[0] == 1:
            x1 = x1.squeeze(0)  # 从 [1, 23, 4] → [23, 4]
        if x1.shape != (23, 4):
            raise ValueError(f"Unexpected shape for x1[{idx}]: {x1.shape}, expected [23, 4]")
        x1 = x1.permute(1, 0).unsqueeze(-1)  # → [4, 23, 1]
        x2 = self.x2[idx]
        y = self.y[idx]
        return x1, x2, y


def load_dataset(pkl_path):
    with open(pkl_path, 'rb') as f:
        X_onehot, X_rnafm, y = pickle.load(f)
    return np.array(X_onehot), np.array(X_rnafm), np.array(y)

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            y_pred = model(x1, x2).view(-1)
            preds.append(y_pred.cpu().numpy())
            labels.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)
    y_true = np.nan_to_num(y_true, nan=np.nanmean(y_true))
    y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred))
    scc = spearmanr(y_true, y_pred)[0]
    pcc = pearsonr(y_true, y_pred)[0]
    return scc, pcc, y_pred

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, target):
        loss = torch.log(torch.cosh(pred - target + 1e-12))
        return loss.mean()

if __name__ == "__main__":
    set_seed(220)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataList = ['WT', 'ESP', 'HELA', 'HF', 'HL60', 'Sniper-Cas9', 'SpCas9-NG', 'xCas', 'HCT116']
    for dataname in dataList:
        X_onehot, X_rnafm, y = load_dataset(f"./processed_data/{dataname}.pkl")
        print(f"{dataname} Loaded: One-hot: {X_onehot.shape}, RNA-FM: {X_rnafm.shape}, Labels: {y.shape}")

        batch_size = 4096
        epochs = 300
        kf = KFold(n_splits=5, shuffle=True, random_state=2024)
        results_df = pd.DataFrame(columns=['Fold', 'scc', 'Pcc'])

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_onehot), 1):
            print(f"\n📦 Fold {fold} Training...")

            train_dataset = CrisprDataset(X_onehot[train_idx], X_rnafm[train_idx], y[train_idx])
            test_dataset = CrisprDataset(X_onehot[test_idx], X_rnafm[test_idx], y[test_idx])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            model = CRISPR_FMC().to(device)
            criterion = LogCoshLoss()
            optimizer = optim.Adamax(model.parameters(), lr=3e-4)
            best_loss = float('inf')
            patience = 15
            wait = 0
            epoch_loss_list = []

            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for x1, x2, target in train_loader:
                    x1, x2, target = x1.to(device), x2.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(x1, x2).squeeze()
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                epoch_loss_list.append(total_loss)
                val_scc, val_pcc, _ = evaluate(model, test_loader, device)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss:.4f} - Val SCC: {val_scc:.4f}, PCC: {val_pcc:.4f}")

                if total_loss < best_loss:
                    best_loss = total_loss
                    wait = 0
                    os.makedirs(f'./RNA-FM_application/{dataname}', exist_ok=True)
                    torch.save(model.state_dict(), f'./RNA-FM_application/{dataname}/CRISPR_FMC_{dataname}_fold{fold}.pt')
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping")
                        break

            # 🎯 绘制 Loss 曲线
            plt.figure()
            plt.plot(range(1, len(epoch_loss_list) + 1), epoch_loss_list, marker='o')
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title(f"{dataname} - Fold {fold} Loss Curve")
            plt.grid(True)
            os.makedirs(f'./RNA-FM_Log/hb/loss_curve/{dataname}', exist_ok=True)
            plt.savefig(f'./RNA-FM_Log/hb/loss_curve/{dataname}/fold{fold}_loss.png')
            plt.close()

            # Evaluation
            model.load_state_dict(torch.load(f'./RNA-FM_application/{dataname}/CRISPR_FMC_{dataname}_fold{fold}.pt'))
            scc, pcc, _ = evaluate(model, test_loader, device)
            print(f"✅ Fold {fold}: SCC = {scc:.4f}, PCC = {pcc:.4f}")
            results_df = pd.concat([results_df, pd.DataFrame([{'Fold': fold, 'scc': scc, 'Pcc': pcc}])], ignore_index=True)

        avg_row = pd.DataFrame([{'Fold': 'Average', 'scc': results_df['scc'].mean(), 'Pcc': results_df['Pcc'].mean()}])
        results_df = pd.concat([results_df, avg_row], ignore_index=True)

        os.makedirs('./RNA-FM_Log/bigru', exist_ok=True)
        time_stamp = time.strftime('%Y%m%d_%H%M%S')
        results_df.to_csv(f'./RNA-FM_Log/bigru/{time_stamp}_lcx_{dataname}.csv', index=False)
        print(results_df)
