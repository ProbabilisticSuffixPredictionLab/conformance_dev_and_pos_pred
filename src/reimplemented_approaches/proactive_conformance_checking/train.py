# src/train_idp_lstm_torch.py
from __future__ import annotations
import json, pathlib, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torch_models import BiLSTM_IDP

DATA = pathlib.Path("data/processed")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH = 256
EPOCHS = 50
PATIENCE = 6
LR = 1e-3
RNG = np.random.default_rng(123)

# ---------- Utils ----------
def undersample_indices(Y: np.ndarray, neg_pos_ratio: int = 3, rng: int | np.random.Generator = 42):
    rng = np.random.default_rng(rng) if isinstance(rng, int) else rng
    pos = (Y.sum(axis=1) > 0)
    pos_idx = np.where(pos)[0]
    neg_idx = np.where(~pos)[0]
    n_neg = int(min(len(neg_idx), neg_pos_ratio * len(pos_idx)))
    if n_neg > 0:
        neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return idx

def compute_pos_weights(Y_train: np.ndarray) -> torch.Tensor:
    n = Y_train.shape[0]
    pos = Y_train.sum(axis=0)
    neg = n - pos
    w = (neg / np.maximum(1.0, pos)).astype("float32")
    w = np.clip(w, 1.0, 50.0)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)  # (L,)

def tune_thresholds_fbeta(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0) -> np.ndarray:
    L = y_true.shape[1]; thr = np.zeros(L, dtype="float32")
    for j in range(L):
        yj = y_true[:, j]
        if yj.min() == yj.max():
            thr[j] = 0.5; continue
        pr, rc, t = precision_recall_curve(yj, y_prob[:, j])
        fbeta = (1+beta**2) * pr * rc / (beta**2 * pr + rc + 1e-12)
        thr[j] = float(t[np.argmax(fbeta[:-1])]) if len(t) else 0.5
    return thr

# ---------- Dataset ----------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, M: np.ndarray, idxs: np.ndarray):
        self.X = X[idxs]   # (N,T,D)
        self.Y = Y[idxs]   # (N,L)
        self.M = M[idxs]   # (N,T)
        self.lengths = self.M.sum(axis=1).astype(np.int64)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float()
        y = torch.from_numpy(self.Y[i]).float()
        length = torch.tensor(self.lengths[i]).long()
        return x, y, length

# ---------- Load data ----------
def load_arrays():
    X = np.load(DATA/"X_seq.npy")     # (N,T,D)
    M = np.load(DATA/"mask.npy")      # (N,T)
    Y = np.load(DATA/"Y.npy")         # (N,L)
    meta = json.loads((DATA/"encoding_meta.json").read_text())
    cases = pd.read_parquet(DATA/"individual_labels.parquet",
                            columns=["case:concept:name"])["case:concept:name"].to_numpy()
    return X, M, Y, meta, cases

def main():
    X, M, Y, meta, groups = load_arrays()
    N,T,D = X.shape; L = Y.shape[1]

    # Split by case (Train/Test), dann Val aus Train
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, Y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=43)
    tr2, va = next(gss2.split(X[tr], Y[tr], groups[tr]))
    tr2, va = tr[tr2], tr[va]

    # Undersampling nur in Train
    tr_final = tr2[undersample_indices(Y[tr2], neg_pos_ratio=3, rng=7)]

    # Datasets / Loaders
    ds_tr = SeqDataset(X, Y, M, tr_final)
    ds_va = SeqDataset(X, Y, M, va)
    ds_te = SeqDataset(X, Y, M, te)
    ld_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True, drop_last=False)
    ld_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False, drop_last=False)
    ld_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False, drop_last=False)

    # Modell
    model = BiLSTM_IDP(D, L).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)

    # Imbalance: pos_weight pro Label in BCEWithLogitsLoss
    pos_w = compute_pos_weights(Y[tr_final])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)  # erwartet logits

    best_val_auc = -np.inf
    best_state = None
    patience = PATIENCE

    # ---------- Training loop ----------
    def run_epoch(loader, train: bool):
        if train: model.train()
        else: model.eval()
        total_loss = 0.0
        ys, ps = [], []
        with torch.set_grad_enabled(train):
            for xb, yb, lb in loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE); lb = lb.to(DEVICE)
                logits = model(xb, lb)               # (B,L)
                loss = criterion(logits, yb)         # scalar
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * xb.size(0)
                ys.append(yb.detach().cpu().numpy())
                ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        Y_all = np.vstack(ys)       # (N,L)
        P_all = np.vstack(ps)       # (N,L)
        # Macro ROC-AUC (über Labels mit beiden Klassen)
        aucs = []
        for j in range(L):
            yj = Y_all[:, j]
            if yj.min() != yj.max():
                aucs.append(roc_auc_score(yj, P_all[:, j]))
        macro_auc = float(np.mean(aucs)) if aucs else float("nan")
        return total_loss/len(loader.dataset), macro_auc, Y_all, P_all

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_auc, _, _ = run_epoch(ld_tr, train=True)
        va_loss, va_auc, Yva, Pva = run_epoch(ld_va, train=False)
        print(f"Epoch {epoch:02d} | loss_tr={tr_loss:.4f} | auc_tr={tr_auc:.4f} | "
              f"loss_va={va_loss:.4f} | auc_va={va_auc:.4f}")

        improved = (va_auc > best_val_auc + 1e-6)
        if improved:
            best_val_auc = va_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = PATIENCE
        else:
            patience -= 1
            if patience == 0:
                print(f"[EarlyStopping] best val_auc_roc={best_val_auc:.4f}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- Threshold-Tuning (Val) und Evaluation (Test) ----------
    # Val-Thresholds per Label (F_beta, recall-fokussiert)
    _, _, Yva, Pva = run_epoch(ld_va, train=False)
    thr = tune_thresholds_fbeta(Yva, Pva, beta=2.0)

    # Test-Predictions
    te_loss, te_auc, Yte, Pte = run_epoch(ld_te, train=False)
    # --- Für Sanity-Check speichern ---
    np.save(DATA/"Y_test.npy", Yte)   # (N, L)
    np.save(DATA/"P_test.npy", Pte)   # (N, L)
    Yhat = (Pte >= thr[None, :]).astype(int)

    # Klassen-basierter Report (wie bisher)
    rep = classification_report(Yte.ravel(), Yhat.ravel(),
                                target_names=["0","1"], output_dict=True, zero_division=0)

    # ----- Dev/NoDev Metriken: Precision, Recall, ROC-AUC -----
    # Dev = mind. ein Label = 1; NoDev = alle 0
    dev_true = (Yte.sum(axis=1) > 0).astype(int)              # (N,)
    # Dev-Score aus Multi-Label-Prob: p_any = 1 - ∏(1 - p_j)
    p_any = 1.0 - np.prod(1.0 - Pte, axis=1)                  # (N,)
    # Best F1-Threshold auf Val auch für p_any:
    p_any_val = 1.0 - np.prod(1.0 - Pva, axis=1)
    pr, rc, t = precision_recall_curve((Yva.sum(axis=1) > 0).astype(int), p_any_val)
    f1 = 2*pr*rc/(pr+rc+1e-12)
    thr_any = float(t[np.argmax(f1[:-1])]) if len(t) else 0.5
    dev_pred = (p_any >= thr_any).astype(int)

    from sklearn.metrics import precision_score, recall_score
    dev_precision = float(precision_score(dev_true, dev_pred, zero_division=0))
    dev_recall    = float(recall_score(dev_true, dev_pred, zero_division=0))
    dev_auc       = float(roc_auc_score(dev_true, p_any))

    # Für NoDev „als positive Klasse“ analog:
    nodev_true = 1 - dev_true
    nodev_score = 1.0 - p_any            # Wahrscheinlichkeit „keine Abweichung“
    nodev_pred  = 1 - dev_pred
    nodev_precision = float(precision_score(nodev_true, nodev_pred, zero_division=0))
    nodev_recall    = float(recall_score(nodev_true, nodev_pred, zero_division=0))
    nodev_auc       = float(roc_auc_score(nodev_true, nodev_score))

    # ---------- Save ----------
    out = {
        "labels": meta["dev_cols"],
        "macro_f1_classes": rep["macro avg"]["f1-score"],
        "val_auc_roc_best": best_val_auc,
        "test_auc_roc_macro_labels": te_auc,
        "thresholds": {k: float(v) for k,v in zip(meta["dev_cols"], thr)},
        "threshold_strategy": "fbeta",
        "threshold_beta": 2.0,
        "dev_metrics": {
            "precision": dev_precision,
            "recall": dev_recall,
            "auc_roc": dev_auc,
            "threshold_any": thr_any
        },
        "nodev_metrics": {
            "precision": nodev_precision,
            "recall": nodev_recall,
            "auc_roc": nodev_auc
        },
        "n_train": int(len(tr_final)), "n_val": int(len(va)), "n_test": int(len(te)),
    }
    (DATA/"idp_lstm_torch_report.json").write_text(json.dumps(out, indent=2))
    torch.save(model.state_dict(), DATA/"idp_lstm_torch_model.pt")
    print("[OK] PyTorch IDP-LSTM gespeichert →", DATA/"idp_lstm_torch_model.pt")
    print("[OK] Report →", DATA/"idp_lstm_torch_report.json")
    print("Dev  | Precision:", round(dev_precision,4), "Recall:", round(dev_recall,4), "AUC:", round(dev_auc,4))
    print("NoDev| Precision:", round(nodev_precision,4), "Recall:", round(nodev_recall,4), "AUC:", round(nodev_auc,4))

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np

# Deine Werte
classes = ["Dev", "NoDev"]
precision = [0.4900, 0.9767]
recall = [0.9835, 0.4032]
auc = [0.8764, 0.8764]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(6,4))
rects1 = ax.bar(x - width, precision, width, label='Precision', color="#FF9F40")
rects2 = ax.bar(x, recall, width, label='Recall', color="#4BC0C0")
rects3 = ax.bar(x + width, auc, width, label='AUC-ROC', color="#9966FF")

ax.set_ylabel('Score')
ax.set_ylim(0, 1.1)
ax.set_title('Dev/NoDev Performance (Test)')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(loc='upper center', ncol=3)

# Werte auf Balken schreiben
for rect in rects1 + rects2 + rects3:
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("results/dev_nodev_barplot.png", dpi=200)
plt.show()

