import os
import sys
from joblib import load
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve, auc
import numpy as np

class PredictionResults:
    def __init__(self, model, test_set):
        self.model = model
        self.test_set = test_set
        
    def get_predictions_targets(self, batch_size: int = 256, shuffle: bool = True, device: torch.device = torch.device('cpu')):
        # Loader:
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        all_targets, all_probs, all_preds = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                x_act, x_res, x_month, x_trace, y = batch
                x_act = x_act.to(device)
                x_res = x_res.to(device)
                x_month = x_month.to(device)
                x_trace = x_trace.to(device)
                y = y.to(device)

                logits = self.model(x_act, x_res, x_month, x_trace, apply_sigmoid=False)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()

                all_probs.append(probs.cpu())    
                all_targets.append(y.cpu())
                all_preds.append(preds.cpu())

        probs = torch.cat(all_probs).numpy()
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        
        return probs, preds, targets
    
class Metrics:
    def __init__(self, preds, targets):
        """
        preds, targets: array-like with shape (N, m), binary {0,1} values.
        """
        self.preds = np.asarray(preds)
        self.targets = np.asarray(targets)
        if self.preds.shape != self.targets.shape:
            raise ValueError("preds and targets must have the same shape (N, m)")

    def _per_label_counts(self):
        # returns arrays of TP, FP, FN per label (for positive class==1)
        tp = np.sum((self.preds == 1) & (self.targets == 1), axis=0).astype(float)
        fp = np.sum((self.preds == 1) & (self.targets == 0), axis=0).astype(float)
        fn = np.sum((self.preds == 0) & (self.targets == 1), axis=0).astype(float)
        tn = np.sum((self.preds == 0) & (self.targets == 0), axis=0).astype(float)
        return tp, fp, fn, tn

    def macro_precision_recall_dev(self):
        """
        Compute macro-averaged precision and recall for deviation predictions.
        Deviation = label==1, TP = pred 1 and target 1.
        Returns dict with per-label arrays and macro averages.
        """
        tp, fp, fn, tn = self._per_label_counts()
        with np.errstate(divide='ignore', invalid='ignore'):
            precision_per_label = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
            recall_per_label = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)

        macro_precision = float(np.nanmean(precision_per_label))
        macro_recall = float(np.nanmean(recall_per_label))

        return {
            "precision_per_label": precision_per_label,
            "recall_per_label": recall_per_label,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall
        }

    def macro_precision_recall_no_dev(self):
        """
        Compute macro-averaged precision and recall for non-deviation predictions (treat 0 as positive).
        Non-deviation TP = pred 0 and target 0.
        """
        tp, fp, fn, tn = self._per_label_counts()
        # For class "0" as positive:
        tp0 = tn
        fp0 = fn  # predicted 0 but target 1 -> false positive for class 0
        fn0 = fp  # predicted 1 but target 0 -> false negative for class 0

        with np.errstate(divide='ignore', invalid='ignore'):
            precision_per_label_0 = np.divide(tp0, tp0 + fp0, out=np.zeros_like(tp0), where=(tp0 + fp0) != 0)
            recall_per_label_0 = np.divide(tp0, tp0 + fn0, out=np.zeros_like(tp0), where=(tp0 + fn0) != 0)

        macro_precision_0 = float(np.nanmean(precision_per_label_0))
        macro_recall_0 = float(np.nanmean(recall_per_label_0))

        return {
            "precision_per_label_non_deviation": precision_per_label_0,
            "recall_per_label_non_deviation": recall_per_label_0,
            "macro_precision_non_deviation": macro_precision_0,
            "macro_recall_non_deviation": macro_recall_0
        }

    def plot_macro_roc_auc(self, prob_scores, label_names=None, figsize=(8, 6)):
        """
        Compute per-label ROC AUC and plot ROC curves.
        prob_scores: array-like shape (N, m) with probabilities for label==1.
        label_names: optional list of m names for legend.
        Returns dict with per-label aucs and macro_auc.
        """
        prob_scores = np.asarray(prob_scores)
        if prob_scores.shape != self.targets.shape:
            raise ValueError("prob_scores must have same shape as preds/targets (N, m)")

        m = prob_scores.shape[1]
        per_label_auc = np.full(m, np.nan)
        plt.figure(figsize=figsize)
        ax = plt.gca()

        for i in range(m):
            y_true_i = self.targets[:, i]
            y_score_i = prob_scores[:, i]
            # Skip labels with single class in y_true
            if np.unique(y_true_i).size < 2:
                per_label_auc[i] = np.nan
                continue
            fpr, tpr, _ = roc_curve(y_true_i, y_score_i)
            lab = label_names[i] if (label_names is not None and i < len(label_names)) else f"label_{i}"
            roc_auc = auc(fpr, tpr)
            per_label_auc[i] = roc_auc
            ax.plot(fpr, tpr, lw=1.5, label=f"{lab} (AUC={roc_auc:.3f})")

        # plot chance
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves per Deviation Label")
        ax.legend(loc="lower right", fontsize="small")
        plt.show()

        macro_auc = float(np.nanmean(per_label_auc))
        return {"per_label_auc": per_label_auc, "macro_auc": macro_auc}

