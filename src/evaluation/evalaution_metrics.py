import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import roc_auc_score, roc_curve

class EvaluationMetrics:
    def __init__(self, target_alignments, predicted_alignments):
        self.target_alignments = target_alignments
        self.pred_alignments = predicted_alignments

    def precision_deviation(self):
        precision_dev = {}

        for pref_len in self.target_alignments.keys():
            model_moves_target = self.target_alignments[pref_len]['model_moves']
            log_moves_target = self.target_alignments[pref_len]['log_moves']

            model_moves_pred = self.pred_alignments[pref_len]['model_moves']
            log_moves_pred = self.pred_alignments[pref_len]['log_moves']

            tp_model = sum(len(set(t) & set(p)) for t, p in zip(model_moves_target, model_moves_pred))
            tp_log   = sum(len(set(t) & set(p)) for t, p in zip(log_moves_target, log_moves_pred))

            fp_model = sum(len(set(p) - set(t)) for t, p in zip(model_moves_target, model_moves_pred))
            fp_log   = sum(len(set(p) - set(t)) for t, p in zip(log_moves_target, log_moves_pred))

            tp = tp_model + tp_log
            fp = fp_model + fp_log

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision_dev[pref_len] = precision
            
        sorted_precision_dev = OrderedDict(sorted(precision_dev.items(), key=lambda item: item[0]))

        return sorted_precision_dev
    
    def recall_deviation(self):
        recall_dev = {}

        for pref_len in self.target_alignments.keys():
            model_moves_target = self.target_alignments[pref_len]['model_moves']
            log_moves_target = self.target_alignments[pref_len]['log_moves']

            model_moves_pred = self.pred_alignments[pref_len]['model_moves']
            log_moves_pred = self.pred_alignments[pref_len]['log_moves']

            tp_model = sum(len(set(t) & set(p)) for t, p in zip(model_moves_target, model_moves_pred))
            tp_log   = sum(len(set(t) & set(p)) for t, p in zip(log_moves_target, log_moves_pred))

            fn_model = sum(len(set(t) - set(p)) for t, p in zip(model_moves_target, model_moves_pred))
            fn_log   = sum(len(set(t) - set(p)) for t, p in zip(log_moves_target, log_moves_pred))

            tp = tp_model + tp_log
            fn = fn_model + fn_log

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_dev[pref_len] = recall
            
        sorted_precision_dev = OrderedDict(sorted(recall_dev.items(), key=lambda item: item[0]))

        return sorted_precision_dev
    
    def auc_roc_deviation(self):
        # Flatten across all cases
        y_true, y_score = [], []
        for pref_len in self.target_alignments:
            t_mod = self.target_alignments[pref_len]['model_moves']
            t_log = self.target_alignments[pref_len]['log_moves']
            p_mod = self.pred_alignments[pref_len]['model_moves']
            p_log = self.pred_alignments[pref_len]['log_moves']
            for tm, tl, pm, pl in zip(t_mod, t_log, p_mod, p_log):
                y_true.append(int(bool(tm or tl)))
                y_score.append(len(pm) + len(pl))
        if len(set(y_true)) < 2:
            return None
        return roc_auc_score(y_true, y_score), roc_curve(y_true, y_score)
    
    def auc_pr_deviation(self, precision_dev, recall_dev):
        # Compute AUC for Precision-Recall (trapz)
        rec = np.array(list(recall_dev.values()))
        prec = np.array(list(precision_dev.values()))
        idx = np.argsort(rec)
        return np.trapezoid(prec[idx], rec[idx])
    
    def precision_no_deviation():
        pass
    
    def recall_no_deviation():
        pass