import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc, roc_auc_score

class EvaluationMetrics:
    def __init__(self, target_alignments, predicted_alignments):
        self.target_alignments = target_alignments
        self.pred_alignments = predicted_alignments

    def precision_deviation(self):
        precision_dev = {}
        total_tp = 0
        total_fp = 0

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

            total_tp += tp
            total_fp += fp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precision_dev[pref_len] = precision

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        sorted_precision_dev = OrderedDict(sorted(precision_dev.items(), key=lambda item: item[0]))

        return sorted_precision_dev, overall_precision
    
    def recall_deviation(self):
        recall_dev = {}
        total_tp = 0
        total_fn = 0

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

            total_tp += tp
            total_fn += fn

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recall_dev[pref_len] = recall
            
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0    
        sorted_precision_dev = OrderedDict(sorted(recall_dev.items(), key=lambda item: item[0]))

        return sorted_precision_dev, overall_recall
    
    def roc_deviation_multilabel(self, average='macro'):
        """
        Multi-label ROC/AUC where each sample's true labels are model_moves ∪ log_moves, and similarly for predictions.

        Returns:
        fpr_dict:    dict[class_label -> array of FPR]
        tpr_dict:    dict[class_label -> array of TPR]
        thresh_dict: dict[class_label -> array of thresholds]
        roc_auc:     float overall AUC (macro or micro)
        Y_true:      np.ndarray (N_cases, M_deviations)
        Y_score:     np.ndarray (N_cases, M_deviations)
        """
        
        # Store here all possibel, different model and log moves 
        all_classes = set()
        # Go through all keys
        for p in self.target_alignments:
            # Target model and log moves:
            m_t = self.target_alignments[p]['model_moves']
            l_t = self.target_alignments[p]['log_moves']
            # Predicted model and log moves:
            m_p = self.pred_alignments[p]['model_moves']
            l_p = self.pred_alignments[p]['log_moves']
            for tgt_seq, pred_seq in zip(m_t + l_t, m_p + l_p):
                all_classes.update(tgt_seq)
                all_classes.update(pred_seq)
        all_classes = sorted(all_classes)
        # Add for each identified deviation an index 0 to z
        cls_to_idx = {c:i for i,c in enumerate(all_classes)}

        # Build Y_true, Y_score by sample (prefix, instance)
        # Both have shape: (N_cases, M_deviations)
        Y_true_rows  = []
        Y_score_rows = []
        
        for p in self.target_alignments:
            # Target model and log moves:
            tgt_mod = self.target_alignments[p]['model_moves']
            tgt_log = self.target_alignments[p]['log_moves']
            # Predicted model and log moves:
            pred_mod= self.pred_alignments[p]['model_moves']
            pred_log= self.pred_alignments[p]['log_moves']

            # iterate each instance i at this prefix
            for Tm, Tl, Pm, Pl in zip(tgt_mod, tgt_log, pred_mod, pred_log):
                # True set (Target) containing all unique model and log moves
                true_set  = set(Tm)  | set(Tl)
                # Score set (Prediction)
                score_set = set(Pm)  | set(Pl)

                true_row  = np.zeros(len(all_classes), dtype=int)
                score_row = np.zeros(len(all_classes), dtype=float)
                # Fill trues with ones in case the deviation is in the true set
                for c in true_set:
                    true_row[ cls_to_idx[c] ] = 1
                # Fill scores with ones
                for c in score_set:
                    score_row[ cls_to_idx[c] ] = 1
                
                Y_true_rows.append(true_row)
                Y_score_rows.append(score_row)

        Y_true  = np.vstack(Y_true_rows)
        Y_score = np.vstack(Y_score_rows)

        # Per–class ROC curves
        fpr_dict    = {}
        tpr_dict    = {}
        thresh_dict = {}
        for idx, cls in enumerate(all_classes):
            fpr, tpr, thr = roc_curve(Y_true[:,idx], Y_score[:,idx])
            fpr_dict[cls]     = fpr
            tpr_dict[cls]     = tpr
            thresh_dict[cls]  = thr

        # Overall multi-label AUC (macro or micro)
        roc_auc = roc_auc_score(Y_true, Y_score, average=average, multi_class='raise')

        return fpr_dict, tpr_dict, thresh_dict, roc_auc, Y_true, Y_score

    def probabilistic_roc_deviation_multilabel(self, deviations_samples_risk, average='macro'):
        all_classes = set()
        # Deviation samples: key: deviation, values probability
        for probs in deviations_samples_risk.values():
            for d in probs['model_moves'] + probs['log_moves']:
                all_classes.update(d.keys())
        all_classes = sorted(all_classes)
        cls_to_idx = {c:i for i,c in enumerate(all_classes)}

        Y_true_rows  = []
        Y_score_rows = []

        for pref_len, probs in deviations_samples_risk.items():
            # Predicted deviations
            model_probs = probs['model_moves']
            log_probs   = probs['log_moves']

            # Target deviations for prefix length
            tgt_mod = self.target_alignments[pref_len]['model_moves']
            tgt_log = self.target_alignments[pref_len]['log_moves']

            for i, (m_prob, l_prob) in enumerate(zip(model_probs, log_probs)):
                true_moves = set(tgt_mod[i]) | set(tgt_log[i])

                # b) build rows
                true_row  = np.zeros(len(all_classes), dtype=int)
                score_row = np.zeros(len(all_classes), dtype=float)

                # fill true
                for c in true_moves:
                    true_row[cls_to_idx[c]] = 1

                # fill score from your risk‐probabilities
                # note: if a class never appeared in the dict, its prob remains 0
                for c, p in m_prob.items():
                    score_row[cls_to_idx[c]] = p
                for c, p in l_prob.items():
                    # take the max if both model/log provide a prob
                    idx = cls_to_idx[c]
                    score_row[idx] = max(score_row[idx], p)

                Y_true_rows .append(true_row)
                Y_score_rows.append(score_row)

        Y_true  = np.vstack(Y_true_rows)
        Y_score = np.vstack(Y_score_rows)

        # Per‐class ROC curves
        fpr_dict    = {}
        tpr_dict    = {}
        thresh_dict = {}
        for idx, cls in enumerate(all_classes):
            fpr, tpr, thr = roc_curve(Y_true[:,idx], Y_score[:,idx])
            fpr_dict[cls]     = fpr
            tpr_dict[cls]     = tpr
            thresh_dict[cls]  = thr

        # Overall AUC
        roc_auc = roc_auc_score(Y_true, Y_score, average=average)

        return fpr_dict, tpr_dict, thresh_dict, roc_auc, Y_true, Y_score
