import numpy as np
from collections import OrderedDict
from typing import Dict, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, roc_auc_score

class EvaluationMetrics:    
    def __init__(self, target_alignments: Optional[Dict]=None, predicted_alignments: Optional[Dict] = None):
        self.target_alignments = target_alignments
        self.pred_alignments = predicted_alignments


    def precision_deviation(self):
        """
        Calculates the precision of the deviation prediction per prefix length and total.
        """
        precision_dev = {}
        total_tp = 0
        total_fp = 0

        for pref_len in self.target_alignments.keys():
            model_moves_target = self.target_alignments[pref_len]['model_moves']
            log_moves_target = self.target_alignments[pref_len]['log_moves']

            if pref_len in self.pred_alignments:
                model_moves_pred = self.pred_alignments[pref_len]['model_moves']
                log_moves_pred = self.pred_alignments[pref_len]['log_moves']
            else:
                model_moves_pred = []
                log_moves_pred = []

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
        """
        Caluclates the recall of the deivation prediction per prefix length and total.
        """
        recall_dev = {}
        total_tp = 0
        total_fn = 0

        for pref_len in self.target_alignments.keys():
            model_moves_target = self.target_alignments[pref_len]['model_moves']
            log_moves_target = self.target_alignments[pref_len]['log_moves']

            if pref_len in self.pred_alignments:
                model_moves_pred = self.pred_alignments[pref_len]['model_moves']
                log_moves_pred = self.pred_alignments[pref_len]['log_moves']
            else:
                model_moves_pred = []
                log_moves_pred = []

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
        sorted_recall_dev = OrderedDict(sorted(recall_dev.items(), key=lambda item: item[0]))

        return sorted_recall_dev, overall_recall
    
    
    def probabilistic_roc_deviation_multilabel(self, average='macro'):
        """
        Calculates the ROC and AUC value per class (model or log move)
        """
        
        # Collect all classes: All model and log moves either predicted or in target or in both
        all_classes = set()
        # Iterate through the predictions and the targets:
        for D in (self.pred_alignments, self.target_alignments):
            for vals in D.values():
                for seq in vals['model_moves'] + vals['log_moves']:
                    all_classes.update(seq)
        # Contains all model and log moves gathered in the predictions and targets
        all_classes = sorted(all_classes)
        # contains all classes (adds an index, starting with 1 and increasing)
        cls_to_idx = {c: i for i, c in enumerate(all_classes)}

        # Fill the lists to get TPR and FPR for each prefix length:
        # Y_true, Y_score: Input for roc curve
        # Y_true contains 0 or 1 for missing or present class
        Y_true_rows, Y_score_rows = [], []
        for pref in self.target_alignments:   
            # For each prefix length get all cases' model and log moves: list of lists
            tgt = self.target_alignments[pref]
            # For each prefix length get all cases model and log move: list of dicts
            pred = self.pred_alignments.get(pref, {'model_moves': [], 'log_moves': []})

            
            # Target and predicted contain same lenght of cases
            # Iterate through all cases deviations: Model and Log moves have same length and can both be seen as classes of deviations
            max_len = len(tgt['model_moves'])
            
            for i in range(max_len):
                
                # Tuples of true deviations in specific cases:
                tgt_model_move = tgt['model_moves'][i]
                tgt_log_move = tgt['log_moves'][i]
                true_set = set(tgt_model_move) | set(tgt_log_move)
                # Zero row for each class in (model & log move) target deviation set
                true_row = np.zeros(len(all_classes), int)
                for c in true_set:
                    # Fill true row with ones for predicted deviation classes:
                    true_row[cls_to_idx[c]] = 1

                # predicted 0/1 scores
                # dict wher key is deviation, value is prob
                pred_model_move = pred['model_moves'][i]
                pred_log_move = pred['log_moves'][i]
                # Prediction deviation dict with log and model moves as key and probs of occurence across samples as value
                pred_dict = pred_model_move | pred_log_move
                # Zero row for each class in (model & log move) predicted deviation set 
                score_row = np.zeros(len(all_classes), float)
                for c, prob in pred_dict.items():
                    score_row[cls_to_idx[c]] =prob

                Y_true_rows.append(true_row)
                Y_score_rows.append(score_row)

        # Contains for all samples a list with length amount classes and its either 0 or 1 (target) or prob (predicted)
        Y_true  = np.vstack(Y_true_rows)
        Y_score = np.vstack(Y_score_rows)
        
        # Filter out only the valid classes:
        # Valid class for ROC needs at least one positive example and at least one negative example in the target across all cases.
        # If a deviation form the prediction never happens in the targets or happens always no ROC can bebuild for that class since variation is required.
        valid_classes = []
        for cls, idx in cls_to_idx.items():
            pos = int(Y_true[:, idx].sum())
            neg = Y_true.shape[0] - pos
            if pos > 0 and neg > 0:
                valid_classes.append(cls)

        fpr_dict, tpr_dict, thr_dict = {}, {}, {}
        for cls in valid_classes:
            idx = cls_to_idx[cls]
            fpr, tpr, thr = roc_curve(Y_true[:, idx], Y_score[:, idx])
            fpr_dict[cls], tpr_dict[cls], thr_dict[cls] = fpr, tpr, thr

        if valid_classes:
            idxs = [cls_to_idx[c] for c in valid_classes]
            Yt_f = Y_true[:, idxs]
            Ys_f = Y_score[:, idxs]
            auc_macro = roc_auc_score(Yt_f, Ys_f, average=average)
        else:
            auc_macro = np.nan
            raise ValueError("No multi-label has both positive and negative samples.")

        return fpr_dict, tpr_dict, thr_dict, auc_macro, Y_true, Y_score, all_classes

    
    def compute_class_coverage(self, target_all: Dict, target_risk: Dict):
        """
        Check how many times target classes are predicted and how many times unseen are predicted.
        """
        # Collect classes in full set
        classes_tgt = set()
        for vals in target_all.values():
            for seq in vals['model_moves'] + vals['log_moves']:
                classes_tgt.update(seq)
        classes_tgt = sorted(classes_tgt)
        classes_prob_tgt = np.zeros(len(classes_tgt), float)
        
        # Collect classes in risk set
        classes_risk = set()
        for vals in target_risk.values():
            for seq in vals['model_moves'] + vals['log_moves']:
                classes_risk.update(seq)
        classes_risk = sorted(classes_risk)
        
        # Unseen deivations: Classes in risk not in deviations:
        classes_unseen = sorted(set([c_risk for c_risk in classes_risk if c_risk not in classes_tgt]))
        classes_prob_unseen = np.zeros(len(classes_unseen), float)
        
        amount_cases = 0
        for _, moves in  target_risk.items():
            model_moves = moves['model_moves'] 
            amount_cases += len(model_moves) 
            log_moves = moves['log_moves']
            
            for i, model_move in enumerate(model_moves):
                model_move_set = set(model_move)
                log_move_set = set(log_moves[i])
                
                # Coun how often the classes in the target are predicted over the samples:
                for i, c in enumerate(classes_tgt):
                    if c in model_move_set or c in log_move_set:
                        classes_prob_tgt[i] += 1
                # Count how often the classes in the predictions/ not in the target are predicted over the samples:        
                for i, c in enumerate(classes_unseen):
                    if c in model_move_set or c in log_move_set:
                        classes_prob_unseen[i] += 1
                        
                
        classes_prob_tgt = classes_prob_tgt / amount_cases
        
        classes_prob_unseen = classes_prob_unseen / amount_cases
        
        return classes_tgt, classes_prob_tgt, classes_unseen, classes_prob_unseen 
        