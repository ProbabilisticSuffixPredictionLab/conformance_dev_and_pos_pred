from collections import Counter, defaultdict
from typing import Iterable, Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

class DeviationEvaluation:
    def __init__(self, deviation_results: Iterable[dict]):
        self.deviation_results = list(deviation_results)
    
    # precision and recall per case -> for all predicted/ target deviation labels compute prec. and rec.  
    def precision_recall_macro_deviations(self,
                                          label_only: bool = True,
                                          label_index: int = 0,
                                          zero_division: float = 1.0
                                          ) -> float:
        """
        Precision = TP / (TP + FP)
        - TP: Deviating predicted and deviating in target.
        - FP: Deviating predicted and non-deviating in target.
        
        Recall = TP / (TP + FN)
        - TP: Deviating predicted and deviating in target.
        - FN: Non-deviating predicted and deviating in target. 
        
        Be class sensitive: 
        - TP: Per case, the predicted deviated class is in the target.
        - FP: The predicted deviated class is not in the target.
        - FN: The target class is not predicted.
        
        Returns (precision_macro, recall_macro) where precision_macro is the mean
        of per-case precision and recall_macro is the mean of per-case recall.
        Conventions:
        - empty pred & empty tgt -> per-case precision = per-case recall = 1.0
        - pred empty & tgt non-empty -> precision_case = 0.0
        - pred non-empty & tgt empty -> recall_case = 0.0
        
        Use macro average: prec, rec per case and take the average over all cases, handles class imbalance.
        (micro: all deviations (tp, fp, ...) of all cases and then compute prec, rec.)
                
        """
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]
                
        prec_scores: List[float] = []
        rec_scores: List[float] = []
        
        for tgts, preds in zip(tgt_deviations, pred_deviations):
            # Counts the occurence of the event labels an stores dict: key: event label, value: count
            p_ctr = Counter(preds)
            t_ctr = Counter(tgts)
            
            # Number of all predicted deviations:
            total_pred = sum(p_ctr.values())
            # Number of all target deviations:
            total_tgt  = sum(t_ctr.values())

            # true positives: sum of the number of labels that appear in the prediction as well as in the target.
            tp_case = sum(min(p_ctr[k], t_ctr[k]) for k in (set(p_ctr) & set(t_ctr)))

            # precision_case
            if total_pred == 0 and total_tgt == 0:
                # decide here if the cases where in both cases correctly no deviation occurs should be counted here!   
                precision_case = 1.0
                # continue
            # In this case tp = 0 and fp > 0:
            elif total_pred == 0:
                precision_case = 0.0
            else:
                # total pred: All deivations that are predicted: TP and FP
                precision_case = tp_case / total_pred

            # recall_case
            if total_pred == 0 and total_tgt == 0:
                # decide here if the cases where in both cases correctly no deviation occurs should be counted here!   
                recall_case = 1.0
                # continue
            # In this case tp = 0 and fn > 0
            elif total_tgt == 0:
                recall_case = 0.0
            else:
                # total tgt: All deivations that are in the target (true): TP and FN
                recall_case = tp_case / total_tgt

            prec_scores.append(precision_case)
            rec_scores.append(recall_case)

        if not prec_scores:
            precision = float(zero_division)
        else:
            precision = float(sum(prec_scores) / len(prec_scores))

        if not rec_scores:
            recall = float(zero_division)
        else:
            recall = float(sum(rec_scores) / len(rec_scores))

        return precision, recall
    
    # Evaluation same as of Grohs/Rehse for comparison:
    # - Dev/ No Dev: Macro Precision, Recall and ROC_AUC 
        
    def precision_recall_macro_by_label_dev(self) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Compute per-label precision and recall only for labels appearing in the target set,
        then return macro-averages across those target labels.

        - macro: Get precision and recall per label and take the mean over all.      
        """
        # collect per-case counters
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # label-level accumulators (across all cases)
        total_pred = defaultdict(int)  # total predicted counts per label across all cases
        total_tgt  = defaultdict(int)  # total target counts per label across all cases
        tp_label   = defaultdict(int)  # true positives per label (summed per-case min)

        # accumulate counts per-case (TP computed as min per case)
        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            p_ctr = Counter(pred_list)
            t_ctr = Counter(tgt_list)
            labels_union = set(p_ctr) | set(t_ctr)
            
            # Go thorugh the labels that occur in prediction and target per case
            for lbl in labels_union:
                # list and count the occurence of all predicted labels
                total_pred[lbl] += p_ctr.get(lbl, 0)
                # list and count the occurence of all target labels
                total_tgt[lbl]  += t_ctr.get(lbl, 0)
                # list and count of true positives: in pred and in tgt 
                tp_label[lbl]   += min(p_ctr.get(lbl, 0), t_ctr.get(lbl, 0))

        # Only evaluate labels that exist in the target set -> According to grohs: "...Such new alignments are ignored during evaluation as no ground truth data to assess against exists."
        target_label_list = sorted(k for k, v in total_tgt.items() if v > 0)
        
        # Precision and Recall
        precision_per_label = {}
        recall_per_label = {}

        for lbl in target_label_list:
            # Number: How many times label in pred and target of same case:
            tp = tp_label.get(lbl, 0)
            # How many times in prediction in total: All true positive and false positive (predicted but not in target)
            tp_fp = total_pred.get(lbl, 0) 
            # How many times in target in total: All true positive and false negative (not predicted but in target)
            tp_fn = total_tgt.get(lbl, 0) 

            # precision: TP / (TP + FP): Since only target labels are counted: Fp > 1 for unseen label not possible.
            precision_per_label[lbl] = tp / tp_fp

            # recall: TP / (TP + FN) where FN = tgt_sum - TP
            if tp_fn == 0:
                # Not predicted but in target -> tp = 0, fn > 0
                recall_per_label[lbl] = 0
            else: 
                recall_per_label[lbl] = tp / tp_fn

        # macro averages over target labels only
        precision_macro = np.mean(list(precision_per_label.values()))
        recall_macro = np.mean(list(recall_per_label.values()))
            
        return precision_macro, recall_macro, precision_per_label, recall_per_label
        
    def precision_recall_macro_by_label_no_dev(self,
                                               zero_division: float = 1.0) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Compute per-label precision and recall for the *no-deviation* event (opposite).
        Positive event = label is NOT present in prediction and NOT present in target.
        
        - macro: Get precision and recall per label and take the mean over all.      
        """
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # accumulators per label
        tp_no = defaultdict(int)         # true positives for "no-deviation" per label
        pred_no_count = defaultdict(int) # predicted no-deviation counts per label (pred_has == False)
        true_no_count = defaultdict(int) # true no-deviation counts per label (tgt_has == False)
        
        total_tgt  = defaultdict(int)  # total target counts per label across all cases

        # collect label universe (union of labels seen anywhere)
        label_universe = set()
        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            p_set = set(pred_list)
            t_set = set(tgt_list)
            
            label_universe |= (p_set | t_set)
            labels_union = (p_set | t_set)
        
            t_ctr = Counter(tgt_list)

            # Go thorugh the labels that occur in prediction and target per case
            for lbl in labels_union:
                # list and count the occurence of all target labels
                total_tgt[lbl]  += t_ctr.get(lbl, 0)          
        
        # iterate cases and update presence/absence counts per label
        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            p_set = set(pred_list)
            t_set = set(tgt_list)
            for lbl in label_universe:
                pred_has = (lbl in p_set)
                tgt_has = (lbl in t_set)

                # predicted no-deviation?
                if not pred_has:
                    pred_no_count[lbl] += 1
                # actually no-deviation?
                if not tgt_has:
                    true_no_count[lbl] += 1
                # true positive for no-deviation:
                if (not pred_has) and (not tgt_has):
                    tp_no[lbl] += 1
                    
        # Only evaluate labels that exist in the target set -> According to grohs: "...Such new alignments are ignored during evaluation as no ground truth data to assess against exists."
        target_label_list = sorted(k for k, v in total_tgt.items() if v > 0)

        # compute per-label precision/recall (for no-deviation)
        precision_per_label = {}
        recall_per_label = {}
        
        for lbl in sorted(target_label_list):
            tp = tp_no.get(lbl, 0)
            pred_pos = pred_no_count.get(lbl, 0)   # predicted no-deviation
            true_pos = true_no_count.get(lbl, 0)  # actual no-deviation

            # precision_no = TP / (TP + FP)   where FP = pred_pos - TP
            if pred_pos == 0:
                precision_per_label[lbl] = float(zero_division)
            else:
                precision_per_label[lbl] = tp / pred_pos

            # recall_no = TP / (TP + FN)  where FN = true_pos - TP
            if true_pos == 0:
                recall_per_label[lbl] = float(zero_division)
            else:
                recall_per_label[lbl] = tp / true_pos

        # macro averages across labels
        precision_macro = float(sum(precision_per_label[lbl] for lbl in precision_per_label) / len(precision_per_label))
        recall_macro = float(sum(recall_per_label[lbl] for lbl in recall_per_label) / len(recall_per_label))

        return precision_macro, recall_macro, precision_per_label, recall_per_label
    
    def roc_auc_macro_by_label(self, plot: bool = True) -> Tuple[float, Dict[str, float]]:
        """
        Compute ROC AUC per-label for the *deviation* event (label present).
        - Positive class = label IS present in target (tgt_has == True).
        - Predicted score = 1.0 if label IS present in prediction (pred_has == True), else 0.0.

        Returns:
        (macro_auc, auc_per_label)
        - macro_auc: mean AUC across labels with a valid AUC (labels with only one class in y_true are ignored).
        - auc_per_label: dict mapping label -> AUC (float('nan') for labels with undefined AUC).
        """
        # gather lists (same structure as in your precision/recall method)
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # build label universe and total_tgt to restrict evaluation to labels that appear in targets
        label_universe = set()
        total_tgt = defaultdict(int)
        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            p_set = set(pred_list)
            t_set = set(tgt_list)
            label_universe |= (p_set | t_set)
            t_ctr = Counter(tgt_list)
            for lbl in (p_set | t_set):
                total_tgt[lbl] += t_ctr.get(lbl, 0)

        # restrict to labels that exist in target (same rule you used for precision/recall)
        target_label_list = sorted(k for k, v in total_tgt.items() if v > 0)

        auc_per_label: Dict[str, float] = {}
        valid_aucs = []

        # for plotting / debugging: keep per-label arrays
        per_label_y_true: Dict[str, np.ndarray] = {}
        per_label_y_score: Dict[str, np.ndarray] = {}

        # for each label build y_true and y_score across cases
        for lbl in target_label_list:
            y_true = []
            y_score = []
            for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
                tgt_has = (lbl in tgt_list)   # True if label present in target for this case
                pred_has = (lbl in pred_list) # True if label present in prediction for this case

                # y_true: 1 if actual "deviation" (label present in target)
                y_true.append(1 if tgt_has else 0)
                # y_score: predicted probability/score for "deviation".
                # binary prediction -> 1.0 for predicted dev, 0.0 otherwise
                y_score.append(1.0 if pred_has else 0.0)

            y_true_arr = np.array(y_true)
            y_score_arr = np.array(y_score)

            # store arrays for plotting
            per_label_y_true[lbl] = y_true_arr
            per_label_y_score[lbl] = y_score_arr

            # If y_true contains only a single class, AUC is undefined
            if np.unique(y_true_arr).size < 2:
                auc_per_label[lbl] = float('nan')
                continue

            try:
                auc = float(roc_auc_score(y_true_arr, y_score_arr))
            except Exception:
                # defensive: if something else goes wrong, mark as nan
                auc = float('nan')

            auc_per_label[lbl] = auc
            if not np.isnan(auc):
                valid_aucs.append(auc)

        # macro: mean over labels with valid AUCs. If none valid, return nan.
        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float('nan')

        # plotting: pass full per-label dicts (method should accept these)
        if plot and hasattr(self, "_{}_plot_roc_per_label".format("__") ) is False:
            # if your plotting helper is named __plot_roc_per_label as before:
            try:
                self.__plot_roc_per_label(per_label_y_true=per_label_y_true, per_label_y_score=per_label_y_score)
            except Exception:
                # swallow plotting errors so metric computation is unaffected
                pass

        return macro_auc, auc_per_label

    def __plot_roc_per_label(self, per_label_y_true: dict, per_label_y_score: dict, labels=None):
        """
        Plot ROC curves for multiple labels.
        
        Args:
            per_label_y_true: dict[label] -> list of 0/1 target values per case
            per_label_y_score: dict[label] -> list of predicted scores per case
            labels: list of labels to plot (default: all in per_label_y_true)
        """
        if labels is None:
            labels = sorted(per_label_y_true.keys())
        
        plt.figure(figsize=(10, 8))
        
        for lbl in labels:
            y_true = np.array(per_label_y_true[lbl])
            y_score = np.array(per_label_y_score[lbl])
            
            # Skip labels with no positive or no negative examples (ROC AUC undefined)
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                print(f"Skipping label {lbl}: ROC undefined (only one class present)")
                continue
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{lbl} (AUC = {roc_auc:.2f})")
        
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')  # diagonal
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves per Label", fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
