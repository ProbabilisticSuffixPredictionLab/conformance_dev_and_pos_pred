from collections import Counter, defaultdict
from typing import Iterable, Any, Callable, Dict, List, Optional, Tuple
import math
from sklearn.metrics import roc_auc_score

class Evaluation:
    def __init__(self, deviation_results: Iterable[dict]):
        self.deviation_results = list(deviation_results)
        
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
    

    def auc_roc_macro_deviations(self,
                                 label_only: bool = True,
                                 label_index: int = 0,
                                 score_extractor: Optional[Callable[[dict], Dict[Any, float]]] = None,
                                 labels: Optional[List[Any]] = None
                                 ) -> Dict[str, Any]:
        """
        Compute macro AUC-ROC (one-vs-rest) across classes.

        Parameters
        - score_extractor: function(case) -> dict {label: score_in_[0,1]}. If None, tries to find 'pred_proba' or 'pred_scores' in each case.
        
        The extractor should return a score for every label you want to evaluate (missing label -> 0.0).
        
        - labels: optional list of labels to evaluate; if None the labels are inferred from targets.

        Returns a dict:
        - "per_label_auc": {label: auc_or_nan, ...},
        - "macro_auc": mean_of_valid_aucs_or_nan,
        - "labels_evaluated": [...],
        - "skipped_labels": [...],  # labels with undefined AUC (no positives or no negatives)
        - "error": error_msg_if_any_or_None

        """
        if roc_auc_score is None:
            return {"error": "sklearn.metrics.roc_auc_score not available in this environment."}
        
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # Infer labels from targets if not provided
        if labels is None:
            lbls = set()
            for tgts_raw in tgt_deviations:
                for t in tgts_raw:
                    lbls.add(t)
            labels = sorted(lbls)

        # Build y_true matrix (cases x labels (all log and model moves present in the target)): 1 if label is in case, else 1 
        y_true_per_label: Dict[Any, List[int]] = {lab: [] for lab in labels}
        for tgts in tgt_deviations:
            tgt_set = set(tgts)
            for lab in labels:
                y_true_per_label[lab].append(1 if lab in tgt_set else 0)



        # default attempt: look for 'pred_proba' or 'pred_scores' in the case
        score_extractor = {}
        mapping = case.get("pred_proba") or case.get("pred_scores") or {}
        # mapping might be dict or list of (label, score)
        if isinstance(mapping, dict):
            for lab in labels:
                score_extractor[lab] = float(mapping.get(lab, 0.0))
        else:
            # try to interpret as iterable of pairs
            for lab in labels:
                score_extractor[lab] = 0.0
            for pair in mapping:
                # accept (label, score)
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    lab = pair[0]
                    score = pair[1]
                    if lab in labels:
                        score_extractor[lab] = float(score)





        # Build y_score per label (list of floats per case)
        y_score_per_label: Dict[Any, List[float]] = {lab: [] for lab in labels}
        for case in self.deviation_results:
            scores_dict = score_extractor(case) or {}
            for lab in labels:
                s = scores_dict.get(lab, 0.0)
                # clamp/convert to float
                try:
                    s = float(s)
                except Exception:
                    s = 0.0
                # ensure within [0,1] is recommended but not enforced here
                y_score_per_label[lab].append(s)

        per_label_auc: Dict[Any, float] = {}
        skipped_labels: List[Any] = []
        for lab in labels:
            y_true = y_true_per_label[lab]
            y_score = y_score_per_label[lab]
            # If y_true is constant (all 0 or all 1), ROC AUC is undefined
            if all(v == 0 for v in y_true) or all(v == 1 for v in y_true):
                per_label_auc[lab] = float("nan")
                skipped_labels.append(lab)
                continue
            try:
                auc = roc_auc_score(y_true, y_score)
            except Exception:
                auc = float("nan")
            per_label_auc[lab] = float(auc)

        # Macro: mean of valid (non-nan) per-label AUCs
        valid_aucs = [v for v in per_label_auc.values() if (not math.isnan(v))]
        macro_auc = float(sum(valid_aucs) / len(valid_aucs)) if valid_aucs else float("nan")

        return {"per_label_auc": per_label_auc,
                "macro_auc": macro_auc,
                "labels_evaluated": labels,
                "skipped_labels": skipped_labels,
                "error": None}
        
        
    def precision_recall_macro_by_label(self,
                                        zero_division: float = 1.0
                                        ) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Compute per-label precision and recall, then return macro-averages across labels.

        Returns:
        precision_macro (float): mean of per-label precisions
        recall_macro    (float): mean of per-label recalls
        precision_per_label (dict): label -> precision
        recall_per_label    (dict): label -> recall
        counts (dict): summary counts: {'TP_total': int, 'FP_total': int, 'FN_total': int}
        Conventions:
        - TP_label computed as sum_over_cases min(pred_count, tgt_count)
        - If denom == 0 for precision/recall, return `zero_division` for that label.
        """
        # collect per-case counters
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # label-level accumulators
        total_pred = defaultdict(int)  # total predicted counts per label across all cases
        total_tgt  = defaultdict(int)  # total target counts per label across all cases
        tp_label   = defaultdict(int)  # true positives per label (summed per-case min)

        # accumulate counts per-case (TP computed as min per case)
        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            p_ctr = Counter(pred_list)
            t_ctr = Counter(tgt_list)
            labels_union = set(p_ctr) | set(t_ctr)
            for lbl in labels_union:
                total_pred[lbl] += p_ctr.get(lbl, 0)
                total_tgt[lbl]  += t_ctr.get(lbl, 0)
                tp_label[lbl]   += min(p_ctr.get(lbl, 0), t_ctr.get(lbl, 0))

        # build per-label precision/recall
        precision_per_label = {}
        recall_per_label = {}
        label_list = sorted(set(list(total_pred.keys()) + list(total_tgt.keys())))

        for lbl in label_list:
            tp = tp_label.get(lbl, 0)
            pred_sum = total_pred.get(lbl, 0)
            tgt_sum = total_tgt.get(lbl, 0)

            # precision: TP / (TP + FP)  where FP = pred_sum - TP
            denom_prec = tp + (pred_sum - tp)  # which is pred_sum
            if denom_prec == 0:
                precision_per_label[lbl] = float(zero_division)
            else:
                precision_per_label[lbl] = tp / denom_prec

            # recall: TP / (TP + FN) where FN = tgt_sum - TP
            denom_rec = tp + (tgt_sum - tp)  # which is tgt_sum
            if denom_rec == 0:
                recall_per_label[lbl] = float(zero_division)
            else:
                recall_per_label[lbl] = tp / denom_rec

        # macro averages over labels
        if label_list:
            precision_macro = float(sum(precision_per_label[lbl] for lbl in label_list) / len(label_list))
            recall_macro = float(sum(recall_per_label[lbl] for lbl in label_list) / len(label_list))
        else:
            # no labels at all -- fallback
            precision_macro = float(zero_division)
            recall_macro = float(zero_division)

        # also return micro/global counts (useful diagnostics)
        TP_total = sum(tp_label.values())
        FP_total = sum(total_pred[lbl] - tp_label[lbl] for lbl in label_list)
        FN_total = sum(total_tgt[lbl] - tp_label[lbl] for lbl in label_list)
        micro_prec = float(TP_total / (TP_total + FP_total)) if (TP_total + FP_total) > 0 else float(zero_division)
        micro_rec  = float(TP_total / (TP_total + FN_total)) if (TP_total + FN_total) > 0 else float(zero_division)

        counts = {
            'TP_total': int(TP_total),
            'FP_total': int(FP_total),
            'FN_total': int(FN_total),
            'micro_precision': micro_prec,
            'micro_recall': micro_rec
        }

        return precision_macro, recall_macro, precision_per_label, recall_per_label, counts
    

    def precision_recall_macro_no_deviations(self,
                                            zero_division: float = 1.0
                                            ) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Compute per-label precision and recall for the *no-deviation* event (opposite).
        Positive event = label is NOT present in prediction and NOT present in target.
        Returns:
        precision_macro (float), recall_macro (float),
        precision_per_label (dict), recall_per_label (dict),
        counts (dict) with global TP/FP/FN and micro-precision/recall.
        Conventions:
        - We operate on presence/absence per case (binary), not counts.
        - Per-label TP_label = sum_cases [not pred_has and not tgt_has].
        - pred_positive_count = sum_cases [not pred_has]  (predicted no-deviation)
        - true_positive_count = sum_cases [not tgt_has]  (actually no-deviation)
        - If denominator == 0 for precision/recall, return `zero_division` for that label.
        """
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # accumulators per label
        tp_no = defaultdict(int)         # true positives for "no-deviation" per label
        pred_no_count = defaultdict(int) # predicted no-deviation counts per label (pred_has == False)
        true_no_count = defaultdict(int) # true no-deviation counts per label (tgt_has == False)

        # collect label universe (union of labels seen anywhere)
        label_universe = set()
        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            p_set = set(pred_list)
            t_set = set(tgt_list)
            label_universe |= (p_set | t_set)

        # If there are no labels observed at all, return default values
        if not label_universe:
            precision_macro = float(zero_division)
            recall_macro = float(zero_division)
            return precision_macro, recall_macro, {}, {}, {
                'TP_total': 0, 'FP_total': 0, 'FN_total': 0,
                'micro_precision': float(zero_division), 'micro_recall': float(zero_division)
            }

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

        # compute per-label precision/recall (for no-deviation)
        precision_per_label = {}
        recall_per_label = {}
        for lbl in sorted(label_universe):
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

        # micro/global counts
        TP_total = sum(tp_no.values())
        FP_total = sum(pred_no_count[lbl] - tp_no[lbl] for lbl in label_universe)   # predicted no-deviation but target had label
        FN_total = sum(true_no_count[lbl] - tp_no[lbl] for lbl in label_universe)   # target no-deviation but pred had label
        micro_prec = float(TP_total / (TP_total + FP_total)) if (TP_total + FP_total) > 0 else float(zero_division)
        micro_rec  = float(TP_total / (TP_total + FN_total)) if (TP_total + FN_total) > 0 else float(zero_division)

        counts = {
            'TP_total': int(TP_total),
            'FP_total': int(FP_total),
            'FN_total': int(FN_total),
            'micro_precision': micro_prec,
            'micro_recall': micro_rec
        }

        return precision_macro, recall_macro, precision_per_label, recall_per_label, counts
