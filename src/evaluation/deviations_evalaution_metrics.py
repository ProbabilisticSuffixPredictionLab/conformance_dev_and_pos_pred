import ast
from collections import Counter, defaultdict
from typing import Iterable, Dict, List, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def load_results(path: Union[str, Path]) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)

class DeviationEvaluation:
    """
    1)Macro average metrics: take the label an compute the precisiona and recall across for a label across all cases
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        (Micro would be to compute TP, FP, ... for all cases and all labels, so not per label but for all labels at once.)
    
    2) ROC_AUC
    
    3) Get suffixes with target deviations and occurence/ position in predicted suffix samples
    """
    def __init__(self, deviation_results: Iterable[dict]):
        self.deviation_results = list(deviation_results)
    
    def precision_recall_macro_by_label_dev(self) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Compute per-label precision and recall only for labels appearing in the target set,
        then return macro-averages across those target labels.
        
        Per label:
        TP: pred 1, tgt 1
        FP: pred 1, tgt 0
        TN: pred 0, tgt 0
        FN: pred 0, tgt 1

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
            if tp_fp == 0:
                # Not predicted but in target -> tp = 0, fn > 0
                precision_per_label[lbl] = 0
            else:
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
        
    def precision_recall_macro_by_label_no_dev(self, zero_division: float = 1.0) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Compute per-label precision and recall for the no-deviation (opposite).
        Positive event = label is NOT present in prediction and NOT present in target.
        
        Per label:
        TP: pred 0, tgt 0
        FP: pred 0, tgt 1
        TN: pred 1, tgt 0
        FN: pred 1, tgt 1
        
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
        
    def plot_macro_roc_auc(self, figsize=(8, 6)):
        pred_lists = [dr.get("pred_deviations", []) for dr in self.deviation_results]
        tgt_lists = [dr.get("tgt_deviations", []) for dr in self.deviation_results]

        total_tgt = defaultdict(int)
        for tgt in tgt_lists:
            for lbl in tgt:
                total_tgt[lbl] += 1

        label_names = sorted(k for k, v in total_tgt.items() if v > 0)
        if not label_names:
            return {"per_label_auc": [], "macro_auc": float("nan")}

        label_to_idx = {lbl: idx for idx, lbl in enumerate(label_names)}
        num_cases = len(tgt_lists)
        num_labels = len(label_names)

        y_true = [[0] * num_labels for _ in range(num_cases)]
        y_scores = [[0] * num_labels for _ in range(num_cases)]

        for row, (pred, tgt) in enumerate(zip(pred_lists, tgt_lists)):
            for lbl in tgt:
                if lbl in label_to_idx:
                    y_true[row][label_to_idx[lbl]] = 1
            for lbl in pred:
                if lbl in label_to_idx:
                    y_scores[row][label_to_idx[lbl]] = 1

        per_label_auc = [float("nan")] * num_labels
        plt.figure(figsize=figsize)
        ax = plt.gca()

        valid_indices = []
        for idx, lbl in enumerate(label_names):
            column_true = [case[idx] for case in y_true]
            if len(set(column_true)) < 2:
                continue
            column_scores = [case[idx] for case in y_scores]
            fpr, tpr, _ = roc_curve(column_true, column_scores)
            roc_auc = auc(fpr, tpr)
            per_label_auc[idx] = roc_auc
            valid_indices.append(idx)
            ax.plot(fpr, tpr, lw=1.5, label=f"{lbl} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves per Deviation Label")
        ax.legend(loc="lower right", fontsize="small")
        plt.show()

        macro_auc = float(
            sum(per_label_auc[i] for i in valid_indices) / len(valid_indices)
        ) if valid_indices else float("nan")

        return {"per_label_auc": per_label_auc, "macro_auc": macro_auc}
    
    # Sequence evaluation: Evaluate the place of occurence of deviation in suffix
    def get_suffix_devs(self):
        # Get the deviation values for which a deviation occured: 
        real_deviations = [{'tgt_deviations':dr.get('tgt_deviations', []),
                            'pred_deviations':dr.get('pred_deviations', []),
                            'tgt_suffix': dr.get('tgt_suffix', []),
                            'pred_suffix': dr.get('pred_suffix', []),
                            'pred_deviations': dr.get('pred_deviations', [])} for dr in self.deviation_results if len(dr.get('tgt_deviations', [])) > 0]
        
        # Target deviation and suffix
        real_tgt_devs = [dr.get('tgt_deviations', []) for dr in real_deviations]
        real_tgt_suffixes = [dr.get('tgt_suffix', []) for dr in real_deviations]
        
        # Predicted deviation and suffix (100 elements per list)
        real_pred_devs = [dr.get('pred_deviations', []) for dr in real_deviations]
        real_pred_suffix_samples = [dr.get('pred_suffix', []) for dr in real_deviations]
        
        assert len(real_tgt_devs) == len(real_tgt_suffixes) == len(real_pred_devs) ==  len(real_pred_suffix_samples)
        
        tgt_suff_dev_pos = []
        pred_suff_dev_pos = []
        for i, suffix in enumerate(real_tgt_suffixes):
            # Target deviation with position in suffix
            tgt_dev_labels = tgt_dev_labels = [a if a != ">>" else b for (a, b) in real_tgt_devs[i]]
            
            # position of target deviations in the target:
            tgt_p = {}
            for t_dev in tgt_dev_labels:
                tgt_p[t_dev] = self._all_indices(suffix, t_dev)
            tgt_suff_dev_pos.append(tgt_p)
            
            # Based on the real position get the position of the pred
            pred_dev_labels = pred_dev_labels = [a if a != ">>" else b for (a, b) in real_pred_devs[i]]
            pred_dev_in_tgt = [d for d in pred_dev_labels if d in tgt_dev_labels]
            
            # Stores for the suffix
            pred_p = {}
            # 1 or multiple target deviations for this suffix
            for p_dev in pred_dev_in_tgt:
                positions = []
                # 100 samples
                for sample in real_pred_suffix_samples[i]:
                    # add the elements to the existing list
                    positions.extend(self._all_indices(sample, p_dev))
                # Counts for each index the occurence of the deviation across the samples, e.g., 'Take in charge ticket': Counter({0: 83, 1: 21, 2: 9, 3: 9, 4: 2})
                count_positions = Counter(positions)
                # sort the indices and keeps an 
                sorted_relative_count_positions = sorted(((p, count_positions[p] / len(real_pred_suffix_samples[i])) for p in count_positions),key=lambda item: item[0])
                pred_p[p_dev] = sorted_relative_count_positions
            pred_suff_dev_pos.append(pred_p)  
        
        return tgt_suff_dev_pos, pred_suff_dev_pos, real_tgt_suffixes, real_pred_suffix_samples
    
    def _all_indices(self, lst, x):
        """Return all indices of x in lst, or [0] if not found."""
        indices = [i for i, val in enumerate(lst) if val == x]
        return indices if indices else [0]
    
    def likelihood_at_target_positions(self,
                                       tgt_suff_dev_poss: List[Dict[str, List[int]]],
                                       pred_suff_dev_poss: List[Dict[str, List[Tuple[int, float]]]]) -> Tuple[List[Dict[str, List[Tuple[int, float]]]], Dict[str, float], float]:
        """
        For each label compute the predicted relative likelihood at the target positions.
        Returns:
        - case_level: list (per case) of dict[label] -> list[(target_pos, predicted_likelihood)]
        - per_label_mean: dict[label] -> mean likelihood across all target positions (includes zeros)
        - weighted_macro: occurrence-weighted mean across all labels
        """
        if len(tgt_suff_dev_poss) != len(pred_suff_dev_poss):
            raise ValueError("tgt_suff_dev_poss and pred_suff_dev_poss must have the same length")

        case_level: List[Dict[str, List[Tuple[int, float]]]] = []
        per_label_scores: Dict[str, List[float]] = defaultdict(list)
        per_label_support: Dict[str, int] = defaultdict(int)

        all_labels = sorted({lbl for case in tgt_suff_dev_poss for lbl in case.keys()})

        for tgt_case, pred_case in zip(tgt_suff_dev_poss, pred_suff_dev_poss):
            case_entry: Dict[str, List[Tuple[int, float]]] = {}
            for label in all_labels:
                tgt_positions = tgt_case.get(label, [])
                if not tgt_positions:
                    continue
                pred_distribution = dict(pred_case.get(label, []))
                position_scores = [(pos, float(pred_distribution.get(pos, 0.0))) for pos in tgt_positions]
                case_entry[label] = position_scores
                per_label_support[label] += len(position_scores)
                for _, score in position_scores:
                    per_label_scores[label].append(score)
            case_level.append(case_entry)

        per_label_mean = {
            label: (float(np.mean(per_label_scores[label])) if per_label_scores[label] else 0.0)
            for label in all_labels
        }

        total_support = sum(per_label_support[label] for label in all_labels)
        weighted_macro = (
            float(sum(sum(per_label_scores[label]) for label in all_labels) / total_support)
            if total_support > 0 else 0.0
        )

        return case_level, per_label_mean, weighted_macro
    
    def plot_suffix_deviation_distribution(self,
                                           suffix_index: int,
                                           label: str,
                                           tgt_suff_dev_poss: List[Dict[str, List[int]]],
                                           pred_suff_dev_poss: List[Dict[str, List[Tuple[int, float]]]],
                                           pred_suffix_samples: List[List[List[str]]]) -> List[int]:
        """
        Visualize, for a single suffix and deviation label, how often the label
        appears at each position across predicted samples, highlighting the true positions.

        Returns:
            List[int]: indices of predicted samples that contain the label at the most likely position.
        """
        if suffix_index < 0 or suffix_index >= len(tgt_suff_dev_poss):
            raise IndexError("suffix_index out of range")

        tgt_case = tgt_suff_dev_poss[suffix_index]
        pred_case = pred_suff_dev_poss[suffix_index]

        true_positions = tgt_case.get(label, [])
        pred_distribution = dict(pred_case.get(label, []))

        if not pred_distribution:
            raise ValueError(f"No predicted occurrences found for label '{label}' in suffix {suffix_index}")

        positions = sorted(pred_distribution.keys())
        percentages = np.array([pred_distribution[pos] * 100.0 for pos in positions])

        top_pos_idx = np.argsort(percentages)[-3:]
        top_pos_set = {positions[idx] for idx in top_pos_idx}

        base_color = "#1f77b4"
        highlight_color = "#2ca02c"
        colors = [highlight_color if pos in top_pos_set else base_color for pos in positions]

        fig = plt.figure(figsize=(12, 6.6))
        gs = fig.add_gridspec(4, 1, height_ratios=[2.4, 1, 1, 1], hspace=0.45)

        ax_hist = fig.add_subplot(gs[0])
        bars = ax_hist.bar(positions, percentages, color=colors, edgecolor="#4d6278", alpha=0.95)
        ax_hist.set_xlabel("Position in suffix", fontsize=11)
        ax_hist.set_ylabel("Occurrence percentage (%)", fontsize=11)
        ax_hist.set_title(f"Deviation '{label}' — suffix #{suffix_index}", fontweight="bold", fontsize=13)

        for bar, pct in zip(bars, percentages):
            ax_hist.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

        for idx, tp in enumerate(true_positions):
            ax_hist.axvline(tp, color="#d95f02", linestyle="--", linewidth=1.8,
                            label="True position" if idx == 0 else None)

        ax_hist.set_ylim(0, max(percentages) * 1.25)
        if true_positions:
            ax_hist.legend(loc="upper right", fontsize=9)
        ax_hist.grid(axis="y", linestyle=":", alpha=0.4)

        most_likely_pos = positions[int(np.argmax(percentages))]
        samples_counter = Counter(tuple(sample) for sample in pred_suffix_samples[suffix_index])
        matching_samples: List[Tuple[int, List[str], int]] = []

        for sample_idx, sample in enumerate(pred_suffix_samples[suffix_index]):
            indices = [i for i, val in enumerate(sample) if val == label]
            if most_likely_pos in indices:
                matching_samples.append((sample_idx, sample, samples_counter[tuple(sample)]))

        unique_samples: Dict[Tuple[str, ...], Tuple[int, List[str], int]] = {}
        for sample_idx, sample, freq in matching_samples:
            key = tuple(sample)
            if key not in unique_samples:
                unique_samples[key] = (sample_idx, sample, freq)

        top_samples = sorted(unique_samples.values(), key=lambda x: x[2], reverse=True)[:3]

        def _draw_process(ax, events: List[str], freq: int, sample_idx: int):
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.1, 1.0)

            if not events:
                ax.text(0.5, 0.45, "(empty)", ha="center", va="center", fontsize=8.5)
                return

            n = len(events)
            x_positions = np.linspace(0.12, 0.88, n)
            box_width = min(0.14, 0.6 / max(n, 2))
            box_height = 0.28

            ax.text(0.04, 0.88, f"sample #{sample_idx}  (freq={freq})",
                    fontsize=8.5, fontweight="semibold", ha="left", color="#24415c")

            for idx, (event, x) in enumerate(zip(events, x_positions)):
                rect = plt.Rectangle((x - box_width / 2, 0.32), box_width, box_height,
                                     linewidth=0.9, edgecolor="#5b6a7f", facecolor="#f1f5fb")
                ax.add_patch(rect)
                ax.text(x, 0.46, event, ha="center", va="center", fontsize=8.3)

                if idx < n - 1:
                    ax.annotate("",
                                xy=(x_positions[idx + 1] - box_width / 2 + 0.005, 0.46),
                                xytext=(x + box_width / 2 - 0.005, 0.46),
                                arrowprops=dict(arrowstyle="->", color="#5b6a7f", linewidth=0.9))
            
        for i, (sample_idx, sample, freq) in enumerate(top_samples):
            ax_proc = fig.add_subplot(gs[i + 1])
            _draw_process(ax_proc, sample, freq, sample_idx)

        fig.tight_layout()
        plt.show()

        print(f"Samples placing '{label}' at position {most_likely_pos}: {[idx for idx, _, _ in matching_samples]}")
