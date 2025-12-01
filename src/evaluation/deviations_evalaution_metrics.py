import ast
import random
from collections import Counter, defaultdict
from typing import Iterable, Dict, List, Tuple, Union, Any
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
    
    def _cases_with_target_deviations(self) -> List[dict]:
        return [
            {
                'tgt_suffix': dr.get('tgt_suffix', []),
                'pred_suffix': dr.get('pred_suffix', []),
                'tgt_aligns': dr.get('tgt_cleaned_aligns', []),
                'pred_aligns': dr.get('pred_cleaned_aligns', []),
                'tgt_deviations': dr.get('tgt_deviations', []),
                'pred_deviations': dr.get('pred_deviations', []),
            }
            for dr in self.deviation_results
            if len(dr.get('tgt_deviations', [])) > 0
        ]

    # Sequence evaluation: Evaluate the position of deviation in suffix
    def get_suffix_devs(self):
        real_deviations = self._cases_with_target_deviations()

        tgt_aligns = [rd.get('tgt_aligns', []) for rd in real_deviations]
        pred_aligns = [rd.get('pred_aligns', []) for rd in real_deviations]
        tgt_suffixes = [rd.get('tgt_suffix', []) for rd in real_deviations]
        pred_suffix_samples = [rd.get('pred_suffix', []) for rd in real_deviations]
        real_tgt_devs = [rd.get('tgt_deviations', []) for rd in real_deviations]
        real_pred_devs = [rd.get('pred_deviations', []) for rd in real_deviations]

        assert len(real_tgt_devs) == len(tgt_suffixes) == len(real_pred_devs) == len(pred_suffix_samples)

        tgt_model_moves, tgt_log_moves = [], []
        pred_model_moves, pred_log_moves = [], []

        for align in tgt_aligns:
            model_positions: Dict[str, List[int]] = defaultdict(list)
            log_positions: Dict[str, List[int]] = defaultdict(list)
            for idx, move in enumerate(align):
                if move[0] == '>>' and move[1] is not None:
                    model_positions[move[1]].append(idx)
                elif move[1] == '>>' and move[0] is not None:
                    log_positions[move[0]].append(idx)
            tgt_model_moves.append(dict(model_positions))
            tgt_log_moves.append(dict(log_positions))

        for align_samples in pred_aligns:
            model_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
            log_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
            for sample in align_samples:
                for idx, move in enumerate(sample):
                    if move[0] == '>>' and move[1] is not None:
                        model_counts[move[1]][idx] += 1
                    elif move[1] == '>>' and move[0] is not None:
                        log_counts[move[0]][idx] += 1
            pred_model_moves.append({
                label: {pos: pos_counts[pos] for pos in sorted(pos_counts)}
                for label, pos_counts in model_counts.items()
            })
            pred_log_moves.append({
                label: {pos: pos_counts[pos] for pos in sorted(pos_counts)}
                for label, pos_counts in log_counts.items()
            })

        return tgt_suffixes, pred_suffix_samples, (tgt_model_moves, tgt_log_moves), (pred_model_moves, pred_log_moves)
    
    def likelihood_at_target_positions(self,
                                       tgt_model_moves: List[Dict[str, List[int]]],
                                       tgt_log_moves: List[Dict[str, List[int]]],
                                       pred_model_moves: List[Dict[str, Dict[int, int]]],
                                       pred_log_moves: List[Dict[str, Dict[int, int]]],
                                       num_samples: int = 100) -> Tuple[Dict[str, List[Dict[str, List[Tuple[int, float]]]]],
                                                                         Dict[str, Dict[str, float]],
                                                                         Dict[str, float]]:
        """
        Compute likelihoods that predicted move positions match target move positions
        for both model moves (('>>', x)) and log moves ((x, '>>')).

        Returns a tuple of three dictionaries keyed by {"model", "log"}:
        - case_level: list per case, each dict[label] -> list[(position, probability)]
        - per_label_mean: per move label mean probability over all its target positions
        - weighted_macro: occurrence-weighted mean probability across labels
        """
        def _compute(tgt_moves: List[Dict[str, List[int]]],
                     pred_moves: List[Dict[str, Dict[int, int]]]
                     ) -> Tuple[List[Dict[str, List[Tuple[int, float]]]], Dict[str, float], float]:
            if len(tgt_moves) != len(pred_moves):
                raise ValueError("Target and predicted move collections must have identical length.")

            case_level: List[Dict[str, List[Tuple[int, float]]]] = []
            per_label_scores: Dict[str, List[float]] = defaultdict(list)
            per_label_support: Dict[str, int] = defaultdict(int)

            all_labels = sorted({lbl for case in tgt_moves for lbl in case.keys()})

            for tgt_case, pred_case in zip(tgt_moves, pred_moves):
                case_entry: Dict[str, List[Tuple[int, float]]] = {}
                for label in all_labels:
                    positions = tgt_case.get(label, [])
                    if not positions:
                        continue
                    counts = pred_case.get(label, {})
                    position_scores: List[Tuple[int, float]] = []
                    for pos in positions:
                        count = counts.get(pos, 0)
                        prob = (count / num_samples) if num_samples > 0 else 0.0
                        position_scores.append((pos, float(prob)))
                        per_label_scores[label].append(float(prob))
                    per_label_support[label] += len(position_scores)
                    case_entry[label] = position_scores
                case_level.append(case_entry)

            per_label_mean = {
                label: (float(np.mean(scores)) if scores else 0.0)
                for label, scores in per_label_scores.items()
            }

            total_support = sum(per_label_support.values())
            weighted_macro = (
                float(sum(sum(scores) for scores in per_label_scores.values()) / total_support)
                if total_support > 0 else 0.0
            )
            return case_level, per_label_mean, weighted_macro

        model_case, model_label_mean, model_weighted = _compute(tgt_model_moves, pred_model_moves)
        log_case, log_label_mean, log_weighted = _compute(tgt_log_moves, pred_log_moves)

        case_level = {"model": model_case, "log": log_case}
        per_label_mean = {"model": model_label_mean, "log": log_label_mean}
        weighted_macro = {"model": model_weighted, "log": log_weighted}

        return case_level, per_label_mean, weighted_macro
    
    def plot_suffix_deviation_distribution(self,
                                           suffix_index: int,
                                           label: str,
                                           move: str,
                                           tgt_suff_move: List[Dict[str, List[int]]],
                                           pred_suff_move: List[Dict[str, Dict[int, int]]],
                                           pred_suffix_samples: List[List[List[str]]],
                                           tgt_suffixes: List[List[str]] = None,
                                           num_samples: int = 100) -> Dict[str, Any]:
        """
        Visualize the distribution of a deviation label for a given suffix case:
        - Blue bars = occurrences at non-top-3 positions, green bars = three most frequent positions.
        - Dashed orange lines mark the true target positions.
        - Shows the target suffix and one predicted sample that exhibits the deviation at a true position.
        """
        if suffix_index < 0 or suffix_index >= len(tgt_suff_move):
            raise IndexError("suffix_index out of range.")

        if tgt_suffixes is None:
            tgt_suffixes = [case['tgt_suffix'] for case in self._cases_with_target_deviations()]

        if suffix_index >= len(tgt_suffixes):
            raise IndexError("suffix_index exceeds available target suffixes.")

        tgt_case_positions = tgt_suff_move[suffix_index]
        pred_case_counts = pred_suff_move[suffix_index]

        true_positions = tgt_case_positions.get(label, [])
        counts_dict = pred_case_counts.get(label, {})
        if not counts_dict:
            raise ValueError(f"No predicted occurrences for label '{label}' in suffix {suffix_index}.")

        all_positions = sorted(counts_dict.keys())
        percentages = [
            (counts_dict[pos] / num_samples) * 100.0 if num_samples > 0 else 0.0
            for pos in all_positions
        ]

        top_positions = sorted(all_positions, key=lambda p: counts_dict[p], reverse=True)[:3]
        top_position_set = set(top_positions)

        fig = plt.figure(figsize=(11, 6.5))
        gs = fig.add_gridspec(3, 1, height_ratios=[2.3, 1, 1], hspace=0.5)

        ax_hist = fig.add_subplot(gs[0])
        colors = ["#2ca02c" if pos in top_position_set else "#1f77b4" for pos in all_positions]
        bars = ax_hist.bar(all_positions, percentages, color=colors, edgecolor="#31465f", width=0.55)
        ax_hist.set_xlabel("Positions in sampled suffixes", fontsize=11)
        ax_hist.set_ylabel("Occurrence in 100 samples (%)", fontsize=11)
        
        if move == 'log':
            ax_hist.set_title(f"Deviation: ({label}, >>)", fontweight="bold", fontsize=13)
        else:
            ax_hist.set_title(f"Deviation: (>>, {label})", fontweight="bold", fontsize=13)

        for bar, pct in zip(bars, percentages):
             ax_hist.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                         f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.5)

        for idx, tp in enumerate(true_positions):
            ax_hist.axvline(tp, color="#d95f02", linestyle="--", linewidth=1.6,
                            label="True position" if idx == 0 else None)

        ax_hist.set_ylim(0, max(percentages) * 1.25 if percentages else 5)
        if true_positions:
            ax_hist.legend(loc="upper right", fontsize=9)
        ax_hist.grid(axis="y", linestyle=":", alpha=0.35)

        samples_case = pred_suffix_samples[suffix_index]
        matching_samples = []
        target_position_set = set(true_positions)
        for sample_idx, sample in enumerate(samples_case):
            match_positions = [pos for pos in true_positions if pos < len(sample) and sample[pos] == label]
            if match_positions:
                matching_samples.append((sample_idx, sample, match_positions))

        if not matching_samples:
            raise ValueError(f"No predicted sample contains '{label}' at any target position for suffix {suffix_index}.")

        chosen_idx, chosen_sample, highlight_positions = random.choice(matching_samples)
        samples_counter = Counter(tuple(s) for s in samples_case)
        chosen_freq = samples_counter[tuple(chosen_sample)]

        def _draw_sequence(ax, events: List[str], highlight_positions: List[int], title: str):
            ax.axis("off")
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.1, 1.0)
            n = len(events)
            if n == 0:
                ax.text(0.5, 0.45, "(empty suffix)", ha="center", va="center", fontsize=9)
                return
            x_positions = np.linspace(0.08, 0.92, n)
            width = min(0.12, 0.65 / max(n, 2))
            ax.text(0.02, 0.92, title, fontsize=9.5, fontweight="semibold", color="#2d4059", ha="left")
            for idx, (event, x) in enumerate(zip(events, x_positions)):
                facecolor = "#ffe4c4" if idx in highlight_positions else "#f7f9fc"
                rect = plt.Rectangle((x - width / 2, 0.35), width, 0.3,
                                    edgecolor="#516173", linewidth=0.9, facecolor=facecolor)
                ax.add_patch(rect)
                ax.text(x, 0.5, event, ha="center", va="center", fontsize=8.4)
                if idx < n - 1:
                    ax.annotate("",
                                xy=(x_positions[idx + 1] - width / 2 + 0.004, 0.5),
                                xytext=(x + width / 2 - 0.004, 0.5),
                                arrowprops=dict(arrowstyle="->", linewidth=0.9, color="#516173"))

        target_events = tgt_suffixes[suffix_index]
        _draw_sequence(fig.add_subplot(gs[1]), target_events, true_positions, "Target suffix (true deviation position)")

        pred_title = f"Exemplary sample #{chosen_idx} (freq={chosen_freq}) with deviation at target position"
        _draw_sequence(fig.add_subplot(gs[2]), chosen_sample, highlight_positions, pred_title)

        fig.tight_layout()
        plt.show()
