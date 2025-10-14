from collections import Counter
from typing import List, Dict, Any

class Evaluation:    
    def __init__(self, deviation_results):
        self.deviation_results = deviation_results
        
        
    def _collect_counts(self,
                        key: str,
                        label_only: bool = True,
                        label_index: int = 0) -> Counter:
        """
        Collect counts of deviation keys across all cases.

        - cases: list of dicts containing 'pred_deviations' / 'tgt_deviations'
        - key: which field to collect (e.g. "pred_deviations" or "tgt_deviations")
        - label_only: if True, use the element at label_index from each deviation tuple
        - label_index: index inside the deviation tuple to use when label_only is True
        """
        ctr = Counter()
        for case in self.deviation_results:
            items = case.get(key, [])
            for it in items:
                if label_only and isinstance(it, (list, tuple)):
                    if label_index < len(it):
                        k = it[label_index]
                    else:
                        # fallback: use whole object if requested index missing
                        k = it
                else:
                    k = it
                ctr[k] += 1
        return ctr        

    def precision_deviations(self,
                             pred_key: str = "pred_deviations",
                             tgt_key: str = "tgt_deviations",
                             label_only: bool = True,
                             label_index: int = 0,
                             zero_division: float = 0.0) -> float:
        """
        Micro-averaged precision across all cases.

        precision = TP / total_predicted
        Returns zero_division if total_predicted == 0.
        """
        pred_counts = self._collect_counts(pred_key, label_only=label_only, label_index=label_index)
        tgt_counts = self._collect_counts(tgt_key, label_only=label_only, label_index=label_index)

        # True positives: sum over labels of min(pred_count[label], tgt_count[label])
        common_labels = set(pred_counts.keys()) & set(tgt_counts.keys())
        tp = sum(min(pred_counts[l], tgt_counts[l]) for l in common_labels)
        total_pred = sum(pred_counts.values())

        if total_pred == 0:
            return float(zero_division)
        return tp / total_pred

    def recall_deviations(self,
                          pred_key: str = "pred_deviations",
                          tgt_key: str = "tgt_deviations",
                          label_only: bool = True,
                          label_index: int = 0,
                          zero_division: float = 0.0) -> float:
        """
        Micro-averaged recall across all cases.

        recall = TP / total_target
        Returns zero_division if total_target == 0.
        """
        pred_counts = self._collect_counts(pred_key, label_only=label_only, label_index=label_index)
        tgt_counts = self._collect_counts(tgt_key, label_only=label_only, label_index=label_index)

        common_labels = set(pred_counts.keys()) & set(tgt_counts.keys())
        tp = sum(min(pred_counts[l], tgt_counts[l]) for l in common_labels)
        total_tgt = sum(tgt_counts.values())

        if total_tgt == 0:
            return float(zero_division)
        return tp / total_tgt

    