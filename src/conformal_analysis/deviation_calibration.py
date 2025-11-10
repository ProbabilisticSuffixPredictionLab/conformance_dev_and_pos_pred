import numpy as np
from collections import Counter
from sklearn.metrics import fbeta_score, precision_recall_curve
from typing import Union
from pathlib import Path
import json

class DeviationPredictionCalibration:
    def __init__(self, risk_conformance_results):
        self.risk_conformance_results = risk_conformance_results
    
    def __get_y(self):
        # Get all tgt alignments
        tgt_aligns = [tgt['suffix_alignment'] for tgt in self.risk_conformance_results['target_conformance']]
        # Remove filler from target alignments
        cleaned_tgt_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in tgt_aligns]
        tgt_deviations = [[(a, b) for (a, b) in align if a != b] for align in cleaned_tgt_alignments]
        
        # Dynamically collect all unique labels (transitions) from the sequences
        all_labels = set()
        for seq in tgt_deviations:
            all_labels.update(seq)

        # Convert to a sorted list for consistent ordering (optional, but good for reproducibility)
        labels = sorted(list(all_labels))

        num_instances = len(tgt_deviations)
        num_labels = len(labels)
        # Now create y_true as a binary matrix: rows = instances, columns = labels: 1 if the label (transition) appears at least once in the sequence, else 0
        y_true = np.zeros((num_instances, num_labels), dtype=int)

        for i, seq in enumerate(tgt_deviations):
            seq_set = set(seq)  # Use set for O(1) lookups and to ignore duplicates
            for j, label in enumerate(labels):
                if label in seq_set:
                    y_true[i, j] = 1
                    
        return labels, y_true
    
    def get_threshold_data(self):
        # Target data:
        labels, y_true = self.__get_y()
        
        # number of cases:
        n = len(self.risk_conformance_results["samples_conformance"])
        # Aggregate probabilistic deivations
        probs = []
        for i in range(n):            
            # Get all samples for this case (prefix)
            smpls = self.risk_conformance_results["samples_conformance"][i]
            
            # number of samples per case:
            total_samples = len(smpls)

            sample_devs = []   
            for smpl in smpls: 
                # Alignment 
                align = smpl["suffix_alignment"]
                cleaned_align = [a for a in align if a != ('>>', None) and a != (None, '>>')]
                    
                # Deviations
                devs = [(a, b) for (a, b) in cleaned_align if a != b]
                # List of deviations across all samples:
                for dev in devs:
                    sample_devs.append(dev)

            # Count frequencies
            counter_devs = Counter(sample_devs) 
            relative_probs = {transition: count / total_samples for transition, count in counter_devs.items()}            
            
            # built prob array based on lables and all devs
            prob_case = []
            relative_probs_labels = list(relative_probs.keys())
            for i, lbl in enumerate(labels):
                if lbl in relative_probs_labels:
                    prob_case.append(relative_probs[lbl])
                else:
                    prob_case.append(0)     
            probs.append(prob_case)
        # numpy array:      
        probs = np.array(probs)  
        return probs, (labels, y_true)
            
    def find_optimal_thresholds(self, beta=1.0, per_label=True):
        # probs = np.array([[1.0, 1.0, 0.55, ...]])# Shape (N: cases, M: number of labels)
        # y_true = np.array([[1, 0, 1, ...]])      # Ground truths
        probs, (labels, y_true) = self.get_threshold_data()
        
        thresholds = {}
        if per_label:
            for label in range(probs.shape[1]):
                prec, rec, thresh = precision_recall_curve(y_true[:, label], probs[:, label])
                fbeta = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec + 1e-10)  # Avoid div by zero
                best_idx = np.argmax(fbeta)
                thresholds[labels[label]] = thresh[best_idx]
        else:  # Global threshold via micro F-beta
            candidates = np.linspace(0.01, 0.99, 99)
            fbetas = []
            for t in candidates:
                preds = (probs >= t).astype(int)
                fbetas.append(fbeta_score(y_true.ravel(), preds.ravel(), beta=beta, average='micro'))
            thresholds['global'] = candidates[np.argmax(fbetas)]
        return thresholds
    
    # Save the trained logistic regression model
    def save(self, path: Union[str, Path], thresholds: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = [{"key": list(k), "value": float(v)} for k, v in thresholds.items()]
        with path.open("w") as f:
            json.dump(payload, f, indent=4)
        
        return str(path)
