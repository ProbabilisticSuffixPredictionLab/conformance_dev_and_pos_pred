import numpy as np
from collections import Counter
from sklearn.metrics import fbeta_score, precision_recall_curve

class DeviationPredictionCalibration:
    def __init__(self, risk_conformance_results):
        self.risk_conformance_results = risk_conformance_results
    
    # Example data (expand to your full validation set)
    # probs = np.array([[1.0, 1.0, 0.55, ...]])  # Shape (N, M)
    # y_true = np.array([[1, 0, 1, ...]])      # Ground truths

    def __get_probs_y(self):
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

        # Now create y_true as a binary matrix: rows = instances, columns = labels
        # 1 if the label (transition) appears at least once in the sequence, else 0
        num_instances = len(tgt_deviations)
        num_labels = len(labels)
        y_true = np.zeros((num_instances, num_labels), dtype=int)

        for i, seq in enumerate(tgt_deviations):
            seq_set = set(seq)  # Use set for O(1) lookups and to ignore duplicates
            for j, label in enumerate(labels):
                if label in seq_set:
                    y_true[i, j] = 1

        # Print the labels and y_true for verification
        print("Labels:")
        for idx, label in enumerate(labels):
            print(f"{idx}: {label}")

        print("\ny_true:")
        print(y_true)

        # Aggregate probabilistic deivations
        results = []
        for i in range(len(tgt_aligns)):            
            # Get all samples for this case (prefix)
            smpls = self.risk_conformance_results["samples_conformance"][i]
            
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
            all_devs_with_prob = [(k, v / total_samples) for k, v in counter_devs.items()]
            # pred_deviations = [k for (k,_) in all_devs_with_prob]
              
        return pred_deviations, tgt_deviations
            
    def find_optimal_thresholds(self, beta=1.0, per_label=True):
        probs, y_true = self.__get_probs_y()
        
        thresholds = {}
        if per_label:
            for label in range(probs.shape[1]):
                prec, rec, thresh = precision_recall_curve(y_true[:, label], probs[:, label])
                fbeta = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec + 1e-10)  # Avoid div by zero
                best_idx = np.argmax(fbeta)
                thresholds[label] = thresh[best_idx]
        else:  # Global threshold via micro F-beta
            candidates = np.linspace(0.01, 0.99, 99)
            fbetas = []
            for t in candidates:
                preds = (probs >= t).astype(int)
                fbetas.append(fbeta_score(y_true.ravel(), preds.ravel(), beta=beta, average='micro'))
            thresholds['global'] = candidates[np.argmax(fbetas)]
        return thresholds