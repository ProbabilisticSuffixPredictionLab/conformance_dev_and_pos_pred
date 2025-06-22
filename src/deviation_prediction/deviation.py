import numpy as np
import pm4py

class Deviation:
    def __init__(self, cases_with_alignments):
        self.cases_with_alignments = cases_with_alignments
    
    def individual_deviations(self, samples_conformance_risk):
        # Define no‑move symbol (null move)
        NULL = '>>'  # same as ≫ in typical notation

        model_moves_dict = {}
        log_moves_dict = {}

        for key, risk_samples in samples_conformance_risk.items():
            # N violating MC sample buckets where mean fitness is not in range of threshold and 1
            for samples in risk_samples:
                # 1000 MC samples
                for sample in samples:
                    alignment = sample['alignment']
                    # Extract model moves: first ≠ NULL, second = NULL
                    model_moves = [ (a, b) for (a, b) in alignment if a != NULL and b == NULL ]
                    # Extract log moves: first = NULL, second ≠ NULL
                    log_moves   = [ (a, b) for (a, b) in alignment if a == NULL and b != NULL ]
    
    def deviation_patterns(self):
        pass