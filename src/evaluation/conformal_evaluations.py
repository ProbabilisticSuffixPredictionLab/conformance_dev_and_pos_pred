from typing import Dict
import numpy as np
import math

class ConformalEvaluation:
    def __init__(self):
        pass
        
    def coverage(self, targets_conformance: Dict, threshold_values: Dict):
        """
        Check if target falls within the miscoverage set
        """
        
        # In case that the threshold values and the all target cases' conformance contain more or less values for prefix lengths:
        # Get sorted list of all unique keys
        all_keys = sorted(set(threshold_values) | set(targets_conformance))
        
        # Value of last prefix length captured in conformal analysis
        last_thresh_value = list(threshold_values.values())[-1]
        
        # Extend threshold_values: Target conformance contain more keys than the threshold dict
        for key in all_keys:
            if key not in threshold_values:
                threshold_values[key] = last_thresh_value

        # Extend targets_conformance: Threshold contain more keys than the target conformance dict
        for key in all_keys:
            if key not in targets_conformance:
                targets_conformance[key] = []
                
        # Total miscovergae value
        total_miscoverage = 0
        
        # Miscoverage and Coverge per prefix length
        miscoverage_pref_len = {}
        coverage_pref_len = {}
        
        coverage_perfect_fitness = {}
        
        # Miscoverage per pref len:
        for pref_len, thresh in threshold_values.items():
            # Miscoverage per pref. len
            m_cov = len([fit for fit in targets_conformance[pref_len] if fit < thresh])
            miscoverage_pref_len[pref_len] = m_cov / len(targets_conformance[pref_len])
            # Add to total miscoverage value:
            total_miscoverage += m_cov
            # Coverage per pref. len
            coverage_pref_len[pref_len] = 1 - miscoverage_pref_len[pref_len]
            
            cov_per_fit = len([fit for fit in targets_conformance[pref_len] if math.isclose(fit, 1.0, rel_tol=1e-9)])
            coverage_perfect_fitness[pref_len] = cov_per_fit / len(targets_conformance[pref_len])
            
        # Total Miscoverage
        total_miscoverage = total_miscoverage / sum([len(value) for value in targets_conformance.values()])
        # Total Coverage
        total_coverage = 1 - total_miscoverage
            
        return  (total_miscoverage,
                 total_coverage,
                 miscoverage_pref_len,
                 coverage_pref_len,
                 coverage_perfect_fitness)
        
    def size(self, cov_set: Dict, miscov_set: Dict):
        avg_size_cov_set = np.mean([len(cons) for _, cons in cov_set.items()])
        avg_size_miscov_set = np.mean([len(cons) for _, cons in miscov_set.items()])
        
        return avg_size_cov_set, avg_size_miscov_set
        