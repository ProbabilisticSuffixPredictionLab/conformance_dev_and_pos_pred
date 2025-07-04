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
        
    
    @staticmethod
    def __build_safe_set_risk(results_all, results_risk) -> Dict:
        """
        Return a dict mapping prefix_len → dict-of-lists,
        containing only those entries in results_all whose test_case_id
        is NOT in the corresponding results_low_risk[test_case_id].
        """
        # Grab all the metric‐names (fields) once
        fields = list(results_all[next(iter(results_all))].keys())
        
        # 2) precompute for each prefix the set of low‐risk IDs
        risk_id_sets = {
            pref: set(v.get('test_case_id', []))
            for pref, v in results_risk.items()
        }
        
        safe_set_risk = {}
        
        # 3) for each prefix in ALL, build your filtered lists
        for pref, vals in results_all.items():
            risk_ids = risk_id_sets.get(pref, set())
            ids = vals['test_case_id']
            
            # build a boolean mask: True if that index is "safe"
            keep_mask = [tc_id not in risk_ids for tc_id in ids]
            if not any(keep_mask):
                continue
            
            # allocate a new dict of empty lists
            filtered = {field: [] for field in fields}
            
            # one pass: for each index i, if keep_mask[i] is True, append vals[field][i]
            for i, keep in enumerate(keep_mask):
                if not keep:
                    continue
                for field in fields:
                    filtered[field].append(vals[field][i])
            
            safe_set_risk[pref] = filtered
        
        return safe_set_risk
    
    def size(self, all_set: Dict, miscov_set: Dict):
        
        cov_set = self.__build_safe_set_risk(results_all=all_set, results_risk=miscov_set)
        
        sizes_cov_sets = {}
        sizes_miscov_set = {}
         
        for pref_len in cov_set.keys():
            sizes_cov_sets[pref_len] = len_cons = len(cov_set[pref_len]['test_case_id'])
            
            if pref_len in miscov_set.keys():
                sizes_miscov_set[pref_len] = len(miscov_set[pref_len]['test_case_id'])
            else:
                sizes_miscov_set[pref_len] = 0 
        
        avg_size_cov_set = np.mean([cons for _, cons in sizes_cov_sets.items()])        
        avg_size_miscov_set = np.mean([cons for _, cons in sizes_miscov_set.items()])
        
        return sizes_cov_sets, avg_size_cov_set, sizes_miscov_set, avg_size_miscov_set
        