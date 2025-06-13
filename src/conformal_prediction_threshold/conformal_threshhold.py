import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import math
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

from conformance_checking.conformance import ConformanceChecking

class ConformalPredictionThreshold:
    def __init__(self, d_con_results: List[Dict[str, Any]], conformance_object:ConformanceChecking , log_name: Optional[str] = ""):
        """
        d_con_results: List of dicts with evaluation results form the probabilistic suffix prediction model on the conformal dataset (validation).
        conformance_object: A ConformanceChecking object -> Implements the chosen (alignment-based) conformance check algorithm.
        log_name: Optional log name for identification.
        """
        self.log_name = log_name
        self.d_con_results = d_con_results
        self.conformance_object = conformance_object
         
    def simple_threshold_q(self, alpha):
        """
        Original single-threaded implementation.
        """
        target_fitness_scores = []
        most_likely_fitness_scores = []
        sampled_fitness_scores = []

        # Step 1: Compute fitness scores
        for results in self.d_con_results:
            for values in results.values():
                target_conformance, most_likely_conformance, samples_conformances = self.conformance_object.conformance_of_sampled_suffixes(log_name=self.log_name, result_values=values)
                
                target_fitness_scores.append(target_conformance['fitness'])
                most_likely_fitness_scores.append(most_likely_conformance['fitness'])
                sampled_fitnessess = np.array([x['fitness'] for x in samples_conformances])
                
                sampled_fitness_scores.append((sampled_fitnessess.mean(), sampled_fitnessess.var(ddof=1)))
        
        # Step 2: Sort based on mean of sampled fitness scores
        target_fitness_scores_sorted = sorted(target_fitness_scores)
        most_likely_fitness_scores_sorted = sorted(most_likely_fitness_scores)
        sampled_fitness_scores_sorted = sorted(sampled_fitness_scores, key=lambda x: x[0])   

        print("Sorted target fitness scores:", target_fitness_scores_sorted)
        print("Sorted most likely fitness scores:", most_likely_fitness_scores_sorted)
        print("Sorted sampled fitness scores:", sampled_fitness_scores_sorted)

        # Step 3: Calculate quantile index (rounding up)
        n = len(target_fitness_scores_sorted)

        # For python indexing: Choose the vlaue (n + 1) * alpha) starting at index 1.
        q_index = math.ceil((n + 1) * alpha) - 1
        q_index = min(max(q_index, 0), n - 1)
        print("Q index: ", q_index)

        q_value_target = target_fitness_scores_sorted[q_index]
        q_value_most_likely = most_likely_fitness_scores_sorted[q_index]
        q_value_samples = sampled_fitness_scores_sorted[q_index][0]

        return q_value_target, q_value_most_likely, q_value_samples
        
    def threshold_q_per_prefix_length(self,alpha: float) -> List[Tuple[int, float, float, float]]:
        """
        For each prefix_length, compute the (target, most_likely, sampled_mean) Q-values.
        Returns a list of tuples: (prefix_length, q_target, q_most_likely, q_sampled_mean) sorted by prefix_length ascending.
        """
        # Group all fitness scores by prefix_length
        groups: Dict[int, Dict[str, List[Any]]] = defaultdict(lambda: {'target': [], 'most_likely': [], 'sampled_means': [], 'sampled_vars': []})
        
        # All pikckles that store results
        for result in self.d_con_results:
            # All cases in a pickle
            for (_, prefix_length), values in result.items():
                # Compute performance for case:
                t_con, m_con, sample_cons = self.conformance_object.conformance_of_sampled_suffixes(log_name=self.log_name, result_values=values)
                
                groups[prefix_length]['target'].append(t_con['fitness'])
                groups[prefix_length]['most_likely'].append(m_con['fitness'])
                
                fitness_samples = np.array([x['fitness'] for x in sample_cons])
                
                # Mean of all samples fitness for one case.
                mean_fitness_samples = fitness_samples.mean()
                # Variance of all samples fitness for one case: The average within batch variance.
                var_fitness_samples = fitness_samples.var(ddof=1)
                
                groups[prefix_length]['sampled_means'].append(mean_fitness_samples)
                groups[prefix_length]['sampled_vars'].append(var_fitness_samples)
                
        mean_tgts_per_prefix_length = []
        mean_ml_per_prefix_length = []
        mean_samples_per_prefix_length = []
        
        qs_per_prefix_length = []
        
        for prefix_length in sorted(groups):
            tgt_scores = sorted(groups[prefix_length]['target'])
            ml_scores  = sorted(groups[prefix_length]['most_likely'])
            sm_scores  = sorted(groups[prefix_length]['sampled_means'])
            
            n = len(sm_scores)
            # conformal quantile index (1-based → 0-based)
            index_f_scores  = math.ceil((n + 1) * alpha) - 1
            index_f_scores = min(max(index_f_scores, 0), n - 1)

            q_tgt  = tgt_scores[index_f_scores]
            q_ml   = ml_scores[index_f_scores]
            q_sm   = sm_scores[index_f_scores]
            # List of dicts: Keys: Prefix length, values: Tuple with target q, most likely q, samples q:
            qs_per_prefix_length.append({prefix_length: (q_tgt, q_ml, q_sm)})
            
            # Mean of all targets over same prefix length
            mean_tgt = np.mean(groups[prefix_length]['target'])
            mean_ml_per_prefix_length.append({prefix_length: mean_tgt})
            
            # Mean of all most-likelies over same prefix length
            mean_ml = np.mean(groups[prefix_length]['most_likely'])
            mean_tgts_per_prefix_length.append({prefix_length: mean_ml})
            
            # Mean of means for all cases with same prefix length.
            mean_means_sm = np.mean(groups[prefix_length]['sampled_means'])
            # Mean of vars for all cases with same prefix length.
            # Variance of batch means:
            var_means_sm = np.var(groups[prefix_length]['sampled_means'])
            # The Average within bacth variance:
            mean_vars_sm = np.mean(groups[prefix_length]['sampled_vars'])
            mean_samples_per_prefix_length.append({prefix_length: (mean_means_sm, (var_means_sm, mean_vars_sm))})

        return qs_per_prefix_length, mean_tgts_per_prefix_length, mean_ml_per_prefix_length, mean_samples_per_prefix_length 
        