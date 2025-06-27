import numpy as np
import math
from collections import defaultdict
from typing import List, Dict, Optional, Any

from conformance_checking.conformance import ConformanceChecking

class ConformalAnalysisThreshold:
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

        # For python indexing: Choose the vlaue (n + 1) * α) starting at index 1.
        q_index = math.floor((n + 1) * alpha) - 1
        q_index = min(max(q_index, 0), n - 1)
        print("Q index: ", q_index)

        q_value_target = target_fitness_scores_sorted[q_index]
        q_value_most_likely = most_likely_fitness_scores_sorted[q_index]
        q_value_samples = sampled_fitness_scores_sorted[q_index][0]

        return q_value_target, q_value_most_likely, q_value_samples
        
    def threshold_q_per_prefix_length(self, alpha_risk, alpha_high_risk: float):
        
        fitness_score_result_groups = defaultdict(lambda: {'target_fitness_score': [],
                                                           'most_likely_fitness_score': [],
                                                           'sampled_case_fitness_scores': [],
                                                           'sampled_case_mean_std_fitness': []
                                                           })
        
        # Iterate thorugh all processed result pickles
        for result in self.d_con_results:
            # Results stored in one pickle
            for (case_name, prefix_length), values in result.items():
                # Conformance Analysis results: case, alignment, cost, fitness:
                t_con, m_con, samples_cons = self.conformance_object.conformance_of_sampled_suffixes(log_name=self.log_name, result_values=values)

                # Fitness score target
                fitness_score_result_groups[prefix_length]['target_fitness_score'].append(t_con['fitness'])
                # Fitness score most likely
                fitness_score_result_groups[prefix_length]['most_likely_fitness_score'].append(m_con['fitness'])

                # All fitness scores for all T samples:
                fitness_samples = np.array([x['fitness'] for x in samples_cons])
                fitness_score_result_groups[prefix_length]['sampled_case_fitness_scores'].append({case_name: fitness_samples})

                # Mean of all fitness scores
                mean_fitness_samples = fitness_samples.mean()
                # Standard deviation of all fitness scores (to mean)
                # All MC samples are the entire population so to compute std use 1/T:
                std_fitness_samples = fitness_samples.std()
                fitness_score_result_groups[prefix_length]['sampled_case_mean_std_fitness'].append({case_name: (mean_fitness_samples, std_fitness_samples)})

        case_name_fitness_scores_per_prefix_length = {}
        mean_tgts_ml_samples_per_prefix_length = {}
        
        # Standard Deviations: 
        # (1): Mean of std within the sample fitness scores (Predictive Uncertainty), 
        # (2): Standard Deviation of the mean fitness scores of the samples (Global Conformal Uncertainty)
        std_samples_per_prefix_length = {}
        
        # Risk Threshold values:
        q_samples_per_prefix_length = {}
        
        # High-Risk Threshold value:
        r_samples_per_prefix_length = {}

        # Iterate through the conformance results by ascending prefix lengths:
        for prefix_length in sorted(fitness_score_result_groups):
            
            # Case: fitness scores, mean, std:
            case_results = {}
            # Iterate through case names and fitness scores
            for d in fitness_score_result_groups[prefix_length]['sampled_case_fitness_scores']:
                for case_name, fitness_scores in d.items():
                    # Find the corresponding mean and std entry
                    mean_std_tuple = next((m_std[case_name] for m_std in fitness_score_result_groups[prefix_length]['sampled_case_mean_std_fitness'] if case_name in m_std), (0, 0))
                    case_results[case_name] = (fitness_scores, mean_std_tuple[0], mean_std_tuple[1])
            # Add dict entry with prefix length: {case name: fitness_scores, mean, std} 
            case_name_fitness_scores_per_prefix_length[prefix_length] = case_results

            # Means:            
            # Mean of all target fitness scores with same prefix length:
            mean_tgt = np.mean(fitness_score_result_groups[prefix_length]['target_fitness_score'])
            # Mean of all most likely fitness scores with same prefix length:
            mean_ml = np.mean(fitness_score_result_groups[prefix_length]['most_likely_fitness_score'])
            
            # Mean of all mean samples fitness scores
            all_mean_stds = [v for d in fitness_score_result_groups[prefix_length]['sampled_case_mean_std_fitness'] for v in d.values()]
            mean_samples = [m for m, _ in all_mean_stds]
            mean_means_sm = np.mean(mean_samples)
            mean_tgts_ml_samples_per_prefix_length[prefix_length] = (mean_tgt, mean_ml, mean_means_sm) 
            
            # Standard Deviations:
            stds = [s for _, s in all_mean_stds]
            # Mean of standard deviations within all samples fitness scores
            within_mean_stds_sm = np.mean(stds)
            # Standard deviation between all mean samlpled fitness scores
            between_std_means_sm = np.std(mean_samples, ddof=1)
            std_samples_per_prefix_length[prefix_length] = (within_mean_stds_sm, between_std_means_sm)

            # Get α (miscoverage: risk and high risk set) thresholds:
            sorted_means = sorted(mean_samples)
            n = len(sorted_means)
            
            # Risk:
            index_f_scores = math.floor((n + 1) * alpha_risk) - 1
            index_f_scores = min(max(index_f_scores, 0), n - 1)
            q_sm = sorted_means[index_f_scores]   
            q_samples_per_prefix_length[prefix_length] = q_sm

            # High Risk:
            index_f_scores = math.floor((n + 1) * alpha_high_risk) - 1
            index_f_scores = min(max(index_f_scores, 0), n - 1)
            r_sm = sorted_means[index_f_scores]   
            r_samples_per_prefix_length[prefix_length] = r_sm

        return (case_name_fitness_scores_per_prefix_length,
                mean_tgts_ml_samples_per_prefix_length,
                std_samples_per_prefix_length,
                q_samples_per_prefix_length,
                r_samples_per_prefix_length)
        