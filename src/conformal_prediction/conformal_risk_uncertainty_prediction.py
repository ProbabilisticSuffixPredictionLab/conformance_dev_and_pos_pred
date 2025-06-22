import numpy as np
import math
from collections import defaultdict, OrderedDict
from typing import List, Dict, Optional, Any, Tuple

from conformance_checking.conformance import ConformanceChecking

class ConformalRiskUncertaintyPrediction:
    def __init__(self, d_inference_results: List[Dict[str, Any]], conformance_object: ConformanceChecking , log_name: Optional[str] = ""):
        """
        Class responsible for methods that sort and categorize test/ inference cases into risk sets for individual deviation, deviation pattern prediction.
        
        d_inference_results: List of dicts with evaluation results form the probabilistic suffix prediction model on the test/ inference dataset.
        conformance_object: A ConformanceChecking object -> Implements the chosen (alignment-based) conformance check algorithm.
        log_name: Optional log name for identification.
        
        """
        self.d_inference_results = d_inference_results
        self.conformance_object = conformance_object
        self.log_name = log_name
        
    def risk_set_prediction(self, mean_conformal_fitness_prefix_lengths: Dict[int, float], r_conformal_fitness_prefix_lengths: Dict[int, float] ) -> Tuple[Dict[int, dict], Dict[int, dict]]:   
        """
        Partition cases into 'risk' and 'high risk' based on conformal fitness thresholds.

        Returns:
            (risk_group, high_risk_group)
        """
        risk_group = defaultdict(lambda: {'test_case_value': [],
                                          'target_conformance': [],
                                          'most_likely_conformance': [],
                                          'samples_conformance': [],
                                          'mean_samples_fitness': [],
                                          'sd_samples_fitness': [],
                                          'se_samples_fitness': [],
                                          'uncertainty_se_fitness': []
                                          })
        high_risk_group = defaultdict(lambda: {'test_case_value': [],
                                               'target_conformance': [],
                                               'most_likely_conformance': [],
                                               'samples_conformance': [],
                                               'mean_samples_fitness': [],
                                               'sd_samples_fitness': [],
                                               'se_samples_fitness': [],
                                               'uncertainty_se_fitness': []
                                               })

        # Iterate through all processed result pickles
        for result in self.d_inference_results:
            for (case_name, prefix_length), values in result.items():
                # Conformance analysis on sampled suffixes
                t_con, ml_con, samples_cons = self.conformance_object.conformance_of_sampled_suffixes(log_name=self.log_name, result_values=values)

                # Select conformal thresholds for this prefix length
                if prefix_length in mean_conformal_fitness_prefix_lengths.keys():
                    mean_cf = mean_conformal_fitness_prefix_lengths[prefix_length]
                    r_cf = r_conformal_fitness_prefix_lengths[prefix_length]
                else:
                    # Use last available thresholds if prefix_length not in dict
                    last_mean = list(mean_conformal_fitness_prefix_lengths.values())[-1]
                    last_r = list(r_conformal_fitness_prefix_lengths.values())[-1]
                    mean_cf = last_mean
                    r_cf = last_r

                # Compute sample fitness statistics
                fitness_samples = np.array([x['fitness'] for x in samples_cons])
                mean_fitness = fitness_samples.mean()
                sd_fitness = fitness_samples.std()
                se_fitness = sd_fitness / np.sqrt(len(fitness_samples))
                uncertainty_se = se_fitness / mean_fitness if mean_fitness != 0 else np.nan

                # Assign to high risk if below tighter threshold, else risk if below looser
                if mean_fitness < r_cf:
                    group = high_risk_group[prefix_length]
                
                elif mean_fitness < mean_cf:
                    group = risk_group[prefix_length]
                
                else:
                    # not in any risk set
                    continue  

                # Append values
                group['test_case_value'].append(values)
                group['target_conformance'].append(t_con)
                group['most_likely_conformance'].append(ml_con)
                group['samples_conformance'].append(samples_cons)
                group['mean_samples_fitness'].append(mean_fitness)
                group['sd_samples_fitness'].append(sd_fitness)
                group['se_samples_fitness'].append(se_fitness)
                group['uncertainty_se_fitness'].append(uncertainty_se)

        # Sort groups by prefix length ascending
        sorted_risk_group = OrderedDict(sorted(risk_group.items(), key=lambda item: item[0]))
        sorted_high_risk_group = OrderedDict(sorted(high_risk_group.items(), key=lambda item: item[0]))

        return sorted_risk_group, sorted_high_risk_group
            
            
            

    def out_of_sd_distribution(self):
        pass