import numpy as np
from collections import defaultdict, OrderedDict
from typing import List, Dict, Optional, Any

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

            
    def conformal_set_predictions(self, mean_conformal_fitness_prefix_lengths: Dict[int, float], r_conformal_fitness_prefix_lengths: Dict[int, float]):   
        """
        Partition cases into 'risk' and 'high risk' based on conformal fitness thresholds.

        Returns:
            all_gropu: 
            risk_group, high_risk_group: 
            discarded_risk_group: 
        """
        
        # All cases:
        all_group = defaultdict(lambda: {'test_case_id': [],
                                     'target_conformance': [],
                                     'most_likely_conformance': [],
                                     'samples_conformance': [],
                                     'mean_samples_fitness': [],
                                     'sd_samples_fitness': []
                                     })
        # Risk-flagged cases: 
        risk_group = defaultdict(lambda: {'test_case_id': [],
                                          'target_conformance': [],
                                          'most_likely_conformance': [],
                                          'samples_conformance': [],
                                          'mean_samples_fitness': [],
                                          'sd_samples_fitness': []
                                        })
        # Discarded cases:
        discarded_risk_group = defaultdict(lambda: {'test_case_id': [],
                                                    'target_conformance': [],
                                                    'most_likely_conformance': [],
                                                    'samples_conformance': [],
                                                    'mean_samples_fitness': [],
                                                    'sd_samples_fitness': []
                                                    })
        # High-Risk flagged cases:
        high_risk_group = defaultdict(lambda: {'test_case_id': [],
                                               'target_conformance': [],
                                               'most_likely_conformance': [],
                                               'samples_conformance': [],
                                               'mean_samples_fitness': [],
                                               'sd_samples_fitness': []
                                            })
        
        # Iterate through all processed result pickles
        for result in self.d_inference_results:
            # All cases stored in a pickle:
            for (case_name, prefix_length), values in result.items():
                # Conformance analysis:
                t_con, ml_con, samples_cons = self.conformance_object.conformance_of_sampled_suffixes(log_name=self.log_name, result_values=values)

                # Compute sample fitness statistics
                fitness_samples = np.array([x['fitness'] for x in samples_cons])
                mean_fitness = fitness_samples.mean()
                sd_fitness = fitness_samples.std()
                
                # All cases:
                all_group[prefix_length]['test_case_id'].append((case_name, prefix_length))
                all_group[prefix_length]['target_conformance'].append(t_con)
                all_group[prefix_length]['most_likely_conformance'].append(ml_con)
                all_group[prefix_length]['samples_conformance'].append(samples_cons)
                all_group[prefix_length]['mean_samples_fitness'].append(mean_fitness)
                all_group[prefix_length]['sd_samples_fitness'].append(sd_fitness)
                
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
                
                # Risk Threshold: Risk-flagged cases:
                if mean_fitness < mean_cf:
                     # Append values
                    risk_group[prefix_length]['test_case_id'].append((case_name, prefix_length))
                    risk_group[prefix_length]['target_conformance'].append(t_con)
                    risk_group[prefix_length]['most_likely_conformance'].append(ml_con)
                    risk_group[prefix_length]['samples_conformance'].append(samples_cons)
                    risk_group[prefix_length]['mean_samples_fitness'].append(mean_fitness)
                    risk_group[prefix_length]['sd_samples_fitness'].append(sd_fitness)
                    
                    # High-Risk Threshold: High-Risk-flagged cases:
                    if mean_fitness < r_cf:
                        # Append values
                        high_risk_group[prefix_length]['test_case_id'].append(values)
                        high_risk_group[prefix_length]['target_conformance'].append(t_con)
                        high_risk_group[prefix_length]['most_likely_conformance'].append(ml_con)
                        high_risk_group[prefix_length]['samples_conformance'].append(samples_cons)
                        high_risk_group[prefix_length]['mean_samples_fitness'].append(mean_fitness)
                        high_risk_group[prefix_length]['sd_samples_fitness'].append(sd_fitness)
                
                else:
                    # Discarded (not flagged as risk)
                    discarded_risk_group[prefix_length]['test_case_id'].append((case_name, prefix_length))
                    discarded_risk_group[prefix_length]['target_conformance'].append(t_con)
                    discarded_risk_group[prefix_length]['most_likely_conformance'].append(ml_con)
                    discarded_risk_group[prefix_length]['samples_conformance'].append(samples_cons)
                    discarded_risk_group[prefix_length]['mean_samples_fitness'].append(mean_fitness)
                    discarded_risk_group[prefix_length]['sd_samples_fitness'].append(sd_fitness)
                
        # Sort groups by prefix length ascending
        sorted_all_group = risk_group = OrderedDict(sorted(all_group.items(), key=lambda item: item[0]))
        sorted_risk_group = OrderedDict(sorted(risk_group.items(), key=lambda item: item[0]))
        sorted_high_risk_group = OrderedDict(sorted(high_risk_group.items(), key=lambda item: item[0]))
        sorted_discarded_risk_group = OrderedDict(sorted(discarded_risk_group.items(), key=lambda item: item[0]))

        return sorted_all_group, sorted_risk_group, sorted_high_risk_group, sorted_discarded_risk_group