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
        
    def full_set_prediction(self):
        """
        Partition cases into 'risk' and 'high risk' based on conformal fitness thresholds.

        Returns:
            (risk_group, high_risk_group)
        """
        group = defaultdict(lambda: {'test_case_id': [],
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
                
                # Append values
                group[prefix_length]['test_case_id'].append((case_name, prefix_length))
                group[prefix_length]['target_conformance'].append(t_con)
                group[prefix_length]['most_likely_conformance'].append(ml_con)
                group[prefix_length]['samples_conformance'].append(samples_cons)
                group[prefix_length]['mean_samples_fitness'].append(mean_fitness)
                group[prefix_length]['sd_samples_fitness'].append(sd_fitness)
                     
        # Sort groups by prefix length ascending
        group = OrderedDict(sorted(group.items(), key=lambda item: item[0]))

        return group
        
    
    def risk_set_prediction(self, mean_conformal_fitness_prefix_lengths: Dict[int, float], r_conformal_fitness_prefix_lengths: Dict[int, float]):   
        """
        Partition cases into 'risk' and 'high risk' based on conformal fitness thresholds.

        Returns:
            (risk_group, high_risk_group)
        """
        risk_group = defaultdict(lambda: {'test_case_id': [],
                                          'target_conformance': [],
                                          'most_likely_conformance': [],
                                          'samples_conformance': [],
                                          'mean_samples_fitness': [],
                                          'sd_samples_fitness': []
                                        })
        
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

                # Risk Threshold: Risk-flagged suffix samples
                if mean_fitness < mean_cf:
                     # Append values
                    risk_group[prefix_length]['test_case_id'].append((case_name, prefix_length))
                    risk_group[prefix_length]['target_conformance'].append(t_con)
                    risk_group[prefix_length]['most_likely_conformance'].append(ml_con)
                    risk_group[prefix_length]['samples_conformance'].append(samples_cons)
                    risk_group[prefix_length]['mean_samples_fitness'].append(mean_fitness)
                    risk_group[prefix_length]['sd_samples_fitness'].append(sd_fitness)
                
                # High-Risk Threshold
                if mean_fitness < r_cf:
                     # Append values
                    high_risk_group[prefix_length]['test_case_id'].append(values)
                    high_risk_group[prefix_length]['target_conformance'].append(t_con)
                    high_risk_group[prefix_length]['most_likely_conformance'].append(ml_con)
                    high_risk_group[prefix_length]['samples_conformance'].append(samples_cons)
                    high_risk_group[prefix_length]['mean_samples_fitness'].append(mean_fitness)
                    high_risk_group[prefix_length]['sd_samples_fitness'].append(sd_fitness)
                
        # Sort groups by prefix length ascending
        sorted_risk_group = OrderedDict(sorted(risk_group.items(), key=lambda item: item[0]))
        sorted_high_risk_group = OrderedDict(sorted(high_risk_group.items(), key=lambda item: item[0]))

        return sorted_risk_group, sorted_high_risk_group
               
    def in_conformal_risk_set(self, r_conformal_fitness_prefix_lengths: Dict[int, float]):
        """
        Partition cases into 'risk' and 'high risk' based on conformal fitness thresholds.

        Returns:
            (risk_group, high_risk_group)
        """
        conformal_group = defaultdict(lambda: {'test_case_id': [],
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

                # Select conformal thresholds for this prefix length
                if prefix_length in r_conformal_fitness_prefix_lengths.keys():
                    r_cf = r_conformal_fitness_prefix_lengths[prefix_length]
                else:
                    # Use last available thresholds if prefix_length not in dict
                    last_r = list(r_conformal_fitness_prefix_lengths.values())[-1]
                    r_cf = last_r

                # Compute sample fitness statistics
                fitness_samples = np.array([x['fitness'] for x in samples_cons])
                mean_fitness = fitness_samples.mean()
                sd_fitness = fitness_samples.std()

                # Conformal Threshold
                if mean_fitness >= r_cf:
                     # Append values
                    conformal_group[prefix_length]['test_case_id'].append(values)
                    conformal_group[prefix_length]['target_conformance'].append(t_con)
                    conformal_group[prefix_length]['most_likely_conformance'].append(ml_con)
                    conformal_group[prefix_length]['samples_conformance'].append(samples_cons)
                    conformal_group[prefix_length]['mean_samples_fitness'].append(mean_fitness)
                    conformal_group[prefix_length]['sd_samples_fitness'].append(sd_fitness)
                
        # Sort groups by prefix length ascending
        sorted_conformal_group = OrderedDict(sorted(conformal_group.items(), key=lambda item: item[0]))

        return sorted_conformal_group