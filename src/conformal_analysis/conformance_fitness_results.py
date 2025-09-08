import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Any

from conformance_checking.conformance import ConformanceChecking

class ConformanceFitnessResults:
    def __init__(self, log_name: Optional[str] = "", d_con_results: List[Dict[str, Any]] = None, conformance_object: ConformanceChecking = None):
        """
        Constructor:
        - log_name: Optional log name for identification.
        - d_con_results: List of dicts with evaluation results form the probabilistic suffix prediction model on the conformal dataset (validation).
        - conformance_object: A ConformanceChecking object -> Implements the chosen (alignment-based) conformance check algorithm.
        """
        self.log_name = log_name
        self.d_con_results = d_con_results
        self.conformance_object = conformance_object
         
    def fitness_scores(self):
        """
        Original single-threaded implementation.
        """
        
        fitness_score_results = {'Case_ID': [],
                                 'target_fitness_score': [],
                                 'most_likely_fitness_score': [],
                                 'sampled_case_fitness_scores': []
                                }
        
        # Step Compute fitness scores
        for results in self.d_con_results:
            for (case_name, prefix_length), values in results.items():
                
                fitness_score_results['Case_ID'].append((case_name, prefix_length))

                # Get the conformance for all cases, prefix + (targets, most likely, and samples (T=1000))
                target_con, ml_con, smpls_con = self.conformance_object.conformance_of_sampled_suffixes(log_name=self.log_name, result_values=values)
                
                # Target fitness
                fitness_score_results['target_fitness_score'].append(target_con['fitness'])
                
                # Most-likely fitness
                fitness_score_results['most_likely_fitness_score'].append(ml_con['fitness'])
                
                # Sampled fitness
                sampled_fitnesses = np.array([x['fitness'] for x in smpls_con])
                fitness_score_results['sampled_case_fitness_scores'].append(sampled_fitnesses)

        return fitness_score_results
    
    
