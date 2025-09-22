import numpy as np
from typing import List, Dict, Optional, Any

from conformance_checking.conformance import ConformanceChecking

class ConformanceResults:
    def __init__(self, log_name: Optional[str] = "", data: List[Dict[str, Any]] = None, conformance_object: ConformanceChecking = None):
        """
        - log_name: Optional log name for identification.
        - d_con_results: List of dicts with evaluation results form the probabilistic suffix prediction model on the conformal dataset (validation).
        - conformance_object: A ConformanceChecking object -> Implements the chosen (alignment-based) conformance check algorithm.
        """
        self.log_name = log_name
        self.data = data
        self.conformance_object = conformance_object
         
    def fitness_scores_calibration(self):
        """
        Original single-threaded implementation.
        """
        
        # Parallelize!

        fitness_score_results = {'Case_ID': [],
                                 'target_fitness_score': [],
                                 'most_likely_fitness_score': [],
                                 'sampled_case_fitness_scores': []}
        
        # Step Compute fitness scores
        for results in self.data:
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
    
    def conformance_predictions(self):   
        """
        Compute the conformance for all cases out of the probabilistic suffi
        """
        
        # Parallelize!
        
        # All cases:
        conformance_results = {'Case_ID': [],
                               'target_conformance': [],
                               'most_likely_conformance': [],
                               'samples_conformance': []}

        # Iterate through all processed result pickles
        for result in self.d_inference_results:
            
            # All cases stored in a pickle:
            for (case_name, prefix_length), values in result.items():
            
                # All cases:
                conformance_results['Case_ID'].append((case_name, prefix_length))
                
                # Conformance analysis:
                t_con, ml_con, samples_cons = self.conformance_object.conformance_of_sampled_suffixes(log_name=self.log_name, result_values=values)
                
                conformance_results['target_conformance'].append(t_con)
                
                conformance_results['most_likely_conformance'].append(ml_con)
                
                conformance_results['samples_conformance'].append(samples_cons)
            
        return conformance_results
    
    
