import numpy as np
import math
from typing import Any, Dict

class ConformalAnalysisModel:
    def __init__(self, mondrian: bool, fitness_score_results: dict):
        """
        - mondrian : bool: True means grouping.
        - fitness_score_results : dict: Dict of fitness score results: 
            - Case_ID: list of (case_name, prefix_length),
            - target_fitness_score: list of fitness scores of target, per case,
            - most_likely_fitness_score: list of fitness score of most-likely, per case,
            - sampled_case_fitness_scores: list of fitness scores of samples (T=1000), per case
        """
        self.mondrian = mondrian
        self.fitness_score_results = fitness_score_results
        
    def _aggregate_samples_fitness(self, samples_fitness: np.ndarray, aggregation: str) -> float:
        if samples_fitness.size == 0:
            raise ValueError("samples_fitness must not be empty")
        
        if aggregation == 'mean':
            agg = float(np.mean(samples_fitness))
        elif aggregation == 'median':
            agg = float(np.median(samples_fitness))
        elif aggregation == 'min':
            agg = float(np.min(samples_fitness))
        elif aggregation == 'max':
            agg = float(np.max(samples_fitness))
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        
        # sample standard deviation (ddof=1 if >1 sample, else 0)
        std = float(np.std(samples_fitness, ddof=1) if samples_fitness.size > 1 else 0.0)

        return (agg, std)
        
    def _value_at_quantiles(self, values: list, q_risk: float, q_highrisk: float) -> Dict[str, Any]:
        """
        Given an unsorted list of floats, return the lower-tail empirical values at q_risk and q_highrisk.
        """
        sorted_vals = sorted(values)  # ascending, smallest is the worst fitness
        n = len(sorted_vals)

        if n <= 0:
            return -1
        
        k_risk = math.floor((n + 1) * q_risk)
        idx_risk = k_risk - 1
        idx_risk = min(max(idx_risk, 0), n - 1)
        
        k_highrisk = math.floor((n + 1) * q_highrisk)
        idx_highrisk = k_highrisk - 1
        idx_risk = min(max(idx_highrisk, 0), n - 1)
        
        val_risk = sorted_vals[idx_risk] if idx_risk != -1 else None
        val_highrisk = sorted_vals[idx_highrisk] if idx_highrisk != -1 else None

        return {'q_risk': val_risk, 'q_high_risk': val_highrisk}
    
    
    def __mondrian_pref_len_grouping(fitness_score_results: dict) -> dict:
        # Ensure that all lists have same size:
        grouped = {}
        for i, case_id in enumerate(fitness_score_results['Case_ID']):
            prefix_len = case_id[1]  # assume Case_ID is (case_name, prefix_len)
            if prefix_len not in grouped:
                grouped[prefix_len] = {
                    'target_fitness_score': [],
                    'most_likely_fitness_score': [],
                    'sampled_case_fitness_scores': []
                }

            grouped[prefix_len]['target_fitness_score'].append(fitness_score_results['target_fitness_score'][i])
            grouped[prefix_len]['most_likely_fitness_score'].append(fitness_score_results['most_likely_fitness_score'][i])
            grouped[prefix_len]['sampled_case_fitness_scores'].append(fitness_score_results['sampled_case_fitness_scores'][i])

        # return with keys sorted ascending
        return dict(sorted(grouped.items()))
    
    def empirical_quantile_thresholds(self, q_risk: float, q_high_risk: float, aggregation: str='mean') -> dict:
        """
        Compute one-sided lower-tail empirical thresholds q_risk and q_high_risk.
        """
        if not self.mondrian:
            # Target
            target_fitness_scores = self.fitness_score_results['target_fitness_score']
            # Get thresholds
            thresholds_target = self._value_at_quantiles(target_fitness_scores, q_risk, q_high_risk)
            
            # Most likely
            ml_fitness_scores = self.fitness_score_results['most_likely_fitness_score']
            # Get thresholds
            thresholds_ml = self._value_at_quantiles(ml_fitness_scores, q_risk, q_high_risk)
            
            # Samples
            sampled_fitness_scores = self.fitness_score_results['sampled_case_fitness_scores']
            # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
            aggragted_sampled_fitness_scores = [self._aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
            # Get thresholds
            thresholds_sampled = self._value_at_quantiles([agg_smp[0] for agg_smp in aggragted_sampled_fitness_scores], q_risk, q_high_risk)
            mean_std_sampled = np.nanmean([agg_smp[1] for agg_smp in aggragted_sampled_fitness_scores])
            thresholds_sampled['mean_std'] = mean_std_sampled
            
            return {'target': thresholds_target, 'most_likely': thresholds_ml, 'sampled': thresholds_sampled}
        
        # mondrian grouping (prefix len for simplicity)
        else:
            grouped_results = {}
            mondrian_grouped_fitness_scores = self.__mondrian_pref_len_grouping(fitness_score_results=self.fitness_score_results)
            # iterate through key: prefix_len, values: fitness_score_results dict
            for key, values in mondrian_grouped_fitness_scores.items():
                target_fitness_scores = values['target_fitness_score']
                # Get thresholds
                thresholds_target = self._value_at_quantiles(target_fitness_scores, q_risk, q_high_risk)
                
                # Most likely
                ml_fitness_scores = values['most_likely_fitness_score']
                # Get thresholds
                thresholds_ml = self._value_at_quantiles(ml_fitness_scores, q_risk, q_high_risk)
                
                # Samples
                sampled_fitness_scores = values['sampled_case_fitness_scores']
                # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
                aggragted_sampled_fitness_scores = [self._aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
                # Get thresholds
                thresholds_sampled = self._value_at_quantiles([agg_smp[0] for agg_smp in aggragted_sampled_fitness_scores], q_risk, q_high_risk)
                mean_std_sampled = np.nanmean([agg_smp[1] for agg_smp in aggragted_sampled_fitness_scores])
                thresholds_sampled['mean_std'] = mean_std_sampled
                
                results = {'target': thresholds_target, 'most_likely': thresholds_ml, 'sampled': thresholds_sampled}    

                grouped_results[key] = results
                
            return grouped_results
    
    def risk_controlled_threshold():
        pass
    
    
            
    def conformal_upper_bound():
        pass
        