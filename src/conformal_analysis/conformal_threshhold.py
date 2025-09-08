import numpy as np
import math
from typing import Any, Dict

class ConformalAnalysisThreshold:
    def __init__(self, mondrian: bool, fitness_score_results: dict):
        """
        - mondrian : bool: True means grouping.
        - fitness_score_results : dict: Dict of fitness score results: 
            - Case_ID: list of (case_name, prefix_length),
            - 'target_fitness_score': list of fitness scores of targets,
            - 'most_likely_fitness_score': list of fitness score of most-likelies,
            - 'sampled_case_fitness_scores': list of fitness scores of samples (T=1000)
        """
        self.mondrian = mondrian
        self.fitness_score_results = fitness_score_results
        
    def _aggregate_samples_fitness(self, samples_fitness: np.ndarray, aggregation: str) -> float:
        if aggregation == 'mean':
            return float(np.mean(samples_fitness))
        if aggregation == 'median':
            return float(np.median(samples_fitness))
        if aggregation == 'var':
            return float(np.var(samples_fitness, ddof=1) if samples_fitness.size > 1 else 0.0)
        if aggregation == 'min':
            return float(np.min(samples_fitness))
        if aggregation == 'max':
            return float(np.max(samples_fitness))
        raise ValueError(f"Unsupported aggregation: {aggregation}")
        
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

    def empirical_quantile_thresholds(self, q_risk: float, q_high_risk: float, aggregation: str='mean') -> dict:
        """
        Compute one-sided lower-tail empirical thresholds q_risk and q_high_risk.
        """
        if not self.mondrian:
            # single (global) dict of lists
            target = self.fitness_score_results['target_fitness_score']
            ml = self.fitness_score_results['most_likely_fitness_score']
            
            # Aggreagate the fitness samples (per case) to determine empirical thresholds:
            sampled = self.fitness_score_results['sampled_case_fitness_scores']
            aggragted_sampled = []
            for smp in sampled:
                aggragted_sampled.append(self._aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation))

            res_target = self._value_at_quantiles(target, q_risk, q_high_risk)
            res_ml = self._value_at_quantiles(ml, q_risk, q_high_risk)
            res_sampled = self._value_at_quantiles(aggragted_sampled, q_risk, q_high_risk)
            
            results = {'target': res_target, 'most_likely': res_ml, 'sampled': res_sampled}
            
        else:
            # First sort according to prefix length:
            pass

        return results
        