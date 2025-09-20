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
        # pkl stores automatically in a list
        self.fitness_score_results = fitness_score_results[0]
        
    def __aggregate_samples_fitness(self, samples_fitness: np.ndarray, aggregation: str) -> float:
        """
        Helper method to aggregate the samples.
        """
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
        
    def __value_at_quantiles(self, values: list, q_risk: float, q_highrisk: float) -> Dict[str, Any]:
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
        val_risk = sorted_vals[idx_risk] if idx_risk != -1 else None
        
        k_highrisk = math.floor((n + 1) * q_highrisk)
        idx_highrisk = k_highrisk - 1
        idx_highrisk = min(max(idx_highrisk, 0), n - 1)
        val_highrisk = sorted_vals[idx_highrisk] if idx_highrisk != -1 else None
        
        return {'q_risk': val_risk, 'q_high_risk': val_highrisk}
    
    def __mondrian_pref_len_grouping(self, fitness_score_results: dict) -> dict:
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
        Compute one-sided lower-tail empirical thresholds for q_risk and q_high_risk.
        """
        if not self.mondrian:
            # Target
            target_fitness_scores = self.fitness_score_results['target_fitness_score']
            # Get thresholds
            thresholds_target = self.__value_at_quantiles(target_fitness_scores, q_risk, q_high_risk)
            
            # Most likely
            ml_fitness_scores = self.fitness_score_results['most_likely_fitness_score']
            # Get thresholds
            thresholds_ml = self.__value_at_quantiles(ml_fitness_scores, q_risk, q_high_risk)
            
            # Samples
            sampled_fitness_scores = self.fitness_score_results['sampled_case_fitness_scores']
            # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
            aggragted_sampled_fitness_scores = [self.__aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
            # Get thresholds
            thresholds_sampled = self.__value_at_quantiles([agg_smp[0] for agg_smp in aggragted_sampled_fitness_scores], q_risk, q_high_risk)
            mean_std_sampled = np.nanmean([agg_smp[1] for agg_smp in aggragted_sampled_fitness_scores])
            thresholds_sampled['mean_std'] = mean_std_sampled
            
            return {'target': thresholds_target,
                    'most_likely': thresholds_ml,
                    'sampled': thresholds_sampled}
        
        # mondrian grouping (prefix len for simplicity)
        else:
            grouped_results = {}
            print(type(self.fitness_score_results))
            mondrian_grouped_fitness_scores = self.__mondrian_pref_len_grouping(fitness_score_results=self.fitness_score_results)
            # iterate through key: prefix_len, values: fitness_score_results dict
            for key, values in mondrian_grouped_fitness_scores.items():
                target_fitness_scores = values['target_fitness_score']
                # Get thresholds
                thresholds_target = self.__value_at_quantiles(target_fitness_scores, q_risk, q_high_risk)
                
                # Most likely
                ml_fitness_scores = values['most_likely_fitness_score']
                # Get thresholds
                thresholds_ml = self.__value_at_quantiles(ml_fitness_scores, q_risk, q_high_risk)
                
                # Samples
                sampled_fitness_scores = values['sampled_case_fitness_scores']
                # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
                aggragted_sampled_fitness_scores_std = [self.__aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
                # Get thresholds
                thresholds_sampled = self.__value_at_quantiles([agg_smp[0] for agg_smp in aggragted_sampled_fitness_scores_std], q_risk, q_high_risk)
                mean_std_sampled = np.nanmean([agg_smp[1] for agg_smp in aggragted_sampled_fitness_scores_std])
                thresholds_sampled['mean_std'] = mean_std_sampled
                
                results = {'target': thresholds_target,
                           'most_likely': thresholds_ml,
                           'sampled': thresholds_sampled}    

                grouped_results[key] = results
                
            return grouped_results
             
    def conformal_bound(self, aggregation: str='mean', alpha: float=0.1):
        """
        Ensure per sample guarantee by using conformal prediction on the fitness residuals.
        """ 
        # Helper to compute Q for given arrays of target and aggregated_sampled (both numpy arrays)
        def _compute_Q_for_arrays(target: np.ndarray, aggregated: np.ndarray) -> float:
            """
            abs conformal residual 1-alpha bound computation
            """
            if target.shape[0] != aggregated.shape[0]:
                raise ValueError(f"Length mismatch in target and aggregated sample lists.")
            n = target.shape[0]
            if n == 0:
                return float('nan')
            
            # Compute residuals
            residuals = np.abs(target - aggregated)  # in [0,1]
            sorted_res = np.sort(residuals)
            
            # finite-sample conformal index (1-based)
            k = math.ceil((1.0 - alpha) * (n + 1))
            
            # cap to [1, n]
            k = max(1, min(k, n))
            idx = k - 1  # 0-based
            Q_val = float(sorted_res[idx])
            
            # clamp to [0,1] for safety
            Q_val = min(max(Q_val, 0.0), 1.0)
            
            return Q_val
        
        if not self.mondrian:
            target_fitness_scores = np.asarray(self.fitness_score_results['target_fitness_score'])
            
            sampled_fitness_scores = self.fitness_score_results['sampled_case_fitness_scores']    
            # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
            aggregated_sampled_pairs = [self.__aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
            aggregated_sampled_fitness = np.asarray([smp[0] for smp in aggregated_sampled_pairs])
            # mean_std_sampled = float(np.nanmean([pair[1] for pair in aggregated_sampled_pairs])) if len(aggregated_sampled_pairs) > 0 else float('nan')
            
            Q = _compute_Q_for_arrays(target=target_fitness_scores, aggregated=aggregated_sampled_fitness)
        
            return Q
        else:
            # mondrian grouped path
            grouped_Qs = {}
            mondrian_grouped_fitness_scores = self.__mondrian_pref_len_grouping(fitness_score_results=self.fitness_score_results)

            for key, values in mondrian_grouped_fitness_scores.items():
                # Expect values to be dicts with 'target_fitness_score' & 'sampled_case_fitness_scores'
                target_fitness_scores = np.asarray(values['target_fitness_score'])
                sampled_fitness_scores = values['sampled_case_fitness_scores']

                # aggregate sampled fitnesses per case for this group
                aggregated_sampled_pairs = [self.__aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
                aggregated_sampled_fitness = np.asarray([pair[0] for pair in aggregated_sampled_pairs])
                # mean std in this group (diagnostic)
                # mean_std_sampled = float(np.nanmean([pair[1] for pair in aggregated_sampled_pairs])) if len(aggregated_sampled_pairs) > 0 else float('nan')
                
                Q_group = _compute_Q_for_arrays(target=target_fitness_scores, aggregated=aggregated_sampled_fitness)

                grouped_Qs[key] = Q_group

            return grouped_Qs
            
    def risk_controlled_threshold(self, t_max, delta, alpha, min_count, eps=1e-4, B=1.0, aggregation: str='mean'):
        """
        Calibrate the empirical thresholds for q_risk and q_high_risk and q_save and deliver a guarentee that the global predicted fitness and the target fitness deviates only by delta with
        a 1-alpha finite sample guarentee using Conformal Risk Control.
        
        def _feasible_lambda(fitness_targets, agg_fitness_samples, lambda_val):
            # update t
            t = t_max - lambda_val
            # check all samples that are samller the threshold -> f elem B
            assigned = (agg_fitness_samples <= t)
            
            count = assigned.sum()
            if count < min_count: 
                return False, {'count': count}
            
            # check all samples that have predicted fitness different to the target fitness by at least delta
            exceed = (np.abs(fitness_targets - agg_fitness_samples) > delta).astype(float)
            
            # L' = (I_assigned & I_exceed) - alpha
            loss = (assigned.astype(float) * exceed) - alpha   
            
            # Mean L' over all validation samples.
            Rn_expected_loss = loss.mean()
            corrected = (len(agg_fitness_samples)/(len(agg_fitness_samples)+1.0))*Rn_expected_loss + (B/(len(agg_fitness_samples)+1.0))
            
            return (corrected <= 0.0), {'count':count, 'Rn':Rn_expected_loss, 'corrected':corrected}
        """
        
        target_fitness_scores = np.asarray(self.fitness_score_results['target_fitness_score'])
            
        sampled_fitness_scores = self.fitness_score_results['sampled_case_fitness_scores']
        # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
        aggregated_sampled_pairs = [self.__aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
        aggregated_sampled_fitness = np.asarray([smp[0] for smp in aggregated_sampled_pairs])
        
        # Guarantee same length of fitness score lists
        assert target_fitness_scores.shape[0] == aggregated_sampled_fitness.shape[0]
        
        # Start with lambda zer0
        lam = 0.0
        # Iterate through all possible lambdas:
        while lam <= t_max:
            # update t
            t = t_max - lam
            # check all samples that are samller the threshold -> f elem B
            assigned = (aggregated_sampled_fitness <= t)
            
            count = assigned.sum()
            if count < min_count: 
                return False, {'count': count}
            
            # check all samples that have predicted fitness different to the target fitness by at least delta
            exceed = (np.abs(target_fitness_scores - aggregated_sampled_fitness) > delta).astype(float)
            
            # L' = (I_assigned & I_exceed) - alpha
            loss = (assigned.astype(float) * exceed) - alpha   
            
            # Mean L' over all validation samples.
            Rn_expected_loss = loss.mean()
            corrected = (len(aggregated_sampled_fitness)/(len(aggregated_sampled_fitness)+1.0))*Rn_expected_loss + (B/(len(aggregated_sampled_fitness)+1.0))
            
            if corrected < 0.0:
                return {'lambda_hat': lam, 't_hat': (t_max -lam)}
            
            lam = lam + eps
        return None

        
        
        
        
        

        