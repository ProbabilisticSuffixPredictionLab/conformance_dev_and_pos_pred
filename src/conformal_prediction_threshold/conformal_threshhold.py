import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import math
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any

class ConformalPredictionThreshold:
    def __init__(self, d_con_results: List[Dict[str, Any]], conformance_object: Any, log_name: Optional[str] = ""):
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
        Original single‐threaded implementation.
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
        
    @staticmethod
    def _compute_fitness(conformance_object, log_name: str, values: Any) -> Tuple[float, float, Tuple[float, float]]:
        """
        Module‐level helper for multiprocessing.
        """
        target_align, mostlikely_align, sample_aligns = (conformance_object.conformance_of_sampled_suffixes(log_name=log_name, result_values=values))
        
        target_fitness = target_align['fitness']
        most_likely = mostlikely_align['fitness']
        sampled_fitnessess = np.array([x['fitness'] for x in sample_aligns])
        
        return target_fitness, most_likely, (sampled_fitnessess.mean(), sampled_fitnessess.var(ddof=1))

    def simple_threshold_q_parallel(self, alpha, max_workers: Optional[int] = None):
        """
        Parallel implementation using ProcessPoolExecutor.
        """
        # 1) Build the flat list of tasks
        tasks = [
            (self.conformance_object, self.log_name, values)
            for result in self.d_con_results
            for values in result.values()
        ]

        # 2) Dispatch in parallel
        max_workers = max_workers or os.cpu_count()
        target_scores = []
        most_likely_scores = []
        sampled_scores = []

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(ConformalPredictionThreshold._compute_fitness, *task): task
                for task in tasks
            }
            for fut in as_completed(futures):
                tf, mf, (s_mean, s_var) = fut.result()
                target_scores.append(tf)
                most_likely_scores.append(mf)
                sampled_scores.append(s_mean)

        # 3) Sort each list ascending
        target_scores.sort()
        most_likely_scores.sort()
        sampled_scores.sort()

        # 4) Compute the Python index for the (n+1)*alpha quantile
        n = len(target_scores)
        q_index = math.ceil((n + 1) * alpha) - 1
        q_index = min(max(q_index, 0), n - 1)

        # 5) Extract the three Q‐values
        q_target = target_scores[q_index]
        q_most_likely = most_likely_scores[q_index]
        q_sampled_mean = sampled_scores[q_index]

        return q_target, q_most_likely, q_sampled_mean
    
    
    def threshold_q_per_prefix_length(self,alpha: float) -> List[Tuple[int, float, float, float]]:
        """
        For each prefix_length, compute the (target, most_likely, sampled_mean) Q-values.
        Returns a list of tuples: (prefix_length, q_target, q_most_likely, q_sampled_mean) sorted by prefix_length ascending.
        """
        # 1) Group all fitness scores by prefix_length
        groups: Dict[int, Dict[str, List[Any]]] = defaultdict(lambda: {
            'target': [],
            'most_likely': [],
            'sampled_means': []
        })

        for result in self.d_con_results:
            for (case_name, prefix_length), values in result.items():
                t_con, m_con, sample_aligns = self.conformance_object.conformance_of_sampled_suffixes(
                    log_name=self.log_name,
                    result_values=values
                )
                # collect
                groups[prefix_length]['target'].append(t_con['fitness'])
                groups[prefix_length]['most_likely'].append(m_con['fitness'])
                
                arr = np.array([x['fitness'] for x in sample_aligns])
                groups[prefix_length]['sampled_means'].append(arr.mean())

        # 2) For each prefix, sort and pick the (n+1)*alpha quantile
        output: List[Tuple[int, float, float, float]] = []
        for prefix_length in sorted(groups):
            tgt_scores = sorted(groups[prefix_length]['target'])
            ml_scores  = sorted(groups[prefix_length]['most_likely'])
            sm_scores  = sorted(groups[prefix_length]['sampled_means'])

            n = len(tgt_scores)
            # conformal quantile index (1-based → 0-based)
            q_idx = math.ceil((n + 1) * alpha) - 1
            q_idx = min(max(q_idx, 0), n - 1)

            q_tgt  = tgt_scores[q_idx]
            q_ml   = ml_scores[q_idx]
            q_sm   = sm_scores[q_idx]

            output.append((prefix_length, q_tgt, q_ml, q_sm))

        return output
    
    
    
    
    
    @staticmethod
    def _compute_fitness_for_prefix(prefix_length: int, conformance_object: Any, log_name: str,values: Any) -> Tuple[int, float, float, float]:
        """
        Worker helper: returns (prefix_length, target_fitness, most_likely_fitness, sampled_mean)
        """
        t_con, m_con, sample_aligns = conformance_object.conformance_of_sampled_suffixes(
            log_name=log_name,
            result_values=values
        )
        target_f = t_con['fitness']
        most_likely_f = m_con['fitness']
        arr = np.array([x['fitness'] for x in sample_aligns])
        return prefix_length, target_f, most_likely_f, arr.mean()

    def threshold_q_per_prefix_length_parallel(
        self,
        alpha: float,
        max_workers: Optional[int] = None
    ) -> List[Tuple[int, float, float, float]]:
        """
        Parallel version: for each prefix_length, compute
          (q_target, q_most_likely, q_sampled_mean) at miscoverage alpha.
        Returns a list of (prefix_length, q_target, q_most_likely, q_sampled_mean),
        sorted by prefix_length.
        """
        # 1) Build tasks: one per (case_name, prefix_length, values)
        tasks = []
        for result in self.d_con_results:
            for (case_name, prefix_length), values in result.items():
                tasks.append((prefix_length, self.conformance_object, self.log_name, values))

        # 2) Launch Pool
        max_workers = max_workers or os.cpu_count()
        grouped: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {
            'target': [], 'most_likely': [], 'sampled_means': []
        })

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(
                    ConformalPredictionThreshold._compute_fitness_for_prefix,
                    *task
                ): task for task in tasks
            }
            for fut in as_completed(futures):
                prefix, t_f, m_f, s_mean = fut.result()
                grouped[prefix]['target'].append(t_f)
                grouped[prefix]['most_likely'].append(m_f)
                grouped[prefix]['sampled_means'].append(s_mean)

        # 3) For each prefix_length, sort & pick quantile
        output: List[Tuple[int, float, float, float]] = []
        for prefix in sorted(grouped):
            tgt = sorted(grouped[prefix]['target'])
            ml  = sorted(grouped[prefix]['most_likely'])
            sm  = sorted(grouped[prefix]['sampled_means'])
            n   = len(tgt)
            idx = math.ceil((n + 1) * alpha) - 1
            idx = min(max(idx, 0), n - 1)

            output.append((prefix, tgt[idx], ml[idx], sm[idx]))

        return output
        