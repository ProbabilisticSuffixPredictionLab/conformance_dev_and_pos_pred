import random
import numpy as np
from typing import Any, Dict, List
import pandas as pd

class PreProcessConformanceResults:
    def __init__(self, conformance_results: Dict[str, List[Any]]):        
        # list of 1000 list of dicts containing: target case, alignment, fitness, cost
        self.conformance_results = conformance_results
        
    def pre_process_for_lr_model(self):
        """
        Returns the list of predicted fitness scores (list of lists) for smpls.
        """
        samples_fitness = [[smpl['fitness'] for smpl in smpls] for smpls in self.conformance_results['samples_conformance']]
        
        rows = []
        for i, samples in enumerate(samples_fitness):
            arr = np.asarray(samples, dtype=float)
            if arr.size == 0:
                raise ValueError(f"Empty sample array at index {i}.")
            mean = float(arr.mean())
            median = float(np.median(arr))
            var = float(arr.var(ddof=0))
            std = float(arr.std(ddof=0))
            mn = float(arr.min())
            mx = float(arr.max())
            q25 = float(np.percentile(arr, 25))
            q75 = float(np.percentile(arr, 75))
            iqr = q75 - q25
            cm2 = float(np.mean((arr - mean) ** 2))
            cm3 = float(np.mean((arr - mean) ** 3))
            cm4 = float(np.mean((arr - mean) ** 4))
            skew = (cm3 / (cm2 ** 1.5)) if cm2 > 0 else 0.0
            kurt = ((cm4 / (cm2 ** 2)) - 3.0) if cm2 > 0 else -3.0
            rows.append([mean, var, std, skew, kurt, median, mn, mx, q25, q75, iqr])
            # rows.append([mean, var, skew, kurt, median])

        columns = ['mean','variance','std','skewness','kurtosis_excess','median','min','max','q25','q75','iqr']
        # columns = ['mean','variance','skewness','kurtosis_excess','median']
        df = pd.DataFrame(rows, columns=columns)
        
        return df

class DeviationPrediction:
    def __init__(self, pred_conf_set):
        # list of dicts containing: target case, alignment, fitness, cost
        self.pred_conf_set = pred_conf_set
        
    def __get_target_aligns_pref_suf(self) -> List[Any]:
        """
        Returns the list of target alignments.
        """
        
        tgt_aligns = [tgt['alignment'] for tgt in self.pred_conf_set['target_conformance']]
        tgt_prefs = [tgt['prefix'] for tgt in self.pred_conf_set['target_conformance']]
        tgt_sufs = [tgt['target_suffix'] for tgt in self.pred_conf_set['target_conformance']]

        return (tgt_aligns, tgt_prefs, tgt_sufs)

    def __get_predicted__aggregated_aligns_pref_suf(self) -> List[List[Any]]:
        """
        Returns the list of predicted alignments (list of lists) for smpls with median fitness score.
        """
        
        # Decide if this behavior is correct
        
        pred_aligns = []
        pred_prefs = []
        pred_sufs = []

        for smpls in self.pred_conf_set["samples_conformance"]:
            # Convert to list if it's a numpy array
            if isinstance(smpls, np.ndarray):
                smpls = smpls.tolist()

            # Skip empty or None entries
            if smpls is None or len(smpls) == 0:
                continue

            # Extract fitness values
            fitness_values = np.array([smpl["fitness"] for smpl in smpls])
            median_fitness = np.median(fitness_values)

            # Find alignments with fitness closest to the median
            abs_diff = np.abs(fitness_values - median_fitness)
            min_diff = np.min(abs_diff)
            
            close_alignments = [smpl["alignment"] for smpl, diff in zip(smpls, abs_diff) if diff == min_diff]
            close_prefixes = [smpl["prefix"] for smpl, diff in zip(smpls, abs_diff) if diff == min_diff]
            close_suffixes = [smpl["sampled_suffix"] for smpl, diff in zip(smpls, abs_diff) if diff == min_diff]
            
            # Randomly choose one alignment (safe even if only one)
            if len(close_alignments) > 0:
                idx = random.randrange(len(close_alignments))
                pred_aligns.append(close_alignments[idx])
                pred_prefs.append(close_prefixes[idx])
                pred_sufs.append(close_suffixes[idx])
            else:
                # Fallback: choose the alignment with smallest difference
                best_idx = int(np.argmin(abs_diff))
                pred_aligns.append(smpls[best_idx]["alignment"])
                pred_prefs.append(smpls[best_idx]["prefix"])
                pred_sufs.append(smpls[best_idx]["sample_suffix"])   

        return pred_aligns, pred_prefs, pred_sufs
    
    ## Check this method again !!
    def get_deviations(self):
        results = []
        
        tgt_aligns, tgt_prefs, tgt_sufs = self.__get_target_aligns_pref_suf()
        pred_aligns, pred_prefs, pred_sufs = self.__get_predicted__aggregated_aligns_pref_suf()
        
        # Clean the alignments by removing ('>>', None) entries
        cleaned_tgt_alignments = [[cleaned_align for cleaned_align  in align if cleaned_align != ('>>', None)] for align in tgt_aligns]
        cleaned_pred_alignments = [[cleaned_align for cleaned_align  in align if cleaned_align != ('>>', None)] for align in pred_aligns]
        
        for i, (t_prefix, p_prefix) in enumerate(zip(tgt_prefs, pred_prefs)):
            # Check if prefixes match
            if t_prefix != p_prefix:
                raise ValueError(f"Prefix mismatch: target prefix {t_prefix} does not match predicted prefix {p_prefix}.")
            
            # Get cleaned aligned element:
            cleaned_tgt_align = cleaned_tgt_alignments[i].copy()
            cleaned_pred_align = cleaned_pred_alignments[i].copy()
            
            for j, p in enumerate(t_prefix):
                if cleaned_tgt_align[j][0] == p or cleaned_tgt_align[j][1] == p:
                    cleaned_tgt_align[j] = None
                
                if cleaned_pred_align[j][0] == p or cleaned_pred_align[j][1] == p:
                     cleaned_pred_align[j] = None 
                
            cleaned_tgt_deviation = [align for align in cleaned_tgt_align if align != None and align[0] != align[1]]
            cleaned_pred_deviation = [align for align in cleaned_pred_align if align != None and align[0] != align[1]]
                    
            result = {"prefix": t_prefix,
                      "tgt_suffix": tgt_sufs[i],
                      "pred_suffix": pred_sufs[i],
                      "tgt_cleaned_aligns": cleaned_tgt_alignments[i],
                      "pred_cleaned_aligns": cleaned_pred_alignments[i],
                      "tgt_deviations": cleaned_tgt_deviation,
                      "pred_deviations": cleaned_pred_deviation}
            
            results.append(result)   
        
        return results


