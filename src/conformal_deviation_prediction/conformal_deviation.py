import random
import numpy as np
from typing import Any, Dict, List, Tuple
import pandas as pd
from collections import Counter

class DeviationPrediction:
    def __init__(self, pred_conf_set):
        # list of dicts containing: target case, alignment, fitness, cost
        self.pred_conf_set = pred_conf_set
        
    def __get_target_aligns_pref_suf(self) -> List[Any]:
        """
        Returns the list of target alignments.
        """
        tgt_aligns = [tgt['suffix_alignment'] for tgt in self.pred_conf_set['target_conformance']]
        tgt_prefs = [tgt['prefix'] for tgt in self.pred_conf_set['target_conformance']]
        tgt_sufs = [tgt['target_suffix'] for tgt in self.pred_conf_set['target_conformance']]

        return tgt_aligns, tgt_prefs, tgt_sufs

    def __get_aggregated_alignments(self) -> List[List[Any]]:
        """
        Gets for each prefix the list of 100 sampled alignments. 
        Then takes an aggregation such as meadian and returns all alignments that have median fitness score.
        """        
        pred_aligns = []
        pred_prefs = []
        pred_sufs = []

        for smpls in self.pred_conf_set["samples_conformance"]:
            # Extract fitness values
            fitness_values = np.array([smpl["suffix_fitness"] for smpl in smpls])
            # Get aggregated fitness score
            fitness = np.median(fitness_values)
            # Get alignments of suffix
            alignments = [smpl["suffix_alignment"] for smpl in smpls if fitness == smpl['fitness']]
            # Get prefix                                  
            prefixes = [smpl["prefix"] for smpl in smpls if fitness == smpl['fitness']]
            # Get sampled suffixes
            suffixes = [smpl["sampled_suffix"] for smpl in smpls if fitness == smpl['fitness']]
            
            # Randomly choose one alignment from median list
            if len(alignments) > 0:
                # Randomness makes results non-deterministic
                idx = random.randrange(len(alignments))
                # Get fixed suffix and alignments
                # idx = 0
                pred_aligns.append(alignments[idx])
                pred_prefs.append(prefixes[idx])
                pred_sufs.append(suffixes[idx])
                
            else:
                # idx = random.randrange(len(smpls))
                idx = 0
                pred_aligns.append([smpl["suffix_alignment"] for smpl in smpls][idx])
                pred_prefs.append([smpl["prefix"] for smpl in smpls][idx])
                pred_sufs.append([smpl["sampled_suffix"] for smpl in smpls][idx])
                
        # return pred_aligns, pred_prefs, pred_sufs, pred_aligns_prob
        return pred_aligns, pred_prefs, pred_sufs
    
    def get_aggregated_deviations(self):
        """
        Return per-case deviation info.
        - Removes ('>>', None), (None, '>>') filler from alignments.
        - Clears entries that belong to the prefix (prefix matches are removed).
        - Collects deviations only in the suffix region (indices >= len(prefix)).
        - Returns cleaned alignments (with prefix slots set to None) and suffix deviations.
        """
        tgt_aligns, tgt_prefs, tgt_sufs = self.__get_target_aligns_pref_suf()
        pred_aligns, pred_prefs, pred_sufs = self.__get_aggregated_alignments()

        # Basic sanity checks for length consistency
        n = len(tgt_prefs)
        if not (len(pred_prefs) == n and len(tgt_aligns) == n and len(pred_aligns) == n and len(tgt_sufs) == n and len(pred_sufs) == n):
            raise ValueError("Mismatched lengths between target/predicted prefixes/aligns/suffixes.")

        # Remove filler form suffix alignments:
        cleaned_tgt_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in tgt_aligns]
        cleaned_pred_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in pred_aligns]
        
        tgt_deviations = [[(a,b) for (a,b) in align if a != b] for align in cleaned_tgt_alignments]
        pred_deviations = [[(a,b) for (a,b) in align if a != b] for align in cleaned_pred_alignments]

        results = []
        for i in range(len(tgt_aligns)):
            result = {"prefix": tgt_prefs[i],
                      "tgt_suffix": tgt_sufs[i],
                      "pred_suffix": pred_sufs[i],
                      # All suffix aligning (synchronous) and deviating moves 
                      "tgt_cleaned_aligns": cleaned_tgt_alignments[i],
                      "pred_cleaned_aligns": cleaned_pred_alignments[i],
                      # All suffix deviating only moves
                      "tgt_deviations": tgt_deviations[i],
                      "pred_deviations": pred_deviations[i]}

            results.append(result)

        return results
    
    def get_probabilistic_deviations(self):
        """
        Like get_aggregated_deviations() but uses probabilistic predicted alignments/suffixes.
        Expects a helper that returns per-prefix lists of alignment variants and their probabilities:
        pred_aligns_all, pred_aligns_prob, pred_prefs, pred_sufs_all, pred_sufs_prob
        Each pred_aligns_all[i] is a list of alignments (alignment = list of (a,b) tuples).
        Each pred_aligns_prob[i] is a list of floats (same length), summing to ~1.0 (if available).
        """
        tgt_aligns, tgt_prefs, tgt_sufs = self.__get_target_aligns_pref_suf()

        # Basic sanity checks for length consistency
        n = len(tgt_prefs)
        if not (len(tgt_aligns) == n and len(tgt_sufs) == n):
            raise ValueError("Mismatched lengths between target prefixes/aligns/suffixes.")

        # Remove filler from target alignments
        cleaned_tgt_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in tgt_aligns]
        tgt_deviations = [[(a, b) for (a, b) in align if a != b] for align in cleaned_tgt_alignments]

        # Aggregate probabilistic deivations
        results = []
        for i in range(n):            
            # Get all samples for this case (prefix)
            smpls = self.pred_conf_set["samples_conformance"][i]
            
            total_samples = len(smpls)

            # Collect cleaned alignments and deviations for each sample
            sampled_suffixes = []
            cleaned_aligns = []
            sample_devs = []   
            for smpl in smpls:
                suffix = smpl["sampled_suffix"]
                sampled_suffixes.append(suffix)
                    
                # Alignment 
                align = smpl["suffix_alignment"]
                cleaned_align = [a for a in align if a != ('>>', None) and a != (None, '>>')]
                cleaned_aligns.append(cleaned_align)
                    
                # Deviations
                devs = [(a, b) for (a, b) in cleaned_align if a != b]
                # List of deviations across all samples:
                for dev in devs:
                    sample_devs.append(dev)

            # Count frequencies
            counter_devs = Counter(sample_devs) 
            all_devs_with_prob = [(k, v / total_samples) for k, v in counter_devs.items()]
            pred_deviations = [k for (k,a) in all_devs_with_prob if a >= 0.50]
                
            results.append({
                "prefix": tgt_prefs[i],
                "tgt_suffix": tgt_sufs[i],
                "pred_suffix": sampled_suffixes,
                # All suffix aligning (synchronous) and deviating moves 
                "tgt_cleaned_aligns": cleaned_tgt_alignments[i],
                "pred_cleaned_aligns": sampled_suffixes,
                # All suffix deviating only moves with probability across all samples
                "tgt_deviations": tgt_deviations[i],
                "pred_deviations": pred_deviations,
                "deviations_prob_per_case": all_devs_with_prob})

        return results
