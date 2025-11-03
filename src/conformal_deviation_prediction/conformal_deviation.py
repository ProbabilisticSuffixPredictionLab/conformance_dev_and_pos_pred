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
                # idx = random.randrange(len(alignments))
                # Get fixed suffix and alignments
                idx = 0
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
    
    
    
    
    
    
    def __get_predicted_probabilistic_aligns_pref_suf(self) -> List[List[Any]]:
        """
        For each prefix-group in self.pred_conf_set["samples_conformance"]:
        - collect all sampled suffixes and alignments (no median selection)
        - compute probability of each unique alignment and each unique suffix
            as (count / total_samples_for_that_prefix)
        Returns:
        pred_aligns_prob: List[List[float]]               # per-prefix: list of probs for each alignment (same order)
        pred_prefs: List[Any]                             # representative prefix for each prefix-group
        pred_sufs: List[List[Tuple[Any,...]]]         # per-prefix: list of unique suffix tuples (e.g. event sequences)
        """

        pred_aligns_prob: List[List[float]] = []
        pred_prefs: List[Any] = []
        pred_sufs: List[List[Tuple[Any, ...]]] = []

        # Go trough all test cases:
        for smpls in self.pred_conf_set["samples_conformance"]:
            # total number of sampled suffixes for this prefix (e.g., ~1000)
            total_samples = len(smpls)

            # If no samples, append empty placeholders
            if total_samples == 0:
                pred_aligns_prob.append([])
                pred_prefs.append(None)
                pred_sufs.append([])
                continue

            # Go through all 1000 alignments across the suffix:
            # First get only suffix alignments:
            
            
            

            

        pass
    
    def get_probabilistic_deviations(self):
        """
        Like get_aggregated_deviations() but uses probabilistic predicted alignments/suffixes.
        Expects a helper that returns per-prefix lists of alignment variants and their probabilities:
        pred_aligns_all, pred_aligns_prob, pred_prefs, pred_sufs_all, pred_sufs_prob
        Each pred_aligns_all[i] is a list of alignments (alignment = list of (a,b) tuples).
        Each pred_aligns_prob[i] is a list of floats (same length), summing to ~1.0 (if available).
        """
        results = []

        # deterministic target
        tgt_aligns, tgt_prefs, tgt_sufs = self.__get_target_aligns_pref_suf()

        # probabilistic predictions (helper must exist)
        (pred_aligns_all,
        pred_aligns_prob,
        pred_prefs,
        pred_sufs_all,
        pred_sufs_prob) = self.__get_predicted_probabilistic_aligns_pref_suf()

        # Basic sanity checks for length consistency
        n = len(tgt_prefs)
        if not (len(pred_prefs) == n and len(tgt_aligns) == n and len(pred_aligns_all) == n
                and len(pred_aligns_prob) == n and len(tgt_sufs) == n and len(pred_sufs_all) == n
                and len(pred_sufs_prob) == n):
            raise ValueError("Mismatched lengths between target/predicted prefixes/aligns/suffixes (probabilistic).")

        # helper to remove filler tokens from an alignment
        def _remove_fillers(alignment: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
            return [a for a in alignment if a != ('>>', None) and a != (None, '>>')]

        for i in range(n):
            # target
            t_prefix = tgt_prefs[i]
            p_prefix = pred_prefs[i]
            if t_prefix != p_prefix:
                raise ValueError(f"Prefix mismatch: target prefix {t_prefix} != predicted prefix {p_prefix} (index {i}).")

            t_suffix = tgt_sufs[i]
            p_suffixes = pred_sufs_all[i]
            p_suffixes_prob = pred_sufs_prob[i]

            # Clean target alignment (remove filler)
            t_align_raw = tgt_aligns[i]
            t_align = _remove_fillers(list(t_align_raw))

            # Clean each predicted alignment variant (remove filler)
            # pred_aligns_all[i] is a list of alignment variants (each variant is list of pairs)
            pred_align_variants = [ _remove_fillers(list(aln)) for aln in pred_aligns_all[i] ]
            pred_align_probs = list(pred_aligns_prob[i])  # copy

            # ensure alignment/prob length consistency for this prefix
            if len(pred_align_variants) != len(pred_align_probs):
                raise ValueError(f"Predicted alignments vs probs length mismatch at index {i}.")

            prefix_len = len(t_prefix)

            # Clear prefix-matching synchronous moves in target alignment (same logic as aggregated version)
            t_align_copy = list(t_align)  # work on modifiable copy
            for j, pref_item in enumerate(t_prefix):
                if j < len(t_align_copy):
                    pair = t_align_copy[j]
                    if pair is not None:
                        a, b = pair
                        if a == pref_item or b == pref_item:
                            t_align_copy[j] = None

            # For predicted variants: clear prefix synchronous moves per-variant
            pred_aligns_cleaned_with_prob: List[Dict[str, Any]] = []
            for aln_variant, prob in zip(pred_align_variants, pred_align_probs):
                aln_copy = list(aln_variant)
                for j, pref_item in enumerate(t_prefix):
                    if j < len(aln_copy):
                        pair = aln_copy[j]
                        if pair is not None:
                            a, b = pair
                            if a == pref_item or b == pref_item:
                                aln_copy[j] = None
                pred_aligns_cleaned_with_prob.append({"alignment": aln_copy, "prob": float(prob)})

            # Collect target deviations in suffix region (idx >= prefix_len): pair not None and pair[0] != pair[1]
            t_deviations = [pair for idx, pair in enumerate(t_align_copy)
                            if pair is not None and idx >= prefix_len and pair[0] != pair[1]]

            # For predicted deviations: aggregate probabilities across variants
            # aggregated_deviation_prob: pair -> summed probability
            aggregated_deviation_prob: Dict[Tuple[Any, Any], float] = defaultdict(float)
            for variant in pred_aligns_cleaned_with_prob:
                prob = variant["prob"]
                aln = variant["alignment"]
                # get deviations in suffix region for this variant
                for idx, pair in enumerate(aln):
                    if pair is not None and idx >= prefix_len and pair[0] != pair[1]:
                        aggregated_deviation_prob[tuple(pair)] += prob

            # If predicted alignments had length < prefix_len or no variants, ensure deterministic empty structures
            # Normalize tiny floating rounding issues: if total prob sums to >0, leave as-is; otherwise keep zeros.
            # Build sorted lists for deterministic output
            pred_deviations_items = sorted(aggregated_deviation_prob.items(), key=lambda kv: (-kv[1], kv[0]))
            pred_deviations = [(pair, float(prob)) for (pair, prob) in pred_deviations_items]

            # Also sort predicted cleaned alignments by probability descending (deterministic tiebreaker by alignment)
            pred_aligns_cleaned_sorted = sorted(pred_aligns_cleaned_with_prob,
                                            key=lambda d: (-d["prob"], tuple(tuple(x) if x is not None else None for x in d["alignment"])))

            result = {
                "prefix": t_prefix,
                "tgt_suffix": t_suffix,
                "pred_suffixes": p_suffixes,
                "pred_suffixes_prob": p_suffixes_prob,
                # cleaned alignments (target single, predicted many with probabilities)
                "tgt_cleaned_aligns": t_align_copy,
                "pred_cleaned_aligns": pred_aligns_cleaned_sorted,
                # deviations (target single list, predicted aggregated distribution)
                "tgt_deviations": t_deviations,
                "pred_deviations": pred_deviations,
                "pred_deviations_dist": dict(aggregated_deviation_prob)
            }

            results.append(result)

        return results
