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
        tgt_aligns = [tgt['alignment'] for tgt in self.pred_conf_set['target_conformance']]
        tgt_prefs = [tgt['prefix'] for tgt in self.pred_conf_set['target_conformance']]
        tgt_sufs = [tgt['target_suffix'] for tgt in self.pred_conf_set['target_conformance']]

        return tgt_aligns, tgt_prefs, tgt_sufs

    def __get_predicted_aggregated_aligns_pref_suf(self) -> List[List[Any]]:
        """
        Gets for each prefix the list of 100 sampled alignments. 
        Then takes an aggregation such as meadian and returns all alignments that have median fitness score.
        """        
        pred_aligns = []
        pred_aligns_prob = []
        pred_prefs = []
        pred_sufs = []

        for smpls in self.pred_conf_set["samples_conformance"]:
            # Extract fitness values
            fitness_values = np.array([smpl["fitness"] for smpl in smpls])
            
            fitness = np.median(fitness_values)
            # fitness = float(fitness_values.max())
            
            alignments = [smpl["alignment"] for smpl in smpls if fitness == smpl['fitness']]
            
            # Correct that for AUC_ROC
            
            # 1) count every tuple across all alignments (single pass)
            counts: Counter = Counter(item for aln in alignments for item in aln)
            total_tokens: int = sum(counts.values())

            # 2) build result: for each alignment, a dict mapping tuple -> relative frequency
            alignments_with_prob: List[Dict[Tuple[Any, ...], float]] = []

            if total_tokens == 0:
                # avoid division by zero
                alignments_with_prob = [{item: 0.0 for item in aln} for aln in alignments]
            else:
                for aln in alignments:
                    # dict: token -> prob (duplicates in aln are collapsed to single key)
                    alignments_with_prob.append({item: counts[item] / total_tokens for item in aln})
            
            """
            # New: Probabilities important for AUC_ROC curve:
            
            # If you prefer a list-of-probabilities that preserves order and duplicates:
            alignments_with_prob_pos: List[List[float]] = []
            if total_tokens == 0:
                alignments_with_prob_pos = [[0.0 for _ in aln] for aln in alignments]
            else:
                alignments_with_prob_pos = [[counts[item] / total_tokens for item in aln] for aln in alignments]  
            """
            
            prefixes = [smpl["prefix"] for smpl in smpls if fitness == smpl['fitness']]
            suffixes = [smpl["sampled_suffix"] for smpl in smpls if fitness == smpl['fitness']]
            
            # Randomly choose one alignment from median list
            if len(alignments) > 0:
                # Randomness makes results non-deterministic
                idx = random.randrange(len(alignments))
                # idx = 0
                pred_aligns.append(alignments[idx])
                pred_prefs.append(prefixes[idx])
                pred_sufs.append(suffixes[idx])
                
            else:
                # Choose random:
                idx = random.randrange(len(smpls))
                # idx = 0
                pred_aligns.append([smpl["alignment"] for smpl in smpls][idx])
                pred_prefs.append([smpl["prefix"] for smpl in smpls][idx])
                pred_sufs.append([smpl["sampled_suffix"] for smpl in smpls][idx])
                
        # return pred_aligns, pred_prefs, pred_sufs, pred_aligns_prob
        return pred_aligns, pred_prefs, pred_sufs
    
    def get_deviations(self):
        """
        Return per-case deviation info.
        - Removes ('>>', None), (None, '>>') filler from alignments.
        - Clears entries that belong to the prefix (prefix matches are removed).
        - Collects deviations only in the suffix region (indices >= len(prefix)).
        - Returns cleaned alignments (with prefix slots set to None) and suffix deviations.
        """
        results = []

        tgt_aligns, tgt_prefs, tgt_sufs = self.__get_target_aligns_pref_suf()
        pred_aligns, pred_prefs, pred_sufs = self.__get_predicted_aggregated_aligns_pref_suf()

        # Basic sanity checks for length consistency
        n = len(tgt_prefs)
        if not (len(pred_prefs) == n and len(tgt_aligns) == n and len(pred_aligns) == n and len(tgt_sufs) == n and len(pred_sufs) == n):
            raise ValueError("Mismatched lengths between target/predicted prefixes/aligns/suffixes.")

        # Remove filler form alignments:
        cleaned_tgt_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in tgt_aligns]
        cleaned_pred_alignments = [[a for a in align if a != ('>>', None) and a != (None, '>>')] for align in pred_aligns]

        for i in range(n):
            # prefix
            t_prefix = tgt_prefs[i]
            p_prefix = pred_prefs[i]
            # Ensure prefixes match (keep original behaviour)
            if t_prefix != p_prefix:
                raise ValueError(f"Prefix mismatch: target prefix {t_prefix} does not match predicted prefix {p_prefix} (index {i}).")
            
            # target suffix
            t_suffix = tgt_sufs[i]
            # predicted suffix
            p_suffix = pred_sufs[i]

            # Work on copies so we can modify and return the modified versions
            t_align = list(cleaned_tgt_alignments[i])
            p_align = list(cleaned_pred_alignments[i])

            prefix_len = len(t_prefix)

            # Clear entries that correspond to the prefix (if they match the prefix element).
            # Be careful with index bounds and None entries.
            for j, pref_item in enumerate(t_prefix):
                # target alignment
                if j < len(t_align):
                    pair = t_align[j]
                    if pair is not None:
                        a, b = pair  # tuple-like expected
                        # If the move is a synchronuous move -> aligning move, set to None since only deviations should be kept.
                        if a == pref_item or b == pref_item:
                            t_align[j] = None
                # predicted alignment
                if j < len(p_align):
                    pair = p_align[j]
                    if pair is not None:
                        a, b = pair
                        if a == pref_item or b == pref_item:
                            # If the move is a synchronuous move -> aligning move, set to None since only deviations should be kept.
                            p_align[j] = None

            # Collect deviations only in suffix region (indices >= prefix_len): A deviation is a log or model move:
            t_deviations = [pair for idx, pair in enumerate(t_align) if pair is not None and idx >= prefix_len and pair[0] != pair[1]]
            p_deviations = [pair for idx, pair in enumerate(p_align) if pair is not None and idx >= prefix_len and pair[0] != pair[1]]

            # Check here, why even if the pred suffix is empty still suffix alignments can take place.
            
            result = {"prefix": t_prefix,
                      "tgt_suffix": t_suffix,
                      "pred_suffix": p_suffix,
                      # All suffix aligning (synchronous) and deviating moves 
                      "tgt_cleaned_aligns": t_align,
                      "pred_cleaned_aligns": p_align,
                      # All suffix deviating only moves
                      "tgt_deviations": t_deviations,
                      "pred_deviations": p_deviations}

            results.append(result)

        return results