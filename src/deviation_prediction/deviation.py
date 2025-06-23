import numpy as np
import pm4py

from collections import defaultdict, OrderedDict, Counter

class Deviation:
    def __init__(self):
        pass
    
    @staticmethod
    def __individual_deviations_helper(conformance: dict):

        deviations = defaultdict(lambda: {'model_moves': [],
                                          'log_moves': []})
        
        # For target and most_likely:
        for pref_len, conformance_dicts in conformance.items():
            # N dicts with one dict per target
            for conformance_dict in conformance_dicts:
                alignment = conformance_dict['alignment']
                
                # Model move
                model_moves = [ (a, b) for (a, b) in alignment if a!= None and b != None and a != '>>' and b == '>>']
                log_moves = [ (a, b) for (a, b) in alignment if a!= None and b != None and a == '>>' and b != '>>']
                
                deviations[pref_len]['model_moves'].append(model_moves)
                deviations[pref_len]['log_moves'].append(log_moves)
    
        sorted_deviations = OrderedDict(sorted(deviations.items(), key=lambda item: item[0]))
    
        return sorted_deviations
    
    def individual_deviations_target(self, target_conformance):
        deviations = self.__individual_deviations_helper(conformance=target_conformance)

        return deviations
    
    def individual_deviations_most_likely(self, most_likely_conformance):
        deviations = self.__individual_deviations_helper(conformance=most_likely_conformance)

        return deviations
    
    @staticmethod
    def __majority_moves(move_lists, beta):
        """
        move_lists: List[List[tuple]]  — one list of moves per sample
        beta: float in (0,1)          — fraction threshold
        returns: List[tuple]          — moves appearing in ≥ beta fraction of samples
        """
        
        N = len(move_lists)
        # Count in how many samples each move appears
        cnt = Counter()
        
        for one_sample in move_lists:
            # Use set() so duplicates in the same sample only count once
            for move in set(one_sample):
                cnt[move] += 1

        # Keep only those moves whose count ≥ beta * N
        thresh = beta * N
        return [ move for move, c in cnt.items() if c >= thresh ]
    
    def individual_deviations_samples(self, samples_conformance, beta_threshold):
        
        deviations = defaultdict(lambda: {'model_moves': [],
                                          'log_moves': []})
        
        # For target and most_likely:
        for pref_len, samples_list in samples_conformance.items():
            # A list of list with MC samples 
            for conformance_dicts in samples_list:
                # 1000 dicts with alignments
                
                # List of list with model moves of all 1000 samples:
                model_moves = []
                log_moves = []
                for conformance_dict in conformance_dicts:
                    # Alignment of one sample
                    alignment = conformance_dict['alignment']
                    
                    # Model move
                    sample_model_moves = [ (a, b) for (a, b) in alignment if a!= None and b != None and a != '>>' and b == '>>']
                    model_moves.append(sample_model_moves)
                    
                    sample_log_moves = [ (a, b) for (a, b) in alignment if a!= None and b != None and a == '>>' and b != '>>']
                    model_moves.append(sample_log_moves)
                    
                
                # now get the “ensemble” deviations at 50% threshold
                top_model_moves = self.__majority_moves(model_moves, beta=beta_threshold)
                top_log_moves   = self.__majority_moves(log_moves, beta=beta_threshold)
                
                deviations[pref_len]['model_moves'].append(top_model_moves)
                deviations[pref_len]['log_moves'].append(top_log_moves)
                
        sorted_deviations = OrderedDict(sorted(deviations.items(), key=lambda item: item[0]))
        
        return sorted_deviations

    def deviation_patterns(self):
        pass