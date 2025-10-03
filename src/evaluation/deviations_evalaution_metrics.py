import numpy as np
from collections import OrderedDict
from typing import Dict, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, roc_auc_score

class EvaluationMetrics:    
    def __init__(self, target_alignments: Optional[Dict]=None, predicted_alignments: Optional[Dict] = None):
        # list of moves per prefix length
        self.target_alignments = target_alignments
        # lists of 1000 lists containing tuples of moves
        self.pred_alignments = predicted_alignments

    def _get_log_moves(self, alignment: list): 
        """
        Alignment is list of tuples: Moves of type (move, '>>')
        """
        return [move for move in alignment if type(move[0]) == str and move[1] == '>>']

    def _get_model_moves(self, alignment: list):
        """
        Alignment is list of tuples: Move of type ('>>', move)
        """
        return [move for move in alignment if move[0] == '>>' and type(move[1]) == str]

    def precision_recall_deviations(self):
        """
        Calculates both precision and recall of the deviation prediction.
        """
        precision_results = {}
        recall_results = {}

        for target_align, pred_samples_align in zip(self.target_alignments, self.pred_alignments):
            # Create sets for target log and model moves (no duplicates)
            target_log_moves = set(self._get_log_moves(target_align))
            target_model_moves = set(self._get_model_moves(target_align))
            target_all_moves = target_log_moves | target_model_moves

            # Create sets for predicted log and model moves with relative frequencies
            pred_log_moves = {}
            pred_model_moves = {}

            for sample_alignment in pred_samples_align:
                for move in self._get_log_moves(sample_alignment):
                    pred_log_moves[move] = pred_log_moves.get(move, 0) + 1
                for move in self._get_model_moves(sample_alignment):
                    pred_model_moves[move] = pred_model_moves.get(move, 0) + 1

            # Convert counts to relative frequencies
            pred_log_moves = {(move, count / len(pred_samples_align)) for move, count in pred_log_moves.items()}
            pred_model_moves = {(move, count / len(pred_samples_align)) for move, count in pred_model_moves.items()}
            predicted_all_moves = pred_log_moves | pred_model_moves

            # Calculate precision and recall for individual deviations
            for move in target_all_moves:
                if move not in recall_results:
                    recall_results[move] = {'tp': 0, 'fn': 0}
                if move not in precision_results:
                    precision_results[move] = {'tp': 0, 'fp': 0}

            for move, prob in predicted_all_moves:
                if move not in precision_results:
                    precision_results[move] = {'tp': 0, 'fp': 0}
                if move in target_all_moves:
                    precision_results[move]['tp'] += prob
                    recall_results[move]['tp'] += prob
                else:
                    precision_results[move]['fp'] += prob

            # Any target move not predicted (or partially predicted)
            for move in target_all_moves:
                prob_pred = sum(prob for m, prob in predicted_all_moves if m == move)
                recall_results[move]['fn'] += max(0, 1 - prob_pred)

        # Compute overall precision
        total_tp_precision = sum(result['tp'] for result in precision_results.values())
        total_fp = sum(result['fp'] for result in precision_results.values())
        overall_precision = total_tp_precision / (total_tp_precision + total_fp) if (total_tp_precision + total_fp) > 0 else 0

        # Compute overall recall
        total_tp_recall = sum(r['tp'] for r in recall_results.values())
        total_fn = sum(r['fn'] for r in recall_results.values())
        overall_recall = total_tp_recall / (total_tp_recall + total_fn) if (total_tp_recall + total_fn) > 0 else 0

        return overall_precision, precision_results, overall_recall, recall_results

        
    
    
    
    
    
    
    
    
    
    
    
    
    def probabilistic_roc_deviation_multilabel(self, average='macro'):
        """
        Calculates the ROC and AUC value per class (model or log move)
        """
        
        # Collect all classes: All model and log moves either predicted or in target or in both
        all_classes = set()
        # Iterate through the predictions and the targets:
        for D in (self.pred_alignments, self.target_alignments):
            for vals in D.values():
                for seq in vals['model_moves'] + vals['log_moves']:
                    all_classes.update(seq)
        # Contains all model and log moves gathered in the predictions and targets
        all_classes = sorted(all_classes)
        # contains all classes (adds an index, starting with 1 and increasing)
        cls_to_idx = {c: i for i, c in enumerate(all_classes)}

        # Fill the lists to get TPR and FPR for each prefix length:
        # Y_true, Y_score: Input for roc curve
        # Y_true contains 0 or 1 for missing or present class
        Y_true_rows, Y_score_rows = [], []
        for pref in self.target_alignments:   
            # For each prefix length get all cases' model and log moves: list of lists
            tgt = self.target_alignments[pref]
            # For each prefix length get all cases model and log move: list of dicts
            pred = self.pred_alignments.get(pref, {'model_moves': [], 'log_moves': []})

            # Target and predicted contain same lenght of cases
            # Iterate through all cases deviations: Model and Log moves have same length and can both be seen as classes of deviations
            max_len = len(tgt['model_moves'])
            
            for i in range(max_len):
                
                # Tuples of true deviations in specific cases:
                tgt_model_move = tgt['model_moves'][i]
                tgt_log_move = tgt['log_moves'][i]
                true_set = set(tgt_model_move) | set(tgt_log_move)
                # Zero row for each class in (model & log move) target deviation set
                true_row = np.zeros(len(all_classes), int)
                for c in true_set:
                    # Fill true row with ones for predicted deviation classes:
                    true_row[cls_to_idx[c]] = 1

                # predicted 0/1 scores
                # dict wher key is deviation, value is prob
                pred_model_move = pred['model_moves'][i]
                pred_log_move = pred['log_moves'][i]
                # Prediction deviation dict with log and model moves as key and probs of occurence across samples as value
                pred_dict = pred_model_move | pred_log_move
                # Zero row for each class in (model & log move) predicted deviation set 
                score_row = np.zeros(len(all_classes), float)
                for c, prob in pred_dict.items():
                    score_row[cls_to_idx[c]] =prob

                Y_true_rows.append(true_row)
                Y_score_rows.append(score_row)

        # Contains for all samples a list with length amount classes and its either 0 or 1 (target) or prob (predicted)
        Y_true  = np.vstack(Y_true_rows)
        Y_score = np.vstack(Y_score_rows)
        
        # Filter out only the valid classes:
        # Valid class for ROC needs at least one positive example and at least one negative example in the target across all cases.
        # If a deviation form the prediction never happens in the targets or happens always no ROC can be build for that class since variation is required.
        valid_classes = []
        for cls, idx in cls_to_idx.items():
            pos = int(Y_true[:, idx].sum())
            neg = Y_true.shape[0] - pos
            if pos > 0 and neg > 0:
                valid_classes.append(cls)

        fpr_dict, tpr_dict, thr_dict = {}, {}, {}
        for cls in valid_classes:
            idx = cls_to_idx[cls]
            fpr, tpr, thr = roc_curve(Y_true[:, idx], Y_score[:, idx])
            fpr_dict[cls], tpr_dict[cls], thr_dict[cls] = fpr, tpr, thr

        if valid_classes:
            idxs = [cls_to_idx[c] for c in valid_classes]
            Yt_f = Y_true[:, idxs]
            Ys_f = Y_score[:, idxs]
            auc_macro = roc_auc_score(Yt_f, Ys_f, average=average)
        else:
            auc_macro = np.nan
            raise ValueError("No multi-label has both positive and negative samples.")

        return fpr_dict, tpr_dict, thr_dict, auc_macro, Y_true, Y_score, all_classes
    
    def compute_class_coverage(self, target_all: Dict, target_risk: Dict):
        """
        Check how many times target classes are predicted and how many times unseen are predicted.
        """
        # Collect classes in full set
        classes_tgt = set()
        for vals in target_all.values():
            for seq in vals['model_moves'] + vals['log_moves']:
                classes_tgt.update(seq)
        classes_tgt = sorted(classes_tgt)
        classes_prob_tgt = np.zeros(len(classes_tgt), float)
        
        # Collect classes in risk set
        classes_risk = set()
        for vals in target_risk.values():
            for seq in vals['model_moves'] + vals['log_moves']:
                classes_risk.update(seq)
        classes_risk = sorted(classes_risk)
        
        # Unseen deivations: Classes in risk not in deviations:
        classes_unseen = sorted(set([c_risk for c_risk in classes_risk if c_risk not in classes_tgt]))
        classes_prob_unseen = np.zeros(len(classes_unseen), float)
        
        amount_cases = 0
        for _, moves in  target_risk.items():
            model_moves = moves['model_moves'] 
            amount_cases += len(model_moves) 
            log_moves = moves['log_moves']
            
            for i, model_move in enumerate(model_moves):
                model_move_set = set(model_move)
                log_move_set = set(log_moves[i])
                
                # Coun how often the classes in the target are predicted over the samples:
                for i, c in enumerate(classes_tgt):
                    if c in model_move_set or c in log_move_set:
                        classes_prob_tgt[i] += 1
                # Count how often the classes in the predictions/ not in the target are predicted over the samples:        
                for i, c in enumerate(classes_unseen):
                    if c in model_move_set or c in log_move_set:
                        classes_prob_unseen[i] += 1
                        
        classes_prob_tgt = classes_prob_tgt / amount_cases
        
        classes_prob_unseen = classes_prob_unseen / amount_cases
        
        return classes_tgt, classes_prob_tgt, classes_unseen, classes_prob_unseen 
        