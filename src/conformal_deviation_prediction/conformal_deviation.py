import numpy as np
from typing import Any, Dict, List, Sequence

from conformal_analysis.conformal_model import DataFrameConstruction, LogisticRegressionModel


"""

class ConformalGrouping:
    def __init__(self, test_conformance_prediction: Dict[str, List[Any]], conformal_thresholds: Dict[str, Any]):
        # list of conformance checking evaluated test cases: contains case_id, target_conformance, ml_conformance, samples_conformance
        self.test_conformance_prediction = test_conformance_prediction

        # Keep the same keys as before; will raise KeyError if missing (same behaviour as original)
        self.agg_method: str = conformal_thresholds['agg_method']

        # Empirical quantile thresholds
        self.q_risk: float = float(conformal_thresholds['q_risk'])
        self.q_highrisk: float = float(conformal_thresholds['q_highrisk'])

        # Conformal bound out of conformal prediction to get intervals around empirical quantiles to ensure coverage
        self.conformal_bound: float = float(conformal_thresholds['conformal_bound'])
        
        # Conformal risk control threshold
        self.crc_thresh: float = float(conformal_thresholds['crc_threshold_risk']['t_hat'])

    def __aggregate_samples_fitness(self, samples_fitness: Sequence[float], aggregation: str) -> float:
        
        arr = np.asarray(samples_fitness, dtype=float)  # safe conversion from list -> ndarray
        if arr.size == 0:
            raise ValueError("samples_fitness must not be empty")

        if aggregation == 'mean':
            agg = float(np.mean(arr))
        elif aggregation == 'median':
            agg = float(np.median(arr))
        elif aggregation == 'min':
            agg = float(np.min(arr))
        elif aggregation == 'max':
            agg = float(np.max(arr))
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

        return agg

    def _append_case(self, dst: Dict[str, List[Any]], case_id: Any, tgt_conf: Any, ml_conf: Any, smpl_conf: Any) -> None:
        
        dst['case_id'].append(case_id)
        dst['target_conformance'].append(tgt_conf)
        dst['ml_conformance'].append(ml_conf)
        dst['samples_conformance'].append(smpl_conf)

    def group_samples_conformance(self, method: str) -> Dict[str, Dict[str, List[Any]]]:
        
        # Sets to store the results:
        save_set = {'case_id': [], 'target_conformance': [], 'ml_conformance': [], 'samples_conformance': []}
        risk_set = {'case_id': [], 'target_conformance': [], 'ml_conformance': [], 'samples_conformance': []}
        highrisk_set = {'case_id': [], 'target_conformance': [], 'ml_conformance': [], 'samples_conformance': []}

        # Test set conformance results (assume lists exist as in original code)
        case_ids = self.test_conformance_prediction['case_id']
        target_confs = self.test_conformance_prediction['target_conformance']
        ml_confs = self.test_conformance_prediction['ml_conformance']
        smpl_confs = self.test_conformance_prediction['samples_conformance']

        # 1) only empirical quantile
        if method == 'emp':
            for case_id, tgt_conf, ml_conf, smpl_conf in zip(case_ids, target_confs, ml_confs, smpl_confs):
                
                agg_smpl_fit = self.__aggregate_samples_fitness(samples_fitness=[smp['fitness'] for smp in smpl_conf],aggregation=self.agg_method)
                
                # [0, highrisk_thresh]
                if agg_smpl_fit >= 0 and agg_smpl_fit <= self.q_highrisk:
                    self._append_case(highrisk_set, case_id, tgt_conf, ml_conf, smpl_conf)
                
                # (highrisk_thresh, risk_thresh]
                elif agg_smpl_fit > self.q_highrisk and agg_smpl_fit <= self.q_risk:
                    self._append_case(risk_set, case_id, tgt_conf, ml_conf, smpl_conf)
                
                # (risk_thresh, 1]
                else:
                    self._append_case(save_set, case_id, tgt_conf, ml_conf, smpl_conf)

        # 2) conformal intervals: empirical quantile +- conformal bound
        elif method == 'conf':
            for case_id, tgt_conf, ml_conf, smpl_conf in zip(case_ids, target_confs, ml_confs, smpl_confs):
                
                agg_smpl_fit = self.__aggregate_samples_fitness(samples_fitness=[smp['fitness'] for smp in smpl_conf],aggregation=self.agg_method)

                agg_smpl_fit_lower = agg_smpl_fit - self.conformal_bound
                agg_smpl_fit_upper = agg_smpl_fit + self.conformal_bound

                # High risk set: If upper bound is smaller equal high risk threshold -> high risk set
                if agg_smpl_fit_upper <= self.q_highrisk:
                    self._append_case(highrisk_set, case_id, tgt_conf, ml_conf, smpl_conf)

                # High risk and risk set:
                elif agg_smpl_fit_upper <= self.q_risk and agg_smpl_fit_lower <= self.q_highrisk:
                    # self._append_case(highrisk_set, case_id, tgt_conf, ml_conf, smpl_conf)
                    # self._append_case(risk_set, case_id, tgt_conf, ml_conf, smpl_conf)
                    pass

                # Risk set:
                elif agg_smpl_fit_lower > self.q_highrisk and agg_smpl_fit_upper <= self.q_risk:
                    self._append_case(risk_set, case_id, tgt_conf, ml_conf, smpl_conf)

                # Risk and save set:
                elif agg_smpl_fit_lower <= self.q_risk:
                    # self._append_case(risk_set, case_id, tgt_conf, ml_conf, smpl_conf)
                    # self._append_case(save_set, case_id, tgt_conf, ml_conf, smpl_conf)
                    pass
                else:
                    self._append_case(save_set, case_id, tgt_conf, ml_conf, smpl_conf)

        # 3) conformal risk control threshold
        elif method == 'crc':
            for case_id, tgt_conf, ml_conf, smpl_conf in zip(case_ids, target_confs, ml_confs, smpl_confs):
                
                agg_smpl_fit = self.__aggregate_samples_fitness(samples_fitness=[smp['fitness'] for smp in smpl_conf],aggregation=self.agg_method)

                if self.crc_thresh > self.q_highrisk:
                    if agg_smpl_fit <= self.q_highrisk:
                        self._append_case(highrisk_set, case_id, tgt_conf, ml_conf, smpl_conf)

                    elif agg_smpl_fit <= self.crc_thresh:
                        self._append_case(risk_set, case_id, tgt_conf, ml_conf, smpl_conf)

                    else:
                        self._append_case(save_set, case_id, tgt_conf, ml_conf, smpl_conf)
                # if crc_thresh > q_highrisk: original code did nothing; preserve that behaviour
                else:
                    raise ValueError("CRC does not work for that dataset and/or credentials!")

        else:
            raise ValueError("Method does not exist")

        return save_set, risk_set, highrisk_set 
"""

class RiskControlledGrouping:
    def __init__(self, test_conformance_prediction: Dict[str, List[Any]], logistic_regression_model: LogisticRegressionModel):
        # list of conformance checking evaluated test cases: contains case_id, target_conformance, ml_conformance, samples_conformance
        self.test_conformance_prediction = test_conformance_prediction
        
        # Logistic regression model to predict risk
        self.logistic_regression_model = logistic_regression_model
        

    def group_risk_and_safe_set(self, risk_threshold: float) -> Dict[str, Dict[str, List[Any]]]:
        # Sets to store the results:
        risk_set = {'case_id': [], 'target_conformance': [], 'ml_conformance': [], 'samples_conformance': []}
        safe_set = {'case_id': [], 'target_conformance': [], 'ml_conformance': [], 'samples_conformance': []}

        # Test set conformance results (assume lists exist as in original code)
        case_ids = self.test_conformance_prediction['case_id']
        
        target_confs = self.test_conformance_prediction['target_conformance']
        # Get target fitnes: 
        target_confs_fitness = [tgt['fitness'] for tgt in target_confs]
        
        ml_confs = self.test_conformance_prediction['ml_conformance']
        # Get ml fitness:
        ml_confs_fitness = [ml['fitness'] for ml in ml_confs]

        smpl_confs = self.test_conformance_prediction['samples_conformance']
        # Get samples fitness:
        smpl_confs_fitness = [[smp['fitness'] for smp in smpls] for smpls in smpl_confs]

        test_set_fitness_score_results = {
            'target_fitness': target_confs_fitness,
            'ml_fitness': ml_confs_fitness,
            'samples_fitness': smpl_confs_fitness
        }

        # Extract features from ml_confs for logistic regression model
        df_init = DataFrameConstruction(fitness_score_results=test_set_fitness_score_results)
        
        # Get feature vectors for logistic regression model
        test_def = df_init.samples_to_dataframe(threshold=risk_threshold)
        
        self.logistic_regression_model.predict_with_threshold(sel.logistic_regression_model.crc_info.get("tau", 0.5))
        

        # Predict risk using the logistic regression model
        predictions = self.logistic_regression_model.classifier.predict(feature_vectors)

        for case_id, tgt_conf, ml_conf, smpl_conf, pred in zip(case_ids, target_confs, ml_confs, smpl_confs, predictions):
            if pred == 1:  # Assuming '1' indicates risk
                risk_set['case_id'].append(case_id)
                risk_set['target_conformance'].append(tgt_conf)
                risk_set['ml_conformance'].append(ml_conf)
                risk_set['samples_conformance'].append(smpl_conf)
            else:
                safe_set['case_id'].append(case_id)
                safe_set['target_conformance'].append(tgt_conf)
                safe_set['ml_conformance'].append(ml_conf)
                safe_set['samples_conformance'].append(smpl_conf)

        return safe_set, risk_set










class ConformalDeviationPrediction:
    def __init__(self, test_conformance_prediction_group: Dict[str, List[Any]]):
        # list of dicts containing: target case, alignment, fitness, cost
        self.conformance_pred_group_target = [tgt for tgt in test_conformance_prediction_group['target_conformance']]
        
        # list of 1000 list of dicts containing: target case, alignment, fitness, cost
        self.conformance_pred_group_samples = [[smpl for smpl in smpls] for smpls in test_conformance_prediction_group['samples_conformance']]
        
    def get_target_alignments(self) -> List[Any]:
        """
        Returns the list of target alignments.
        """
        
        return [tgt['alignment'] for tgt in self.conformance_pred_group_target]
    
    def get_predicted_alignments_with_median_fitness(self) -> List[List[Any]]:
        """
        Returns the list of predicted alignments (list of lists) for smpls with median fitness score.
        """
        
        predicted_alignments = []
        for smpls in self.conformance_pred_group_samples:
            if smpls:
                median_fitness = np.median([smpl['fitness'] for smpl in smpls])
                median_alignments = [smpl['alignment'] for smpl in smpls if smpl['fitness'] == median_fitness]
                predicted_alignments.append(median_alignments)
            else:
                predicted_alignments.append([])
        return predicted_alignments


