import numpy as np
from typing import Any, Dict, List, Sequence

class ConformalDeviationPrediction:
    def __init__(self, test_conformance_prediction: Dict[str, List[Any]], conformal_thresholds: Dict[str, Any]):
        self.test_conformance_prediction = test_conformance_prediction

        # Keep the same keys as before; will raise KeyError if missing (same behaviour as original)
        self.agg_method: str = conformal_thresholds['agg_method']

        self.q_risk: float = float(conformal_thresholds['q_risk'])
        self.q_highrisk: float = float(conformal_thresholds['q_highrisk'])

        self.conformal_lbound: float = float(conformal_thresholds['conformal_bound'])

        self.crc_thresh: float = float(conformal_thresholds['crc_threshold_risk']['t_hat'])

    def __aggregate_samples_fitness(self, samples_fitness: Sequence[float], aggregation: str) -> float:
        """
        Helper method to aggregate the samples.
        Accepts a sequence or numpy array and returns a float aggregation.
        """
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
        """Small helper to append case data to one of the sets."""
        dst['case_id'].append(case_id)
        dst['target_conformance'].append(tgt_conf)
        dst['ml_conformance'].append(ml_conf)
        dst['samples_conformance'].append(smpl_conf)

    def group_samples_conformance(self, method: str) -> Dict[str, Dict[str, List[Any]]]:
        """
        Group test cases into save / risk / highrisk according to the requested method.
        Supported methods (preserve original logic): 'emp', 'conf', 'crc'.
        Returns a dict with keys: 'save', 'risk', 'highrisk' each mapping to a dict with lists.
        """
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

                agg_smpl_fit_lower = agg_smpl_fit - self.conformal_lbound
                agg_smpl_fit_upper = agg_smpl_fit + self.conformal_lbound

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
