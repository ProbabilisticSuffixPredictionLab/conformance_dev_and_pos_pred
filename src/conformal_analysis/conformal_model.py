from __future__ import annotations
import math
from typing import Optional, List, Union, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
               
class DataFrameConstruction:
    def __init__(self, conformance_results: dict):
        self.res_target_conf = conformance_results['target_conformance']
        # self.res_target_conf_fit = [res['fitness'] for res in self.res_target_conf]
        self.res_target_conf_suffix_fit = [res['suffix_fitness'] for res in self.res_target_conf]

        self.res_ml_conf = conformance_results['ml_conformance']
        # self.res_ml_conf_fit =  [res['fitness'] for res in self.res_ml_conf]
        self.res_ml_conf_suffix_fit = [res['suffix_fitness'] for res in self.res_ml_conf]

        self.res_smpl_conf = conformance_results['samples_conformance']
        # self.res_smpl_conf_fit = [[r['fitness'] for r in res] for res in self.res_smpl_conf]
        self.res_smpl_conf_suffix_fit = [[r['suffix_fitness'] for r in res] for res in self.res_smpl_conf]
        
    def __aggregate_samples_fitness(self, samples_fitness: np.ndarray, aggregation: str) -> float:
        """
        Helper method to aggregate the samples using various moment metrics.
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
        elif aggregation == 'variance':
            agg = float(np.var(samples_fitness, ddof=1))  # sample variance
        elif aggregation == 'std':
            agg = float(np.std(samples_fitness, ddof=1))  # sample standard deviation
        elif aggregation == 'skewness':
            agg = float((np.mean((samples_fitness - np.mean(samples_fitness))**3)) / (np.std(samples_fitness, ddof=1)**3))
        elif aggregation == 'kurtosis':
            agg = float((np.mean((samples_fitness - np.mean(samples_fitness))**4)) / (np.std(samples_fitness, ddof=1)**4) - 3)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        
        # sample standard deviation (ddof=1 if >1 sample, else 0)
        std = float(np.std(samples_fitness, ddof=1) if samples_fitness.size > 1 else 0.0)

        return (agg, std)
        
    def __value_at_quantiles(self, values: list, alpha_risk: float) -> Dict[str, Any]:
        """
        Given an unsorted list of floats, return the lower-tail empirical values at q_risk and q_highrisk.
        """
        sorted_vals = sorted(values)  # ascending, smallest is the worst fitness
        n = len(sorted_vals)

        if n <= 0:
            return -1
        
        k_risk = math.floor((n + 1) * alpha_risk)
        idx_risk = k_risk - 1
        idx_risk = min(max(idx_risk, 0), n - 1)
        q_risk = sorted_vals[idx_risk] if idx_risk != -1 else None
        
        return {'q_risk': q_risk}
    
    def empirical_quantile_thresholds(self, alpha_risk: float, aggregation: str='mean') -> dict:
        """
        Compute one-sided lower-tail empirical thresholds for q_risk and q_highrisk.
        """
        
        # Extend the dict with high-risk if needed
        
        # Target (suffix)
        target_fitness_scores = self.res_target_conf_suffix_fit
        # Get thresholds
        thresholds_target = self.__value_at_quantiles(target_fitness_scores, alpha_risk)
            
        # Most likely (suffix)
        ml_fitness_scores = self.res_ml_conf_suffix_fit
        # Get thresholds
        thresholds_ml = self.__value_at_quantiles(ml_fitness_scores, alpha_risk)
            
        # Samples
        sampled_fitness_scores = self.res_smpl_conf_suffix_fit
        
        # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
        aggragted_sampled_fitness_scores = [self.__aggregate_samples_fitness(samples_fitness=np.array(smp), aggregation=aggregation) for smp in sampled_fitness_scores]
        # Get thresholds
        thresholds_sampled = self.__value_at_quantiles([agg_smp[0] for agg_smp in aggragted_sampled_fitness_scores], alpha_risk)
        mean_std_sampled = np.nanmean([agg_smp[1] for agg_smp in aggragted_sampled_fitness_scores])
        thresholds_sampled['mean_std'] = mean_std_sampled
            
        return {'target': thresholds_target,
                'most_likely': thresholds_ml,
                'samples': thresholds_sampled}
        
        
    def samples_to_dataframe(self, q_risk: float = 0.0, target_col: str = "y"):
        # Target suffix fitness scores
        targets = self.res_target_conf_suffix_fit
        # Predicted samples fitness scores
        predicted_samples = self.res_smpl_conf_suffix_fit

        if len(targets) != len(predicted_samples):
            raise ValueError("Length mismatch between targets and predicted_samples.")

        rows = []
        # 1000 samples
        for i, samples in enumerate(predicted_samples):
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
        df[target_col] = [1 if t >= q_risk else 0 for t in targets]
        return df

class LogisticRegressionModel:
    def __init__(self, 
                 alpha_quantile_risk: float = 0.5,
                 risk_fitness_threshold: float = 1.0,
                 classifier: Optional[LogisticRegression] = None):
        # Risk fitness threshold (for reference, not used in fitting)
        self.alpha_quantile_risk = alpha_quantile_risk
        self.risk_fitness_threshold = risk_fitness_threshold
        self.empiricial_risk_values: Dict[str, Any] = {'alpha_quantile_risk': alpha_quantile_risk, 'risk_fitness_threshold': risk_fitness_threshold}
        
        # Standard LR params:
        self.classifier = classifier if classifier is not None else LogisticRegression(max_iter=1000)
        self.pipeline = None
        self.feature_names: Optional[List[str]] = None
        self.trained = False

        # CRC info stored after calibration
        self.crc_info: Dict[str, Any] = {}
        
        self.metadata: Dict[str, Any] = {}

    def get_crc_info(self) -> Dict[str, Any]:
        return dict(self.crc_info)

    def get_crc_value(self) -> Optional[float]:
        return self.crc_info.get("tau", None)

    def get_feature_names(self) -> Optional[List[str]]:
        return self.feature_names

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.metadata)

    def _build_pipeline(self, calibrate: bool):
        scaler = StandardScaler()
        if calibrate:
            base = self.classifier
            clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            clf = self.classifier
        self.pipeline = make_pipeline(scaler, clf)

    # Logistic regression fitting
    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            feature_names: Optional[List[str]] = None,
            calibrate: bool = False,
            **fit_kwargs) -> "LogisticRegressionModel":
        """
        Fit model. If X is numpy array, feature_names must be provided.
        If X is DataFrame and feature_names provided, they must be subset of columns.
        """
        if isinstance(X, np.ndarray) and feature_names is None:
            raise ValueError("Provide feature_names when X is numpy array.")
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                self.feature_names = list(X.columns)
            else:
                # validate supplied feature_names are present in dataframe
                missing = [f for f in feature_names if f not in X.columns]
                if missing:
                    raise ValueError(f"Provided feature_names missing from X columns: {missing}")
                self.feature_names = list(feature_names)
        else:
            self.feature_names = feature_names

        self._build_pipeline(calibrate)
        # Pipeline is fit with X and y and contains scaler and classifier:
        self.pipeline.fit(X, y, **fit_kwargs)
        self.trained = True
        self.metadata.update({
            "n_features": len(self.feature_names) if self.feature_names is not None else None,
            "calibrated": bool(calibrate),
        })
        return self

    def fit_from_dataframe(self,
                           df: pd.DataFrame,
                           target_col: str = "y",
                           features: Optional[List[str]] = None,
                           calibrate: bool = False,
                           **fit_kwargs) -> "LogisticRegressionModel":
        """
        Thin wrapper: extracts X and y from df and calls fit.
        """
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not in dataframe")
        features = features or [c for c in df.columns if c != target_col]
        X = df[features]
        y = df[target_col]
        return self.fit(X, y, feature_names=features, calibrate=calibrate, **fit_kwargs)

    # Prediction
    def _ensure_feature_order(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                raise RuntimeError("Model has no stored feature names.")
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame missing features: {missing}")
            return X[self.feature_names]
        # numpy array: assume caller ensured correct column order
        return X

    # CRC threshold calibration:
    def calibrate_crc_threshold_false_positive(self,
                                            X_cal: Union[pd.DataFrame, np.ndarray],
                                            y_cal: Union[pd.Series, np.ndarray],
                                            alpha: float = 0.05,
                                            n_grid: int = 1000,
                                            ) -> Dict[str, Any]:
        """
        Calibrate the prediction probability such that the:
        false positive rate (FPR) of safe (== 1) is controlled:
        -> Bound P(pred safe | true risk) <= alpha.
        -> Most important to catch all risks, so control the conditional FPR_safe.
        
        Perform grid search over threshold in [0,1] to find the minimal threshold that
        guarantees the adjusted empirical risk <= alpha (per paper, least conservative).
        
        Parameters:
        X_cal, y_cal : calibration dataset (not used in training or probability calibration)
        alpha : max expected FPR_safe (e.g. 0.05)
        n_grid : size of threshold grid to search over [0,1]
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before CRC calibration.")
        
        # predicted probabilities: >= threshold safe (=1), < threshold risk (=0)
        # probabilities for each calibration case (P(safe))
        probs = np.asarray(self.predict_proba(X_cal)).ravel()
        
        # true targets
        y = np.asarray(y_cal).ravel()
        if len(probs) != len(y):
            raise ValueError("Length mismatch between X_cal and y_cal")
        # Filter to true risk samples for conditional risk control
        risk_mask = (y == 0)
        n_risk = np.sum(risk_mask)
        
        if n_risk == 0:
            crc_entry = {"note": "No true risk samples in calibration data; using conservative threshold",
                        "params": {"alpha": float(alpha), "n_cal": 0, "B": 1.0},
                        "lambda": 1.0,
                        "threshold": 1.0}
            self.crc_info = crc_entry
            return dict(self.crc_info)
        
        # Get all probabilities of cases that have true risk:
        probs_risk = probs[risk_mask]
        # n for conditional risk
        n_cal = n_risk  
        
        # The max loss is 1.0: B as an upper bound on the loss.
        B = 1.0
        
        # build threshold grid (lambda in paper): increasing from 0 to 1 (larger = more conservative, lower risk)
        thresholds = np.linspace(0.0, 1.0, n_grid)
        
        # compute empirical average loss R_n(threshold) for each threshold (decreasing)
        R_n = np.empty_like(thresholds, dtype=float)
        for j, t in enumerate(thresholds):
            # predict safe when prob >= t, and loss when y == 0 (false safe)
            # since filtered to y==0, just count predicted safe
            k_fpr_safe = np.sum(probs_risk >= t)
            R_n[j] = k_fpr_safe / n_cal
        
        # paper adjusted quantity: (n/(n+1)) * R_n + B/(n+1)
        adj = (n_cal / (n_cal + 1.0)) * R_n + (B / (n_cal + 1.0))
        
        # Pick the infimum lambda (smallest threshold) where adj <= alpha: this is the first index j where adj[j] <= alpha (least conservative satisfying the bound)
        feasible_idx = np.where(adj <= alpha)[0]
        if feasible_idx.size > 0:
            j0 = int(feasible_idx[0])
            lambda_hat = float(thresholds[j0])
            # lambda is the threshold here
            t_hat = lambda_hat  
            crc_entry = {"note": None,
                        "params": {"alpha": float(alpha), "n_cal": int(n_cal), "B": float(B)},
                        "lambda": lambda_hat,
                        "threshold": t_hat}
        else:
            # if no lambda satisfies, set lambda_hat = lambda_max (most conservative)
            lambda_hat = float(thresholds[-1])
            t_hat = lambda_hat
            crc_entry = {"note": "no lambda satisfied inequality; using lambda_max per paper",
                        "params": {"alpha": float(alpha), "n_cal": int(n_cal), "B": float(B)},
                        "lambda": lambda_hat,
                        "threshold": t_hat}
        
        self.crc_info = crc_entry
        
        return dict(self.crc_info)
    
    def calibrate_threshold_for_f1(self,
                               X_cal: Union[pd.DataFrame, np.ndarray],
                               y_cal: Union[pd.Series, np.ndarray],
                               n_grid: int = 1000,
                               pos_label: int = 1) -> Dict[str, Any]:
        """
        Calibrate the threshold to maximize the F1 score on the calibration set.
        Assumes binary classification with classes 0 (risk) and 1 (safe).
        Parameters:
        X_cal, y_cal : calibration dataset
        n_grid : size of threshold grid to search over [0,1]
        pos_label : the positive class for F1 computation (default 1 for safe)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before calibration.")
        
        # Probabilities for the safe class (assuming predict_proba returns P(safe))
        probs = np.asarray(self.predict_proba(X_cal)).ravel()
        y = np.asarray(y_cal).ravel()
        if len(probs) != len(y):
            raise ValueError("Length mismatch between X_cal and y_cal")
        n_cal = len(y)
        
        # Build threshold grid
        thresholds = np.linspace(0.0, 1.0, n_grid)
        
        # Compute F1 for each threshold
        f1_scores = np.empty_like(thresholds, dtype=float)
        for j, t in enumerate(thresholds):
            preds = (probs >= t).astype(int)  # 1 if >= t (safe), 0 otherwise
            if np.sum(preds == pos_label) == 0:  # Avoid division by zero
                f1_scores[j] = 0.0
            else:
                precision = np.sum((preds == pos_label) & (y == pos_label)) / np.sum(preds == pos_label)
                recall = np.sum((preds == pos_label) & (y == pos_label)) / np.sum(y == pos_label) if np.sum(y == pos_label) > 0 else 0.0
                f1_scores[j] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Pick the threshold with max F1
        max_idx = np.argmax(f1_scores)
        t_hat = float(thresholds[max_idx])
        f1_hat = float(f1_scores[max_idx])
        cal_entry = {"note": "Threshold selected to maximize empirical F1 on calibration set",
                    "params": {"n_cal": int(n_cal), "pos_label": int(pos_label)},
                    "threshold": t_hat,
                    "f1_score": f1_hat}
        
        self.crc_info = cal_entry  # Or store in a new attribute
        
        return dict(self.crc_info)
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        X_in = self._ensure_feature_order(X)
        return self.pipeline.predict(X_in)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Returns probability for the *safe* class (label == 1).
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")
        X_in = self._ensure_feature_order(X)
        proba = self.pipeline.predict_proba(X_in)
        # assume binary: [:,1] is P(y=1)
        return np.asarray(proba)[:, 1]
    
    def predict_without_threshold(self,
                                  X: Union[pd.DataFrame, np.ndarray]
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels.
        Returns labels (and optionally probabilities).
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")

        # Predicted probabilities: Calls the fitted pipeline and predict proabilities.
        probs = self.predict_proba(X)
        
        labels = np.zeros_like(probs, dtype=int)
        # when using threshold: label is 1 if prob >= t_hat -> safe (1), else risk (0)
        labels[probs >= 0.5] = 1
        
        # Return labels and probabilities
        return (labels, probs)
        

    # Predict with CRC threshold
    def predict_with_threshold(self,
                               X: Union[pd.DataFrame, np.ndarray]
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict labels using stored CRC threshold.
        Returns labels (and optionally probabilities).
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")
        if not self.crc_info:
            raise RuntimeError("No CRC info found. Call calibrate_crc_marginal_false_positive first.")

        # Predicted probabilities: Calls the fitted pipeline and predict proabilities.
        probs = self.predict_proba(X)
        # CRC threshold
        t_hat = self.crc_info.get("threshold", None)
        
        labels = np.zeros_like(probs, dtype=int)
        # when using threshold: label is 1 if prob >= t_hat -> safe (1), else risk (0)
        labels[probs >= t_hat] = 1
        
        # Return labels and probabilities
        return (labels, probs)

    # Save the trained logistic regression model
    def save(self, path: Union[str, Path]):
        if not self.trained:
            raise RuntimeError("Train model before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {# Store risk fitness threshold
                   "empiricial_risk_values": {"alpha_quantile_risk": self.alpha_quantile_risk, "risk_fitness_threshold": self.risk_fitness_threshold},
                   # Pipeline stores the scaler and classifier:
                   "pipeline": self.pipeline,
                   # feature names to ensure correct order during prediction
                   "feature_names": self.feature_names,
                   "metadata": self.metadata,
                   # CRC information for conformal prediction
                   "crc_info": self.crc_info
               }
        joblib.dump(payload, path)
        return str(path)
    
    # Load a trained logistic regression model
    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogisticRegressionModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        lm = cls()
        lm.empiricial_risk_values = payload.get("empiricial_risk_values", {})
        lm.pipeline = payload.get("pipeline")
        lm.feature_names = payload.get("feature_names")
        lm.metadata = payload.get("metadata", {})
        lm.crc_info = payload.get("crc_info", {})
        lm.trained = True
        return lm