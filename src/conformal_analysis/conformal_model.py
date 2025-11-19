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

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
               
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
    
class ConformalAnalysisVisualizations:
    def __init__(self, sampled_fitness, target_fitness, ml_fitness):
        self.samples_fitness = sampled_fitness
        self.target_fitness = target_fitness
        self.ml_fitness = ml_fitness

    def __aggregate_samples_fitness(self, samples_fitness, aggregation: str) -> np.ndarray:
        """
        Aggregate each element in samples_fitness using aggregation.
        Accepts:
         - list/tuple of array-like (each inner element is an array of samples)
         - 2D numpy array shaped (n, m) -> aggregates across axis=1
         - 1D array (assumed already aggregated) -> returned as-is (cast to float array)
        """
        if samples_fitness is None:
            raise ValueError("samples_fitness is None")

        # If it's a numpy array and 1D numeric, assume already aggregated
        arr = np.asarray(samples_fitness, dtype=object)

        # helper map
        agg_funcs = {
            'mean': np.mean,
            'median': np.median,
            'min': np.min,
            'max': np.max
        }
        if aggregation not in agg_funcs:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        agg_f = agg_funcs[aggregation]

        # If arr is object dtype but each element is array-like -> iterate
        if arr.dtype == object or arr.ndim == 1 and any(hasattr(x, '__iter__') for x in arr):
            out = []
            for x in arr:
                x_a = np.asarray(x)
                if x_a.size == 0:
                    raise ValueError("no calibration value exist!")
                out.append(float(agg_f(x_a)))
            return np.array(out, dtype=float)

        # If it's a 2D numeric array, aggregate per-row
        arr_num = np.asarray(samples_fitness, dtype=float)
        if arr_num.ndim == 2:
            if arr_num.shape[1] == 0:
                raise ValueError("no calibration value exist!")
            if aggregation == 'mean':
                return np.nanmean(arr_num, axis=1)
            elif aggregation == 'median':
                return np.nanmedian(arr_num, axis=1)
            elif aggregation == 'min':
                return np.min(arr_num, axis=1)
            elif aggregation == 'max':
                return np.max(arr_num, axis=1)

        # If it's 1D numeric, return it (ensure float dtype)
        if arr_num.ndim == 1:
            if arr_num.size == 0:
                raise ValueError("no calibration value exist!")
            return arr_num.astype(float)

        raise ValueError("Unsupported shape for samples_fitness")
    
    def plot_distribution(self,
                          aggregation: Optional[str] = None,
                          bins=30,
                          show_kde=True,
                          alpha_risk: float = 1.0):
        """
        1) Plot distribution of target fitness and of aggregated sample fitness and (optional).
        2) Plot vertical lines for risk fitness score threshold based on alpha level.
        """
        # aggregate samples
        smpls_fit = None
        if aggregation is not None:
            smpls_fit = self.__aggregate_samples_fitness(samples_fitness=self.samples_fitness, aggregation=aggregation)

        # handle target_fitness
        target_fit = None
        if self.target_fitness is not None:
            target_fit = np.asarray(self.target_fitness, dtype=float).flatten()
            if target_fit.size == 0:
                target_fit = None

        # Print summary statistics (safe casting to float for formatting)
        def print_stats(name, arr):
            if arr is None:
                print(f"{name}: None")
                return
            arr = np.asarray(arr, dtype=float).flatten()
            n = arr.size
            mean = float(np.mean(arr)) if n > 0 else float('nan')
            median = float(np.median(arr)) if n > 0 else float('nan')
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            if n > 1:
                q1, q3 = np.percentile(arr, [25, 75])
            elif n == 1:
                q1 = q3 = float(arr[0])
            else:
                q1 = q3 = float('nan')
            print(f"{name}: n={n}, mean={mean:.4f}, median={median:.4f}, std={std:.4f}, Q1={q1:.4f}, Q3={q3:.4f}")

        if aggregation is not None:
            print_stats("Aggregated samples fitness statistics", smpls_fit)
        print_stats("Target fitness statistics", target_fit)

        # compute requested empirical quantiles (on aggregated samples)
        q_risk = None
 
        # alpha risk
        if not (0.0 <= alpha_risk <= 1.0):
            raise ValueError("alpha_risk must be in [0,1]")
        
        q_risk = float(np.quantile(target_fit, alpha_risk))
        print("Risk threshold fitness score: ",q_risk)

        # Histogram (density) overlay
        plt.figure(figsize=(9, 5))
        if smpls_fit is not None:
            plt.hist(smpls_fit, bins=bins, density=True, alpha=0.6, edgecolor='black', linewidth=0.5, label='aggregated samples fitness', color='blue')
        if target_fit is not None:
            plt.hist(target_fit, bins=bins, density=True, alpha=0.5, edgecolor='black', linewidth=0.5, label='target fitness', color='green')

        # KDE overlays (only if >1 sample)
        if show_kde:
            try:
                valid_xmin, valid_xmax = float('inf'), float('-inf')

                if smpls_fit is not None and smpls_fit.size > 1:
                    valid_xmin = min(valid_xmin, float(smpls_fit.min()))
                    valid_xmax = max(valid_xmax, float(smpls_fit.max()))

                if target_fit is not None and target_fit.size > 1:
                    valid_xmin = min(valid_xmin, float(target_fit.min()))
                    valid_xmax = max(valid_xmax, float(target_fit.max()))

                if valid_xmin < valid_xmax:  # Ensure valid range for KDE
                    x = np.linspace(valid_xmin, valid_xmax, 400)

                if smpls_fit is not None and smpls_fit.size > 1:
                    kde_means = gaussian_kde(smpls_fit)
                    plt.plot(x, kde_means(x), lw=2, label='KDE (samples)', color='blue')

                if target_fit is not None and target_fit.size > 1:
                    kde_target = gaussian_kde(target_fit)
                    plt.plot(x, kde_target(x), lw=2, label='KDE (target)', color='green')

            except Exception as e:
                print(f"Error in KDE computation: {e}")
                # Silently skip KDE if it fails (e.g., scipy missing or KDE error)
                pass

        # draw quantile vertical lines + annotate their numeric values on the x-axis
        ax = plt.gca()
        ylim = ax.get_ylim()
        # small offset above bottom
        y_text_pos = ylim[0] + 0.03 * (ylim[1] - ylim[0])

        if q_risk is not None:
            ax.axvline(q_risk, color='red', linestyle='--', linewidth=2, label=f'risk fitness threshold (alpha-risk quantile={alpha_risk})')
            ax.text(q_risk, y_text_pos, f"{q_risk:.3f}", color='red', ha='center', va='bottom', fontsize=9, backgroundcolor='white')

        plt.xlabel('(aggregated) fitness score')
        plt.ylabel('density')
        if aggregation is not None:
            plt.title(f'Distribution of aggregated samples and target fitness scores')
        else:
            plt.title('Distribution of target fitness scores')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.show()
        
        return q_risk

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