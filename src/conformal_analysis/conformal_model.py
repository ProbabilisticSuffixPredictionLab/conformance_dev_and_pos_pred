
from __future__ import annotations
import math
from typing import Any, Dict
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
import joblib
from scipy.stats import beta

# Check if scipy is available
try:
    import scipy
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

class ConformalAnalysisModel:
    def __init__(self, fitness_score_results: dict):
        """
        fitness_score_results : dict: Dict of fitness score results: 
            - case_id: list of (case_name, prefix_length),
            - target_fitness: list of fitness scores of target, per case,
            - ml_fitness: list of fitness score of most-likely, per case,
            - samples_fitness: list of fitness scores of samples (T=1000), per case
        """
        self.fitness_score_results = fitness_score_results
        
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
        
    def __value_at_quantiles(self, values: list, q_risk: float, q_highrisk: float) -> Dict[str, Any]:
        """
        Given an unsorted list of floats, return the lower-tail empirical values at q_risk and q_highrisk.
        """
        sorted_vals = sorted(values)  # ascending, smallest is the worst fitness
        n = len(sorted_vals)

        if n <= 0:
            return -1
        
        k_risk = math.floor((n + 1) * q_risk)
        idx_risk = k_risk - 1
        idx_risk = min(max(idx_risk, 0), n - 1)
        val_risk = sorted_vals[idx_risk] if idx_risk != -1 else None
        
        k_highrisk = math.floor((n + 1) * q_highrisk)
        idx_highrisk = k_highrisk - 1
        idx_highrisk = min(max(idx_highrisk, 0), n - 1)
        val_highrisk = sorted_vals[idx_highrisk] if idx_highrisk != -1 else None
        
        return {'q_risk': val_risk, 'q_highrisk': val_highrisk}
    
    def empirical_quantile_thresholds(self, q_risk: float, q_highrisk: float, aggregation: str='mean') -> dict:
        """
        Compute one-sided lower-tail empirical thresholds for q_risk and q_highrisk.
        """
        # Target
        target_fitness_scores = self.fitness_score_results['target_fitness']
        # Get thresholds
        thresholds_target = self.__value_at_quantiles(target_fitness_scores, q_risk, q_highrisk)
            
        # Most likely
        ml_fitness_scores = self.fitness_score_results['ml_fitness']
        # Get thresholds
        thresholds_ml = self.__value_at_quantiles(ml_fitness_scores, q_risk, q_highrisk)
            
        # Samples
        sampled_fitness_scores = self.fitness_score_results['samples_fitness']
        # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
        aggragted_sampled_fitness_scores = [self.__aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
        # Get thresholds
        thresholds_sampled = self.__value_at_quantiles([agg_smp[0] for agg_smp in aggragted_sampled_fitness_scores], q_risk, q_highrisk)
        mean_std_sampled = np.nanmean([agg_smp[1] for agg_smp in aggragted_sampled_fitness_scores])
        thresholds_sampled['mean_std'] = mean_std_sampled
            
        return {'target': thresholds_target,
                'most_likely': thresholds_ml,
                'samples': thresholds_sampled}
                     
    def conformal_bound(self, aggregation: str='mean', alpha: float=0.1):
        """
        Ensure per sample guarantee by using conformal prediction on the fitness residuals.
        """ 
        # Helper to compute Q for given arrays of target and aggregated_sampled (both numpy arrays)
        def _compute_Q_for_arrays(target: np.ndarray, aggregated: np.ndarray) -> float:
            """
            abs conformal residual 1-alpha bound computation
            """
            if target.shape[0] != aggregated.shape[0]:
                raise ValueError(f"Length mismatch in target and aggregated sample lists.")
            n = target.shape[0]
            if n == 0:
                return float('nan')
            
            # Compute residuals
            residuals = np.abs(target - aggregated)  # in [0,1]
            sorted_res = np.sort(residuals)
            
            # finite-sample conformal index (1-based)
            k = math.ceil((1.0 - alpha) * (n + 1))
            
            # cap to [1, n]
            k = max(1, min(k, n))
            idx = k - 1  # 0-based
            Q_val = float(sorted_res[idx])
            
            # clamp to [0,1] for safety
            Q_val = min(max(Q_val, 0.0), 1.0)
            
            return Q_val
        
        target_fitness_scores = np.asarray(self.fitness_score_results['target_fitness'])
            
        sampled_fitness_scores = self.fitness_score_results['samples_fitness']    
        # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
        aggregated_sampled_pairs = [self.__aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
        aggregated_sampled_fitness = np.asarray([smp[0] for smp in aggregated_sampled_pairs])
        # mean_std_sampled = float(np.nanmean([pair[1] for pair in aggregated_sampled_pairs])) if len(aggregated_sampled_pairs) > 0 else float('nan')
            
        Q = _compute_Q_for_arrays(target=target_fitness_scores, aggregated=aggregated_sampled_fitness)
        
        return Q
            
    def risk_controlled_threshold(self, t_max, delta, alpha, min_count, eps=1e-4, B=1.0, aggregation: str='mean'):
        """
        Calibrate the empirical thresholds for q_risk and q_high_risk and q_save and deliver a guarentee that the global predicted fitness and the target fitness deviates only by delta with
        a 1-alpha finite sample guarentee using Conformal Risk Control.
        """
        target_fitness_scores = np.asarray(self.fitness_score_results['target_fitness'])
            
        sampled_fitness_scores = self.fitness_score_results['samples_fitness']
        # Aggreagate the fitness samples (per case): Add tuples (aggregated, std)
        aggregated_sampled_pairs = [self.__aggregate_samples_fitness(samples_fitness=smp, aggregation=aggregation) for smp in sampled_fitness_scores]
        aggregated_sampled_fitness = np.asarray([smp[0] for smp in aggregated_sampled_pairs])
        
        # Guarantee same length of fitness score lists
        assert target_fitness_scores.shape[0] == aggregated_sampled_fitness.shape[0]
        
        # Start with lambda zer0
        lam = 0.0
        # Iterate through all possible lambdas:
        while lam <= t_max:
            # update t
            t = t_max - lam
            # check all samples that are samller the threshold -> f elem B
            assigned = (aggregated_sampled_fitness <= t)
            
            count = assigned.sum()
            if count < min_count: 
                return False, {'count': count}
            
            # check all samples that have predicted fitness different to the target fitness by at least delta
            exceed = (np.abs(target_fitness_scores - aggregated_sampled_fitness) > delta).astype(float)
            
            # L' = (I_assigned & I_exceed) - alpha
            loss = (assigned.astype(float) * exceed) - alpha   
            
            # Mean L' over all validation samples.
            Rn_expected_loss = loss.mean()
            corrected = (len(aggregated_sampled_fitness)/(len(aggregated_sampled_fitness)+1.0))*Rn_expected_loss + (B/(len(aggregated_sampled_fitness)+1.0))
            
            if corrected < 0.0:
                return {'lambda_hat': lam, 't_hat': (t_max -lam)}
            
            lam = lam + eps
        return None
    

class DataFrameConstruction:
    def __init__(self, fitness_score_results: dict):
        """
        fitness_score_results : dict: Dict of fitness score results: 
            - case_id: list of (case_name, prefix_length),
            - target_fitness: list of fitness scores of target, per case,
            - ml_fitness: list of fitness score of most-likely, per case,
            - samples_fitness: list of fitness scores of samples (T=1000), per case
        """
        self.fitness_score_results = fitness_score_results
        
    def samples_to_dataframe(self, threshold:float=0):
        """
        Build a pandas DataFrame of distributional features from predicted samples.
        
        Args:
        targets : sequence-like, length N Float target fitness values in [0,1].
        predicted_samples : sequence-like, length N Each element is an array-like of sample floats (e.g., 1000 samples per case).
        
        threshold : Threshold to determine if sample is assigned as risk (1) or safe (0) float, default=0: Every sample is safe (0)
        
        Label rule: If target > threshold -> save: 0, else: -> risk: 1 

        Returns
        pd.DataFrame
            - Columns: ['mean','variance','std','skewness','kurtosis_excess', 'median','min','max','q25','q75','iqr','y']
        """
        targets = self.target_fitness_results = self.fitness_score_results['target_fitness']
        
        predicted_samples = self.fitness_score_results['samples_fitness']
        
        if len(targets) != len(predicted_samples):    
            raise ValueError("Length mismatch: targets and predicted_samples must have same length.")

        rows = []
        for i, samples in enumerate(predicted_samples):
            arr = np.asarray(samples, dtype=float)
            if arr.size == 0:
                raise ValueError(f"Empty sample array at index {i}.")

            mean = float(arr.mean())
            median = float(np.median(arr))
            
            # population variance
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

        columns = ['mean','variance','std','skewness','kurtosis_excess', 'median','min','max','q25','q75','iqr']
        df = pd.DataFrame(rows, columns=columns)

        # Label: 0 if target <= threshold, else 1
        df['y'] = [0 if t > threshold else 1 for t in targets]

        return df

class LogisticRegressionModel:
    def __init__(self, classifier: Optional[LogisticRegression] = None):
        self.classifier = classifier if classifier is not None else LogisticRegression(max_iter=1000)
        self.pipeline = None
        self.feature_names: Optional[List[str]] = None
        self.trained = False
        # For storing CRC calibration info
        self.crc_info: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        
    # small helpers
    def get_crc_info(self) -> Dict[str, Any]:
        """
        CRC info stored here after calibrate_crc_marginal_false_positive:
        {
           "tau": float or None,
           "method": "hoeffding"|"clopper",
           "alpha": float,
           "delta": float,
           "n_cal": int,
           "hoeffding_penalty": float (if method hoeffding),
           "randomized": bool,
           "rand_info": { "tau_lo": , "tau_hi": , "p_mix": , "L_lo": , "L_hi": }  # if randomized
        }
        """
        return dict(self.crc_info)
    
    def get_crc_value(self) -> float:
        return self.crc_info.get("tau", None)

    def get_feature_names(self) -> Optional[List[str]]:
        return self.feature_names

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.metadata)

    def _build_pipeline(self, calibrate: bool):
        scaler = StandardScaler()
        if calibrate:
            base = self.classifier
            clf = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        else:
            clf = self.classifier
        self.pipeline = make_pipeline(scaler, clf)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray], feature_names: Optional[List[str]] = None, calibrate: bool = False, **fit_kwargs) -> "LogisticModel":
        if isinstance(X, np.ndarray) and feature_names is None:
            raise ValueError("Provide feature_names when X is numpy array.")
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns) if feature_names is None else feature_names
        else:
            self.feature_names = feature_names

        self._build_pipeline(calibrate)
        self.pipeline.fit(X, y, **fit_kwargs)
        self.trained = True
        self.metadata.update({
            "n_features": len(self.feature_names) if self.feature_names else None,
            "calibrated": bool(calibrate),
        })
        return self

    def fit_from_dataframe(self, df: pd.DataFrame, target_col: str = "y", features: Optional[List[str]] = None, calibrate: bool = False, **fit_kwargs) -> "LogisticModel":
        if target_col not in df.columns:
            raise ValueError(f"target_col '{target_col}' not in dataframe")
        features = features or [c for c in df.columns if c != target_col]
        X = df[features]
        y = df[target_col]
        return self.fit(X, y, feature_names=features, calibrate=calibrate, **fit_kwargs)

    def _ensure_feature_order(self, X: Union[pd.DataFrame, np.ndarray]):
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                raise RuntimeError("Model has no stored feature names.")
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame missing features: {missing}")
            return X[self.feature_names]
        return X

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        X_in = self._ensure_feature_order(X)
        return self.pipeline.predict(X_in)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained.")
        X_in = self._ensure_feature_order(X)
        proba = self.pipeline.predict_proba(X_in)
        return proba[:, 1]

    # -----------------------------
    # CRC calibration for marginal false-positive risk
    # -----------------------------
    def calibrate_crc_marginal_false_positive(self,
                                              X_cal: Union[pd.DataFrame, np.ndarray],
                                              y_cal: Union[pd.Series, np.ndarray],
                                              alpha: float = 0.05,
                                              delta: float = 0.05,
                                              method: str = "hoeffding",
                                              grid: Optional[np.ndarray] = None,
                                              n_grid: int = 1000,
                                              randomized_if_needed: bool = False
                                              ) -> Dict[str, Any]:
        """
        Find threshold tau such that P(pred=1 AND y=0) <= alpha with probability >= 1-delta.

        Parameters:
          - X_cal, y_cal: calibration set (must not be used for training or probability calibration)
          - alpha: desired marginal false-positive bound
          - delta: confidence parameter
          - method: "hoeffding" (distribution-free) or "clopper" (Clopper-Pearson binomial UB; requires scipy)
          - grid: optional thresholds to test; if None, created from unique probs + uniform grid
          - n_grid: size of uniform grid if grid is None
          - randomized_if_needed: if True, and no deterministic tau meets bound,
                                 compute randomized mixing (on calibration empirical rates)
                                 between two adjacent taus to achieve target alpha in expectation.
                                 **Warning**: randomized mixing only matches expectation on the
                                 calibration calibration distribution; it is not a distribution-free
                                 high-probability guarantee unless further analysis is done.

        Returns:
          - dict: updated crc_info (also stored in self.crc_info)
        """
        if not self.trained:
            raise RuntimeError("Model must be trained before CRC calibration.")

        # get probabilities for positive class
        probs = np.asarray(self.predict_proba(X_cal)).ravel()
        y = np.asarray(y_cal).ravel()
        if len(probs) != len(y):
            raise ValueError("Length mismatch between X_cal and y_cal")

        n_cal = len(y)
        # prepare grid
        if grid is None:
            uniq = np.unique(probs)
            lin = np.linspace(0.0, 1.0, n_grid)
            grid = np.unique(np.concatenate((uniq, lin)))
            grid.sort()

        # compute empirical false-positive counts for each tau
        taus = []
        false_counts = []
        L_hat_list = []
        for tau in grid:
            mask_pos = probs >= tau
            fp = int(np.sum(mask_pos & (y == 0)))
            L_hat = fp / n_cal
            taus.append(float(tau))
            false_counts.append(fp)
            L_hat_list.append(L_hat)

        taus = np.array(taus)
        false_counts = np.array(false_counts)
        L_hat_list = np.array(L_hat_list)

        # compute UB depending on method
        ub_list = np.full_like(L_hat_list, fill_value=np.inf, dtype=float)

        if method == "hoeffding":
            from math import sqrt, log
            penalty = sqrt(log(2.0 / delta) / (2.0 * n_cal))
            ub_list = L_hat_list + penalty
            extra = {"hoeffding_penalty": float(penalty)}
        elif method == "clopper":
            if not _HAVE_SCIPY:
                raise RuntimeError("Clopper-Pearson method requested but scipy not available.")
            # one-sided upper bound at confidence level 1-delta:
            # upper = beta.ppf(1 - delta, k+1, n-k)  (if k<n), else 1.0
            ub = []
            for k in false_counts:
                k = int(k)
                if k >= n_cal:
                    upper = 1.0
                else:
                    upper = float(beta.ppf(1.0 - delta, k + 1, n_cal - k))
                    upper = float(beta.ppf(1.0 - delta, k + 1, n_cal - k))
                ub.append(upper)
            ub_list = np.array(ub)
            extra = {"clopper_used": True}
        else:
            raise ValueError("method must be 'hoeffding' or 'clopper'")

        # find feasible taus (smallest tau with ub <= alpha)
        feasible_idx = np.where(ub_list <= alpha)[0]
        chosen_tau = None
        randomized_info = None

        if feasible_idx.size > 0:
            idx0 = feasible_idx[0]
            chosen_tau = float(taus[idx0])
            self.crc_info = {
                "tau": chosen_tau,
                "method": method,
                "alpha": float(alpha),
                "delta": float(delta),
                "n_cal": int(n_cal),
                **extra,
                "randomized": False
            }
        else:
            # no deterministic threshold satisfies bound
            if randomized_if_needed:
                # Attempt randomized mixing between two adjacent thresholds where L_hat crosses alpha (empirical)
                # Find smallest tau where empirical L_hat <= alpha; and preceding tau where > alpha (or vice versa)
                # We'll use empirical L_hat to mix and achieve alpha in expectation on calibration set.
                # Note: this does not preserve the high-probability guarantee; user should be warned.
                # Find indices sorted by tau increasing: note L_hat decreases as tau increases.
                # We want L_lo < alpha < L_hi for adjacent points -> mix between them.
                # We'll search for index where L_hat drops from >alpha to <alpha.
                idxs = np.argsort(taus)
                L_sorted = L_hat_list[idxs]
                taus_sorted = taus[idxs]
                cross_idx = None
                for i in range(1, len(L_sorted)):
                    if (L_sorted[i-1] > alpha) and (L_sorted[i] < alpha):
                        cross_idx = (idxs[i-1], idxs[i])
                        break
                # edge cases: maybe all L_sorted > alpha or all L_sorted < alpha
                if cross_idx is None:
                    # try if any L_hat < alpha (then choose that tau deterministic)
                    under_idx = np.where(L_hat_list <= alpha)[0]
                    if under_idx.size > 0:
                        chosen_tau = float(taus[under_idx[0]])
                        self.crc_info = {
                            "tau": chosen_tau,
                            "method": method,
                            "alpha": float(alpha),
                            "delta": float(delta),
                            "n_cal": int(n_cal),
                            **extra,
                            "randomized": False
                        }
                    else:
                        # cannot find crossing; give up
                        self.crc_info = {
                            "tau": None,
                            "method": method,
                            "alpha": float(alpha),
                            "delta": float(delta),
                            "n_cal": int(n_cal),
                            **extra,
                            "randomized": False,
                            "note": "no tau found; randomized mixing not possible because no crossing detected"
                        }
                else:
                    i_lo, i_hi = cross_idx
                    tau_lo = float(taus[i_lo])
                    tau_hi = float(taus[i_hi])
                    L_lo = float(L_hat_list[i_lo])
                    L_hi = float(L_hat_list[i_hi])
                    # p_mix s.t. p * L_hi + (1-p) * L_lo = alpha => p = (alpha - L_lo) / (L_hi - L_lo)
                    if (L_hi - L_lo) == 0:
                        p_mix = 0.0
                    else:
                        p_mix = float((alpha - L_lo) / (L_hi - L_lo))
                        p_mix = max(0.0, min(1.0, p_mix))
                    randomized_info = {
                        "tau_lo": tau_lo,
                        "tau_hi": tau_hi,
                        "L_lo": L_lo,
                        "L_hi": L_hi,
                        "p_mix": p_mix
                    }
                    self.crc_info = {
                        "tau": None,
                        "method": method,
                        "alpha": float(alpha),
                        "delta": float(delta),
                        "n_cal": int(n_cal),
                        **extra,
                        "randomized": True,
                        "rand_info": randomized_info
                    }
            else:
                # deterministic not found and randomized not requested
                self.crc_info = {
                    "tau": None,
                    "method": method,
                    "alpha": float(alpha),
                    "delta": float(delta),
                    "n_cal": int(n_cal),
                    **extra,
                    "randomized": False,
                    "note": "no deterministic tau found; try increasing alpha or enlarging calibration set"
                }

        # Also store the calibration results table for inspection (small-ish)
        results_df = pd.DataFrame({
            "tau": taus,
            "false_positives": false_counts,
            "empirical_loss": L_hat_list,
            "ub": ub_list
        })
        # store some diagnostics
        self.crc_info["results_preview"] = results_df.sort_values("tau").head(100).to_dict(orient="list")
        return dict(self.crc_info)

    def predict_with_threshold(self,
                               X: Union[pd.DataFrame, np.ndarray],
                               return_proba: bool = False,
                               rng: Optional[np.random.RandomState] = None
                               ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict labels using stored CRC threshold (or randomized mixing if configured).

        If randomized mixing is configured, each sample uses the same random draw logic:
          - compute p_mix and with probability p_mix use tau_hi, otherwise tau_lo.
        Note: randomized mixing uses the stored p_mix computed from calibration set.

        Returns:
          labels (and optionally proba)
        """
        if not self.trained:
            raise RuntimeError("Model not trained.")
        if not self.crc_info:
            raise RuntimeError("No CRC info found. Call calibrate_crc_marginal_false_positive first.")

        probs = self.predict_proba(X)
        # deterministic threshold
        if self.crc_info.get("randomized", False) is False:
            tau = self.crc_info.get("tau", None)
            if tau is None:
                raise RuntimeError("No deterministic tau stored. Calibration did not find tau.")
            labels = (probs >= tau).astype(int)
            return (labels, probs) if return_proba else labels

        # randomized mixing:
        rand_info = self.crc_info.get("rand_info")
        if rand_info is None:
            raise RuntimeError("CRC randomized info missing.")
        tau_lo = rand_info["tau_lo"]
        tau_hi = rand_info["tau_hi"]
        p_mix = float(rand_info["p_mix"])
        # generate random draw per sample if desired; if rng not provided use global np.random
        rng_local = rng if rng is not None else np.random
        # decide per-sample whether to use hi or lo threshold
        draws = rng_local.rand(len(probs))
        use_hi = draws < p_mix
        labels = np.zeros_like(probs, dtype=int)
        labels[use_hi] = (probs[use_hi] >= tau_hi).astype(int)
        labels[~use_hi] = (probs[~use_hi] >= tau_lo).astype(int)
        return (labels, probs) if return_proba else labels

    # -----------------------------
    # Persistence
    # -----------------------------
    def save(self, path: Union[str, Path]):
        if not self.trained:
            raise RuntimeError("Train model before saving.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "crc_info": self.crc_info
        }
        joblib.dump(payload, path)
        return str(path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LogisticRegressionModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        lm = cls()
        lm.pipeline = payload.get("pipeline")
        lm.feature_names = payload.get("feature_names")
        lm.metadata = payload.get("metadata", {})
        lm.crc_info = payload.get("crc_info", {})
        lm.trained = True
        return lm


        

        