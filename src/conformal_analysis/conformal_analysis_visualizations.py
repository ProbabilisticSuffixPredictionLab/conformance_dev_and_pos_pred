import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import Optional

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



