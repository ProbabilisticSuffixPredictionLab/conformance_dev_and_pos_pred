import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class ConformalAnalysisVisualizations:
    def __init__(self, sampled_fitness, target_fitness, ml_fitness):
        # remove trailing commas (they created tuples)
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
                          aggregation='mean',
                          bins=30,
                          show_kde=True,
                          alpha_risk: float = None,
                          alpha_highrisk: float = None):
        """
        Plot distribution of aggregated sample fitness and (optional) target fitness.
        Additionally: draw red vertical lines and annotate x-axis values for the empirical
        quantiles given by alpha_risk and alpha_highrisk (if provided).
        """
        # aggregate samples
        smpls_fit = self.__aggregate_samples_fitness(samples_fitness=self.samples_fitness,
                                                     aggregation=aggregation)

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

        print_stats("Aggregated samples fitness statistics", smpls_fit)
        print_stats("Target fitness statistics", target_fit)

        # compute requested empirical quantiles (on aggregated samples)
        q_risk = None
        q_highrisk = None
        
        if smpls_fit is None or smpls_fit.size == 0:
            # nothing to plot; raise or return gracefully
            raise ValueError("No aggregated sample fitness values to plot.")
        if alpha_risk is not None:
            if not (0.0 <= alpha_risk <= 1.0):
                raise ValueError("alpha_risk must be in [0,1]")
            q_risk = float(np.quantile(smpls_fit, alpha_risk))
        if alpha_highrisk is not None:
            if not (0.0 <= alpha_highrisk <= 1.0):
                raise ValueError("alpha_highrisk must be in [0,1]")
            q_highrisk = float(np.quantile(smpls_fit, alpha_highrisk))

        # 1) Histogram (density) overlay
        plt.figure(figsize=(9, 5))
        plt.hist(smpls_fit, bins=bins, density=True, alpha=0.6,
                 edgecolor='black', linewidth=0.5, label='aggregated samples fitness', color='blue')
        if target_fit is not None:
            plt.hist(target_fit, bins=bins, density=True, alpha=0.5,
                     edgecolor='black', linewidth=0.5, label='target fitness', color='green')

        # 2) KDE overlays (only if >1 sample)
        if show_kde:
            try:
                valid_xmin = float(smpls_fit.min())
                valid_xmax = float(smpls_fit.max())
                if target_fit is not None:
                    valid_xmin = min(valid_xmin, float(target_fit.min()))
                    valid_xmax = max(valid_xmax, float(target_fit.max()))
                x = np.linspace(valid_xmin, valid_xmax, 400)

                if smpls_fit.size > 1:
                    kde_means = gaussian_kde(smpls_fit)
                    plt.plot(x, kde_means(x), lw=2, label='KDE (samples)', color='blue')
                if target_fit is not None and target_fit.size > 1:
                    kde_target = gaussian_kde(target_fit)
                    plt.plot(x, kde_target(x), lw=2, label='KDE (target)', color='green')
            except Exception:
                # silently skip KDE if it fails (scipy missing or KDE error)
                pass

        #  --- draw quantile vertical lines + annotate their numeric values on the x-axis ---
        ax = plt.gca()
        ylim = ax.get_ylim()
        y_text_pos = ylim[0] + 0.03 * (ylim[1] - ylim[0])  # small offset above bottom

        if q_risk is not None:
            ax.axvline(q_risk, color='red', linestyle='--', linewidth=2, label=f'quantile (risk={alpha_risk})')
            # add numeric annotation slightly above the x-axis (centered)
            ax.text(q_risk, y_text_pos, f"{q_risk:.3f}", color='red', ha='center', va='bottom', fontsize=9,
                    backgroundcolor='white')

        if q_highrisk is not None:
            ax.axvline(q_highrisk, color='red', linestyle='-', linewidth=1.5,
                       label=f'quantile (high risk={alpha_highrisk})')
            ax.text(q_highrisk, y_text_pos, f"{q_highrisk:.3f}", color='red', ha='center', va='bottom', fontsize=9,
                    backgroundcolor='white')

        plt.xlabel('(aggregated) fitness score')
        plt.ylabel('density')
        plt.title('Distribution of aggregated samples and target fitness scores')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.35)
        plt.tight_layout()
        plt.show()

        # 3) Boxplot (both distributions if present)
        datasets = [smpls_fit]
        labels = ['Data']
        if target_fit is not None:
            datasets.append(target_fit)
            labels.append('Target')

        plt.figure(figsize=(8, 2.8))
        box = plt.boxplot(datasets, vert=False, patch_artist=True, widths=0.6, labels=labels)

        # style: color first blue, second green (if present)
        colors = ['blue', 'green']
        for i, b in enumerate(box['boxes']):
            b.set(facecolor=colors[i % len(colors)])

        # Optionally mark quantiles on boxplot axis as vertical lines as well
        ax2 = plt.gca()
        if q_risk is not None:
            ax2.axvline(q_risk, color='red', linestyle='--', linewidth=2)
            ax2.text(q_risk, 0.95, f"{q_risk:.3f}", color='red', ha='center', va='top', fontsize=8,
                     rotation=90, backgroundcolor='white')
        if q_highrisk is not None:
            ax2.axvline(q_highrisk, color='red', linestyle='-', linewidth=1.5)
            ax2.text(q_highrisk, 0.95, f"{q_highrisk:.3f}", color='red', ha='center', va='top', fontsize=8,
                     rotation=90, backgroundcolor='white')

        plt.setp(box['whiskers'], color='black')
        plt.setp(box['caps'], color='black')
        plt.setp(box['medians'], color='red')
        plt.xlabel('Aggregated fitness score')
        plt.title('Boxplot: aggregated data vs target')
        plt.grid(axis='x', linestyle='--', alpha=0.25)
        plt.tight_layout()
        plt.show()
  
  
    def __empirical_coverage(self, lo, hi, y):
        """Return fraction of y in [lo, hi] and boolean mask."""
        lo = np.asarray(lo); hi = np.asarray(hi); y = np.asarray(y)
        inside = (y >= lo) & (y <= hi)
        return float(np.mean(inside)), inside


    # Put this method inside your class (uses self.__aggregate_samples_fitness and self.target_fitness)
    def plot_preds_vs_true_intervals(self,
                                    aggregation: str = 'mean',
                                    conformal_threshold: float = 0.05,
                                    max_points: int = 2000,
                                    clip_bounds: tuple = (0.0, 1.0),
                                    show_diagonal: bool = True,
                                    start_at_observed_min: bool = True,
                                    ):
        """
        For each aggregated sample fitness value (prediction) form interval
        [pred - conformal_threshold, pred + conformal_threshold]
        and check whether the corresponding target_fitness value lies inside that interval.
        Plot vertical interval lines (blue = predictions) and target points (green).
        Uncovered targets are marked with red 'x'.

        Requirements:
        - self.samples_fitness is provided (list/2D/1D handled by __aggregate_samples_fitness)
        - self.target_fitness is provided and has same length as aggregated samples (one target per sample)
        """
        
        # --- aggregate predicted samples ---
        preds = self.__aggregate_samples_fitness(samples_fitness=self.samples_fitness,
                                                aggregation=aggregation)
        preds = np.asarray(preds, dtype=float).flatten()

        # --- handle target_fitness ---
        if self.target_fitness is None:
            raise ValueError("self.target_fitness is None; need target values to check coverage.")
        target = np.asarray(self.target_fitness, dtype=float).flatten()

        if preds.size != target.size:
            raise ValueError(f"Length mismatch: aggregated samples ({preds.size}) vs target ({target.size}). "
                            "They must align 1:1.")

        # --- compute intervals (symmetric) ---
        lo = preds - float(conformal_threshold)
        hi = preds + float(conformal_threshold)
        # optional clipping (use same domain as your fitness values)
        lo = np.clip(lo, clip_bounds[0], clip_bounds[1])
        hi = np.clip(hi, clip_bounds[0], clip_bounds[1])

        # --- diagnostics ---
        cov, inside = self.__empirical_coverage(lo, hi, target)
        widths = hi - lo
        mean_width = float(np.mean(widths))
        median_width = float(np.median(widths))
        n = preds.size

        # --- plotting (use same colorway as empirical distribution) ---
        # Blue = aggregated samples / intervals, Green = target points, Red = uncovered targets
        # Limit number of points plotted for readability
        if n > max_points:
            sel_idx = np.linspace(0, n-1, max_points).astype(int)
        else:
            sel_idx = np.arange(n)

        covered_idx = sel_idx[inside[sel_idx]]
        uncovered_idx = sel_idx[~inside[sel_idx]]

        # --- compute axis limits: start at observed min if requested (with a small margin) ---
        observed_min = float(min(np.min(preds), np.min(target)))
        observed_max = float(max(np.max(preds), np.max(target)))
        obs_range = max(observed_max - observed_min, 1e-6)
        margin = max(0.02 * obs_range, 1e-4)

        if start_at_observed_min:
            x_left = max(clip_bounds[0], observed_min - margin)
        else:
            x_left = clip_bounds[0]
        x_right = min(clip_bounds[1], observed_max + margin)

        # set same y-limits as well to square the plot on the same domain (useful for y=pred diagonal)
        y_bottom = x_left
        y_top = x_right

        plt.figure(figsize=(7, 7))

        # Draw vertical interval lines for each chosen index (blue)
        plt.vlines(preds[sel_idx], ymin=lo[sel_idx], ymax=hi[sel_idx],
                colors='tab:blue', alpha=0.6, linewidth=1, label='Interval (pred ± bound)', zorder=1)

        # Optionally draw a small horizontal tick at the prediction value to show x-position
        tick_half = (x_right - x_left) * 0.002  # tiny horizontal tick relative scale
        for idx in sel_idx:
            plt.plot([preds[idx] - tick_half, preds[idx] + tick_half],
                    [preds[idx], preds[idx]], color='tab:blue', alpha=0.6, linewidth=0.8, zorder=2)

        # Plot target true values as green dots (full drawn indices)
        plt.scatter(preds[covered_idx], target[covered_idx], c='tab:green', s=28, label='Target (covered)', zorder=4)
        # Mark uncovered targets with red 'x'
        if uncovered_idx.size > 0:
            plt.scatter(preds[uncovered_idx], target[uncovered_idx], c='tab:red', s=50,
                        marker='x', label='Target (uncovered)', zorder=5)

        # draw diagonal y = pred line if desired (helps spot bias)
        if show_diagonal:
            plt.plot([x_left, x_right], [x_left, x_right], linestyle='--', color='gray', label='y = pred', zorder=0)

        plt.xlabel('Aggregated sample fitness (prediction)')
        plt.ylabel('Target fitness (true)')
        title: str = 'Aggregated samples ± conformal threshold (conformal fitnes score interval) vs target'
        plt.title(f"{title}\nemp. coverage = {cov:.3f} (n={n}), mean width = {mean_width:.4f}")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.xlim(x_left, x_right)
        plt.ylim(y_bottom, y_top)
        plt.tight_layout()
        plt.show()


    def plot_coverage_by_pred_bin(self,
                                aggregation: str = 'mean',
                                conformal_threshold: float = 0.05,
                                n_bins: int = 10,
                                min_count_for_bin: int = 5,
                                title: str = 'Coverage by predicted-value bin',
                                start_at_observed_min: bool = True,
                                clip_bounds: tuple = (0.0, 1.0)):
        """
        Aggregate samples_fitness using `aggregation`, build symmetric conformal intervals
        [pred - conformal_threshold, pred + conformal_threshold], then compute & plot
        empirical coverage per predicted-value bin and mean interval width per bin.

        Additionally computes and displays the global (mean) empirical coverage across all examples
        and the overall mean interval width.

        Returns a dict with bin centers, counts, coverage per bin, mean width per bin, bin edges,
        plus global_coverage and mean_width_overall.
        """
        # --- aggregate predicted samples ---
        preds = self.__aggregate_samples_fitness(samples_fitness=self.samples_fitness, aggregation=aggregation)
        preds = np.asarray(preds, dtype=float).flatten()

        # --- handle target_fitness ---
        if self.target_fitness is None:
            raise ValueError("self.target_fitness is None; need target values to check coverage.")
        target = np.asarray(self.target_fitness, dtype=float).flatten()

        if preds.size != target.size:
            raise ValueError(f"Length mismatch: aggregated samples ({preds.size}) vs target ({target.size}). They must align 1:1.")

        # --- compute symmetric intervals and clip to allowed domain ---
        lo = preds - float(conformal_threshold)
        hi = preds + float(conformal_threshold)
        lo = np.clip(lo, clip_bounds[0], clip_bounds[1])
        hi = np.clip(hi, clip_bounds[0], clip_bounds[1])

        # --- inside mask and widths ---
        inside = (target >= lo) & (target <= hi)
        widths = hi - lo

        # overall/global diagnostics
        global_coverage = float(np.mean(inside)) if preds.size > 0 else float('nan')
        mean_width_overall = float(np.mean(widths)) if preds.size > 0 else float('nan')

        # --- compute observed min/max and build bins starting at observed_min if requested ---
        observed_min = float(min(np.min(preds), np.min(target)))
        observed_max = float(max(np.max(preds), np.max(target)))
        obs_range = max(observed_max - observed_min, 1e-6)
        margin = max(0.02 * obs_range, 1e-4)

        if start_at_observed_min:
            left = max(clip_bounds[0], observed_min - margin)
        else:
            left = clip_bounds[0]
        right = min(clip_bounds[1], observed_max + margin)

        # guard the degenerate case where left >= right
        if right <= left:
            left = max(clip_bounds[0], observed_min - margin)
            right = min(clip_bounds[1], observed_min + margin + 1e-3)

        bins = np.linspace(left, right, n_bins + 1)
        bin_idx = np.digitize(preds, bins) - 1
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        cov_by_bin = np.full(n_bins, np.nan)
        width_by_bin = np.full(n_bins, np.nan)
        counts = np.zeros(n_bins, dtype=int)

        for b in range(n_bins):
            sel = (bin_idx == b)
            c = int(sel.sum())
            counts[b] = c
            if c >= int(min_count_for_bin):
                cov_by_bin[b] = float(np.mean(inside[sel]))
                width_by_bin[b] = float(np.mean(widths[sel]))
            else:
                cov_by_bin[b] = np.nan
                width_by_bin[b] = np.nan

        # --- Plot coverage and mean width per bin ---
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(bin_centers, cov_by_bin, marker='o', linestyle='-', label='empirical coverage', color='tab:blue')
        ax1.axhline(0.90, linestyle='--', color='gray', label='nominal 0.90')
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Predicted value (bin center)')
        ax1.set_ylabel('Empirical coverage')
        ax1.set_title(title)
        ax1.grid(alpha=0.2)

        # annotate counts above points
        for x, c in zip(bin_centers, counts):
            ax1.text(x, 0.03, f"n={c}", ha='center', va='bottom', fontsize=8, color='black')

        # draw global coverage horizontal line and annotate with a boxed label
        ax1.axhline(global_coverage, linestyle=':', color='black', linewidth=1.2,
                    label=f'global coverage = {global_coverage:.3f}')
        # text box on upper-right of the axes
        bbox_text = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black')
        ax1.text(0.98, 0.98,
                f"global cov = {global_coverage:.3f}\nmean width = {mean_width_overall:.3f}",
                transform=ax1.transAxes, ha='right', va='top', fontsize=9, bbox=bbox_text)

        # Secondary axis: mean width per bin as bars (green)
        ax2 = ax1.twinx()
        if np.all(np.isnan(width_by_bin)):
            ylim2 = 1.0
        else:
            ylim2 = float(np.nanmax(width_by_bin) * 1.4)
        ax2.bar(bin_centers, width_by_bin, width=(bins[1] - bins[0]) * 0.7, alpha=0.35, color='tab:green', label='mean width')
        ax2.set_ylabel('Mean interval width')
        ax2.set_ylim(0, ylim2)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        plt.show()



