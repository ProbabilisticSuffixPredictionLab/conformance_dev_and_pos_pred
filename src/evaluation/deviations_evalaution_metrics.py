from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import textwrap
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Union, Iterable

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import textwrap
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.ticker import MaxNLocator

def load_results(path: Union[str, Path]) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)

class DeviationEvaluation:
    """
    1) Macro average precision and recall per true deviaiton label
    2) Get suffixes with target deviations and occurrence/position in predicted suffix samples -> updated to use SEQUENCE-INDEX positions (more robust)
    3) Likelihood evaluation:
       - keeps mean likelihood at target positions.
       - adds hitProb (fraction of samples that hit any true position).
    4) Visualization
    """
    def __init__(self, deviation_results: Iterable[dict]):
        self.deviation_results = list(deviation_results)
    
    # updated
    def precision_recall_macro_by_label_dev(self) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Compute per-label precision and recall (BINARY presence per case) only for labels appearing in the target set,
        then return macro-averages across those target labels.

        Per label (per case):
        TP: pred 1, tgt 1
        FP: pred 1, tgt 0
        FN: pred 0, tgt 1
        TN: pred 0, tgt 0   (not needed for prec/rec)

        Notes:
        - Uses set() per case (binary), not counts. -> Only chek if label is in dev at least once!
        - Only evaluates labels that appear in the TARGET set (total_tgt > 0).
        """
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        # pred_deviations: is dev in samples has higher prob of occurence than threshold
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # label-level accumulators across cases (binary presence)
        tp_label = defaultdict(int)
        fp_label = defaultdict(int)
        fn_label = defaultdict(int)
        total_tgt = defaultdict(int)  # number of cases where label is present in target (for filtering)

        # first pass: determine target labels (binary presence)
        for tgt_list in tgt_deviations:
            t_set = set(tgt_list)
            for lbl in t_set:
                total_tgt[lbl] += 1

        target_label_list = sorted(k for k, v in total_tgt.items() if v > 0)

        # second pass: compute TP/FP/FN per label (binary per case)
        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            t_set = set(tgt_list)
            p_set = set(pred_list)
            for lbl in target_label_list:
                t = (lbl in t_set)
                p = (lbl in p_set)
                if p and t:
                    tp_label[lbl] += 1
                elif p and (not t):
                    fp_label[lbl] += 1
                elif (not p) and t:
                    fn_label[lbl] += 1

        precision_per_label = {}
        recall_per_label = {}
        for lbl in target_label_list:
            denom_p = tp_label[lbl] + fp_label[lbl]
            denom_r = tp_label[lbl] + fn_label[lbl]
            precision_per_label[lbl] = tp_label[lbl] / denom_p if denom_p > 0 else 0.0
            recall_per_label[lbl] = tp_label[lbl] / denom_r if denom_r > 0 else 0.0

        precision_macro = float(np.mean(list(precision_per_label.values()))) if precision_per_label else 0.0
        recall_macro = float(np.mean(list(recall_per_label.values()))) if recall_per_label else 0.0
            
        return precision_macro, recall_macro, precision_per_label, recall_per_label
        
    # updated
    def precision_recall_macro_by_label_no_dev(self, zero_division: float = 1.0) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, int]]:
        """
        Compute per-label precision and recall for the no-deviation (opposite) using BINARY presence per case.

        Positive event = label is NOT present in prediction AND NOT present in target (per case).

        Per label (per case):
        TP: pred_no=1, tgt_no=1   (both absent)
        FP: pred_no=1, tgt_no=0   (absent in pred, present in tgt)
        FN: pred_no=0, tgt_no=1   (present in pred, absent in tgt)
        TN: pred_no=0, tgt_no=0

        Only evaluates labels that appear in the TARGET set (same filter as dev).
        """
        tgt_deviations = [dr.get('tgt_deviations', []) for dr in self.deviation_results]
        pred_deviations = [dr.get('pred_deviations', []) for dr in self.deviation_results]

        # determine target labels (binary presence)
        total_tgt = defaultdict(int)
        for tgt_list in tgt_deviations:
            for lbl in set(tgt_list):
                total_tgt[lbl] += 1
        target_label_list = sorted(k for k, v in total_tgt.items() if v > 0)

        # accumulators for "no deviation" positives (binary per case)
        tp_no = defaultdict(int)
        pred_no_count = defaultdict(int)  # #cases where pred says "no deviation" (lbl absent)
        true_no_count = defaultdict(int)  # #cases where true is "no deviation" (lbl absent)

        for tgt_list, pred_list in zip(tgt_deviations, pred_deviations):
            t_set = set(tgt_list)
            p_set = set(pred_list)
            for lbl in target_label_list:
                pred_has = (lbl in p_set)
                tgt_has = (lbl in t_set)

                pred_no = (not pred_has)
                true_no = (not tgt_has)

                if pred_no:
                    pred_no_count[lbl] += 1
                if true_no:
                    true_no_count[lbl] += 1
                if pred_no and true_no:
                    tp_no[lbl] += 1

        precision_per_label = {}
        recall_per_label = {}
        for lbl in target_label_list:
            denom_p = pred_no_count[lbl]
            denom_r = true_no_count[lbl]

            precision_per_label[lbl] = (tp_no[lbl] / denom_p) if denom_p > 0 else float(zero_division)
            recall_per_label[lbl] = (tp_no[lbl] / denom_r) if denom_r > 0 else float(zero_division)

        precision_macro = float(sum(precision_per_label.values()) / len(precision_per_label)) if precision_per_label else 0.0
        recall_macro = float(sum(recall_per_label.values()) / len(recall_per_label)) if recall_per_label else 0.0

        return precision_macro, recall_macro, precision_per_label, recall_per_label
    
    # data selection
    def _cases_with_target_deviations(self) -> List[dict]:
        return [{'prefix': dr.get('prefix', []),
                'tgt_suffix': dr.get('tgt_suffix', []),
                'pred_suffix': dr.get('pred_suffix', []),
                'tgt_aligns': dr.get('tgt_cleaned_aligns', []),
                'pred_aligns': dr.get('pred_cleaned_aligns', []),
                'tgt_deviations': dr.get('tgt_deviations', []),
                'pred_deviations': dr.get('pred_deviations', [])} 
                
                for dr in self.deviation_results 
                if len(dr.get('tgt_deviations', [])) > 0
                ]

    # updated: positions now in sequence-index; also optionally return extras for hitProb
    def get_suffix_devs(self, return_extras: bool = False):
        """
        Default (return_extras=False) keeps your original return shape:
        return tgt_suffixes, pred_suffix_samples, (tgt_model_moves, tgt_log_moves), (pred_model_moves, pred_log_moves)

        If return_extras=True, additionally returns: (pred_model_sets, pred_log_sets), num_samples_per_case, prefixes
        """
        # get all cases where the target contains a deviation:
        real_deviations = self._cases_with_target_deviations()

        prefixes = [rd.get('prefix', []) for rd in real_deviations]
        tgt_aligns = [rd.get('tgt_aligns', []) for rd in real_deviations]
        pred_aligns = [rd.get('pred_aligns', []) for rd in real_deviations]
        tgt_suffixes = [rd.get('tgt_suffix', []) for rd in real_deviations]
        pred_suffix_samples = [rd.get('pred_suffix', []) for rd in real_deviations]

        # target positions (sequence-index)
        tgt_model_moves, tgt_log_moves = [], []
        for align in tgt_aligns:
            mpos, lpos = _extract_positions_sequence_index(align)
            tgt_model_moves.append(mpos)
            tgt_log_moves.append(lpos)

        # predicted counts + per-sample sets (sequence-index)
        pred_model_moves, pred_log_moves = [], []
        pred_model_sets, pred_log_sets = [], []
        num_samples_per_case = []

        for align_samples in pred_aligns:
            cm, cl, sm, sl = _aggregate_sample_positions_sequence_index(align_samples)
            pred_model_moves.append(cm)
            pred_log_moves.append(cl)
            pred_model_sets.append(sm)
            pred_log_sets.append(sl)
            num_samples_per_case.append(len(align_samples))

        base = (tgt_suffixes, pred_suffix_samples, (tgt_model_moves, tgt_log_moves), (pred_model_moves, pred_log_moves))
        if not return_extras:
            return base

        extras = ((pred_model_sets, pred_log_sets), num_samples_per_case, prefixes)
        return base + extras

    # (B) UPDATED: likelihood kept + hitProb added (optional)
    def likelihood_at_target_positions(self,
                                       tgt_model_moves: List[Dict[str, List[int]]],
                                       tgt_log_moves: List[Dict[str, List[int]]],
                                       pred_model_moves: List[Dict[str, Dict[int, int]]],
                                       pred_log_moves: List[Dict[str, Dict[int, int]]],
                                       return_hitprob: bool = False,
                                       pred_model_sets: List[List[Dict[str, set]]] = None,
                                       pred_log_sets: List[List[Dict[str, set]]] = None,
                                       num_samples_per_case: List[int] = None):
        """
        Keeps your mean likelihood at target positions.
        Optionally adds hitProb.

        If return_hitprob=False:
          returns (case_level, per_label_mean, weighted_macro)  [same as before]

        If return_hitprob=True:
          returns (case_level, per_label_mean, weighted_macro, per_label_hitprob, weighted_macro_hitprob)

        Notes:
          - If you pass pred_model_sets/pred_log_sets + num_samples_per_case, hitProb uses them.
          - Otherwise, we will try to compute them from self.get_suffix_devs(return_extras=True).
          - Likelihood uses num_samples_per_case if provided; otherwise uses constant num_samples.
        """

        if return_hitprob and (pred_model_sets is None or pred_log_sets is None or num_samples_per_case is None):
            # compute required extras from the same underlying self.deviation_results
            _, _, (_, _), (_, _), (pred_model_sets, pred_log_sets), num_samples_per_case, _ = self.get_suffix_devs(return_extras=True)

        def _compute_likelihood(
            tgt_moves: List[Dict[str, List[int]]],
            pred_moves: List[Dict[str, Dict[int, int]]],
        ):
            if len(tgt_moves) != len(pred_moves):
                raise ValueError("Target and predicted move collections must have identical length.")

            case_level: List[Dict[str, List[Tuple[int, float]]]] = []
            per_label_scores: Dict[str, List[float]] = defaultdict(list)
            per_label_support: Dict[str, int] = defaultdict(int)

            all_labels = sorted({lbl for case in tgt_moves for lbl in case.keys()})

            for i, (tgt_case, pred_case) in enumerate(zip(tgt_moves, pred_moves)):
                T = num_samples_per_case[i] if (num_samples_per_case is not None) else num_samples
                case_entry: Dict[str, List[Tuple[int, float]]] = {}

                for label in all_labels:
                    positions = tgt_case.get(label, [])
                    if not positions:
                        continue
                    counts = pred_case.get(label, {})
                    position_scores: List[Tuple[int, float]] = []
                    for pos in positions:
                        c = counts.get(pos, 0)
                        prob = (c / T) if T > 0 else 0.0
                        position_scores.append((pos, float(prob)))
                        per_label_scores[label].append(float(prob))
                    per_label_support[label] += len(position_scores)
                    case_entry[label] = position_scores

                case_level.append(case_entry)

            per_label_mean = {
                label: (float(np.mean(scores)) if scores else 0.0)
                for label, scores in per_label_scores.items()
            }

            total_support = sum(per_label_support.values())
            weighted_macro = (
                float(sum(sum(scores) for scores in per_label_scores.values()) / total_support)
                if total_support > 0 else 0.0
            )
            return case_level, per_label_mean, weighted_macro, per_label_support

        def _compute_hitprob(tgt_moves: List[Dict[str, List[int]]],
                             pred_sets: List[List[Dict[str, set]]],
                             per_label_support: Dict[str, int]):
            per_label_hits: Dict[str, List[float]] = defaultdict(list)

            all_labels = sorted({lbl for case in tgt_moves for lbl in case.keys()})

            for i, tgt_case in enumerate(tgt_moves):
                T = num_samples_per_case[i] if (num_samples_per_case is not None) else len(pred_sets[i])
                T = max(1, T)
                sample_sets_case = pred_sets[i]

                for label in all_labels:
                    true_positions = tgt_case.get(label, [])
                    if not true_positions:
                        continue
                    true_set = set(true_positions)

                    hits = 0
                    for s in sample_sets_case:
                        pos_set = s.get(label, set())
                        if pos_set & true_set:
                            hits += 1
                    per_label_hits[label].append(hits / T)

            per_label_hitprob = {
                label: float(np.mean(vals)) if vals else 0.0
                for label, vals in per_label_hits.items()
            }

            total_support = sum(per_label_support.values())
            weighted_macro_hitprob = (
                float(sum(per_label_hitprob[lbl] * per_label_support.get(lbl, 0) for lbl in per_label_hitprob) / total_support)
                if total_support > 0 else 0.0)

            return per_label_hitprob, weighted_macro_hitprob

        model_case, model_label_mean, model_weighted, model_support = _compute_likelihood(tgt_model_moves, pred_model_moves)
        log_case, log_label_mean, log_weighted, log_support = _compute_likelihood(tgt_log_moves, pred_log_moves)

        case_level = {"model": model_case, "log": log_case}
        per_label_mean = {"model": model_label_mean, "log": log_label_mean}
        weighted_macro = {"model": model_weighted, "log": log_weighted}

        if not return_hitprob:
            return case_level, per_label_mean, weighted_macro

        model_hit, model_hit_weighted = _compute_hitprob(tgt_model_moves, pred_model_sets, model_support)
        log_hit, log_hit_weighted = _compute_hitprob(tgt_log_moves, pred_log_sets, log_support)

        per_label_hitprob = {"model": model_hit, "log": log_hit}
        weighted_macro_hitprob = {"model": model_hit_weighted, "log": log_hit_weighted}

        return case_level, per_label_mean, weighted_macro, per_label_hitprob, weighted_macro_hitprob








    def plot_suffix_deviation_distribution(
        self,
        suffix_index: int,
        label: str,
        move: str,
        tgt_suff_move,
        pred_suff_move,
        pred_suffix_samples,
        pred_suff_sets=None,
        tgt_suffixes=None,
        num_samples: int = 100,
        figsize=(12, 5),
        dpi: int = 220,
        style: str = "paper",   # "paper" | "dense" | "auto"
        show: bool = True,
        return_objects: bool = False,
    ):
        if not (0 <= suffix_index < len(tgt_suff_move)):
            raise IndexError("suffix_index out of range.")

        cases = self._cases_with_target_deviations()
        case = cases[suffix_index] if suffix_index < len(cases) else {}

        prefix = case.get("prefix") or []
        if tgt_suffixes is None:
            tgt_suffixes = [(c.get("tgt_suffix") or []) for c in cases]
        tgt_suffix = tgt_suffixes[suffix_index] if suffix_index < len(tgt_suffixes) else []

        samples_case = pred_suffix_samples[suffix_index] if suffix_index < len(pred_suffix_samples) else []
        sets_case = (pred_suff_sets[suffix_index] if (pred_suff_sets and suffix_index < len(pred_suff_sets)) else None)

        is_log = (move == "log")
        # Use the *actual* number of available samples for normalization.
        # Using the num_samples argument here can artificially deflate rates when fewer samples exist.
        denom_req = max(1, len(sets_case) if (sets_case is not None) else len(samples_case))
        highlight_tok = str(label) if is_log else None

        PAPER = dict(fs_title=13.5, fs_h=11, fs_t=10, fs_s=9,
                    max_chars=92, max_chars_s=88,
                    max_ev_prefix=26, max_ev_suffix=34, max_ev_sample=38,
                    gap_h=0.030, gap_b=0.020)
        DENSE = dict(fs_title=13, fs_h=10, fs_t=9, fs_s=8,
                    max_chars=108, max_chars_s=102,
                    max_ev_prefix=32, max_ev_suffix=42, max_ev_sample=46,
                    gap_h=0.024, gap_b=0.017)

        if style not in {"paper", "dense", "auto"}:
            style = "paper"
        if style == "auto":
            longish = (len(prefix) + len(tgt_suffix) >= 55) or any(len(s or []) >= 40 for s in (samples_case[:10] or []))
            cfg = DENSE if longish else PAPER
        else:
            cfg = PAPER if style == "paper" else DENSE

        fs_title, fs_h, fs_t, fs_s = cfg["fs_title"], cfg["fs_h"], cfg["fs_t"], cfg["fs_s"]
        GAP_H, GAP_B = cfg["gap_h"], cfg["gap_b"]

        # ---------- helpers ----------
        def _wrap(s: str, w: int) -> str:
            s = str(s)
            return s if len(s) <= w else "\n".join(textwrap.wrap(s, width=w, break_long_words=False, break_on_hyphens=False))

        def _clip(tokens, n, tail=False):
            t = [str(x) for x in (tokens or [])]
            if len(t) <= n:
                return t
            return (["…"] + t[-(n - 1):]) if tail else (t[: n - 1] + ["…"])

        def _token_lines(tokens, *, max_events, max_chars, tail=False):
            toks = _clip(tokens, max_events, tail=tail)
            out, cur, cur_len = [], [], 0
            for tok in toks:
                add = len(tok) + (3 if cur else 0)  # " → "
                if cur and (cur_len + add > max_chars):
                    out.append(cur)
                    cur, cur_len = [tok], len(tok)
                else:
                    cur.append(tok)
                    cur_len += add
            if cur:
                out.append(cur)
            return out or [["(empty)"]]

        def _draw_tokens(ax, x, y_top, tokens, *, max_events, max_chars, tail=False, fontsize=10, hi=None, accent="tab:orange"):
            base = "black"
            lines = _token_lines(tokens, max_events=max_events, max_chars=max_chars, tail=tail)

            v = []
            for ln in lines:
                h = []
                for j, tok in enumerate(ln):
                    if j:
                        h.append(TextArea(" → ", textprops={"fontsize": fontsize, "color": base}))
                    col = accent if (hi is not None and tok == str(hi)) else base
                    h.append(TextArea(tok, textprops={"fontsize": fontsize, "color": col}))
                v.append(HPacker(children=h, align="baseline", pad=0, sep=0))

            box = VPacker(children=v, align="left", pad=0, sep=2)
            abox = AnchoredOffsetbox(
                loc="upper left",
                child=box,
                frameon=False,
                bbox_to_anchor=(x, y_top),
                bbox_transform=ax.transAxes,
                borderpad=0.0,
            )
            abox.set_clip_on(True)
            abox.set_clip_path(ax.patch)
            ax.add_artist(abox)
            return abox

        def _after_artist(fig, ax, artist, gap_axes: float) -> float:
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = artist.get_window_extent(renderer=renderer)
            y0 = ax.transAxes.inverted().transform((bbox.x0, bbox.y0))[1]
            return y0 - gap_axes

        def _topk_at_pos(samples, pos, token, k=2):
            if not samples:
                return []
            ctr = Counter(tuple(s) for s in samples if s and len(s) > pos and str(s[pos]) == str(token))
            return [(c, c / denom_req, list(seq)) for seq, c in ctr.most_common(k)]

        true_pos = sorted({int(p) for p in ((tgt_suff_move[suffix_index] or {}).get(label) or [])})
        counts_dict = dict(((pred_suff_move[suffix_index] or {}).get(label) or {}))

        def _rate_from_sets(label_sets):
            pos_counts = defaultdict(int)
            for smap in (label_sets or []):
                for p in (smap.get(label) or set()):
                    pos_counts[int(p)] += 1
            pos = sorted(pos_counts)
            denom = max(1, len(label_sets or []))
            return pos, [pos_counts[p] / denom for p in pos], dict(pos_counts), denom

        if sets_case is not None:
            positions, probs, counts_shown, denom_shown = _rate_from_sets(sets_case)
        elif is_log:
            pos_counts = defaultdict(int)
            for seq in samples_case:
                for p, tok in enumerate(seq or []):
                    if str(tok) == str(label):
                        pos_counts[int(p)] += 1
            positions = sorted(pos_counts)
            probs = [pos_counts[p] / denom_req for p in positions]
            counts_shown, denom_shown = dict(pos_counts), denom_req
        elif counts_dict:
            denom = max(1, len(samples_case))
            positions = sorted(counts_dict)
            probs = [counts_dict[p] / denom for p in positions]
            counts_shown, denom_shown = dict(counts_dict), denom
        else:
            positions, probs, counts_shown, denom_shown = [], [], {}, denom_req

        # ---------- seaborn “camera-ready” theme (local) ----------
        palette = sns.color_palette("colorblind")
        bar_color, accent = palette[0], palette[1]

        rc = {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "grid.linestyle": ":",
            "grid.alpha": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
        ctx = "paper" if style != "dense" else "notebook"
        font_scale = 1.06 if style == "paper" else 0.95

        with sns.axes_style("ticks", rc=rc), sns.plotting_context(ctx, font_scale=font_scale, rc=rc):
            # deterministic layout (avoid constrained_layout quirks with offsetboxes)
            fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
            fig.subplots_adjust(left=0.035, right=0.99, top=0.89, bottom=0.12, wspace=0.10)

            gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1.0])

            ax_txt = fig.add_subplot(gs[0, 0])
            ax_prob = fig.add_subplot(gs[0, 1])

            # shrink + lift the bar axis -> visually “shorter” plot and better balance
            shrink = 0.70 if (len(true_pos) <= 1 and len(prefix) <= 2) else 0.78
            lift   = 0.14 if shrink < 0.78 else 0.10
            p = ax_prob.get_position()
            ax_prob.set_position([p.x0, p.y0 + p.height * lift, p.width, p.height * shrink])

            title_move = "log" if is_log else "model"
            fig.suptitle(
                f"Deviation position distribution — label='{label}', move='{title_move}'",
                fontsize=fs_title,
                fontweight="bold",
                y=0.965,
            )

            # ---------- left text panel ----------
            ax_txt.axis("off")
            ax_txt.set_xlim(0, 1)
            ax_txt.set_ylim(0, 1)

            X, y = 0.01, 0.985  # tiny padding looks better than flush-left

            def add_header(text: str, y: float) -> float:
                t = ax_txt.text(X, y, _wrap(text, cfg["max_chars"]),
                                ha="left", va="top", fontsize=fs_h, fontweight="bold", clip_on=True)
                return _after_artist(fig, ax_txt, t, GAP_H)

            def add_tokens(tokens, y: float, *, max_events, max_chars, tail=False, hi=None) -> float:
                ab = _draw_tokens(ax_txt, X, y, tokens,
                                max_events=max_events, max_chars=max_chars,
                                tail=tail, fontsize=fs_t, hi=hi, accent=accent)
                return _after_artist(fig, ax_txt, ab, GAP_B)

            y = add_header(f"Prefix (len={len(prefix)})", y)
            y = add_tokens(prefix, y, max_events=cfg["max_ev_prefix"], max_chars=cfg["max_chars"], tail=True, hi=None)
            y -= GAP_B * 0.20

            dev_str = ", ".join(map(str, true_pos)) if true_pos else "(none)"
            y = add_header(f"Target suffix (len={len(tgt_suffix)}) — deviating positions: {dev_str}", y)
            y = add_tokens(tgt_suffix, y, max_events=cfg["max_ev_suffix"], max_chars=cfg["max_chars"],
                        tail=False, hi=highlight_tok)
            y -= GAP_B * 0.20

            y = add_header("Top sampled suffix sequences at deviating positions (top 2)", y)

            if not samples_case:
                t = ax_txt.text(X, y, "(no samples available)", ha="left", va="top", fontsize=fs_t, alpha=0.85, clip_on=True)
                y = _after_artist(fig, ax_txt, t, GAP_B)
            elif not true_pos:
                t = ax_txt.text(X, y, "(no deviating positions for this label)", ha="left", va="top", fontsize=fs_t, alpha=0.85, clip_on=True)
                y = _after_artist(fig, ax_txt, t, GAP_B)
            else:
                for p0 in true_pos:
                    if y < 0.08:
                        ax_txt.text(X, max(0.02, y), "(truncated to fit figure)",
                                    ha="left", va="bottom", fontsize=fs_s, alpha=0.6, clip_on=True)
                        break

                    y = add_header(f"Pos {p0}", y)
                    topk = _topk_at_pos(samples_case, pos=p0, token=label, k=2)

                    if not topk:
                        t = ax_txt.text(X, y, "(none)", ha="left", va="top", fontsize=fs_t, alpha=0.85, clip_on=True)
                        y = _after_artist(fig, ax_txt, t, GAP_B)
                        continue

                    for i, (c, frac, seq) in enumerate(topk, 1):
                        t = ax_txt.text(X, y, f"{i}. {c}/{denom_req}  ({100*frac:.0f}%)",
                                        ha="left", va="top", fontsize=fs_t, fontweight="bold", clip_on=True)
                        y = _after_artist(fig, ax_txt, t, GAP_B * 0.65)
                        y = add_tokens(seq, y, max_events=cfg["max_ev_sample"], max_chars=cfg["max_chars_s"],
                                    tail=False, hi=highlight_tok)
                    y -= GAP_B * 0.15

            # --- right bar panel (equal spacing) ---
            ax_prob.set_title(f"Predicted rate of '{label}'", fontsize=fs_h, fontweight="bold", pad=3)
            ax_prob.set_xlabel("Position (0-based)")
            ax_prob.set_ylabel("Rate")
            ax_prob.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax_prob.yaxis.set_major_locator(MaxNLocator(6))
            ax_prob.grid(axis="y", linestyle=":", alpha=0.22)
            ax_prob.grid(axis="x", visible=False)

            if positions:
                pos_vals = list(map(int, positions))
                xs = list(range(len(pos_vals)))             # equally spaced bar locations

                bars = ax_prob.bar(xs, probs, color=bar_color, edgecolor="0.25", linewidth=0.8, width=0.78, zorder=3)
                ax_prob.set_xticks(xs)
                ax_prob.set_xticklabels([str(p) for p in pos_vals])

                # True deviating positions -> map to categorical index
                pos_to_x = {p: i for i, p in enumerate(pos_vals)}
                for tp in true_pos:
                    if tp in pos_to_x:
                        ax_prob.axvline(pos_to_x[tp], linestyle="--", linewidth=1.1, color=accent, alpha=0.9, zorder=2)

                ymax = max(probs) if probs else 0.0
                ax_prob.set_ylim(0, min(1.0, max(0.10, ymax * 1.18)))
                ax_prob.set_xlim(-0.6, len(xs) - 0.4)

                # Count labels with subtle white background (prevents collisions)
                for b, p in zip(bars, pos_vals):
                    c = int(counts_shown.get(int(p), 0))
                    ax_prob.text(
                        b.get_x() + b.get_width() / 2,
                        b.get_height() + 0.004,
                        f"{c}/{denom_shown}",
                        ha="center", va="bottom",
                        fontsize=fs_s, alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
                        zorder=4,
                    )
            else:
                ax_prob.text(0.5, 0.5, "(no predicted occurrences)", ha="center", va="center", fontsize=fs_h)
                ax_prob.set_xticks([]); ax_prob.set_yticks([])

            sns.despine(ax=ax_prob)

            if show:
                plt.show()

            if return_objects:
                return fig, (ax_txt, ax_prob)
            return None







# -----------------------
# helpers for robust (sequence-index) deviation positions
# -----------------------
def _is_real_event(x: Any) -> bool:
    return (x is not None) and (x != ">>")


def _extract_positions_sequence_index(align: List[Tuple[Any, Any]]):
    """
    Convert alignment indices to *sequence indices* (stable under insertions/deletions).

    Returns:
      model_positions[label] = list of model-seq indices where ('>>', label) occurs
      log_positions[label]   = list of log-seq indices   where (label, '>>') occurs
    """
    model_positions: Dict[str, List[int]] = defaultdict(list)
    log_positions: Dict[str, List[int]] = defaultdict(list)

    log_i = -1
    model_i = -1

    for (log_move, model_move) in align:
        if _is_real_event(log_move):
            log_i += 1
        if _is_real_event(model_move):
            model_i += 1

        # model deviation: ('>>', x)
        if (log_move == ">>") and _is_real_event(model_move):
            model_positions[str(model_move)].append(model_i)

        # log deviation: (x, '>>')
        elif (model_move == ">>") and _is_real_event(log_move):
            log_positions[str(log_move)].append(log_i)

    return dict(model_positions), dict(log_positions)


def _aggregate_sample_positions_sequence_index(align_samples: List[List[Tuple[Any, Any]]]):
    """
    For a list of alignment samples:
    - counts_model[label][pos] = number of samples where ('>>', label) occurs at model-pos
    - counts_log[label][pos]   = number of samples where (label, '>>') occurs at log-pos
    - sets_model[sample_idx][label] = set(model positions) for that sample
    - sets_log[sample_idx][label]   = set(log positions) for that sample
    """
    counts_model: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    counts_log: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    sets_model: List[Dict[str, set]] = []
    sets_log: List[Dict[str, set]] = []

    for sample in align_samples:
        mpos, lpos = _extract_positions_sequence_index(sample)

        msets = {lbl: set(pos_list) for lbl, pos_list in mpos.items()}
        lsets = {lbl: set(pos_list) for lbl, pos_list in lpos.items()}
        sets_model.append(msets)
        sets_log.append(lsets)

        for lbl, pos_list in mpos.items():
            for p in pos_list:
                counts_model[lbl][p] += 1

        for lbl, pos_list in lpos.items():
            for p in pos_list:
                counts_log[lbl][p] += 1

    # normalize
    counts_model = {
        lbl: {pos: int(c) for pos, c in sorted(d.items())}
        for lbl, d in counts_model.items()
    }
    counts_log = {
        lbl: {pos: int(c) for pos, c in sorted(d.items())}
        for lbl, d in counts_log.items()
    }
    return counts_model, counts_log, sets_model, sets_log
