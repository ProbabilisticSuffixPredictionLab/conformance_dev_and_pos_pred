# src/encoding.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import numpy as np
import pandas as pd

@dataclass
class PrefixEncoder:
    max_prefix_len: int = 30
    use_time_since_start: bool = True
    use_time_since_prev: bool = True
    add_bias: bool = True

    # werden in fit_vocab() befüllt
    act_vocab: Dict[str, int] = field(default_factory=dict)
    res_vocab: Dict[str, int] = field(default_factory=dict)

    # ----- 1) Vokabulare lernen -----
    def fit_vocab(self, prefixes_df: pd.DataFrame) -> None:
        acts, ress = set(), set()
        for seq in prefixes_df["act_seq"]:
            acts.update(seq)
        for seq in prefixes_df["res_seq"]:
            ress.update(seq)
        self.act_vocab = {a: i for i, a in enumerate(sorted(acts))}
        self.res_vocab = {r: i for i, r in enumerate(sorted(ress))}

    # ----- 2) Dimension pro Schritt -----
    @property
    def step_dim(self) -> int:
        d = len(self.act_vocab) + len(self.res_vocab)
        if self.use_time_since_start: d += 1
        if self.use_time_since_prev:  d += 1
        if self.add_bias:             d += 1
        return d

    # ----- 3) Einzelnen Schritt encoden -----
    def _encode_step(self, act: str | None, res: str | None,
                     t_since_start: float | None, t_since_prev: float | None) -> np.ndarray:
        v = np.zeros(self.step_dim, dtype=np.float32)
        idx = 0

        # One-Hot Activity
        if act in self.act_vocab:
            v[self.act_vocab[act]] = 1.0
        idx += len(self.act_vocab)

        # One-Hot Resource
        if res in self.res_vocab:
            v[idx + self.res_vocab[res]] = 1.0
        idx += len(self.res_vocab)

        # Zeitfeatures
        if self.use_time_since_start:
            v[idx] = 0.0 if t_since_start is None else np.log1p(max(0.0, float(t_since_start))) / 10.0
            idx += 1
        if self.use_time_since_prev:
            v[idx] = 0.0 if t_since_prev is None else np.log1p(max(0.0, float(t_since_prev))) / 10.0
            idx += 1

        if self.add_bias:
            v[idx] = 1.0
        return v

    # ----- 4) Hauptfunktion: encoden -----
    def encode(self, labeled_df: pd.DataFrame):
        """
        Erwartet Spalten:
          - 'case:concept:name', 'prefix_len', 'act_seq', 'res_seq', 'time_since_start_seq'
          - Zielspalten: beginnen mit 'y_'
        Rückgabe:
          X_seq  : (N, T, D)
          X_flat : (N, T*D + trace_dim)  [trace_dim = 2: norm. prefix_len + norm. elapsed]
          Y      : (N, L)
          mask   : (N, T)  (1=echter Schritt, 0=Padding)
          meta   : dict (dev_cols, step_dim, T, act_vocab, res_vocab, trace_features)
        """
        assert {"act_seq","res_seq","time_since_start_seq","prefix_len"}.issubset(labeled_df.columns), \
            "labeled_df hat nicht alle benötigten Spalten."

        # Zielspalten (y_individual)
        dev_cols = [c for c in labeled_df.columns if c.startswith("y_")]
        L = len(dev_cols)
        N = len(labeled_df)
        T = self.max_prefix_len
        D = self.step_dim

        # Trace-Features (minimal): normiertes prefix_len + normierte elapsed_time
        def _elapsed(lst):
            return float(lst[-1]) if (isinstance(lst, list) and len(lst) > 0) else 0.0

        pfx_len = labeled_df["prefix_len"].values.astype(np.float32)
        elapsed = labeled_df["time_since_start_seq"].apply(_elapsed).values.astype(np.float32)

        pfx_len_norm = (pfx_len / np.maximum(1.0, T)).reshape(-1, 1)
        elapsed_norm = (np.log1p(np.maximum(0.0, elapsed)) / 10.0).reshape(-1, 1)
        X_trace = np.hstack([pfx_len_norm, elapsed_norm]).astype(np.float32)  # (N, 2)

        # Container
        X_seq = np.zeros((N, T, D), dtype=np.float32)
        mask  = np.zeros((N, T),    dtype=np.float32)

        # >>> Y direkt aus DataFrame ziehen (robust ggü. ':' / Leerzeichen)
        Y = labeled_df[dev_cols].astype(np.float32).to_numpy(copy=True)  # (N, L)

        # Sequenz-Features füllen
        for i, row in enumerate(labeled_df.itertuples(index=False)):
            acts = list(getattr(row, "act_seq"))
            ress = list(getattr(row, "res_seq"))
            tss  = list(getattr(row, "time_since_start_seq"))

            # t_since_prev aus t_since_start ableiten
            tsp = [None]
            for k in range(1, len(tss)):
                tsp.append(float(tss[k]) - float(tss[k-1]))

            real_len = min(len(acts), T)
            for t in range(real_len):
                X_seq[i, t, :] = self._encode_step(
                    acts[t] if t < len(acts) else None,
                    ress[t] if t < len(ress) else None,
                    tss[t]  if t < len(tss)  else None,
                    tsp[t]  if t < len(tsp)  else None
                )
            mask[i, :real_len] = 1.0

        # Sequenz flatten + Trace-Features anhängen
        X_flat = np.hstack([X_seq.reshape(N, T * D), X_trace]).astype(np.float32)

        meta = {"dev_cols": dev_cols,
                "step_dim": D,
                "T": T,
                "act_vocab": self.act_vocab,
                "res_vocab": self.res_vocab,
                "trace_features": ["prefix_len_norm", "elapsed_norm"]}
        
        return X_seq, X_flat, Y, mask, meta

