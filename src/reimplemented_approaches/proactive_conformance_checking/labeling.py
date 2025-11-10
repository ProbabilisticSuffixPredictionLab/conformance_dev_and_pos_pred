import pandas as pd
import os, json
from tqdm import trange 
from typing import List, Dict, Tuple

import pm4py
from pm4py.objects.conversion.log import converter as log_converter

from collections import defaultdict
from typing import List, Tuple, Dict, Set 

class DeviationLabeling():
    def __init__(self, path_event_log, log_name, path_process_model):
        self.path_event_log = path_event_log
        self.log_name = log_name
        
        self.path_process_model = path_process_model
        
    def load_log_csv(self, path_csv: str):    
        df = pd.read_csv(path_csv)
        
        if self.log_name == 'Helpdesk':
            rename = {
                "Case ID": "case:concept:name",
                "Activity": "concept:name",
                "Complete Timestamp": "time:timestamp",
                "Resource": "org:resource"
            }
        df = df.rename(columns=rename)
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

        ev_log = log_converter.apply(df,
                                     variant=log_converter.Variants.TO_EVENT_LOG,
                                     parameters={log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"})
        return df, ev_log  
    
    def pre_proces_process_model(self):
        # Read BPMN model to pm4py object
        pm = pm4py.read.read_bpmn(self.path_process_model)
        # Convert BPMN to Petri Net: Return to Petri net, Initial Marking, Final Marking
        pn, im, fm = pm4py.convert.convert_to_petri_net(pm)
        return pn, im, fm

    def _alignment_steps_list(self, res: dict):
        """
        # Get alignment results
        """
        aln = (res.get("alignment") or
            res.get("alignment_info") or
            res.get("diag_alignment") or
            res.get("aligned_trace") or
            res.get("alignment_result"))
        return aln

    def _choose_best_run_with_deviations(self, ev_log, runs: List[List[dict]]) -> Tuple[List[dict], Dict]:
        """
        Selects the best alignment run according to the paper's criteria:
        1. Minimize |D_L,B| — the total number of unique deviation types across the log.
        2. Tie-breaker: minimize the total number of deviation steps in the log.

        Additionally computes:
        - dev_pos_by_case: case_id → [(log_index, deviation_type), ...]
        - dev_types: sorted list of all unique deviation types

        Returns:
            (best_diagnostics, metadata_dict)
        """
        INVISIBLE_MOVES: Set[str] = {None, "None", "", "τ", "tau"}  # invisible/noise moves to ignore

        best_diags = None
        best_score = (10**9, 10**9)  # (|D|, total_dev_steps)
        best_metadata = {}

        for diags in runs:
            dev_pos_by_case: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
            dev_types: Set[str] = set()
            log_idx = 0

            for trace, res in zip(ev_log, diags):
                # Extract case identifier
                cid = (
                    trace.attributes.get("concept:name")
                    or trace.attributes.get("case:concept:name")
                )

                # Get alignment steps as list of (model_move, log_move)
                aln_steps = self._alignment_steps_list(res)
                if not aln_steps:
                    continue

                # Parse deviations directly while iterating
                for m, l in aln_steps:
                    # Count real log steps (even if later discarded as deviation)
                    if l != ">>":
                        log_idx += 1

                    # Model move: required by model but missing in log
                    if l == ">>" and m not in INVISIBLE_MOVES and m != ">>":
                        dev_type = f"model:{m}"
                        dev_pos_by_case[cid].append((log_idx, dev_type))
                        dev_types.add(dev_type)

                    # Log move: present in log but not allowed by model
                    if m == ">>" and l not in INVISIBLE_MOVES and l != ">>":
                        dev_type = f"log:{l}"
                        dev_pos_by_case[cid].append((log_idx, dev_type))
                        dev_types.add(dev_type)

            # Compute scores
            total_dev_steps = sum(len(positions) for positions in dev_pos_by_case.values())
            current_score = (len(dev_types), total_dev_steps)

            # Update best if improved
            if current_score < best_score:
                best_score = current_score
                best_diags = diags
                best_metadata = {
                    "dev_pos_by_case": dict(dev_pos_by_case),
                    "dev_types": sorted(dev_types),
                    "D_size": best_score[0],
                    "total_dev_steps": best_score[1],
                }

        return best_diags, best_metadata

    def generate_individual_labels(self, 
                                   df_raw: pd.DataFrame,
                                   ev_log,
                                   net, im, fm,
                                   max_prefix_len: int | None = None,
                                   n_runs: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute alignment-based conformance checking on log:
        - Gorhs et.al.: 100 runs, choose the run with the least deviations in D_{L,B}
        """
        # mehrfache Alignments (100 Runs wie Grohs), mit optionaler Progressbar
        os.makedirs("artifacts", exist_ok=True)

        # run alignments 100 times:
        runs = []
        for _ in trange(max(1, n_runs), desc="Alignments"):
            diags = pm4py.conformance.conformance_diagnostics_alignments(ev_log, net, im, fm, parameters={"activity_key": ACT, "case_id_key": CASE})
            runs.append(diags)

        # choose best run 
        diags, info = self._choose_best_run_with_deviations(ev_log, runs)
        dev_pos_by_case = info["dev_pos_by_case"]
        dev_types = info["dev_types"]
        
        # Best-Run für Reproduzierbarkeit speichern (optional, aber empfohlen)
        with open("artifacts/best_run_alignments.json", "w") as f:
            json.dump(diags, f)
        with open("artifacts/best_run_meta.json", "w") as f:
            json.dump(info, f)

        # 3) Prefix-Labels bauen
        df_sorted = df_raw.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
        out_rows = []
        for cid, g in df_sorted.groupby("case:concept:name", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)
            # Default: bis n-1 (Prefix darf nicht der komplette Trace sein)
            # Bis zum vollen Trace (n) encodieren – wie im Paper 4.2
            T = min(max_prefix_len or n, n)

            for k in range(1, T + 1):
                y = {f"y_{dt}": 0 for dt in dev_types}
                for pos, dt in dev_pos_by_case.get(cid, []):
                    # Standardregel: Abweichung liegt in der Zukunft des Prefix
                    if pos > k:
                        y[f"y_{dt}"] = 1
                    # Paper-Spezialfall: model move "nach dem Ende" -> volles Prefix (k == n) soll 1 bekommen
                    elif (k == n) and dt.startswith("model:") and (pos == n):
                        y[f"y_{dt}"] = 1

                out_rows.append({
                    "case:concept:name": cid,
                    "prefix_len": k,
                    "act_seq": g.loc[:k-1, "concept:name"].tolist(),
                    "res_seq": g.loc[:k-1, "org:resource"].tolist(),
                    "time_since_start_seq": (g.loc[:k-1, "time:timestamp"] - g.loc[0, "time:timestamp"]).dt.total_seconds().fillna(0.0).tolist(),
                    **y})

        labeled_df = pd.DataFrame(out_rows)
        meta = {"dev_types": dev_types,
                "n_deviation_types": len(dev_types),
                "best_run_D_size": info["D_size"],
                "best_run_total_dev_steps": info["total_dev_steps"]}

        return labeled_df, meta
