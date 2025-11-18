import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import pandas as pd
import numpy as np
from tqdm import trange
import re
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import OneSidedSelection
import torch
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
try:
    from torch.serialization import add_safe_globals
except ImportError:
    add_safe_globals = None

class DeviationLabeling:
    """
    Simplified deviation labeling:
    - Builds prefix dataset (X) and targets (y = list of deviation types that will occur in the suffix).
    - Saves .csv dataframe
    """
    def __init__(self, log_name: str, path_event_log: str, path_process_model: str):
        self.log_name = log_name
        self.path_event_log = path_event_log
        self.path_process_model = path_process_model

    def _load_log_csv(self):
        df = pd.read_csv(self.path_event_log)
        if self.log_name == "Helpdesk":
            # Rename for example important for helpdesk:
            rename = {
                "CaseID": "case:concept:name",
                "Activity": "concept:name",
                "CompleteTimestamp": "time:timestamp",
                "Resource": "org:resource",
            }
            df = df.rename(columns=rename)
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

        ev_log = log_converter.apply(
            df,
            variant=log_converter.Variants.TO_EVENT_LOG,
            parameters={log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"},
        )
        return df, ev_log

    def _pre_process_process_model(self):
        pm = pm4py.read.read_bpmn(self.path_process_model)
        pn, im, fm = pm4py.convert.convert_to_petri_net(pm)
        return pn, im, fm
    
    def _sanitize_label(self, s: str) -> str:
        return re.sub(r'[^0-9A-Za-z_]', '_', str(s))

    def _extract_deviations_from_alignment(self, ev_log, alignment_results: List[dict]):
        """
        From one alignment result per trace, produce:
          - dev_pos_by_case: case -> list of (position_index, dev_type)
          - dev_types: set of all deviation types encountered
        """
        dev_pos_by_case: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        dev_types: Set[str] = set()

        # iterate through the alignments in each trace:
        for trace, res in zip(ev_log, alignment_results):
            # Get the case id:
            cid = trace.attributes.get("concept:name")
            trace_event_labels = [t.get("concept:name") for t in trace]
            
            # Get the alignments
            aln_steps = res.get("alignment")
            aln_cleaned_steps = [(a,b) for a,b in aln_steps if a not in (None, "None") and b not in (None, "None")]
            
            if not aln_cleaned_steps:
                continue
            
            assert len(trace_event_labels) <= len(aln_cleaned_steps), \
                "Alignment length mismatch: trace has {} events, alignment has {}".format(trace_event_labels, aln_cleaned_steps)
            
            log_idx = 0
            for a, b in aln_cleaned_steps:
                # (a,b) and b == a: synchronuous move:
                if a == b:
                    log_idx += 1
                    continue 
                # (a, >>) log move on activity a:  activity a executed in the trace should not have been executed according to the model.
                if a != ">>" and b == ">>":
                    dt = str((a,b))
                    dev_pos_by_case[cid].append((log_idx, dt))
                    dev_types.add(dt) 
                # (>>, b) model move on b:activity b is prescribed by model but missing in the trace.
                elif a == ">>" and b != ">>":
                    dt = str((a,b))
                    dev_pos_by_case[cid].append((log_idx, dt))
                    dev_types.add(dt)
                else:
                    continue
                    
        return sorted(dev_types), dict(dev_pos_by_case), 

    def generate_individual_labels(self,
                                   trace_attr: List[str],
                                   max_prefix_cap: int = None,
                                   conf_runs: int = 100) -> Tuple[pd.DataFrame, List[dict], List[List[str]]]:
        """
        Run conformance and produce:
          - labeled_df: DataFrame with prefixes and binary columns for each dev type
        """
        df_raw, ev_log = self._load_log_csv()
        net, im, fm = self._pre_process_process_model()

        # Run 100 conformance alignment:
        best_D = ([], {})  # (dev_types, dev_pos_by_case)
        for _ in trange(conf_runs):
            conformance = pm4py.conformance.conformance_diagnostics_alignments(ev_log, net, im, fm, multi_processing=False)
            deviations, dev_pos_by_case = self._extract_deviations_from_alignment(ev_log, conformance)
            # Choose the smallest deviation set:
            if len(best_D[0]) == 0 or len(best_D[0]) >= len(deviations):
                best_D = (deviations, dev_pos_by_case)

        dev_types, dev_pos_by_case = best_D
        
        # Build prefixes and targets:
        df_sorted = df_raw.sort_values(["case:concept:name", "time:timestamp"]).reset_index(drop=True)
        
        # vocabularies for activities/resources
        activities = df_sorted["concept:name"].fillna("NA").astype(str).unique().tolist()
        resources  = df_sorted["org:resource"].fillna("NA").astype(str).unique().tolist()
        act2idx = {act: (i+1) for i, act in enumerate(sorted(activities))}   # 0 reserved for PAD
        res2idx = {res: (i+1) for i, res in enumerate(sorted(resources))}
        
        # case attribute encoders
        case_attr_list = trace_attr or []
        case_attr_encoders: Dict[str, LabelEncoder] = {}
        for ca in case_attr_list:
            le = LabelEncoder()
            if ca not in df_sorted.columns:
                df_sorted[ca] = "NA"
            col = df_sorted.groupby("case:concept:name")[ca].first().fillna("NA").astype(str).values
            le.fit(col)
            case_attr_encoders[ca] = le
            
        # determine L_max (cap if requested)
        case_lengths = df_sorted.groupby("case:concept:name").size().to_dict()
        L_max = max(case_lengths.values()) if len(case_lengths) > 0 else 0
        if max_prefix_cap is not None:
            L_max = min(L_max, max_prefix_cap)
            
        # build rows (store lists inside DataFrame cells)
        rows = []
        for cid, g in df_sorted.groupby("case:concept:name", sort=False):
            g = g.reset_index(drop=True)
            n = len(g)

            # encode case attrs once per case
            encoded_case_attrs = {}
            for ca in case_attr_list:
                raw = g[ca].iloc[0] if ca in g.columns else "NA"
                raw = "NA" if pd.isna(raw) else raw
                le = case_attr_encoders[ca]
                try:
                    enc = int(le.transform([str(raw)])[0])
                except Exception:
                    enc = len(le.classes_)
                encoded_case_attrs[ca] = enc

            start_ts = g.loc[0, "time:timestamp"]

            for i in range(n):  # prefix ends at index i
                # prepare lists of length L_max
                act_row = [0] * L_max
                res_row = [0] * L_max
                month_row = [0] * L_max  # only month number

                for p in range(1, L_max + 1):
                    if p-1 <= i:
                        # values from the log
                        act = str(g.loc[p-1, "concept:name"]) if "concept:name" in g.columns else "NA"
                        res = str(g.loc[p-1, "org:resource"]) if "org:resource" in g.columns else "NA"
                        ts  = g.loc[p-1, "time:timestamp"]
                        # encoded values
                        act_row[p-1] = act2idx.get(act, 0)
                        res_row[p-1] = res2idx.get(res, 0)

                        if pd.isna(ts) or pd.isna(start_ts):
                            month_row[p-1] = 0
                        else:
                            month_row[p-1] = int(ts.month)
                    else:
                        # PAD remains 0
                        pass

                # for prefixes build classes for suffix labels
                devs_in_suffix = []
                for pos, dt in dev_pos_by_case.get(cid, []):
                    # When first activity produces log move:
                    # if pos == 0 and i == 0:
                    #    devs_in_suffix.append(dt) 
                    # deviations and last possible model moves
                    if pos > i or (pos >= i and i == n - 1):
                        devs_in_suffix.append(dt)

                # base row fields
                base = {"case_id": str((cid, i+1)),
                        "activities": act_row,
                        "resources": res_row,
                        "months": month_row}
                
                # add case_attr_<name> columns
                for ca in case_attr_list:
                    base[f"trace_attr_{ca}"] = encoded_case_attrs[ca]

                # add y_<dev_label> target columns
                for dt in dev_types:
                    base[f"y_{dt}"] = 1 if dt in devs_in_suffix else 0

                rows.append(base)

        df_flat = pd.DataFrame(rows)

        # encoders: values to ids and vice-versa
        encoders = {"activity_ids": act2idx,
                    "resource_ids": res2idx,
                    "trace_attr_encoders": case_attr_encoders,
                    "deviations": dev_types,
                    "L_max": L_max}

        return df_flat, encoders
        
class TrainTestSplit:
    def __init__(self, df_labled_deviations):
        self.df_labeled_deviations = df_labled_deviations
        
    def data_split(self,
                   seed: int = 42,
                   train_frac: float = 2/3):
        
        cases = self.df_labeled_deviations["case_id"].unique()
        n_cases = len(cases)

        # Random, reproducible sampling
        rng = np.random.default_rng(seed)
        train_cases = rng.choice(cases,
                                 size=int(n_cases * train_frac),
                                 replace=False)

        # Build splits
        train_df = self.df_labeled_deviations[self.df_labeled_deviations["case_id"].isin(train_cases)]
        test_df  = self.df_labeled_deviations[~self.df_labeled_deviations["case_id"].isin(train_cases)]
        
        return train_df, test_df
    
class Undersampling:
    def __init__(self, train_df, list_dynamic_cols):
        self.train_df = train_df
        self.list_dynamic_cols = list_dynamic_cols

    @staticmethod
    def _flat_list_cols(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        df_flat = df.copy()
        mapping = {}
        for col in cols:
            if col not in df_flat.columns:
                continue
            expanded = pd.DataFrame(df_flat[col].tolist(), index=df_flat.index)
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]
            df_flat = pd.concat([df_flat.drop(columns=[col]), expanded], axis=1)
            mapping[col] = expanded.columns.tolist()
        return df_flat, mapping

    @staticmethod
    def _reflat_list_cols(df_flat: pd.DataFrame, mapping: Dict[str, List[str]], original_cols: List[str]) -> pd.DataFrame:
        df_restored = df_flat.copy()
        for original_col, expanded_cols in mapping.items():
            existing = [c for c in expanded_cols if c in df_restored.columns]
            if not existing:
                continue
            df_restored[original_col] = df_restored[existing].values.tolist()
            df_restored = df_restored.drop(columns=existing)
        df_restored = df_restored[[c for c in original_cols if c in df_restored.columns]]
        return df_restored

    def one_sided_selection_undersampling(self):
        """
        we undersample with respect to all columns that start with y_
        """
        original_cols = self.train_df.columns.tolist()
        list_cols = self.list_dynamic_cols
        train_df_flat, flattened_mapping = self._flat_list_cols(self.train_df, list_cols)

        feature_cols = [col for col in train_df_flat.columns if not col.startswith('y_') and col != 'case_id']
        target_cols = [col for col in train_df_flat.columns if col.startswith('y_')]

        resampled_indices = []
        no_true_class = []

        for target_col in target_cols:
            y = train_df_flat[target_col].values
            if np.sum(y == 1) == 0:
                no_true_class.append(target_col)
                continue

            X = train_df_flat[feature_cols].values
            oss = OneSidedSelection(sampling_strategy='auto', random_state=42)
            oss.fit_resample(X, y)
            resampled_indices.append(set(oss.sample_indices_))

        if not resampled_indices:
            df_resampled = self.train_df.drop(columns=no_true_class, errors="ignore").copy()
            df_resampled = df_resampled[[c for c in original_cols if c in df_resampled.columns]]
            return df_resampled, no_true_class

        common_indices = sorted(set.intersection(*resampled_indices))
        df_resampled_flat = train_df_flat.iloc[common_indices].reset_index(drop=True)
        df_resampled_flat = df_resampled_flat.drop(columns=no_true_class, errors="ignore")

        df_restored = self._reflat_list_cols(df_resampled_flat, flattened_mapping, original_cols)
        
        return df_restored, no_true_class
    
class CleanDatasets:
    def __init__(self, train_undersmpl_df: pd.DataFrame, test_df: pd.DataFrame, undersmpl_removed_cols: List[str]):
        self.train_undersmpl_df: pd.DataFrame = train_undersmpl_df
        self.test_df: pd.DataFrame = test_df
        self.undersmpl_removed_cols: List[str] = undersmpl_removed_cols
            
    def clean(self):
        # removed columns not used in undersampling
        for rc in self.undersmpl_removed_cols:
            if rc in self.train_undersmpl_df.columns:
                self.train_undersmpl_df = self.train_undersmpl_df.drop(rc, axis=1)
            if rc in self.test_df.columns:
                self.test_df = self.test_df.drop(rc, axis=1)
                
        # Check column values
        y_cols_train = [col for col in self.train_undersmpl_df.columns if col.startswith('y_')]
        y_cols_test = [col for col in self.test_df.columns if col.startswith('y_')]
            
        assert y_cols_train == y_cols_test
            
        for y_col in y_cols_train:
            y_col_values_train = self.train_undersmpl_df[y_col].tolist()
            y_col_values_test = self.test_df[y_col].tolist()
            # check if is true in the data
            if (1 in y_col_values_train) and (1 in y_col_values_test):
                continue
            # remove form the data if only false
            else:
                self.train_undersmpl_df = self.train_undersmpl_df.drop(y_col, axis=1)
                self.test_df = self.test_df.drop(y_col, axis=1)
            
        return self.train_undersmpl_df, self.test_df
                
class PrefixDataset(Dataset):
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, activity_col: str, resource_col: str, month_col: str, trace_cols, y_cols: List[str]):
        # datasets
        self.df_train = df_train.reset_index(drop=True)
        self.df_test = df_test.reset_index(drop=True)
        # column values
        self.activity_col = activity_col
        self.resource_col = resource_col
        self.month_col = month_col
        self.trace_cols = trace_cols
        self.y_cols = y_cols
        self._tensor_dataset = None

    def __len__(self, train: bool = True):
        if train:
            return len(self.df_train)
        else:
            return len(self.df_test)

    def __getitem__(self, idx, train: bool = True):
        if train:
            row = self.df_train.iloc[idx]
        else:
            row = self.df_test.iloc[idx]
            
        x_act = torch.tensor(row[self.activity_col], dtype=torch.long)
        x_res = torch.tensor(row[self.resource_col], dtype=torch.long)
        x_month = torch.tensor(row[self.month_col], dtype=torch.long)

        if self.trace_cols:
            trace_vals = np.asarray(row[self.trace_cols], dtype=np.int64)
            trace_feats = torch.tensor(trace_vals, dtype=torch.long)
        else:
            trace_feats = torch.zeros(0, dtype=torch.long)

        if self.y_cols:
            y_vals = np.asarray(row[self.y_cols], dtype=np.int64)
            y = torch.tensor(y_vals, dtype=torch.long)
        else:
            y = torch.zeros(0, dtype=torch.long)

        return x_act, x_res, x_month, trace_feats, y

    def _to_tensor_dataset(self, df, device=None, cache_key=None):
        device = torch.device(device) if device is not None else torch.device("cpu")
        
        # if cache_key is not None and device.type == "cpu" and cache_key in self._tensor_dataset:
        #     return self._tensor_dataset[cache_key]

        act_arr = np.asarray(df[self.activity_col].tolist(), dtype=np.int64)
        res_arr = np.asarray(df[self.resource_col].tolist(), dtype=np.int64)
        month_arr = np.asarray(df[self.month_col].tolist(), dtype=np.int64)

        if self.trace_cols:
            trace_arr = df[self.trace_cols].to_numpy(dtype=np.int64, copy=True)
        else:
            trace_arr = np.zeros((len(df), 0), dtype=np.int64)

        if self.y_cols:
            y_arr = df[self.y_cols].to_numpy(dtype=np.int64, copy=True)
        else:
            y_arr = np.zeros((len(df), 0), dtype=np.int64)

        x_act = torch.tensor(act_arr, dtype=torch.long, device=device)
        x_res = torch.tensor(res_arr, dtype=torch.long, device=device)
        x_month = torch.tensor(month_arr, dtype=torch.long, device=device)
        trace_tensor = torch.tensor(trace_arr, dtype=torch.long, device=device)
        y_tensor = torch.tensor(y_arr, dtype=torch.long, device=device)

        dataset = TensorDataset(x_act, x_res, x_month, trace_tensor, y_tensor)

        # if cache_key is not None and device.type == "cpu":
        #     self._tensor_dataset[cache_key] = dataset
        
        return dataset
    
    def tensor_datset_encoding(self, device):
        train_dataset = self._to_tensor_dataset(self.df_train, device)
        test_dataset = self._to_tensor_dataset(self.df_test, device)
        
        return train_dataset, test_dataset

    @staticmethod
    def save_datasets(train_dataset, test_dataset, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        train_path = os.path.join(save_path, "train_set.pkl")
        test_path = os.path.join(save_path, "test_set.pkl")

        torch.save(train_dataset, train_path)
        torch.save(test_dataset, test_path)

        return train_path, test_path

    @staticmethod
    def load_datasets(save_path: str, map_location=None):
        train_path = os.path.join(save_path, "train_set.pkl")
        test_path = os.path.join(save_path, "test_set.pkl")

        if add_safe_globals is not None:
            add_safe_globals([TensorDataset])

        def _torch_load(path):
            load_kwargs = {}
            if map_location is not None:
                load_kwargs["map_location"] = map_location
            try:
                return torch.load(path, weights_only=False, **load_kwargs)
            except TypeError:
                return torch.load(path, **load_kwargs)

        train_dataset = _torch_load(train_path)
        test_dataset = _torch_load(test_path)

        return train_dataset, test_dataset
