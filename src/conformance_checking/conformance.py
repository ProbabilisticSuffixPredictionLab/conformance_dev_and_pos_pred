import pm4py
import pandas as pd
import pickle

class ConformanceChecking():
    def __init__(self, path_process_model: str, path_prob_suffix_pred: str):
        self.path_process_model = path_process_model
        self.path_prob_suffix_pred = path_prob_suffix_pred
        
    def get_case_ids(self):
        with open(self.path_prob_suffix_pred, 'rb') as f:
            predicted_results = pickle.load(f)
        
        # Key: (Case name, prefix_length)
        case_keys_dict = predicted_results.keys()
        
        return case_keys_dict
    
    def __pre_proces_process_model(self):
        # Read BPMN model to pm4py object
        pm = pm4py.read.read_bpmn(self.path_process_model)
        
        # Convert BPMN to Petri Net: Return to Petri net, Initial Marking, Final Marking
        pn, im, fm = pm4py.convert.convert_to_petri_net(pm)
        
        return pn, im, fm
        
    def __pre_process_sampled_suffixes(self):
        with open(self.path_prob_suffix_pred, 'rb') as f:
            predicted_results = pickle.load(f)
        
        return predicted_results
    
    def __create_df(self, cases: list):
        # Single, deterministic base timestamp
        base_time = pd.Timestamp("2025-01-01 00:00:00")
        
        df = pd.DataFrame({
                'case:concept:name': range(len(cases)),
                'concept:name': cases
            })
        
        df['case:concept:name'] = df['case:concept:name'].astype(str)
        
        df = df.explode('concept:name', ignore_index=True)
        # Ensure concept:name is a string and not None
        df['concept:name'] = df['concept:name'].apply(lambda x: str(x) if pd.notnull(x) else 'UNKNOWN')
            
        # Compute offset grouped by case
        df['offset'] = df.groupby('case:concept:name').cumcount()
        
        df['time:timestamp'] = base_time + pd.to_timedelta(df['offset'], unit='min')
        # Fill missing timestamps with the last valid one
        last_valid_timestamp = df['time:timestamp'].dropna().max()
        df['time:timestamp'] = df['time:timestamp'].fillna(last_valid_timestamp)
        
        return df
        
    def conformance_of_sampled_suffixes(self, case_concept_name: str):
        f = lambda v, k : list(sorted(list(filter(lambda k: k[0] == v, k)), key = lambda x : x[1]))
        
        pn, im, fm = self.__pre_proces_process_model()
        
        predicted_results = self.__pre_process_sampled_suffixes()
        case_keys = f(case_concept_name, predicted_results.keys())
        
        list_predicted_sample_case_alignment = []
        list_mean_sample_case_alignment = []
        list_target_case_alignment = []
        
        for case_key in case_keys:                        
            # Prediction of case/ prefix: Prefix, Target Suffix, Most-likely prediction, All sampled suffixes - 1000 MC samples
            prefix, suffix, mean_prediction, predicted_sampled_suffixes = predicted_results[case_key]
            
            prefix_activities = [p['concept:name'] for p in prefix]
            suffix_activities = [s['concept:name'] for s in suffix]
            # Target Cases: Activity/ event labels
            target_case = prefix_activities + suffix_activities
            
            mean_suffix_activities =  [s['concept:name'] for s in mean_prediction]
            mean_case = prefix_activities + mean_suffix_activities
            
            predicted_sampled_suffixes_activities = [[s['concept:name'] for s in predicted_suffix] for predicted_suffix in predicted_sampled_suffixes]
            # Cases Activity/ Event labels of all prefix + samples suffixes
            predicted_sampled_cases = [prefix_activities + predicted_suffix for predicted_suffix in predicted_sampled_suffixes_activities]                    
            
            # Sampled cases
            df_sampled_cases = self.__create_df(predicted_sampled_cases)
            # Apply the alignments algorithm between a log and a process model. This method returns the full alignment diagnostics: Alignment, cost, fitness
            cf_sampled_cases = pm4py.conformance.conformance_diagnostics_alignments(df_sampled_cases, pn, im, fm, multi_processing=True)
            predicted_sample_case_alignment = [{"sampled case": predicted_sampled_cases[i],
                                                "alignment": cf_res['alignment'],
                                                "cost": cf_res['cost'],
                                                "fitness": cf_res['fitness']} for i, cf_res in enumerate(cf_sampled_cases)]
            list_predicted_sample_case_alignment.append(predicted_sample_case_alignment)
            
            # Most likely case
            df_mean_case = self.__create_df([mean_case])
            cf_mean_cases = pm4py.conformance.conformance_diagnostics_alignments(df_mean_case, pn, im, fm, multi_processing=True)
            mean_case_alignment = {"sampled case": mean_case,
                                    "alignment": cf_mean_cases[0]['alignment'],
                                    "cost": cf_mean_cases[0]['cost'],
                                    "fitness": cf_mean_cases[0]['fitness']}
            list_mean_sample_case_alignment.append(mean_case_alignment)
            
            # Target case
            df_target_case = self.__create_df([target_case])
            cf_target_cases = pm4py.conformance.conformance_diagnostics_alignments(df_target_case, pn, im, fm, multi_processing=True)
            target_case_alignment = {"sampled case": target_case,
                                      "alignment": cf_target_cases[0]['alignment'],
                                      "cost": cf_target_cases[0]['cost'],
                                      "fitness": cf_target_cases[0]['fitness']}
            list_target_case_alignment.append(target_case_alignment)
        
        return list_predicted_sample_case_alignment, list_mean_sample_case_alignment, list_target_case_alignment                
        
    
    