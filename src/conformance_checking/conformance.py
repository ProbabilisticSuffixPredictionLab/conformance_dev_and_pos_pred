import pm4py
import pandas as pd

class ConformanceChecking():
    def __init__(self, path_process_model):
        self.path_process_model = path_process_model
    
    def pre_proces_process_model(self):
        # Read BPMN model to pm4py object
        pm = pm4py.read.read_bpmn(self.path_process_model)
        # Convert BPMN to Petri Net: Return to Petri net, Initial Marking, Final Marking
        pn, im, fm = pm4py.convert.convert_to_petri_net(pm)
        return pn, im, fm
                
    def conformance_of_sampled_suffixes(self, log_name: str, result_values: list):        
        # To-Be Process model
        pn, im, fm = self.pre_proces_process_model()
            
        if log_name == 'Helpdesk':
            concept_name = 'Activity'
        else:
            concept_name = 'concept:name'
            
        prefix, suffix, mostlikely_prediction, predicted_sampled_suffixes = result_values
        
        prefix_activities = [p[concept_name] for p in prefix]
        
        # Target Cases
        suffix_activities = [s[concept_name] for s in suffix]
        target_case = prefix_activities + suffix_activities
        df_target_case = self.__create_df(cases=[target_case])
        
        cf_target_cases = pm4py.conformance.conformance_diagnostics_alignments(df_target_case, pn, im, fm, multi_processing=False)
        target_case_alignment = {"prefix": prefix_activities,
                                 "target_suffix": suffix_activities,
                                 "alignment": cf_target_cases[0]['alignment'],
                                 "cost": cf_target_cases[0]['cost'],
                                 "fitness": cf_target_cases[0]['fitness']}
            
        # Most likely case
        mostlikely_suffix_activities =  [s[concept_name] for s in mostlikely_prediction]
        mostlikely_case = prefix_activities + mostlikely_suffix_activities
        df_mostlikely_case = self.__create_df(cases=[mostlikely_case])
        
        cf_mostlikely_cases = pm4py.conformance.conformance_diagnostics_alignments(df_mostlikely_case, pn, im, fm, multi_processing=False)
        mostlikely_case_alignment = {"prefix": prefix_activities,
                                     "ml_suffix": mostlikely_suffix_activities,
                                     "alignment": cf_mostlikely_cases[0]['alignment'],
                                     "cost": cf_mostlikely_cases[0]['cost'],
                                     "fitness": cf_mostlikely_cases[0]['fitness']}
            
        # Sampled cases
        predicted_sampled_suffixes_activities = [[s[concept_name] for s in predicted_suffix] for predicted_suffix in predicted_sampled_suffixes]
        predicted_sampled_cases = [prefix_activities + predicted_suffix for predicted_suffix in predicted_sampled_suffixes_activities]  
        df_sampled_cases = self.__create_df(cases=predicted_sampled_cases)

        cf_sampled_cases = pm4py.conformance.conformance_diagnostics_alignments(df_sampled_cases, pn, im, fm, multi_processing=False)
        predicted_sample_case_alignment = [{"prefix": prefix_activities,
                                            "sampled_suffix": predicted_sampled_suffixes_activities[i],
                                            "alignment": cf_res['alignment'],
                                            "cost": cf_res['cost'],
                                            "fitness": cf_res['fitness']} for i, cf_res in enumerate(cf_sampled_cases)]
        
        return target_case_alignment, mostlikely_case_alignment, predicted_sample_case_alignment
              
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
    