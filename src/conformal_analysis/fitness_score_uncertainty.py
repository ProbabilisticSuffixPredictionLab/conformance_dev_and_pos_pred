
class FitnessUncertainty():
    def __init__(self, dropout_fitness_std, no_dropout_fitness_std):
        self.dropout_fitness_std = dropout_fitness_std
        self.no_dropout_fitness_std = no_dropout_fitness_std
    
    def total_fitness_score_sd(self):
        return self.dropout_fitness_std
    
    def aleatoric_fitness_score_sd(self):
        return self.no_dropout_fitness_std
    
    def epistemic_fitness_score(self):
        return [float(drop_sd_pref_length - self.no_dropout_fitness_std[i]) for i, drop_sd_pref_length in enumerate(self.dropout_fitness_std)]