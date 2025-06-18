import numpy as np
import math
from collections import defaultdict
from typing import List, Dict, Optional, Any

from conformance_checking.conformance import ConformanceChecking

class ConformalRiskUncertaintyPrediction:
    def __init__(self, d_inf_results: List[Dict[str, Any]], conformance_object:ConformanceChecking , log_name: Optional[str] = ""):
        """
        d_con_results: List of dicts with evaluation results form the probabilistic suffix prediction model on the conformal dataset (validation).
        conformance_object: A ConformanceChecking object -> Implements the chosen (alignment-based) conformance check algorithm.
        log_name: Optional log name for identification.
        """
        self.log_name = log_name
        self.d_inf_results = d_inf_results
        self.conformance_object = conformance_object
         
        