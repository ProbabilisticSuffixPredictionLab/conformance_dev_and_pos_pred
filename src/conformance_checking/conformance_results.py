import os
import numpy as np
from typing import List, Dict, Optional, Any

# Parallelizing
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Call the conformance checking logic
from conformance_checking.conformance import ConformanceChecking

# Module-level globals populated by initializer
_CONF_OBJ = None
_LOG_NAME = None

def _init_worker(conformance_obj: ConformanceChecking, log_name: str):
    """
    Runs once in each worker process. Put the conformance object into a global
    so worker tasks can call its method directly without re-pickling per task.
    """
    global _CONF_OBJ, _LOG_NAME
    # optional: try to use fork start method on Unix for faster spawn (no-op if disallowed)
    try:
        multiprocessing.set_start_method('fork', force=False)
    except Exception:
        pass

    _CONF_OBJ = conformance_obj
    _LOG_NAME = log_name

def _get_pid():
    """Small noop to force worker creation and return its pid."""
    return os.getpid()

def _worker(task):
    """
    Worker receives: task = (case_name, prefix_length, values)
    Calls the original conformance method on the worker-local conformance object.
    Returns: (case_key, target_con, ml_con, sampled_cons)
    """
    global _CONF_OBJ, _LOG_NAME
    if _CONF_OBJ is None:
        raise RuntimeError("Worker not initialized with conformance object")

    case_name, prefix_length, values = task

    # Call the original method exactly as-is
    target_con, ml_con, smpls_con = _CONF_OBJ.conformance_of_sampled_suffixes(log_name=_LOG_NAME, result_values=values)

    return (case_name, prefix_length), target_con, ml_con, smpls_con

# Class that provides methods to evaluate suffix prediction using alignment-based conformance checking:
class ConformanceResults:
    def __init__(self, log_name: Optional[str] = "", data: List[Dict[str, Any]] = None, conformance_object: ConformanceChecking = None):
        """
        - log_name: Optional log name for identification.
        - d_con_results: List of dicts with evaluation results form the probabilistic suffix prediction model on the conformal dataset (validation).
        - conformance_object: A ConformanceChecking object -> Implements the chosen (alignment-based) conformance check algorithm.
        """
        self.log_name = log_name
        self.data = data
        self.conformance_object = conformance_object
         
    def conformance_results(self, target_workers: int = 64):
        """
        Use same parallelization as for the fitness score calibration: Evaluate the test cases and keep
        fitness, costs, alignments
        """
        
        # Build task list: (case_name, prefix_length, values)
        tasks = []
        
        for results in self.data:
            for (case_name, prefix_length), values in results.items():
                tasks.append((case_name, prefix_length, values))

        if not tasks:
            return {'case_id': [], 'target_conformance': [], 'ml_conformance': [], 'samples_conformance': []}

        system_cpus = os.cpu_count() or 1
        workers = min(target_workers, system_cpus)

        conformance_results = {'case_id': [], 'target_conformance': [], 'ml_conformance': [], 'samples_conformance': []}

        # IMPORTANT: if running on Windows, call this under `if __name__ == '__main__':`
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(self.conformance_object, self.log_name)) as exc:

            # Ensure workers are actually started and capture their PIDs
            noop_futures = [exc.submit(_get_pid) for _ in range(workers)]
            worker_pids = [f.result() for f in noop_futures]

            # Bind each worker to a unique core (Linux only)
            if hasattr(os, "sched_setaffinity"):
                for idx, pid in enumerate(worker_pids):
                    core_id = idx  # assign core 0..workers-1
                    try:
                        os.sched_setaffinity(pid, {core_id})
                    except PermissionError:
                        # lack of permission — ignore and continue
                        pass
                    except Exception:
                        # If pid disappeared or other OS issues, ignore
                        pass
            else:
                # Not Linux: affinity not available here; consider using psutil on Windows/macOS
                pass

            # Chunksize tuning: moderate chunks reduce scheduling overhead but keep load balanced
            chunksize = max(1, len(tasks) // (workers * 4))

            # Map tasks to worker (preserves order), no as_completed usage
            for case_key, target_con, ml_con, smpls_con in exc.map(_worker, tasks, chunksize=chunksize):
                
                conformance_results['case_id'].append(case_key)
                conformance_results['target_conformance'].append(target_con)
                conformance_results['ml_conformance'].append(ml_con)
                
                sampled_fitnesses = np.array([x for x in smpls_con])
                conformance_results['samples_conformance'].append(sampled_fitnesses)

        return conformance_results
    
    # Testing pupose: 
    def quick_fitness(self):
        conf_obj = self.conformance_object
        tasks = []
        for (case_name, prefix_length), values in self.data.items():
            # Call the conformance method for testing.
            cfr = conf_obj.conformance_of_sampled_suffixes(log_name=self.log_name, result_values=values)
            break
        if not tasks:
            return {'case_id': [], 'target_fitness': [], 'ml_fitness': [], 'samples_fitness': []}
        

        
        
    
    
