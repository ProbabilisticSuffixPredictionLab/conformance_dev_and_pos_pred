import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

def load_results(path_d_con_results: str) -> None:
        """
        Serially load all .pkl files from the given directory into self.d_con_results.
        """
        
        i = 0
        
        results = []
        for file_path in Path(path_d_con_results).glob("*.pkl"):
            with file_path.open('rb') as f:
                data = pickle.load(f)
                results.append(data)
                print(f"Loaded conformal result of .pkl: {file_path.name}")
                
                i = i+1
                
                if i == 2:
                    break
                
        print("Loaded all conformal results!")
        return results  
        
def _load_single_pickle(path_str: str):
    """
    Helper for parallel loading: unpickle a single file.
    Returns (filename, unpickled_data).
    """
    with open(path_str, 'rb') as f:
        data = pickle.load(f)
    return Path(path_str).name, data

def load_results_parallel(path_d_con_results: str, max_workers: Optional[int] = None) -> None:
    """
    Load all .pkl files in parallel using ProcessPoolExecutor.
    """
    results = []
    pkl_paths = list(Path(path_d_con_results).glob("*.pkl"))
    max_workers = max_workers or len(pkl_paths)

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        # schedule all loads
        futures = {
            exe.submit(_load_single_pickle, str(path)): path
            for path in pkl_paths
        }
        # collect as they finish
        for fut in as_completed(futures):
            filename, data = fut.result()
            results.append(data)
            # print(f"Loaded conformal result of .pkl: {filename}")
    print("Loaded all conformal results in parallel!")
    return results
    