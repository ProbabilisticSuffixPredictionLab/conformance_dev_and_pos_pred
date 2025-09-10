import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

def load_results(path: str) -> None:
        """
        Load all .pkl files from the given directory into self.d_con_results.
        """
        
        # i = 0
        
        results = []
        print(f"Looking for .pkl files in: {path}")
        print(f"Resolved absolute path: {Path(path).resolve()}")
        file_paths = list(Path(path).glob("*.pkl"))
        print(f"Found {len(file_paths)} files")

        for file_path in file_paths:
            with file_path.open('rb') as f:
                data = pickle.load(f)
                results.append(data)
                print(f"Loaded conformal result of .pkl: {file_path.name}")
                
                # i = i+1
                # if i == 1:
                   # break
                
        print("Loaded all conformal results!")
        return results 
    