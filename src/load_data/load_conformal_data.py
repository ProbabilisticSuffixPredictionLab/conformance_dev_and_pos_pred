import pickle
from pathlib import Path

def load_results(path: str) -> None:
        """
        Load all .pkl files from the given directory into self.d_con_results.
        """
        results = []
        print(f"Looking for .pkl files in: {path}")
        print(f"Resolved absolute path: {Path(path).resolve()}")
        file_paths = list(Path(path).glob("*.pkl"))
        print(f"Found {len(file_paths)} files")

        for file_path in file_paths:
            with file_path.open('rb') as f:
                data = pickle.load(f)
                if len(file_paths) == 1:
                    results = data
                else:
                    results.append(data)
                
        print("Loaded all conformal results!")
        return results 
    