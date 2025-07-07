"""
Utility functions for VQE optimization baselines.
"""

import json
from typing import Dict, Any


def load_and_analyze_results(json_file: str) -> Dict[str, Any]:
    """
    Load and analyze results from a JSON file.
    
    Args:
        json_file: Path to the JSON results file
    
    Returns:
        dict: Loaded results dictionary
    """
    with open(json_file, 'r') as f:
        loaded_results = json.load(f)

    print("Loaded results for molecular distances:", list(loaded_results.keys()))

    print("\nSummary of results:")
    for dist in loaded_results:
        print(f"\nDistance {dist} Å:")
        for opt in loaded_results[dist]:
            results = loaded_results[dist][opt]
            print(
                f"  {opt}: Min Energy = {results['min_energy']:.8f} Ha, Error = {results['min_error']:.8f} Ha, lr = {results['learning_rate']}")
            if results['computational_accuracy_evals'] is not None:
                print(f"      Reached computational accuracy after {results['computational_accuracy_evals']} evaluations")
            else:
                print(f"      Did not reach computational accuracy threshold")

    return loaded_results


def print_time(start_time, message):
    """
    Print elapsed time since start_time.
    
    Args:
        start_time: Start time from time.time()
        message: Message to print with the elapsed time
    """
    import time
    end_time = time.time()
    print(f"{message}: {end_time - start_time:.2f} seconds")


def convert_to_json_serializable(obj):
    """
    Convert objects to JSON serializable format.
    
    Args:
        obj: Object to convert
    
    Returns:
        JSON serializable object
    """
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj 