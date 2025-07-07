#!/usr/bin/env python3
"""
Main entry point for VQE optimization baselines.
 
"""

import sys
import os
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization_baselines import run, OptimizationConfig, get_default_config, parse_arguments


def main():
    """Main function that mimics the original optimization_baselines.py behavior."""
     
    try:
        config = parse_arguments()
    except SystemExit: 
        config = get_default_config()
    
    # Run the optimization
    results = run(
        molecule=config.molecule,
        mode=config.mode,
        distances=config.distances,
        optimizers=config.optimizers,
        learning_rate=config.learning_rate,
        max_iterations=config.max_iterations,
        spsa_max_iterations=config.spsa_max_iterations,
        computational_accuracy=config.computational_accuracy,
        no_improvement_threshold=config.no_improvement_threshold,
        improvement_threshold=config.improvement_threshold
    )
    
    print("\nTo load and analyze these results later, use:")
    print(f"results = load_and_analyze_results('{config.results_dir}/vqe_results_{config.molecule}_L{config.ansatz_layer}_lr{config.learning_rate}.json')")
    
    return results


if __name__ == "__main__":
    main() 