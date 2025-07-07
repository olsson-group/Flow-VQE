"""
Quantum VQE Optimization Baselines Package

This package provides a structured approach to running VQE optimizations
with different molecules, optimizers, and initialization strategies.
"""

import warnings
warnings.filterwarnings("ignore", message="Couldn't find `optuna`, `cmaes`, or `nevergrad`")

from .config import OptimizationConfig, get_default_config, parse_arguments
from .runner import OptimizationRunner
from .plotter import ResultPlotter
from .utils import load_and_analyze_results

__all__ = [
    'OptimizationConfig',
    'get_default_config', 
    'parse_arguments',
    'OptimizationRunner',
    'ResultPlotter',
    'load_and_analyze_results',
    'run'
]

def run(molecule='H2O', mode='default', distances=None, optimizers=None, 
        learning_rate=0.02, max_iterations=151, **kwargs):
    """
    Main entry point for running VQE optimizations.
    
    This function provides the same interface as the original optimization_baselines.py
    but uses the new structured approach internally.
    
    Args:
        molecule (str): Molecule type ('H4', 'H2O', 'NH3', 'C6H6')
        mode (str): Optimization mode ('default', 'warm_up', 'parameter_transfer')
        distances: Custom distances or None for defaults
        optimizers (list): List of optimizers to use
        learning_rate (float): Learning rate for optimizers
        max_iterations (int): Maximum optimization iterations
        **kwargs: Additional configuration parameters
    
    Returns:
        dict: Results dictionary containing all optimization data
    """
    # Create configuration
    config = get_default_config(molecule, mode, distances, optimizers, 
                               learning_rate, max_iterations, **kwargs)
    
    # Create runner and execute
    runner = OptimizationRunner(config)
    results = runner.run()
    
    # Create plotter and generate plots
    plotter = ResultPlotter(config, results)
    plotter.generate_all_plots()
    
    return results 