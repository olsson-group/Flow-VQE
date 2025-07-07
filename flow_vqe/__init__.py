"""
Flow VQE Package

A unified implementation of Flow VQE that combines single-distance and multi-distance training modes.
"""
import warnings 
warnings.filterwarnings("ignore", message="Couldn't find `optuna`, `cmaes`, or `nevergrad`")

from .config import get_molecule_defaults, parse_distance_argument, generate_experiment_name
from .molecule_utils import get_all_hamiltonians, calculate_hf_energies
from .circuit_utils import create_all_circuits
from .flow_training import pretrain_flow, train_single_distance_models, train_flow_with_unified_hamiltonians
from .evaluation import generate_parameters, evaluate_parameters, generate_potential_energy_surface
from .warm_up import run_warm_up_evaluation, sample_parameters_and_energies, plot_potential_energy_surface as plot_warm_up_pes, plot_energy_distribution
from .plotting import (
    plot_training_loss, 
    plot_potential_energy_surface, 
    plot_learning_curves,
    plot_single_distance_training_errors
)
from .utils import setup_environment, tensor_to_serializable, save_single_distance_final_results

__all__ = [
    # Configuration
    'get_molecule_defaults',
    'parse_distance_argument',
    'generate_experiment_name',
    
    # Molecule utilities
    'get_all_hamiltonians',
    'calculate_hf_energies',
    
    # Circuit utilities
    'create_all_circuits',
    
    # Flow training
    'pretrain_flow',
    'train_single_distance_models',
    'train_flow_with_unified_hamiltonians',
    
    # Evaluation
    'generate_parameters',
    'evaluate_parameters',
    'generate_potential_energy_surface',
    
    # Warm-up evaluation
    'run_warm_up_evaluation',
    'sample_parameters_and_energies',
    'plot_warm_up_pes',
    'plot_energy_distribution',
    
    # Plotting
    'plot_training_loss',
    'plot_potential_energy_surface',
    'plot_learning_curves',
    'plot_single_distance_training_errors',
    
    # Utilities
    'setup_environment',
    'tensor_to_serializable',
    'save_single_distance_final_results',
] 