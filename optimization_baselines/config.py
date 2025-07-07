"""
Configuration management for VQE optimization baselines.
"""

import os
os.environ["OMP_NUM_THREADS"] = "12"
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from flow_vqe.config import get_molecule_defaults, parse_distance_argument


@dataclass
class OptimizationConfig:
    """Configuration class for VQE optimization runs."""
    
    # Basic settings
    molecule: str = 'H2O'
    mode: str = 'default'
    distances: np.ndarray = field(default_factory=lambda: np.array([]))
    optimizers: List[str] = field(default_factory=lambda: ['ADAM'])
    
    # Optimization parameters
    learning_rate: float = 0.02
    max_iterations: int = 151
    spsa_max_iterations: int = 20001
    
    # Thresholds
    computational_accuracy: float = 1.6e-3
    no_improvement_threshold: int = 50
    improvement_threshold: float = 0.0001
    
    # Circuit configuration
    ansatz_layer: int = 1
    ansatz_type: str = 'GIVENS'
    use_tapering: bool = True
    
    # File paths
    warm_up_data: Optional[str] = None
    parameter_transfer_file: Optional[str] = None
    
    # Results directory
    results_dir: str = ""
    timestamp: str = ""
    
    # Internal flags
    use_warm_up_params: bool = False
    use_parameter_transfer: bool = False
    
    def __post_init__(self):
        """Initialize derived attributes after object creation."""
        # Set timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get molecule-specific defaults
        molecule_defaults = get_molecule_defaults(self.molecule)
        self.ansatz_layer = molecule_defaults['ansatz_layer']
        self.ansatz_type = molecule_defaults['ansatz_type']
        self.use_tapering = molecule_defaults['use_tapering']
        
        # Setup mode-specific configurations
        self._setup_mode_config()
        
        # Setup results directory
        self._setup_results_directory()
    
    def _setup_mode_config(self):
        """Setup mode-specific configurations."""
        if self.mode == 'default':
            self._setup_default_mode()
        elif self.mode == 'warm_up':
            self._setup_warm_up_mode()
        elif self.mode == 'parameter_transfer':
            self._setup_parameter_transfer_mode()
        else:
            raise ValueError(f"Invalid optimization mode: {self.mode}")
    
    def _setup_default_mode(self):
        """Setup default optimization mode."""
        default_distances = {
            'H4': 'np.linspace(0.6, 2.6, 8)',
            'H2O': 'np.linspace(0.8, 1.8, 6)',
            'NH3': 'np.linspace(-0.55, 0.55, 4)',
            'C6H6': 'np.linspace(-0.4, 0.7, 10)'
        }
        
        if self.distances is None or len(self.distances) == 0:
            self.distances = parse_distance_argument(default_distances[self.molecule])
        
        self.use_warm_up_params = False
        self.use_parameter_transfer = False
        self.warm_up_data = None
        self.parameter_transfer_file = None
    
    def _setup_warm_up_mode(self):
        """Setup warm-up optimization mode."""
        warm_up_distances = {
            'H2O': np.array([1.9]),
            'H4': np.array([2.5755102040816324]),
            'C6H6': np.linspace(-0.4, 0.7, 10),
            'NH3': np.linspace(-0.55, 0.55, 10)
        }
        
        if self.distances is None or len(self.distances) == 0:
            self.distances = warm_up_distances[self.molecule]
        
        self.use_warm_up_params = True
        self.use_parameter_transfer = False
        
        # Warm-up parameter files
        warm_up_files = {
            'H2O': 'a_store_data/warm_up/warm_up_params_h2o.json',
            'H4': 'a_store_data/warm_up/warm_up_params_h4.json',
            'C6H6': 'a_store_data/warm_up/warm_up_params_c6h6.json',   
            'NH3': 'a_store_data/warm_up/warm_up_params_nh3.json'
        }
        self.warm_up_data = warm_up_files[self.molecule]
        self.parameter_transfer_file = None
    
    def _setup_parameter_transfer_mode(self):
        """Setup parameter transfer optimization mode."""
        pt_distances = { 
            'C6H6': np.linspace(-0.4, 0.7, 10),
            'NH3': np.linspace(-0.55, 0.55, 10)
        }
        
        if self.molecule not in pt_distances:
            raise ValueError(f"Parameter transfer mode is not supported for {self.molecule}. Only NH3 and C6H6 are supported.")
        
        if self.distances is None or len(self.distances) == 0:
            self.distances = pt_distances[self.molecule]
        
        self.use_warm_up_params = False
        self.use_parameter_transfer = True
        
        # Parameter transfer files
        pt_files = { 
            'C6H6': 'a_store_data/pt_optimization_results_c6h6/single_qflows_optimization/individual_models/distance_0.0/best_parameters.json',
            'NH3': 'a_store_data/pt_optimization_results_nh3/single_qflows_optimization/individual_models/distance_0.0/best_parameters.json'
        }
        self.warm_up_data = None
        self.parameter_transfer_file = pt_files[self.molecule]
    
    def _setup_results_directory(self):
        """Setup results directory structure."""
        if self.mode == 'warm_up':
            main_results_dir = f"warm_optimization_results_{self.molecule.lower()}"  
        elif self.mode == 'parameter_transfer':
            main_results_dir = f"pt_optimization_results_{self.molecule.lower()}"
        else:
            main_results_dir = f"vqe_optimization_results_{self.molecule.lower()}"  
        
        if not os.path.exists(main_results_dir):
            os.makedirs(main_results_dir)
        
        self.results_dir = os.path.join(main_results_dir, self.timestamp)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def print_config(self):
        """Print current configuration."""
        print("\nConfiguration:")
        print(f"- Molecule: {self.molecule}")
        print(f"- Optimization mode: {self.mode}")
        print(f"- Distances: {self.distances}")
        print(f"- Using warm-up parameters: {self.use_warm_up_params}")
        print(f"- Using parameter transfer: {self.use_parameter_transfer}")
        print(f"- Optimizers: {self.optimizers}")
        print(f"- Learning rate: {self.learning_rate}")
        print(f"- Max iterations: {self.max_iterations}")
        print(f"- Circuit layers: {self.ansatz_layer}")
        print(f"- Ansatz type: {self.ansatz_type}")
        print(f"- Use tapering: {self.use_tapering}")
        print(f"Results will be saved in: {self.results_dir}")


def get_default_config(molecule='H2O', mode='default', distances=None, 
                      optimizers=None, learning_rate=0.02, max_iterations=151, **kwargs):
    """
    Create a default configuration object.
    
    Args:
        molecule (str): Molecule type
        mode (str): Optimization mode
        distances: Custom distances or None for defaults
        optimizers (list): List of optimizers
        learning_rate (float): Learning rate
        max_iterations (int): Maximum iterations
        **kwargs: Additional configuration parameters
    
    Returns:
        OptimizationConfig: Configuration object
    """
    # Convert distances to numpy array if provided as string or list
    if distances is not None and not isinstance(distances, np.ndarray):
        if isinstance(distances, str):
            distances = parse_distance_argument(distances)
        else:
            distances = np.array(distances)
    
    # Set default optimizers
    if optimizers is None:
        optimizers = ['ADAM']
    
    # Create config object
    config = OptimizationConfig(
        molecule=molecule,
        mode=mode,
        distances=distances,
        optimizers=optimizers,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        **kwargs
    )
    
    return config


def parse_arguments():
    """Parse command line arguments for backward compatibility."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum VQE Optimization Baselines')
    
    # Molecule selection
    parser.add_argument('--molecule', type=str, default='H2O', 
                       choices=['H4', 'H2O', 'NH3', 'C6H6'],
                       help='Molecule type to optimize')
    
    # Optimization mode
    parser.add_argument('--mode', type=str, default='default',
                       choices=['default', 'warm_up', 'parameter_transfer'],
                       help='Optimization mode')
    
    # Distance specification
    parser.add_argument('--distances', type=str, default=None,
                       help='Custom distances. Can be comma-separated values or np.linspace format.')
    
    # Optimizer selection
    parser.add_argument('--optimizers', nargs='+', default=['ADAM'],
                       choices=['ADAM', 'GD', 'QNSPSA'],
                       help='Optimizers to use')
    
    # Optimization parameters
    parser.add_argument('--learning_rate', type=float, default=0.02,
                       help='Learning rate for optimizers')
    parser.add_argument('--max_iterations', type=int, default=151,
                       help='Maximum optimization iterations')
    parser.add_argument('--spsa_max_iterations', type=int, default=20001,
                       help='Maximum iterations for QNSPSA')
    
    # Optimization thresholds
    parser.add_argument('--computational_accuracy', type=float, default=1.6e-3,
                       help='Computational accuracy threshold')
    parser.add_argument('--no_improvement_threshold', type=int, default=50,
                       help='Number of iterations without improvement before stopping')
    parser.add_argument('--improvement_threshold', type=float, default=0.0001,
                       help='Minimum improvement threshold')
    
    args = parser.parse_args()
    
    # Convert to config
    return get_default_config(
        molecule=args.molecule,
        mode=args.mode,
        distances=args.distances,
        optimizers=args.optimizers,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        spsa_max_iterations=args.spsa_max_iterations,
        computational_accuracy=args.computational_accuracy,
        no_improvement_threshold=args.no_improvement_threshold,
        improvement_threshold=args.improvement_threshold
    ) 