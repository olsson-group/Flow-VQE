"""
Configuration module for Flow VQE.

Contains argument parsing, molecule-specific defaults, and distance parsing utilities.
"""

import argparse
import numpy as np
from datetime import datetime
from molecule_configs import MOLECULE_CONFIGS


def get_molecule_defaults(molecule_type):
    """Get default training and test ranges for different molecules"""
    defaults = {
        'H4': {
            'training_range': 'np.linspace(0.6, 2.6, 8)',
            'test_range': 'np.linspace(0.6, 2.8, 50)',
            'use_tapering': True,
            'ansatz_type': 'HEA',
            'ansatz_layer': 10,
            'pretrain_epochs': 0
        },
        'H2O': {
            'training_range': 'np.linspace(0.8, 1.8, 6)',
            'test_range': 'np.linspace(0.75, 1.9, 50)',
            'use_tapering': False,
            'ansatz_type': 'GIVENS',
            'ansatz_layer': 1,
            'pretrain_epochs': 201
        },
        'NH3': {
            'training_range': 'np.linspace(-0.5, 0.5, 4)',
            'test_range': 'np.linspace(-0.55, 0.55, 10)',
            'use_tapering': False,
            'ansatz_type': 'GIVENS',
            'ansatz_layer': 1,
            'pretrain_epochs': 201
        },
        'C6H6': {
            'training_range': 'np.linspace(-0.3, 0.6, 4)',
            'test_range': 'np.linspace(-0.4, 0.7, 10)',
            'use_tapering': False,
            'ansatz_type': 'GIVENS',
            'ansatz_layer': 1,
            'pretrain_epochs': 101
        }
    }
    return defaults.get(molecule_type, {
        'training_range': 'np.linspace(0.8, 1.8, 6)',   # Use H2O's training range as fallback
        'test_range': 'np.linspace(0.75, 1.9, 50)',     # Use H2O's test range as fallback
        'use_tapering': False,                          # Default to False for unknown molecules
        'ansatz_type': 'GIVENS',                        # Default to GIVENS for unknown molecules
        'ansatz_layer': 1,                              # Default to 1 layer for unknown molecules
        'pretrain_epochs': 201                          # Default to 201 for unknown molecules
    })


def parse_distance_argument(distance_arg):
    """Parse distance argument that can be either comma-separated values or np.linspace format"""
    if distance_arg.startswith('np.linspace(') and distance_arg.endswith(')'): 
        try: 
            args_str = distance_arg[12:-1]  # Remove 'np.linspace(' and ')'
            args = [float(arg.strip()) for arg in args_str.split(',')]
            if len(args) == 3:
                start, stop, num = args
                return np.linspace(start, stop, int(num)).tolist()
            else:
                raise ValueError(f"np.linspace requires exactly 3 arguments, got {len(args)}")
        except Exception as e:
            raise ValueError(f"Invalid np.linspace format: {distance_arg}. Error: {e}")
    else: 
        try:
            return [float(d.strip()) for d in distance_arg.split(',')]
        except Exception as e:
            raise ValueError(f"Invalid distance format: {distance_arg}. Error: {e}")


def generate_experiment_name(molecule_type=None):
    """Generate experiment name with molecule type prefix"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if molecule_type:
        return f"{molecule_type}_experiment_{timestamp}"
    else:
        return f"experiment_{timestamp}"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Quantum VQE with Conditional Normalizing Flow')

    # General settings
    parser.add_argument('--save_dir', type=str, default='flow_vqe_results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Molecule configuration
    available_molecules = list(MOLECULE_CONFIGS.keys())
    molecules_help = (f'Molecule type: {", ".join(available_molecules)}. ' 
                      f'For H2/H4: bond distance between adjacent atoms. '
                      f'For H2O: O-H bond length with fixed angle of 104.5°')
    parser.add_argument('--molecules', type=str, default='H4', help=molecules_help)
    parser.add_argument('--symbols', type=str, default=None, help='Optional: Override auto-detected atomic symbols')
    parser.add_argument('--active_electrons', type=int, default=None, help='Optional: Override auto-detected number of active electrons')
    parser.add_argument('--active_orbitals', type=int, default=None, help='Optional: Override auto-detected number of active orbitals')
    parser.add_argument('--use_tapering', type=bool, default=None, help='Whether to use symmetry tapering (auto-detected based on molecule if not specified)')
    parser.add_argument('--ansatz_type', type=str, default=None, choices=['HEA', 'GIVENS'],
                        help='Type of ansatz to use: HEA (Hardware Efficient Ansatz) or GIVENS (Givens Rotation) (auto-detected based on molecule if not specified)')
    parser.add_argument('--num_coeffs', type=int, default=10000, help='Number of coefficients to use as flow context (after reordering)')

    # Training mode selection
    parser.add_argument('--training_mode', type=str, default='auto', choices=['auto', 'single', 'multi'],
                       help='Training mode: auto (determined by number of training distances), single (individual models per distance), multi (shared model for all distances)')

    # Get molecule-specific defaults
    molecule_defaults = get_molecule_defaults('H4')  # Default to H4

    # Training range defaults
    training_range_help = (
        'Training bond lengths. Can be comma-separated values or np.linspace format. '
        'Examples: "1.3" for single distance, "1.0,1.5,2.0" for multiple distances, '
        '"np.linspace(0.6, 2.6, 8)" for evenly spaced range. '
        'Paper defaults: H4: np.linspace(0.6, 2.6, 8), H2O: np.linspace(0.8, 1.8, 6), '
        'NH3: np.linspace(-0.5, 0.5, 4), C6H6: np.linspace(-0.3, 0.6, 4)'
    )
    parser.add_argument('--training_range', type=str, default=molecule_defaults['training_range'], help=training_range_help)
    
    # Test range defaults
    test_range_help = (
        'Test range for potential energy surface generation. Can be comma-separated values or np.linspace format. '
        'Examples: "0.8,1.0,1.2" or "np.linspace(0.6, 2.6, 10)". '
        'Paper defaults: H4: np.linspace(0.6, 2.8, 50), H2O: np.linspace(0.75, 1.9, 50), '
        'NH3: np.linspace(-0.55, 0.55, 10), C6H6: np.linspace(-0.4, 0.7, 10)'
    )
    parser.add_argument('--test_range', type=str, default=molecule_defaults['test_range'], help=test_range_help)
    
    # Quantum circuit settings
    parser.add_argument('--ansatz_layer', type=int, default=None, help='Number of layers in the ansatz (auto-detected based on molecule if not specified)')
    
    # Flow model settings
    parser.add_argument('--n_flows', type=int, default=20, help='Number of flow layers')
    parser.add_argument('--flow_hidden_dim', type=int, default=256, help='Hidden dimension for flow layers')
    parser.add_argument('--prior_std', type=float, default=0.01, help='Standard deviation for prior distribution')
    parser.add_argument('--components', type=int, default=32, help='Number of Gaussian components in the flow model')
    parser.add_argument('--context_type', type=str, default='hamiltonian_coeffs', choices=['hamiltonian_coeffs', 'bond_length'],
                        help='Type of context for the conditional flow (hamiltonian_coeffs or bond_length)')

    # Training settings
    parser.add_argument('--n_epochs', type=int, default=3001, help='Number of training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=None, help='Number of pretraining epochs (auto-detected based on molecule if not specified)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--buffer_size', type=int, default=2, help='Memory buffer size')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for Adam optimizer')
    parser.add_argument('--samples_per_point', type=int, default=16, help='Number of generated samples per bond length for evaluation')
    parser.add_argument('--training_noise', type=float, default=0.001, help='Standard deviation of Gaussian noise added during training')

    # Mixed precision settings
    parser.add_argument('--use_mixed_precision', type=bool, default=True, help='Whether to use mixed precision training')
    parser.add_argument('--use_dynamic_loss_scaling', type=bool, default=True, help='Whether to use dynamic loss scaling')

    # Experiment name  
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument('--experiment_name', type=str, default=f'experiment_{timestamp}',
                        help='Name of the experiment (will be prefixed with molecule type)')

    args = parser.parse_args()
    
    # Apply molecule-specific defaults for parameters that weren't explicitly set
    molecule_defaults = get_molecule_defaults(args.molecules)
    
    if args.use_tapering is None:
        args.use_tapering = molecule_defaults['use_tapering']
    if args.ansatz_type is None:
        args.ansatz_type = molecule_defaults['ansatz_type']
    if args.ansatz_layer is None:
        args.ansatz_layer = molecule_defaults['ansatz_layer']
    if args.pretrain_epochs is None:
        args.pretrain_epochs = molecule_defaults['pretrain_epochs']
    
    return args 