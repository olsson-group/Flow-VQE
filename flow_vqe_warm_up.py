#!/usr/bin/env python3
"""
Flow VQE Warm-up Evaluation Script

This script provides a command-line interface for running warm-up evaluation
on trained Flow VQE models. It loads pre-trained models and performs offline
sampling to evaluate their performance.

Usage:
    python flow_vqe_warm_up.py --molecules H4
    python flow_vqe_warm_up.py --molecules H2O --model_path path/to/model.pt
    python flow_vqe_warm_up.py --molecules C6H6 --training_range "0.8,1.0,1.2" --test_range "np.linspace(0.75,1.9,50)"
"""

import argparse
import sys
from flow_vqe import run_warm_up_evaluation, get_molecule_defaults


def parse_arguments():
    """Parse command line arguments with molecule-specific defaults"""
    parser = argparse.ArgumentParser(
        description='Warm-up evaluation for Flow VQE models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with molecule defaults
  python flow_vqe_warm_up.py --molecules H4
  
  # Custom model path and ranges
  python flow_vqe_warm_up.py --molecules H2O --model_path path/to/model.pt --test_range "np.linspace(0.8,1.8,30)"
  
  # Custom parameters
  python flow_vqe_warm_up.py --molecules C6H6 --batch_size 32 --num_coeffs 1000
        """
    )
    
    # General settings
    parser.add_argument('--save_dir', type=str, default='warm_up', help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'], help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Molecule configuration
    parser.add_argument('--molecules', type=str, required=True, choices=['H4', 'H2O', 'NH3', 'C6H6'],
                       help='Molecule type')
    parser.add_argument('--symbols', type=str, default=None, help='Optional: Override auto-detected atomic symbols')
    parser.add_argument('--active_electrons', type=int, default=None, help='Optional: Override auto-detected number of active electrons')
    parser.add_argument('--active_orbitals', type=int, default=None, help='Optional: Override auto-detected number of active orbitals')
    parser.add_argument('--use_tapering', type=bool, default=None, help='Whether to use symmetry tapering (if None, uses molecule defaults)')
    parser.add_argument('--ansatz_type', type=str, default=None, choices=['HEA', 'GIVENS'],
                       help='Type of ansatz to use: HEA (Hardware Efficient Ansatz) or GIVENS (Givens Rotation)')
    parser.add_argument('--ansatz_layer', type=int, default=None, help='Number of layers in the ansatz')
    parser.add_argument('--num_coeffs', type=int, default=2000, help='Number of coefficients to use as flow context')
    
    # Model path
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to the trained model. If not provided, will use default path based on molecule type')
    parser.add_argument('--training_num_coeffs', type=int, default=None,
                       help='Number of coefficients used during training. If not provided, will use --num_coeffs value')
    
    # Flow model settings
    parser.add_argument('--n_flows', type=int, default=20, help='Number of flow layers')
    parser.add_argument('--flow_hidden_dim', type=int, default=256, help='Hidden dimension for flow layers')
    parser.add_argument('--components', type=int, default=32, help='Number of Gaussian components in the flow model')
    
    # Training and test ranges
    parser.add_argument('--training_range', type=str, default=None,
                       help='Training bond lengths. Can be comma-separated values or np.linspace format. If not provided, will use molecule-specific defaults')
    parser.add_argument('--test_range', type=str, default=None,
                       help='Test range for evaluation. Can be comma-separated values or np.linspace format. If not provided, will use molecule-specific defaults')
    
    # Sampling settings
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for sampling')
    
    # Experiment name
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name of the experiment (will be prefixed with molecule type)')
    
    args = parser.parse_args()
    
    # Get molecule-specific defaults for display
    molecule_defaults = get_molecule_defaults(args.molecules)
    
    # Display molecule defaults if not explicitly set
    print(f"\nMolecule: {args.molecules}")
    print("Molecule-specific defaults:")
    for key, value in molecule_defaults.items():
        print(f"  {key}: {value}")
    
    # Show what will be used
    print(f"\nParameters to be used:")
    print(f"  Training range: {args.training_range or molecule_defaults['training_range']}")
    print(f"  Test range: {args.test_range or molecule_defaults['test_range']}")
    print(f"  Use tapering: {args.use_tapering if args.use_tapering is not None else molecule_defaults['use_tapering']}")
    print(f"  Ansatz type: {args.ansatz_type or molecule_defaults['ansatz_type']}")
    print(f"  Ansatz layers: {args.ansatz_layer or molecule_defaults['ansatz_layer']}")
    print(f"  Model path: {args.model_path or f'a_store_data/flow_vqe_m_{args.molecules.lower()}_results/best_flow_model.pt'}")
    
    return args


def main():
    """Main function to run warm-up evaluation"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Run warm-up evaluation
        results = run_warm_up_evaluation(
            molecule_type=args.molecules,
            model_path=args.model_path,
            training_range=args.training_range,
            test_range=args.test_range,
            save_dir=args.save_dir,
            device=args.device,
            seed=args.seed,
            symbols=args.symbols,
            active_electrons=args.active_electrons,
            active_orbitals=args.active_orbitals,
            use_tapering=args.use_tapering,
            ansatz_type=args.ansatz_type,
            ansatz_layer=args.ansatz_layer,
            num_coeffs=args.num_coeffs,
            training_num_coeffs=args.training_num_coeffs,
            n_flows=args.n_flows,
            flow_hidden_dim=args.flow_hidden_dim,
            components=args.components,
            batch_size=args.batch_size,
            experiment_name=args.experiment_name
        )
        
        print(f"\nWarm-up evaluation completed successfully!")
        print(f"Results saved to: {args.save_dir}")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during warm-up evaluation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 