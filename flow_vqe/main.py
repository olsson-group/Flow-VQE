"""
Main script for Flow VQE.

This script orchestrates the entire Flow VQE workflow, including training and evaluation.
"""

import os
from molecule_configs import MOLECULE_CONFIGS
from .config import parse_arguments, get_molecule_defaults, parse_distance_argument, generate_experiment_name
from .molecule_utils import get_all_hamiltonians, calculate_hf_energies
from .circuit_utils import create_all_circuits
from .flow_training import train_single_distance_models, train_flow_with_unified_hamiltonians
from .evaluation import generate_potential_energy_surface
from .plotting import (
    plot_training_loss, 
    plot_potential_energy_surface, 
    plot_learning_curves,
    plot_single_distance_training_errors
)
from .utils import setup_environment, save_single_distance_final_results


def main():
    """Main function that orchestrates the Flow VQE workflow"""
    args = parse_arguments()

    args.experiment_name = generate_experiment_name(args.molecules)

    device, save_dir = setup_environment(args)

    # Get molecule-specific defaults and apply them if not explicitly set
    molecule_defaults = get_molecule_defaults(args.molecules)
    
    # Check if training_range and test_range are using the default values (from H4)
    h4_defaults = get_molecule_defaults('H4')
    if args.training_range == h4_defaults['training_range']:
        args.training_range = molecule_defaults['training_range']
        print(f"Using molecule-specific training range for {args.molecules}: {args.training_range}")
    
    if args.test_range == h4_defaults['test_range']:
        args.test_range = molecule_defaults['test_range']
        print(f"Using molecule-specific test range for {args.molecules}: {args.test_range}")

    training_distances = parse_distance_argument(args.training_range)

    # Determine training mode
    if args.training_mode == 'auto':
        if len(training_distances) == 1:
            training_mode = 'single'
        else:
            training_mode = 'multi'
    else:
        training_mode = args.training_mode

    print(f"Training mode: {training_mode}")
    print(f"Training on {len(training_distances)} bond lengths: {training_distances}")
    print(f"Results will be saved to: {save_dir}")

    if args.molecules in MOLECULE_CONFIGS:
        config = MOLECULE_CONFIGS[args.molecules]
    else:
        raise ValueError(f"Unknown molecule type: {args.molecules}. Supported types: {list(MOLECULE_CONFIGS.keys())}")
    
    symbols = [s.strip() for s in args.symbols.split(',')] if args.symbols else config['symbols']
    
    active_electrons = args.active_electrons if args.active_electrons is not None else config['active_electrons']
    active_orbitals = args.active_orbitals if args.active_orbitals is not None else config['active_orbitals']
    
    print(f"Using molecule configuration:")
    print(f"  Molecule type: {args.molecules}")
    print(f"  Symbols: {symbols}")
    print(f"  Active electrons: {active_electrons}")
    print(f"  Active orbitals: {active_orbitals}")

    # Prepare unified Hamiltonians and ground state data
    print("\n=== Preparing Unified Hamiltonians ===")
    unified_hamiltonians, gs_energies, hf_states, unified_pauli_strings, pauli_to_idx, unified_coeffs, num_coeffs = get_all_hamiltonians(
        training_distances,  # Only use training distances for single mode
        device,
        symbols=symbols,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        use_tapering=args.use_tapering,
        save_dir=os.path.join(save_dir, 'hdf5_files'),
        molecule_type=args.molecules,
        num_coeffs=args.num_coeffs
    )

    # Create quantum circuits for all Hamiltonians
    print("\n=== Creating Quantum Circuits ===")
    circuits, param_dim = create_all_circuits(
        unified_hamiltonians,
        hf_states,
        L=args.ansatz_layer,
        device=device,
        ansatz_type=args.ansatz_type,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals
    )

    print("\n=== Calculating Hartree-Fock Energies ===")
    hf_energies = calculate_hf_energies(unified_hamiltonians, hf_states, device)

    if training_mode == 'single':
        # Single-distance training mode
        print("\n=== Training Individual Flow Models (Single-Distance Mode) ===")
        all_models, all_histories, all_best_params, all_best_energies = train_single_distance_models(
            unified_hamiltonians=unified_hamiltonians,
            unified_coeffs=unified_coeffs,
            circuits=circuits,
            gs_energies=gs_energies,
            training_distances=training_distances,
            device=device,
            save_dir=save_dir,
            param_dim=param_dim,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            pretrain_epochs=args.pretrain_epochs,
            weight_decay=args.weight_decay,
            n_flows=args.n_flows,
            flow_hidden_dim=args.flow_hidden_dim,
            prior_std=args.prior_std,
            components=args.components,
            training_noise=args.training_noise,
            num_coeffs=num_coeffs
        )

        print("\n=== Plotting Single-Distance Training Errors ===")
        plot_single_distance_training_errors(all_histories, save_dir)

        print("\n=== Saving Single-Distance Final Results ===")
        save_single_distance_final_results(all_histories, all_best_energies, gs_energies, save_dir)

    else:
        # Multi-distance training mode
        test_distances = parse_distance_argument(args.test_range)
        all_distances = sorted(list(set(list(training_distances) + list(test_distances))))

        # Re-prepare Hamiltonians with all distances for multi-mode
        print("\n=== Re-preparing Unified Hamiltonians for Multi-Distance Mode ===")
        unified_hamiltonians, gs_energies, hf_states, unified_pauli_strings, pauli_to_idx, unified_coeffs, num_coeffs = get_all_hamiltonians(
            all_distances,
            device,
            symbols=symbols,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            use_tapering=args.use_tapering,
            save_dir=os.path.join(save_dir, 'hdf5_files'),
            molecule_type=args.molecules,
            num_coeffs=args.num_coeffs
        )

        # Re-create circuits for all distances
        circuits, param_dim = create_all_circuits(
            unified_hamiltonians,
            hf_states,
            L=args.ansatz_layer,
            device=device,
            ansatz_type=args.ansatz_type,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals
        )

        hf_energies = calculate_hf_energies(unified_hamiltonians, hf_states, device)

        print(f"Will test on {len(test_distances)} bond lengths: {test_distances}")

        print("\n=== Training Shared Flow Model (Multi-Distance Mode) ===")
        best_model, history, best_params, best_energies, energy_history, circuit_eval_history = train_flow_with_unified_hamiltonians(
            unified_hamiltonians=unified_hamiltonians,
            unified_coeffs=unified_coeffs,
            circuits=circuits,
            gs_energies=gs_energies,
            training_distances=training_distances,
            device=device,
            save_dir=save_dir,
            param_dim=param_dim,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            pretrain_epochs=args.pretrain_epochs,
            weight_decay=args.weight_decay,
            n_flows=args.n_flows,
            flow_hidden_dim=args.flow_hidden_dim,
            prior_std=args.prior_std,
            components=args.components,
            training_noise=args.training_noise,
            num_coeffs=num_coeffs,
            context_type=args.context_type
        )

        print("\n=== Plotting Training MLE Loss Curve ===")
        plot_training_loss(history, save_dir)

        print("\n=== Generating Potential Energy Surface ===")
        pes_data, all_generated_params = generate_potential_energy_surface(
            best_model,
            unified_hamiltonians,
            unified_coeffs,
            circuits,
            gs_energies,
            hf_energies,
            test_distances,
            device,
            save_dir,
            samples_per_point=args.samples_per_point,
            num_coeffs=args.num_coeffs,
            context_type=args.context_type
        )

        print("\n=== Plotting Potential Energy Surface ===")
        plot_potential_energy_surface(pes_data, training_distances, save_dir, gs_energies)

        print("\n=== Plotting Learning Curves ===")
        plot_learning_curves(energy_history, circuit_eval_history, gs_energies, training_distances, save_dir)

    print(f"\nTraining and evaluation completed. All results saved to: {save_dir}")


if __name__ == "__main__":
    main() 