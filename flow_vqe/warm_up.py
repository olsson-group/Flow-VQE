"""
Warm-up evaluation module for Flow VQE.

Contains functions for offline model loading and parameter sampling.
This module is designed for post-training evaluation and analysis.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np 
import os
import json
from datetime import datetime
import zuko 
import matplotlib as mpl
import random
from .config import get_molecule_defaults, parse_distance_argument, generate_experiment_name
from .molecule_utils import get_all_hamiltonians
from .circuit_utils import create_all_circuits
from .utils import tensor_to_serializable


def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(model_path, param_dim, context_dim, n_flows, flow_hidden_dim, components, device):
    """Load a saved model from disk"""
    print(f"Loading model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        flow_model = zuko.flows.GF(
            features=param_dim,
            context=context_dim,
            transforms=n_flows,
            components=components,
            hidden_features=(flow_hidden_dim, flow_hidden_dim, flow_hidden_dim)
        )
        flow_model.load_state_dict(torch.load(model_path, map_location=device))
        flow_model = flow_model.to(device)
        print(f"Model loaded successfully with param_dim={param_dim}, context_dim={context_dim}")
        return flow_model
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print(f"Current model config: param_dim={param_dim}, context_dim={context_dim}")
        print("This usually means the model was trained with different parameters.")
        print("Please check the training configuration or use --num_coeffs to match the training setup.")
        raise


def sample_parameters_and_energies(flow_model, unified_hamiltonians, unified_coeffs, circuits, 
                                 bond_lengths, gs_energies, hf_states, device, batch_size=16):
    """Sample parameters and energies for given bond lengths"""
    results = {}
    
    for dist in bond_lengths:
        print(f"\nSampling for bond length {dist} Å")
        
        # Get Hamiltonian components
        H = unified_hamiltonians[dist]
        circuit = circuits[dist]
        coeffs_tensor = unified_coeffs[dist]
        gs_energy = gs_energies[dist]
        hf_state = hf_states[dist]
        
        # Calculate HF energy using initial parameters (all zeros) 
        x, _ = flow_model(coeffs_tensor).rsample_and_log_prob((1,))
        num_params = x.shape[1]
        initial_params = torch.zeros(num_params, device=device)
        hf_energy = float(torch.tensor(circuit(initial_params), device=device).item())
        
        # Generate samples from the flow model
        coeffs_batch = coeffs_tensor.repeat(batch_size, 1)
        x, log_probs = flow_model(coeffs_tensor).rsample_and_log_prob((batch_size,))
        params_batch = x
        
        # Evaluate energies for each parameter set
        energies_batch = torch.zeros(batch_size, device=device)
        for j, params in enumerate(params_batch):
            energies_batch[j] = torch.tensor(circuit(params), device=device)
        
        # Calculate mean and min energies
        mean_energy = float(energies_batch.mean().item())
        min_idx = torch.argmin(energies_batch)
        min_energy = float(energies_batch[min_idx].item())
        best_params = params_batch[min_idx].cpu().detach().numpy().tolist()
        
        # Calculate errors
        mean_error = float(mean_energy - gs_energy)
        min_error = float(min_energy - gs_energy)
        hf_error = float(hf_energy - gs_energy)
        
        # Store sampled energies and their log probabilities
        sampled_energies = energies_batch.cpu().detach().numpy().tolist()
        sampled_log_probs = log_probs.cpu().detach().numpy().tolist()
        
        results[str(dist)] = {
            "mean_energy": mean_energy,
            "min_energy": min_energy,
            "hf_energy": hf_energy,
            "best_params": best_params,
            "gs_energy": float(gs_energy),
            "mean_error": mean_error,
            "min_error": min_error,
            "hf_error": hf_error,
            "sampled_energies": sampled_energies,
            "sampled_log_probs": sampled_log_probs  
        }
        
        print(f"HF energy: {hf_energy:.8f}")
        print(f"Mean energy: {mean_energy:.8f}")
        print(f"Min energy: {min_energy:.8f}")
        print(f"Ground state energy: {gs_energy:.8f}")
        print(f"HF error: {hf_error:.8f}")
        print(f"Mean error: {mean_error:.8f}")
        print(f"Min error: {min_error:.8f}")
    
    return results


def plot_potential_energy_surface(results, save_dir, training_distances, gs_energies):
    """Plot potential energy surface using the sampled results"""
    distances = [float(d) for d in results.keys()]
    mean_energies = [results[str(d)]["mean_energy"] for d in distances]
    min_energies = [results[str(d)]["min_energy"] for d in distances]
    hf_energies = [results[str(d)]["hf_energy"] for d in distances]
    gs_energies_list = [gs_energies[float(d)] for d in distances]
    mean_errors = [results[str(d)]["mean_error"] for d in distances]
    min_errors = [results[str(d)]["min_error"] for d in distances]
    
    train_energies = [gs_energies[d] for d in training_distances]
    train_distances = training_distances
    
    print(f"Training points found: {len(train_distances)}")
    print(f"Training distances: {train_distances}")
    print(f"Training energies: {train_energies}")
    
    # Sort by distance
    sorted_indices = np.argsort(distances)
    distances = np.array(distances)[sorted_indices]
    mean_energies = np.array(mean_energies)[sorted_indices]
    min_energies = np.array(min_energies)[sorted_indices]
    hf_energies = np.array(hf_energies)[sorted_indices]
    gs_energies_list = np.array(gs_energies_list)[sorted_indices]
    mean_errors = np.array(mean_errors)[sorted_indices]
    min_errors = np.array(min_errors)[sorted_indices]
    
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Helvetica'],
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 15,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
     
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
     
    exact_color = '#0055A4'  
    hf_color = '#E69F00'   
    min_face = '#B7E7A7'      
    min_edge = '#228B22'    
    mean_face = '#F7C4D4'   
    mean_edge = '#E75480'    
    train_color = '#E75480'    
     
    ax = axes[0]

    ax.plot(distances, gs_energies_list, '-', color=exact_color, label='Exact', linewidth=2)
    ax.plot(distances, hf_energies, '-', color=hf_color, label='HF', linewidth=2)
    ax.scatter(distances, mean_energies, label='Generated (mean)',
               marker='s', s=80, facecolor=mean_face, edgecolor=mean_edge, linewidth=1.5, zorder=3)
    ax.scatter(distances, min_energies, label='Generated (min)',
               marker='o', s=80, facecolor=min_face, edgecolor=min_edge, linewidth=1.5, zorder=4)

    if len(train_distances) > 0:   
        ax.scatter(train_distances, train_energies, marker='*', linestyle='None', 
                  color=train_color, s=100, label='Training Points', zorder=10)
    ax.set_xlabel('Coordinate (Å)')
    ax.set_ylabel('Energy (Ha)')
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
     
    ax2 = axes[1]
    ax2.scatter(distances, np.abs(mean_errors), label='Mean Energy Error',
                marker='s', s=80, facecolor=mean_face, edgecolor=mean_edge, linewidth=1.5, zorder=3)
    ax2.scatter(distances, np.abs(min_errors), label='Min Energy Error',
                marker='o', s=80, facecolor=min_face, edgecolor=min_edge, linewidth=1.5, zorder=4)
    ax2.axhline(y=1.6e-3, color='gray', linestyle='--', linewidth=2, label='Computational Accuracy')
    ax2.set_xlabel('Coordinate (Å)')  
    ax2.set_ylabel('Error (Ha)')
    ax2.set_yscale('log')
    ax2.legend(frameon=False)
    for spine in ax2.spines.values():
        spine.set_visible(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'potential_energy_surface_nature.pdf'), dpi=600)
    plt.close(fig)
    
    print(f"Potential energy surface plot saved to {os.path.join(save_dir, 'potential_energy_surface_nature.pdf')}")


def plot_energy_distribution(results, save_dir, gs_energies, batch_size=128):
    """Plot energy distribution as boxplot"""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Helvetica'],
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 15,
        'axes.linewidth': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'figure.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linestyle': '--',
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })
 
    box_facecolor = '#00BFC4'   
    box_edgecolor = '#007b8a'  
    acc_color = '#E75480'
 
    all_x = []
    all_y = []
    for dist in results.keys():
        dist_float = float(dist)
        gs_energy = gs_energies[dist_float]
        energies = np.array(results[dist]["sampled_energies"])
        errors = np.abs(energies - gs_energy)  
        all_x.extend([dist_float] * len(errors))
        all_y.extend(errors)
 
    bond_lengths = sorted(set(all_x))
    data = []
    for bl in bond_lengths:
        errors_bl = [e for x, e in zip(all_x, all_y) if x == bl]
        data.append(errors_bl)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    box = ax.boxplot(
        data,
        positions=bond_lengths,
        widths=0.02,  
        patch_artist=True,
        showfliers=False,  
        whis=[0, 100],  
        medianprops=dict(color='black', linewidth=1),
        boxprops=dict(facecolor=box_facecolor, color=box_edgecolor, linewidth=1, alpha=0.7),
        whiskerprops=dict(color=box_edgecolor, linewidth=1),
        capprops=dict(color=box_edgecolor, linewidth=1)
    )
 
    ax.set_xlim(min(bond_lengths) - 0.05, max(bond_lengths) + 0.05)
    xticks_major = np.linspace(min(bond_lengths), max(bond_lengths), 6)
    ax.set_xticks(xticks_major)
    ax.set_xticklabels([f'{x:.2f}' for x in xticks_major])
 
    acc = 1.6e-3
    ymin = 1e-4
    ax.set_ylim(bottom=ymin)
    ax.axhspan(ymin, acc, color=acc_color, alpha=0.18, label='Computational Accuracy', zorder=1)

    ax.set_xlabel('Bond Length (Å)')
    ax.set_ylabel('Error (Ha)')
    ax.set_yscale('log')
    ax.tick_params(axis='both', labelsize=16, width=1.2, length=6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)

    ax.legend(frameon=False, loc='best', fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_distribution_all_bonds_boxplot.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Error distribution plot saved to {os.path.join(save_dir, 'error_distribution_all_bonds_boxplot.pdf')}")


def run_warm_up_evaluation(
    molecule_type,
    model_path=None,
    training_range=None,
    test_range=None,
    save_dir='warm_up',
    device='auto',
    seed=42,
    symbols=None,
    active_electrons=None,
    active_orbitals=None,
    use_tapering=None,
    ansatz_type=None,
    ansatz_layer=None,
    num_coeffs=2000,
    training_num_coeffs=None,
    n_flows=20,
    flow_hidden_dim=256,
    components=32,
    batch_size=16,
    experiment_name=None
): 
    # Set random seed
    set_all_seeds(seed)
    
    # Setup device
    if device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Get molecule-specific defaults
    molecule_defaults = get_molecule_defaults(molecule_type)
    
    # Apply defaults for parameters not explicitly set
    if training_range is None:
        training_range = molecule_defaults['training_range']
    if test_range is None:
        test_range = molecule_defaults['test_range']
    if use_tapering is None:
        use_tapering = molecule_defaults['use_tapering']
    if ansatz_type is None:
        ansatz_type = molecule_defaults['ansatz_type']
    if ansatz_layer is None:
        ansatz_layer = molecule_defaults['ansatz_layer']
    
    # Set default model path if not provided
    if model_path is None:
        model_path = f"a_store_data/flow_vqe_m_{molecule_type.lower()}_results/best_flow_model.pt"
    
    # Generate experiment name if not provided
    if experiment_name is None:
        experiment_name = generate_experiment_name(molecule_type)
    
    # Create save directory
    full_save_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(full_save_dir, exist_ok=True)
    print(f"Results will be saved in: {full_save_dir}")
    
    # Get molecule configuration
    from molecule_configs import MOLECULE_CONFIGS
    config = MOLECULE_CONFIGS[molecule_type]
    symbols = symbols if symbols is not None else config['symbols']
    active_electrons = active_electrons if active_electrons is not None else config['active_electrons']
    active_orbitals = active_orbitals if active_orbitals is not None else config['active_orbitals']
     
    training_distances = parse_distance_argument(training_range)
    test_distances = parse_distance_argument(test_range)
     
    all_distances = sorted(list(set(list(training_distances) + list(test_distances))))
    
    print(f"\nMolecule: {molecule_type}")
    print(f"Training distances: {training_distances}")
    print(f"Test distances: {test_distances}")
    print(f"Model path: {model_path}")
    
    print("\nPreparing Hamiltonians and circuits...")
    unified_hamiltonians, gs_energies, hf_states, unified_pauli_strings, pauli_to_idx, unified_coeffs, num_coeffs_actual = get_all_hamiltonians(
        all_distances,
        device,
        symbols=symbols,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        use_tapering=use_tapering,
        save_dir=os.path.join(full_save_dir, 'hdf5_files'),
        molecule_type=molecule_type,
        num_coeffs=num_coeffs
    )
    
    circuits, param_dim = create_all_circuits(
        unified_hamiltonians,
        hf_states,
        L=ansatz_layer,
        device=device,
        ansatz_type=ansatz_type,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals
    )
      
    context_dim = training_num_coeffs if training_num_coeffs is not None else num_coeffs_actual
    print(f"Using context_dim={context_dim} for model loading (training used {training_num_coeffs or num_coeffs_actual} coefficients)")
     
    flow_model = load_model(
        model_path,
        param_dim,
        context_dim,
        n_flows,
        flow_hidden_dim,
        components,
        device
    )
     
    print("\nSampling parameters and energies...")
    results = sample_parameters_and_energies(
        flow_model,
        unified_hamiltonians,
        unified_coeffs,
        circuits,
        test_distances,  
        gs_energies,
        hf_states,
        device,
        batch_size=batch_size
    )
      
    output_path = os.path.join(full_save_dir, "warm_up_params.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults successfully saved to {output_path}")
         
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                loaded_results = json.load(f)
            print("File verification successful")
        else:
            print("Error: File was not created")
    except Exception as e:
        print(f"Error saving results: {str(e)}")
     
    print("\nGenerating potential energy surface plot...")
    plot_potential_energy_surface(results, full_save_dir, training_distances, gs_energies)
 
    print("\nGenerating energy distribution plots...")
    plot_energy_distribution(results, full_save_dir, gs_energies)
    
    return results 