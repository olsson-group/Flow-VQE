"""
Evaluation module for Flow VQE.

Contains functions for generating parameters and evaluating model performance.
"""

import torch
import numpy as np
import os
import json
from .utils import tensor_to_serializable


def generate_parameters(model, unified_coeffs, distance, device, num_samples=1, num_coeffs=300, context_type='hamiltonian_coeffs'):
    """Generate parameters for a specific bond length using trained model"""
    if context_type == 'hamiltonian_coeffs':
        context_tensor = unified_coeffs[distance][:num_coeffs]
    elif context_type == 'bond_length':
        context_tensor = torch.tensor([distance], dtype=torch.float32, device=device)
    else:
        raise ValueError(f"Unsupported context_type: {context_type}")

    x, log_prob_batch = model(context_tensor).rsample_and_log_prob((num_samples,))
    params = x

    return params


def evaluate_parameters(params, unified_hamiltonians, circuits, gs_energies, dist, device):
    """Evaluate parameters for a given distance."""
    if not isinstance(params, torch.Tensor):
        params = torch.tensor(params, device=device)
    
    circuit = circuits[dist]
    H = unified_hamiltonians[dist]
    
    if isinstance(params, torch.Tensor):
        params_np = params.detach().cpu().numpy()
    else:
        params_np = np.asarray(params)
    
    energy = circuit(params_np)
    
    if not isinstance(energy, torch.Tensor):
        energy = torch.tensor(energy, device=device)
    
    error = torch.abs(energy - gs_energies[dist])
    
    return energy, error


def generate_potential_energy_surface(
        model, unified_hamiltonians, unified_coeffs, circuits,
        gs_energies, hf_energies, test_distances, device, save_dir,
        samples_per_point, num_coeffs=300, context_type='hamiltonian_coeffs'):
    """Generate potential energy surface data for all test distances"""
    print(f"Generating potential energy surface for {len(test_distances)} bond lengths...")

    exact_energies = []
    generated_min_energies = []
    generated_mean_energies = []
    errors = []
    hf_energies_list = []
    all_generated_params = {} 

    for i, dist in enumerate(test_distances):
        print(f"Processing bond length {i + 1}/{len(test_distances)}: {dist} Å")

        params = generate_parameters(
            model, unified_coeffs, dist, device, samples_per_point,
            num_coeffs=num_coeffs, context_type=context_type
        )

        all_generated_params[str(dist)] = params.detach().cpu()

        energies_list = []
        for param_set in params:
            energy, _ = evaluate_parameters(
                param_set, unified_hamiltonians, circuits, gs_energies, dist, device
            )
            energies_list.append(energy)

        energies = torch.stack(energies_list)

        best_param_idx = torch.argmin(energies)
        best_param = params[best_param_idx].detach().cpu()
        best_energy = energies[best_param_idx]
        mean_energy = energies.mean()

        error = torch.abs(best_energy - gs_energies[dist])

        exact_energies.append(float(gs_energies[dist]))
        generated_min_energies.append(float(best_energy))
        generated_mean_energies.append(float(mean_energy))
        errors.append(float(error))
        hf_energies_list.append(float(hf_energies[dist]))

        print(f"  Exact energy: {gs_energies[dist]:.8f} Ha")
        print(f"  HF energy: {hf_energies[dist]:.8f} Ha")
        print(f"  Generated min energy: {best_energy:.8f} Ha")
        print(f"  Error: {error:.8f} Ha {'(chemical accuracy)' if abs(error) < 1.6e-3 else ''}")

    pes_data = {
        'bond_lengths': test_distances.tolist() if isinstance(test_distances, np.ndarray) else test_distances,
        'exact_energies': exact_energies,
        'generated_min_energies': generated_min_energies,
        'generated_mean_energies': generated_mean_energies,
        'errors': errors,
        'hf_energies': hf_energies_list
    }

    pes_data = tensor_to_serializable(pes_data)

    with open(os.path.join(save_dir, 'pes_data.json'), 'w') as f:
        json.dump(pes_data, f, indent=4)

    all_params_serializable = tensor_to_serializable(all_generated_params)
    with open(os.path.join(save_dir, 'generated_parameters.json'), 'w') as f:
        json.dump(all_params_serializable, f, indent=4)

    best_params = {}
    for i, dist in enumerate(test_distances):
        dist_str = str(dist)
        energies_tensor = torch.stack([evaluate_parameters(param_set, unified_hamiltonians,
                                                         circuits, gs_energies, dist, device)[0]
                                     for param_set in all_generated_params[dist_str]])
        best_idx = torch.argmin(energies_tensor)
        best_params[dist_str] = all_generated_params[dist_str][best_idx].tolist()

    with open(os.path.join(save_dir, 'best_parameters.json'), 'w') as f:
        json.dump(best_params, f, indent=4)

    return pes_data, all_generated_params 