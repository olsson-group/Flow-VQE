"""
Plotting module for Flow VQE.

Contains functions for creating various plots and visualizations.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
import os
from .utils import tensor_to_serializable


def plot_training_loss(history, save_dir):
    """Plot MLE loss curve"""
    history = tensor_to_serializable(history)

    epochs = [h['epoch'] for h in history]
    losses = [h['loss'] for h in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, '-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Log-likelihood loss')
    plt.grid(True)

    # Add moving average to smooth the curve
    window_size = min(20, len(losses) // 5) if len(losses) > 20 else 1
    if window_size > 1:
        moving_avg = np.convolve(losses, np.ones(window_size) / window_size, mode='valid')
        plt.plot(epochs[window_size - 1:], moving_avg, 'r-', linewidth=2,
                 label=f'{window_size}-epoch moving average')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'), dpi=300)
    plt.close()


def plot_potential_energy_surface(pes_data, training_distances, save_dir, gs_energies):
    pes_data = tensor_to_serializable(pes_data)

    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Helvetica',  ],
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

    bond_lengths = pes_data['bond_lengths']
    exact_energies = pes_data['exact_energies']
    generated_min_energies = pes_data['generated_min_energies']
    generated_mean_energies = pes_data['generated_mean_energies']
    errors = pes_data['errors']
    hf_energies = pes_data['hf_energies']

    mean_errors = [generated_mean - exact for generated_mean, exact in zip(generated_mean_energies, exact_energies)]
    min_errors = [error for error in errors]

    exact_color = '#0055A4'
    hf_color = '#E69F00'
    min_face = '#B7E7A7'
    min_edge = '#228B22'
    mean_face = '#F7C4D4'
    mean_edge = '#E75480'
    train_color = '#E75480'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # PES subplot
    ax = axes[0]
    ax.plot(bond_lengths, exact_energies, '-', color=exact_color, label='Exact', linewidth=2)
    ax.plot(bond_lengths, hf_energies, '-', color=hf_color, label='HF', linewidth=2)
    ax.scatter(bond_lengths, generated_mean_energies, label='Generated (mean)',
               marker='s', s=80, facecolor=mean_face, edgecolor=mean_edge, linewidth=1.5, zorder=3)
    ax.scatter(bond_lengths, generated_min_energies, label='Generated (min)',
               marker='o', s=80, facecolor=min_face, edgecolor=min_edge, linewidth=1.5, zorder=4)

    if training_distances:
        training_exact_y = [gs_energies[dist] for dist in training_distances]
        ax.plot(training_distances, training_exact_y, marker='*', linestyle='None', color=train_color, markersize=12, label='Training points', zorder=10)

    ax.set_xlabel('Bond length (Å)')
    ax.set_ylabel('Energy (Ha)')
    ax.legend(frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(True)

    # Error subplot
    ax2 = axes[1]
    ax2.scatter(bond_lengths, mean_errors, label='Mean energy error', marker='s', s=80, facecolor=mean_face, edgecolor=mean_edge, linewidth=1.5, zorder=3)
    ax2.scatter(bond_lengths, min_errors, label='Min energy error', marker='o', s=80, facecolor=min_face, edgecolor=min_edge, linewidth=1.5, zorder=4)
    ax2.axhline(y=1.6e-3, color='gray', linestyle='--', linewidth=2, label=' Computational accuracy')
    ax2.set_xlabel('Bond length (Å)')
    ax2.set_ylabel('Error (Ha)')
    ax2.set_yscale('log')
    ax2.legend(frameon=False)
    for spine in ax2.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'potential_energy_surface.pdf'), dpi=600)
    plt.close(fig)


def plot_learning_curves(energy_history, circuit_eval_history, gs_energies, training_distances, save_dir):

    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Helvetica',  ],
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

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(training_distances)))

    chemical_accuracy = 1.6e-3  # Ha
    circuit_evals_at_chemical_accuracy = {}

    for i, dist in enumerate(training_distances):
        errors = [abs(e - gs_energies[dist]) for e in energy_history[dist]]
        circuit_evals = circuit_eval_history[dist]
        
        dist_str = f"{dist:.2f} Å"
        
        ax.plot(circuit_evals, errors, '-', color=colors[i], label=dist_str, linewidth=2)

        for j, error in enumerate(errors):
            if error < chemical_accuracy:
                circuit_evals_at_chemical_accuracy[dist] = circuit_evals[j]
                break

    ax.axhline(y=1.6e-3, color='gray', linestyle='--', linewidth=2, label='Computational accuracy')

    ax.set_xlabel('Circuit evaluations')
    ax.set_ylabel('Energy error (Ha)')
    ax.set_yscale('log')  
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    plt.tight_layout() 
    plt.savefig(os.path.join(save_dir, 'learning_curves.pdf'), dpi=600, bbox_inches='tight')
    plt.close()

    final_errors = {
        dist: abs(energy_history[dist][-1] - gs_energies[dist]) 
        for dist in training_distances
    }

    chemical_accuracy_data = {
        'circuit_evaluations_at_chemical_accuracy': circuit_evals_at_chemical_accuracy,
        'final_errors': final_errors
    }
    
    with open(os.path.join(save_dir, 'chemical_accuracy_data.json'), 'w') as f:
        json.dump(chemical_accuracy_data, f, indent=4)


def plot_single_distance_training_errors(all_histories, save_dir):
    """Plot training error curves for all bond lengths in single-distance mode"""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Helvetica'],
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
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

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(all_histories)))
    
    for (dist, history), color in zip(all_histories.items(), colors):
        history = tensor_to_serializable(history)
        evaluations = [h['total_evaluations'] for h in history]
        errors = [abs(h['error']) for h in history]
        
        dist_str = f"{float(dist):.2f} Å"
        ax.plot(evaluations, errors, '-', color=color, label=dist_str, linewidth=2)
    
    ax.axhline(y=1.6e-3, color='gray', linestyle='--', linewidth=2, label='Computational accuracy')
    
    ax.set_yscale('log')
    
    ax.set_xlabel('Circuit evaluations')
    ax.set_ylabel('Energy error (Ha)')
    
    ax.legend(frameon=False, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.pdf'), dpi=1200, bbox_inches='tight')
    plt.close() 