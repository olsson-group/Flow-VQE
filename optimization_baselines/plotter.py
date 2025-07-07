"""
Plotting utilities for VQE optimization results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from .config import OptimizationConfig


class ResultPlotter:
    """Class for generating plots from optimization results."""
    
    def __init__(self, config: OptimizationConfig, results: Dict[str, Any]):
        """
        Initialize the plotter.
        
        Args:
            config: Optimization configuration object
            results: Results dictionary from optimization
        """
        self.config = config
        self.results = results
        
        # Setup matplotlib style
        self._setup_plot_style()
        
        # Define colors for different optimizers
        self.colors = {
            "GD": "#0055A4",     
            "ADAM": "#228B22",   
            "QNSPSA": "#8B4513"  
        }
    
    def _setup_plot_style(self):
        """Setup matplotlib plotting style."""
        plt.rcParams.update({
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
    
    def generate_all_plots(self):
        """Generate all plots for the optimization results."""
        print("\nGenerating plots...")
        
        # Generate individual optimization plots
        self._generate_optimization_plots()
        
        # Generate potential energy surface plot
        self._generate_potential_energy_surface()
        
        # Generate summary table
        self._generate_summary_table()
        
        print("All plots generated successfully!")
    
    def _generate_optimization_plots(self):
        """Generate individual optimization plots for each distance."""
        for dist in self.config.distances:
            dist_results = self.results[dist]
            
            plt.figure(figsize=(10, 6))
            
            for optimizer_type in self.config.optimizers:
                results = dist_results[optimizer_type]
                cost_history = results["cost_history"]
                computational_accuracy_evals = results["computational_accuracy_evals"]

                # Calculate x values based on optimizer type
                if optimizer_type == "QNSPSA":
                    x_values = np.array([6 * (n + 1) for n in range(len(cost_history))])
                else:
                    num_params = len(results["min_cost_params"])
                    x_values = np.array([num_params * 2 * (n + 1) for n in range(len(cost_history))])

                label = f'{optimizer_type}'
                if computational_accuracy_evals is not None:
                    label += f' (comp: {computational_accuracy_evals})'
                
                plt.plot(x_values, cost_history, '-', 
                         color=self.colors[optimizer_type],
                         linewidth=2,
                         label=label)

            plt.axhline(y=self.config.computational_accuracy, color='#808080', 
                       linestyle='--', linewidth=2, 
                       label=f'Computational Accuracy ({self.config.computational_accuracy})')

            plt.xlabel('Circuit Evaluations', fontsize=18)
            plt.ylabel('Error (Ha)', fontsize=18)
            plt.yscale('log')  
            plt.title(f'Optimization Error for Molecular Distance {dist} Å', fontsize=20)
            plt.legend(frameon=False)
            plt.grid(True, alpha=0.25, linestyle='--')

            plt.savefig(f"{self.config.results_dir}/optimization_dist{dist}_{self.config.molecule}_L{self.config.ansatz_layer}_lr{self.config.learning_rate}.pdf", 
                        format='pdf', dpi=600, bbox_inches='tight')
            plt.close()
    
    def _generate_potential_energy_surface(self):
        """Generate potential energy surface plot."""
        min_energies = {}
        for optimizer_type in self.config.optimizers:
            min_energies[optimizer_type] = [self.results[dist][optimizer_type]["min_energy"] for dist in self.config.distances]

        exact_energies = [self.results[dist][self.config.optimizers[0]]["min_energy"] - 
                         self.results[dist][self.config.optimizers[0]]["min_error"] 
                         for dist in self.config.distances]

        plt.figure(figsize=(10, 6))

        plt.plot(self.config.distances, exact_energies, 'o-', color='#0055A4', 
                linewidth=2, markersize=8, label='Exact')

        for optimizer_type in self.config.optimizers:
            plt.plot(self.config.distances, min_energies[optimizer_type], 's--', 
                     color=self.colors[optimizer_type], 
                     linewidth=2, 
                     markersize=8, 
                     label=optimizer_type)

        plt.xlabel('Bond Length (Å)', fontsize=18)
        plt.ylabel('Energy (Ha)', fontsize=18)
        plt.title(f'Potential Energy Surface', fontsize=20)
        plt.legend(frameon=False)
        plt.grid(True, alpha=0.25, linestyle='--')

        plt.savefig(f"{self.config.results_dir}/potential_energy_surface_{self.config.molecule}_L{self.config.ansatz_layer}_lr{self.config.learning_rate}.pdf", 
                    format='pdf', dpi=600, bbox_inches='tight')
        plt.close()
    

    
    def _generate_summary_table(self):
        """Generate summary table and save to JSON."""
        summary_data = {
            "distances": self.config.distances.tolist() if hasattr(self.config.distances, 'tolist') else self.config.distances,
            "results": []
        }

        for dist in self.config.distances:
            dist_results = self.results[dist]
            for optimizer_type in self.config.optimizers:
                results = dist_results[optimizer_type]
                # Calculate total evaluations based on optimizer type
                if optimizer_type == "QNSPSA":
                    total_evaluations = int(results["iterations"] * 6)
                else:
                    total_evaluations = int(results["iterations"] * len(results["min_cost_params"]) * 2)
                
                summary_data["results"].append({
                    "distance": float(dist),
                    "optimizer": optimizer_type,
                    "final_accuracy": float(results["min_error"]),
                    "total_evaluations": total_evaluations,
                    "evaluations_to_computational_accuracy": int(results["computational_accuracy_evals"]) if results["computational_accuracy_evals"] is not None else None,
                    "evaluations_to_no_improvement": int(results["no_improvement_evals"]) if results["no_improvement_evals"] is not None else None,
                    "energy_at_no_improvement": float(results["no_improvement_energy"]) if results["no_improvement_energy"] is not None else None,
                    "final_energy": float(results["min_energy"]),
                    "ground_state_energy": float(results["min_energy"] - results["min_error"])
                })

        summary_file = f"{self.config.results_dir}/summary_table_{self.config.molecule}_L{self.config.ansatz_layer}_lr{self.config.learning_rate}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4)

        print(f"\nSummary table saved to: {summary_file}")
        self._print_summary_table(summary_data)
    
    def _print_summary_table(self, summary_data):
        """Print summary table to console."""
        print("\nSummary Table:")
        print("-" * 120)
        print(f"{'Distance (Å)':<10} {'Optimizer':<10} {'Final Accuracy (Ha)':<20} {'Total Evals':<15} {'Evals to Comp':<15} {'Evals to No Improvement':<15} {'Energy at No Improvement (Ha)':<25} {'Final Energy (Ha)':<20}")
        print("-" * 120)

        for result in summary_data["results"]:
            evals_to_comp = result["evaluations_to_computational_accuracy"] if result["evaluations_to_computational_accuracy"] is not None else "N/A"
            evals_to_no_improvement = result["evaluations_to_no_improvement"] if result["evaluations_to_no_improvement"] is not None else "N/A"
            energy_at_no_improvement = result["energy_at_no_improvement"] if result["energy_at_no_improvement"] is not None else "N/A"
            
            print(f"{result['distance']:<10.4f} {result['optimizer']:<10} {result['final_accuracy']:<20.8f} {result['total_evaluations']:<15} "
                  f"{evals_to_comp:<15} {evals_to_no_improvement:<15} "
                  f"{energy_at_no_improvement if isinstance(energy_at_no_improvement, str) else f'{energy_at_no_improvement:<25.8f}'} "
                  f"{result['final_energy']:<20.8f}")

        print("-" * 120) 