"""
Optimization runner for VQE calculations.
"""

import os
import json
import time
import pennylane as qml
import numpy as np
import pennylane.numpy as pnp
from pennylane import qchem
from molecule_configs import MOLECULE_CONFIGS
from ansatz import get_ansatz_circuit, get_ansatz_parameter_count
from .config import OptimizationConfig


class OptimizationRunner:
    """Main class for running VQE optimizations."""
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialize the optimization runner.
        
        Args:
            config: Optimization configuration object
        """
        self.config = config
        self.warm_up_params = {}
        self._load_warm_up_params()
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        qml.numpy.random.seed(42)
        
        print(f"Results will be saved in: {config.results_dir}")
        config.print_config()
    
    def _load_warm_up_params(self):
        """Load warm-up parameters if enabled."""
        if self.config.use_warm_up_params:
            try:
                with open(self.config.warm_up_data, 'r') as f:
                    self.warm_up_params = json.load(f)
                print("Successfully loaded warm-up parameters")
            except FileNotFoundError:
                print(f"Warning: {self.config.warm_up_data} not found. Using zero initialization.")
                self.config.use_warm_up_params = False
    
    def run(self):
        """
        Run the complete optimization process.
        
        Returns:
            dict: Results dictionary containing all optimization data
        """
        all_results = {}
        
        for dist in self.config.distances:
            print(f"\n{'=' * 50}")
            print(f"Testing molecular distance dist = {dist}")
            print(f"{'=' * 50}\n")
            
            # Create Hamiltonian and get reference values
            H, hf_state, gs_eigenvalues = self._get_molecule_with_tapering(dist)
            nqb = len(H.wires)
            
            print(f"Using {nqb} qubits after tapering")
            print(f"Ground state energy: {gs_eigenvalues[0]:.12f} Ha")
            
            # Create quantum device and circuit
            dev = qml.device("lightning.qubit", wires=H.wires)
            circuit = self._create_circuit(dev, H, hf_state)
            
            # Initialize parameters
            init_coeffs = self._get_initial_parameters(dist, H, hf_state)
            
            # Evaluate initial energy
            initial_energy = circuit(init_coeffs)
            print('# Parameters:', len(init_coeffs), ', Initial energy:', initial_energy)
            print('Initial error:', initial_energy - gs_eigenvalues[0], 'Ha')
            
            # Run optimizations for each optimizer
            dist_results = {}
            for optimizer_type in self.config.optimizers:
                print(f"\nRunning with {optimizer_type} optimizer...")
                results = self._run_circuit_with_optimizer(
                    circuit, init_coeffs, optimizer_type, gs_eigenvalues[0]
                )
                dist_results[optimizer_type] = results
            
            all_results[dist] = dist_results
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _get_molecule_with_tapering(self, dist):
        """Create molecule with tapering applied."""
        config = MOLECULE_CONFIGS[self.config.molecule]
        symbols = config['symbols']
        active_electrons = config['active_electrons']
        active_orbitals = config['active_orbitals']
        basis_name = config['basis']
        coordinates = config['get_coordinates'](dist)

        molecule = qml.qchem.Molecule(
            unit='angstrom',
            symbols=symbols,
            coordinates=coordinates,
            charge=0,
            mult=1,
            basis_name=basis_name,
            load_data=True
        )

        H0, qubits = qchem.molecular_hamiltonian(
            molecule,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            mapping="jordan_wigner",
            method="openfermion",
        )

        coeffs, obs = H0.terms()
        H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc", method='rlf')

        print(f"Original Hamiltonian uses {len(H.wires)} qubits")

        if not self.config.use_tapering:
            print("Tapering disabled, using full Hamiltonian")
            H_sparse = qml.SparseHamiltonian(H.sparse_matrix(), wires=H.wires)
            gs_eigenvalues = qml.eigvals(H_sparse, k=2)
            print(f"Ground state energy: {gs_eigenvalues.min():.8f} Ha")

            dev = qml.device("lightning.qubit", wires=H.wires) 
            hf_state = qml.qchem.hf_state(active_electrons, len(H.wires))

            @qml.qnode(dev, interface="torch", diff_method='parameter-shift')
            def hf_circuit():
                sorted_wires = sorted(H.wires.tolist())
                qml.BasisState(hf_state, wires=sorted_wires)
                return qml.expval(H)

            hf_energy = hf_circuit()
            print(f"HF energy: {hf_energy:.8f} Ha, error: {hf_energy - gs_eigenvalues.min():.8f} Ha")
            print(f"HF state: {hf_state}")
            return H, hf_state, gs_eigenvalues

        generators = qml.symmetry_generators(H)
        paulixops = qml.paulix_ops(generators, qubits)
        paulix_sector = qml.qchem.optimal_sector(H, generators, active_electrons)
        H_tapered = qml.taper(H, generators, paulixops, paulix_sector)

        hf_state_tapered = qml.qchem.taper_hf(generators, paulixops, paulix_sector,
                                          num_electrons=active_electrons, num_wires=len(H.wires))

        print(f"Tapered Hamiltonian uses {len(H_tapered.wires)} qubits")

        H_sparse = qml.SparseHamiltonian(H_tapered.sparse_matrix(), wires=H_tapered.wires)
        gs_eigenvalues = qml.eigvals(H_sparse, k=2)
        print(f"Ground state energy: {gs_eigenvalues.min():.8f} Ha")

        dev = qml.device("lightning.qubit", wires=H_tapered.wires)

        @qml.qnode(dev, interface="torch")
        def hf_circuit():
            sorted_wires = sorted(H_tapered.wires.tolist())
            qml.BasisState(hf_state_tapered, wires=sorted_wires)
            return qml.expval(H_tapered)

        hf_energy = hf_circuit()
        print(f"HF energy: {hf_energy:.8f} Ha, error: {hf_energy - gs_eigenvalues.min():.8f} Ha")
        print(f"HF state: {hf_state_tapered}")

        return H_tapered, hf_state_tapered, gs_eigenvalues
    
    def _create_circuit(self, dev, H, hf_state):
        """Create the quantum circuit."""
        @qml.qnode(dev, interface='autograd', diff_method='adjoint')
        def circuit(weights, layers=self.config.ansatz_layer):
            sorted_wires = sorted(H.wires.tolist())
            
            # Get the ansatz circuit based on ANSATZ_TYPE
            if self.config.ansatz_type == 'GIVENS':
                config = MOLECULE_CONFIGS[self.config.molecule]
                active_electrons = config['active_electrons']
                active_orbitals = config['active_orbitals']
                singles, doubles = qml.qchem.excitations(active_electrons, active_orbitals * 2)
                singles = [[sorted_wires[i] for i in single] for single in singles]
                doubles = [[sorted_wires[i] for i in double] for double in doubles]
                ansatz_circuit = get_ansatz_circuit(self.config.ansatz_type, weights, sorted_wires, hf_state, 
                                                  singles=singles, doubles=doubles, layers=layers)
            else:
                ansatz_circuit = get_ansatz_circuit(self.config.ansatz_type, weights, sorted_wires, hf_state, layers=layers)
             
            ansatz_circuit()
                
            return qml.expval(H)
        
        return circuit
    
    def _get_initial_parameters(self, dist, H, hf_state):
        """Get initial parameters from warm-up results, parameter transfer, or zeros."""
        # Calculate parameter count
        if self.config.ansatz_type == 'GIVENS':
            config = MOLECULE_CONFIGS[self.config.molecule]
            active_electrons = config['active_electrons']
            active_orbitals = config['active_orbitals']
            singles, doubles = qml.qchem.excitations(active_electrons, active_orbitals * 2)
            size = get_ansatz_parameter_count(self.config.ansatz_type, len(H.wires), 
                                            singles=singles, doubles=doubles, layers=self.config.ansatz_layer)
        else:
            size = get_ansatz_parameter_count(self.config.ansatz_type, len(H.wires), layers=self.config.ansatz_layer)
        
        # Try parameter transfer first
        if self.config.use_parameter_transfer and self.config.parameter_transfer_file is not None:
            try:
                with open(self.config.parameter_transfer_file, 'r') as f:
                    params = json.load(f)
                    if len(params) == size:
                        print(f"Using parameters from {self.config.parameter_transfer_file} for initialization")
                        return pnp.array(params, requires_grad=True)
                    else:
                        print(f"Warning: Parameter length mismatch. Expected {size}, got {len(params)}. Using zeros instead.")
            except Exception as e:
                print(f"Warning: Error loading parameter file: {e}. Using zeros instead.")
        
        # Try warm-up parameters
        if self.config.use_warm_up_params:
            # Try exact string match first
            dist_str = str(dist)
            if dist_str in self.warm_up_params:
                params = self.warm_up_params[dist_str]["best_params"]
                if len(params) == size:
                    print(f"Using warm-up parameters for distance {dist_str}")
                    return pnp.array(params, requires_grad=True)
            
            # If exact match fails, try to find the closest match
            available_distances = list(self.warm_up_params.keys())
            try:
                available_distances_float = [float(d) for d in available_distances]
                closest_idx = min(range(len(available_distances_float)), 
                                key=lambda i: abs(available_distances_float[i] - dist))
                closest_dist = available_distances[closest_idx]
                closest_dist_float = available_distances_float[closest_idx]
                
                # Only use if the difference is very small (within 1e-10)
                if abs(closest_dist_float - dist) < 1e-10:
                    params = self.warm_up_params[closest_dist]["best_params"]
                    if len(params) == size:
                        print(f"Using warm-up parameters for distance {closest_dist} (closest to {dist})")
                        return pnp.array(params, requires_grad=True)
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not find suitable warm-up parameters for distance {dist}: {e}")
        
        # Default to zeros
        return pnp.array(np.zeros(size), requires_grad=True)
    
    def _run_circuit_with_optimizer(self, circuit, init_coeffs, optimizer_type, gs_energy):
        """Run optimization with a specific optimizer."""
        params = init_coeffs.copy()

        # Initialize PennyLane optimizers and set iteration limits
        if optimizer_type == "GD":
            opt = qml.GradientDescentOptimizer(stepsize=self.config.learning_rate)
            max_iterations = self.config.max_iterations
        elif optimizer_type == "ADAM":
            opt = qml.AdamOptimizer(stepsize=self.config.learning_rate)
            max_iterations = self.config.max_iterations
        elif optimizer_type == "QNSPSA":
            opt = qml.QNSPSAOptimizer(stepsize=self.config.learning_rate) 
            max_iterations = self.config.spsa_max_iterations
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # For tracking history
        param_history = [params.copy()]
        cost_history = [circuit(params) - gs_energy]
        computational_accuracy_evals = None
        no_improvement_evals = None 
        no_improvement_energy = None  
        no_improvement_counter = 0 
        last_best_energy = cost_history[0]  

        print(f"\nStarting optimization with {optimizer_type}...")
        batch_start_time = time.time()
        
        for n in range(max_iterations):
            if optimizer_type == "QNSPSA":
                params = opt.step(circuit, params)
                energy = circuit(params)
                evals = 6 * (n + 1)
            else:
                params = opt.step(circuit, params)
                energy = circuit(params)
                evals = len(params) * 2 * (n + 1)

            param_history.append(params.copy())
            error = np.abs(energy - gs_energy)
            cost_history.append(error)

            # Track when energy hasn't improved significantly
            if error < last_best_energy - self.config.improvement_threshold:  
                last_best_energy = error
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                if no_improvement_counter >= self.config.no_improvement_threshold and no_improvement_evals is None:
                    no_improvement_evals = evals
                    no_improvement_energy = error
                    print(f'Energy hasn\'t improved significantly for {self.config.no_improvement_threshold} iterations. Error={error:.12f} Ha, evals={evals}')

            if n % 50 == 0 and n > 0:
                batch_time = time.time() - batch_start_time
                print(f"Iter={n}, Energy={energy:.12f} Ha, Error={error:.12f} Ha, evals={evals}")
                print(f"Last 50 iterations time: {batch_time:.2f} seconds")
                batch_start_time = time.time()

            # Track computational accuracy
            if error < self.config.computational_accuracy and computational_accuracy_evals is None:
                computational_accuracy_evals = evals
                print(f'Step {n + 1} reached computational accuracy. Error={error:.12f} Ha, evals={evals}')

            if n == self.config.max_iterations - 1:
                print(f'Final error={error:.12f} Ha')

        min_cost_index = np.argmin(cost_history)
        min_cost_params = param_history[min_cost_index]
        min_error = cost_history[min_cost_index]
        min_energy = circuit(min_cost_params)

        print(f"\nMin energy at iter {min_cost_index} = {min_energy:.12f} Ha")
        print(f"Min error from ground state = {min_error:.12f} Ha")

        return {
            "cost_history": cost_history,
            "param_history": param_history,
            "min_cost_params": min_cost_params,
            "min_energy": min_energy,
            "min_error": min_error,
            "min_cost_index": min_cost_index,
            "computational_accuracy_evals": computational_accuracy_evals,
            "no_improvement_evals": no_improvement_evals,
            "no_improvement_energy": no_improvement_energy, 
            "iterations": n + 1,
            "learning_rate": self.config.learning_rate
        }
    
    def _save_results(self, all_results):
        """Save results to JSON file."""
        json_file = f"{self.config.results_dir}/vqe_results_{self.config.molecule}_L{self.config.ansatz_layer}_lr{self.config.learning_rate}.json"

        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        with open(json_file, 'w') as f:
            json.dump(convert_to_json_serializable(all_results), f, indent=4)

        print(f"\nAll results saved to {self.config.results_dir}/")
        print(f"Data file: vqe_results_{self.config.molecule}_L{self.config.ansatz_layer}_lr{self.config.learning_rate}.json") 