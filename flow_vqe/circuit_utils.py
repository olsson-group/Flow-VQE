"""
Circuit utilities module for Flow VQE.

Contains functions for creating quantum circuits and ansatz.
"""

import torch
import pennylane as qml
from ansatz import get_ansatz_circuit, get_ansatz_parameter_count


def create_circuit_for_hamiltonian(H, hf_state, L=6, device="cpu", ansatz_type='HEA', active_electrons=None, active_orbitals=None):
    """Create quantum circuit for the given Hamiltonian"""
    if torch.device(device).type == "cuda":
        dev = qml.device("lightning.gpu", wires=H.wires)
    else:
        dev = qml.device("lightning.qubit", wires=H.wires)
 
    @qml.qnode(dev, interface='torch', diff_method=None)
    def circuit(weights):
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
            
        sorted_wires = sorted(H.wires.tolist())
        
        if ansatz_type == 'GIVENS':
            if active_electrons is None or active_orbitals is None:
                raise ValueError("active_electrons and active_orbitals must be provided when using Givens rotation ansatz")
            singles, doubles = qml.qchem.excitations(active_electrons, active_orbitals * 2)
            singles = [[sorted_wires[i] for i in single] for single in singles]
            doubles = [[sorted_wires[i] for i in double] for double in doubles]
            ansatz_circuit = get_ansatz_circuit('GIVENS', weights, sorted_wires, hf_state, 
                                              singles=singles, doubles=doubles, layers=L) 
        else:  
            ansatz_circuit = get_ansatz_circuit('HEA', weights, sorted_wires, hf_state, layers=L)
        
        ansatz_circuit()
        return qml.expval(H)

    return circuit


def create_all_circuits(unified_hamiltonians, all_hf_states, L=6, device="cpu", ansatz_type='HEA', active_electrons=None, active_orbitals=None):
    """Create circuits for all Hamiltonians"""
    circuits = {}
    param_dims = {}

    for dist, H in unified_hamiltonians.items():
        hf_state = all_hf_states[dist]
        circuit = create_circuit_for_hamiltonian(H, hf_state, L=L, device=device, 
                                               ansatz_type=ansatz_type,
                                               active_electrons=active_electrons,
                                               active_orbitals=active_orbitals)
        circuits[dist] = circuit
        
        if ansatz_type == 'GIVENS':
            if active_electrons is None or active_orbitals is None:
                raise ValueError("active_electrons and active_orbitals must be provided when using Givens rotation ansatz")
            singles, doubles = qml.qchem.excitations(active_electrons, active_orbitals * 2)
            param_dim = get_ansatz_parameter_count(ansatz_type, len(H.wires), singles=singles, doubles=doubles, layers=L)
        else:
            param_dim = get_ansatz_parameter_count(ansatz_type, len(H.wires), layers=L)
            
        param_dims[dist] = param_dim

    if len(set(param_dims.values())) > 1:
        raise ValueError("Inconsistent parameter dimensions across different bond lengths")

    return circuits, param_dims[list(param_dims.keys())[0]] 