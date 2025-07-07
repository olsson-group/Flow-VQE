"""
Molecule utilities module for Flow VQE.

Contains functions for generating Hamiltonians and calculating Hartree-Fock energies.
"""

import torch
import numpy as np
import pennylane as qml
from pennylane import qchem
import os
import json
import re
from molecule_configs import MOLECULE_CONFIGS


def parse_pauli_string(pauli_str):
    """Parse Pauli string representation"""
    if 'Tensor' in pauli_str:
        return qml.Identity(0)

    if 'Identity' in pauli_str or pauli_str.startswith('I('):
        match = re.search(r'I\((\d+)\)', pauli_str)
        if match:
            wire = int(match.group(1))
        else:
            wire = 0  
        return qml.Identity(wire)

    # Handle Pauli operators
    op_mapping = {
        'PauliX': qml.PauliX,
        'PauliY': qml.PauliY,
        'PauliZ': qml.PauliZ,
        'X': qml.PauliX,
        'Y': qml.PauliY,
        'Z': qml.PauliZ,
    }

    for op_name, op_class in op_mapping.items():
        if op_name in pauli_str:
            pattern = r'{}\((\d+)\)'.format(op_name)
            match = re.search(pattern, pauli_str)
            if match:
                wire = int(match.group(1))
                return op_class(wire)

    print(f"Warning: Could not parse Pauli string: {pauli_str}, defaulting to Identity(0)")
    return qml.Identity(0)


def get_all_hamiltonians(distances, device, symbols=['H', 'H', 'H', 'H'],
                         active_electrons=4, active_orbitals=4, use_tapering=True, save_dir='hdf5_files',
                         molecule_type='H4', num_coeffs=300):
    """
    Create Hamiltonians for all specified distances with ordered Pauli strings
    """
    print(f"Preparing unified Hamiltonians for {len(distances)} bond lengths...")
    os.makedirs(save_dir, exist_ok=True)

    all_pauli_strings = set()
    original_hamiltonians = {}
    all_hf_states = {}

    print("First pass: collecting all unique Pauli strings...")
    for i, dist in enumerate(distances):
        print(f"  Processing bond length {i + 1}/{len(distances)}: {dist} Å")

        config = MOLECULE_CONFIGS[molecule_type]
        symbols = config['symbols']
        active_electrons = config['active_electrons']
        active_orbitals = config['active_orbitals']
        basis_name = config['basis']
        
        coordinates = config['get_coordinates'](dist)

        molecule = qchem.Molecule(
            unit='angstrom',
            symbols=symbols,
            coordinates=coordinates,
            charge=0,
            mult=1,
            basis_name=basis_name,
            load_data=True,
            name=f'{save_dir}/{dist}_molecule',
        )

        H0, qubits = qchem.molecular_hamiltonian(
            molecule,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            mapping="jordan_wigner",
            method="openfermion"
        )

        coeffs, obs = H0.terms()
        H = qml.Hamiltonian(coeffs, obs, grouping_type="qwc", method='rlf')

        if use_tapering:
            generators = qml.symmetry_generators(H)
            paulixops = qml.paulix_ops(generators, qubits)
            paulix_sector = qml.qchem.optimal_sector(H, generators, active_electrons)
            H_tapered = qml.taper(H, generators, paulixops, paulix_sector)

            hf_state_tapered = qml.qchem.taper_hf(generators, paulixops, paulix_sector,
                                                  num_electrons=active_electrons, num_wires=len(H.wires))
            hf_state = torch.tensor(hf_state_tapered, device=device)

            H_final = H_tapered
        else:
            hf_state = qml.qchem.hf_state(active_electrons, qubits)
            hf_state = torch.tensor(hf_state, device=device)
            H_final = H

        coeffs, obs = H_final.terms()
        for o in obs:
            all_pauli_strings.add(str(o))

        original_hamiltonians[dist] = H_final
        all_hf_states[dist] = hf_state

    unified_pauli_strings = sorted(list(all_pauli_strings))
    pauli_to_idx = {pauli: idx for idx, pauli in enumerate(unified_pauli_strings)}

    actual_num_coeffs = len(unified_pauli_strings)
    if num_coeffs > actual_num_coeffs:
        print(f"Warning: Requested {num_coeffs} coefficients but only {actual_num_coeffs} are available. Using {actual_num_coeffs} coefficients.")
        num_coeffs = actual_num_coeffs

    print(f"Found {len(unified_pauli_strings)} unique Pauli strings across all Hamiltonians")
    print(f"Using {num_coeffs} coefficients for flow model input")

    unified_hamiltonians = {}
    unified_coeffs = {}
    all_gs_energies = {}   

    print("Second pass: creating unified Hamiltonians...")
    for dist in distances:
        print(f"  Restructuring Hamiltonian for bond length {dist} Å")

        H = original_hamiltonians[dist]
        coeffs, obs = H.terms()

        unified_coeff_array = np.zeros(len(unified_pauli_strings))

        for c, o in zip(coeffs, obs):
            pauli_str = str(o)
            if pauli_str in pauli_to_idx:
                idx = pauli_to_idx[pauli_str]
                unified_coeff_array[idx] = c
            else:
                print(f"Warning: Pauli string {pauli_str} not found in unified list")

        unified_hamiltonians[dist] = H
        unified_coeffs[dist] = torch.tensor(unified_coeff_array[:num_coeffs], dtype=torch.float32, device=device)

        H_sparse = qml.SparseHamiltonian(H.sparse_matrix(), wires=H.wires)
        gs_eigenvalues = qml.eigvals(H_sparse, k=2)
        gs_energy = gs_eigenvalues.min()
        all_gs_energies[dist] = gs_energy

    with open(os.path.join(save_dir, 'unified_pauli_strings.json'), 'w') as f:
        json.dump({
            'unified_pauli_strings': unified_pauli_strings,
            'pauli_to_idx': {k: v for k, v in pauli_to_idx.items() if isinstance(k, str)}
        }, f, indent=4)

    with open(os.path.join(save_dir, 'ground_state_energies.json'), 'w') as f:
        json.dump({str(k): float(v) for k, v in all_gs_energies.items()}, f, indent=4)

    return unified_hamiltonians, all_gs_energies, all_hf_states, unified_pauli_strings, pauli_to_idx, unified_coeffs, num_coeffs


def calculate_hf_energies(unified_hamiltonians, all_hf_states, device):
    """Calculate Hartree-Fock energies for all Hamiltonians"""
    hf_energies = {}

    print("Calculating HF energies for all bond lengths...")
    for dist, H in unified_hamiltonians.items():
        hf_state = all_hf_states[dist]

        if torch.device(device).type == "cuda":
            dev = qml.device("lightning.gpu", wires=H.wires)
        else:
            dev = qml.device("lightning.qubit", wires=H.wires)
 
        @qml.qnode(dev, interface='torch')
        def hf_circuit():
            sorted_wires = sorted(H.wires.tolist())
            qml.BasisState(hf_state, wires=sorted_wires)
            return qml.expval(H)

        # Calculate and store HF energy
        hf_energy = hf_circuit()
        hf_energies[dist] = float(hf_energy)

    return hf_energies 