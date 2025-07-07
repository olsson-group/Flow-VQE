import pennylane as qml
import pennylane.numpy as pnp


def hardware_efficient_ansatz(weights, wires, layers, hf_state):
    k = 0
    n_qubits = len(wires) 

    for i in wires:
        if k >= len(weights):
            break
        qml.RY(weights[k], wires=i)
        k += 1
     
    for l in range(layers): 
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
         
        for i in wires:
            if k >= len(weights):
                break
            qml.RY(weights[k], wires=i)
            k += 1
     
    if hasattr(hf_state, 'numpy'):
        hf_state = hf_state.cpu().numpy()
    qml.BasisState(hf_state, wires=wires)

def givens_ansatz(weights, wires, hf_state, singles, doubles, layers):
   
    if hasattr(hf_state, 'numpy'):
        hf_state = hf_state.cpu().numpy()
    qml.BasisState(hf_state, wires=wires)
    j = 0
    for _ in range(layers):
        for s in singles:
            qml.SingleExcitation(weights[j], wires=s)
            j += 1
        for d in doubles:
            qml.DoubleExcitation(weights[j], wires=d)
            j += 1
            

def get_ansatz_circuit(ansatz_type, weights, wires, hf_state, layers=None, singles=None, doubles=None):
    """
    Factory function to get the appropriate ansatz circuit based on the type.
    
    Args:
        ansatz_type (str): Type of ansatz ('HEA' or 'GIVENS')
        weights (array): Parameter array for the ansatz
        wires (list): List of qubit wires
        hf_state (array): Hartree-Fock state to initialize the circuit
        layers (int, optional): Number of layers for HEA ansatz
        singles (list, optional): List of single excitation indices for Givens ansatz
        doubles (list, optional): List of double excitation indices for Givens ansatz
    
    Returns:
        function: The ansatz circuit function
    """
    if ansatz_type == 'HEA':
        if layers is None:
            raise ValueError("Number of layers must be specified for HEA ansatz")
        return lambda: hardware_efficient_ansatz(weights, wires, layers, hf_state)
    
    elif ansatz_type == 'GIVENS':
        if singles is None or doubles is None:
            raise ValueError("Singles and doubles must be specified for Givens ansatz")
        return lambda: givens_ansatz(weights, wires, hf_state, singles, doubles, layers)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")

def get_ansatz_parameter_count(ansatz_type, n_qubits, layers=None, singles=None, doubles=None):
    """
    Calculate the number of parameters required for a given ansatz type.
    
    Args:
        ansatz_type (str): Type of ansatz ('HEA' or 'GIVENS')
        n_qubits (int): Number of qubits
        layers (int, optional): Number of layers for HEA ansatz
        singles (list, optional): List of single excitation indices for Givens ansatz
        doubles (list, optional): List of double excitation indices for Givens ansatz
    
    Returns:
        int: Number of parameters required for the ansatz
    """
    if ansatz_type == 'HEA':
        if layers is None:
            raise ValueError("Number of layers must be specified for HEA ansatz") 
        return n_qubits * (layers + 1)
    
    elif ansatz_type == 'GIVENS':
        if singles is None or doubles is None:
            raise ValueError("Singles and doubles must be specified for Givens ansatz")
        return (len(singles) + len(doubles)) * layers
    
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}") 