# Generative flow-based warm start of the variational quantum eigensolver

This repository contains the implementation of Flow-VQE, a novel approach combining variational quantum eigensolvers (VQE) with normalizing flows for molecular ground state optimization and warm-start. The code supports the research presented in [arXiv:2507.01726](https://arxiv.org/abs/2507.01726).

## Overview

The project includes two main components:
- **Optimization Baselines**: Traditional VQE optimization methods with various optimizers, and post-training by invoking warm-start or parameter-transfer parameters
- **Flow-VQE**: Main procedure using normalizing flows as surrogate models for VQE training

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Data Repository: Due to data capacity limitations, the experimental results and trained models are stored in the [a_store_data](https://drive.google.com/drive/folders/1mDTOIzxq2T6BhY_qCFSTw9EszD9fsJdH?usp=sharing). (Download it for the warm-start function.)

## Quick Start

### Optimization Baselines

Run traditional VQE optimization for different molecules:

```bash
# H4 molecule with multiple optimizers  (default bond-length range)
python optimization_baselines_main.py --molecule H4 --mode default --optimizers ADAM GD QNSPSA

# H2O molecule optimization  (default bond-length range)
python optimization_baselines_main.py --molecule H2O --mode default --optimizers ADAM GD QNSPSA
```

### Flow-VQE Training

Train Flow-VQE models for molecular systems:

```bash
# Single distance training for H4
python flow_vqe_main.py --molecules H4 --training_range 2.6 --training_mode single --n_flows 7  --n_epochs 3001

# Multi-distance training for H2O 
python flow_vqe_main.py --molecules H2O --training_range "np.linspace(0.8, 1.8, 6)" --training_mode multi --n_epochs 5001 
```

### Warm-up Parameter Generation

Generate optimized parameters using trained Flow-VQE models:

```bash
# Generate warm-up parameters for H2O using the trained flow model 
python flow_vqe_warm_up.py --molecules H2O --test_range "np.linspace(0.75, 1.9, 50)"
```
 
## Complete Usage Examples

**For detailed usage examples and reproduction of all paper results, please refer to `example_usage.py`**. This file contains:

- All default optimization experiments (Fig. 2)
- Warm-up optimization examples (Table I) 
- Parameter transfer comparisons (Fig. 4)
- Flow-VQE training configurations (Fig. 2, 3, 4)
- Complete command-line examples for every experiment in the paper

Simply run:
```bash
python example_usage.py
```

This will display all available commands and their descriptions for reproducing the complete experimental results from the paper.

## File Structure

### Main Scripts
- `flow_vqe_main.py` - Main script for Flow-VQE training and evaluation
- `flow_vqe_warm_up.py` - Script for generating warm-up parameters using trained Flow-VQE models
- `optimization_baselines_main.py` - Script for running traditional VQE optimization baselines
- `example_usage.py` - Comprehensive examples for reproducing all paper results
- `ansatz.py` - Quantum circuit ansatz definitions and implementations
- `molecule_configs.py` - Molecular system configurations and parameters
- `read_draw.ipynb` - Jupyter notebook for data analysis and visualization

### Core Packages
- `flow_vqe/` - Main Flow-VQE implementation package
  - `main.py` - Core Flow-VQE training logic
  - `flow_training.py` - Normalizing flow training procedures
  - `molecule_utils.py` - Molecular system utilities and setup
  - `circuit_utils.py` - Quantum circuit manipulation utilities
  - `evaluation.py` - Model evaluation and metrics
  - `plotting.py` - Visualization and plotting functions
  - `utils.py` - General utility functions
  - `warm_up.py` - Warm-start parameter generation
  - `config.py` - Configuration management

- `optimization_baselines/` - Traditional VQE optimization package
  - `runner.py` - Optimization algorithm implementations
  - `config.py` - Baseline optimization configurations
  - `plotter.py` - Results visualization
  - `utils.py` - Utility functions for baselines

### Data and Results 
- `a_store_data/` - Experimental results and trained models, find [here](https://drive.google.com/drive/folders/1mDTOIzxq2T6BhY_qCFSTw9EszD9fsJdH?usp=sharing)
  - `flow_vqe_m_*/` - Flow-VQE-M training results
  - `flow_vqe_s_optimization/` - Flow-VQE-S optimization experiments
  - `pt_optimization_results_*/` - Parameter transfer optimization results
  - `vqe_optimization_results_*/` - Traditional VQE optimization results
  - `warm_optimization_*/` - Warm-start optimization results
  - `warm_up/` - Generated warm-up parameters

## Contact

If you have any questions or other issues, please contact me at: hangzo@chalmers.se

## Cite this paper
```bibtex
@article{zou2025generative,
      title={Generative flow-based warm start of the variational quantum eigensolver}, 
      author={Hang Zou and Martin Rahm and Anton Frisk Kockum and Simon Olsson},
      year={2025},
      eprint={2507.01726},
      archivePrefix={arXiv}
}
```
