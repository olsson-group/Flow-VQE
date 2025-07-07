#!/usr/bin/env python3
"""
Usage Examples for Optimization Baselines & Flow VQE
==========================================================

This script shows all default experiments in the paper: https://arxiv.org/abs/2507.01726
"""

def print_example(title, command, description):
    print(f"\n{title}")
    print("-" * 50)
    print(f"Command: {command}")
    print(f"Description: {description}")

def main():
    print("Usage Examples - Optimization Baselines & Flow-VQE")
    print("=" * 80)
    
###################################################################################################
###################################################################################################
    print("\n" + "="*80)
    print("“optimization_baselines” EXAMPLES")
    print("="*80)

###################################################################################################
    print("\n--------------  Default optimization examples for H4, H2O. See Fig. 2 in the paper ----------------")
    print_example(
        "1. Direct optimization for H4 molecule",
        "python optimization_baselines_main.py --molecule H4 --mode default --optimizers ADAM GD QNSPSA",
        "Runs direct optimization for H4 molecule with ADAM, GD, QNSPSA optimizers, learning rate 0.02, max 151 iterations"
    )
    
    print_example(
        "2. Direct optimization for H2O molecule", 
        "python optimization_baselines_main.py --molecule H2O --mode default --optimizers ADAM GD QNSPSA",
        "Runs direct optimization for H2O molecule with ADAM, GD, QNSPSA optimizers, learning rate 0.02, max 151 iterations"
    )

###################################################################################################
    print("\n--------------  Warm-up optimization examples for H2O, H4. See Table I in the paper ----------------")
    print_example( 
        "3. Warm-up optimization for H2O",
        "python optimization_baselines_main.py --molecule H2O --mode warm_up --learning_rate 0.02 --max_iterations 1000", 
        "Runs warm-up optimization for H2O at distance 1.9Å using generated parameters, with learning rate 0.02"
    )
    
    print_example( 
        "4. Warm-up optimization for H4",
        "python optimization_baselines_main.py --molecule H4 --mode warm_up --learning_rate 0.02 --max_iterations 1000", 
        "Runs warm-up optimization for H4 at distance 2.5755102040816324Å using generated parameters, with learning rate 0.02"
    )

###################################################################################################
    print("\n--------------  Warm-up vs. PT for NH3, C6H6. See Fig. 4 in the paper ----------------")
    print_example(
        "5. Warm-up optimization for NH3",
        "python optimization_baselines_main.py --molecule NH3 --mode warm_up",
        "Runs warm-up optimization for NH3 across multiple distances using pre-trained parameters"
    )
    
    print_example(
        "6. Warm-up optimization for C6H6",
        "python optimization_baselines_main.py --molecule C6H6 --mode warm_up",
        "Runs warm-up optimization for C6H6 across multiple distances using pre-trained parameters"
    )
    
    print_example(
        "7. Parameter transfer optimization for NH3",
        "python optimization_baselines_main.py --molecule NH3 --mode parameter_transfer",
        "Runs parameter transfer optimization for NH3 using optimized parameters from distance 0.0Å"
    )
    
    print_example(
        "8. Parameter transfer optimization for C6H6",
        "python optimization_baselines_main.py --molecule C6H6 --mode parameter_transfer",
        "Runs parameter transfer optimization for C6H6 using optimized parameters from distance 0.0Å"
    )
    
###################################################################################################
###################################################################################################
    print("\n" + "="*80)
    print("“flow_vqe” EXAMPLES")
    print("="*80)

###################################################################################################
    print("\n--------------  Flow-VQE-S training for H4, H2O. See Fig. 2 in the paper ----------------")
    print_example(
        "9. H4 single distance training in [0.6, 0.89, 1.17, 1.46, 1.74, 2.03, 2.31, 2.6] Å",
        "python flow_vqe_main.py --molecules H4 --training_range 2.6 --training_mode single --n_flows 7",
        "Just change '--training_range' when using default settings"
    )
    
    print_example(
        "10. H2O single distance training in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8] Å",
        "python flow_vqe_main.py --molecules H2O --training_range 1.8 --training_mode single --n_flows 10",
        "Just change '--training_range' when using default settings"
    )

###################################################################################################
    print("\n--------------  Flow-VQE-M training for all four molecules  ----------------")
    print("H4: np.linspace(0.6, 2.6, 8), H2O: np.linspace(0.8, 1.8, 6). See Fig. 3 & 2 in the paper") 
    print("NH3: np.linspace(-0.5, 0.5, 4), C6H6: np.linspace(-0.3, 0.6, 4). See Fig. 4 in the paper")
    
    print_example( 
        "11. H4 multi-distance training",
        "python flow_vqe_main.py --molecules H4 --training_mode multi --n_epochs 5001",
        "Just change '--molecules' when using default settings"
    ) 

###################################################################################################
    print("\n--------------  Read the trained model and warm-up generation for all four molecules  ----------------")
    print("H4: np.linspace(0.6, 2.8, 50), H2O: np.linspace(0.75, 1.9, 50). See Fig. 3 in the paper") 
    print("NH3: np.linspace(-0.55, 0.55, 10), C6H6: np.linspace(-0.4, 0.7, 10). See Fig. 4 in the paper")

    print_example(
        "12. H2O warm-up generation",
        "python flow_vqe_warm_up.py --molecules H2O",
        "Just change '--molecules' when using default settings"
    )
###################################################################################################

    print("\n" + "="*80)
    print("RESULT DIRECTORIES")
    print("="*80)
    
    print("\nOptimization Baselines Result Directories:")
    print("- vqe_optimization_results_*/ (default mode)")
    print("- warm_optimization_results_*/ (warm-up mode)")
    print("- pt_optimization_results_*/ (parameter transfer mode)")
    
    print("\nFlow VQE Result Directories:")
    print("- flow_vqe_results_*/ (training mode)")
    print("- flow_vqe_warm_up_results_*/ (warm-up mode)")

if __name__ == "__main__":
    main() 