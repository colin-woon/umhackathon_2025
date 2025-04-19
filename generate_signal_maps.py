# python generate_signal_maps.py --run_dir outputs/run_20250419_171148 --states 8 --config config.yaml --n_calls 1000 --patience 100 --optimize_metric total_return
import itertools
# import random # No longer needed
import math
# import subprocess # No longer needed
import yaml       # To read/write config files
import re         # To parse output for score
import os         # To handle file paths and deletion
import sys        # To check python executable path
import joblib     # <<< Import joblib
import argparse   # <<< Import argparse
import pandas as pd # <<< Import pandas
import numpy as np # <<< Import numpy
from skopt import gp_minimize # <<< Import skopt components
from skopt.space import Integer
from skopt.utils import use_named_args
import functools # <<< Import functools
from skopt.callbacks import EarlyStopper # <<< Import EarlyStopper

# --- Import necessary functions from main_strategy --- <<< NEW SECTION
try:
    # Assuming main_strategy.py is in the same directory
    from main_strategy import generate_signals, run_backtest, calculate_performance, load_config
except ImportError as e:
    print(f"Error: Could not import functions from main_strategy.py: {e}", file=sys.stderr)
    print("Ensure main_strategy.py is in the same directory or accessible via PYTHONPATH.", file=sys.stderr)
    exit(1)
# --- END NEW SECTION ---

# --- Function to load artifacts --- <<< NEW FUNCTION
def load_artifacts(artifacts_dir):
    """Loads pre-computed artifacts needed for signal evaluation."""
    print(f"Loading artifacts from: {artifacts_dir}")
    try:
        states_bt = joblib.load(os.path.join(artifacts_dir, 'states_backtest.pkl'))
        states_ft = joblib.load(os.path.join(artifacts_dir, 'states_forwardtest.pkl'))
        df_feat_bt = joblib.load(os.path.join(artifacts_dir, 'df_features_backtest.pkl'))
        df_feat_ft = joblib.load(os.path.join(artifacts_dir, 'df_features_forwardtest.pkl'))
        # selected_features = joblib.load(os.path.join(artifacts_dir, 'selected_features.pkl')) # Not strictly needed here
        print("Artifacts loaded successfully.")
        return states_bt, states_ft, df_feat_bt, df_feat_ft
    except FileNotFoundError as e:
        print(f"Error loading artifacts: File not found - {e}. Did you run main_strategy.py first?", file=sys.stderr)
        return None, None, None, None
    except Exception as e:
        print(f"Error loading artifacts: {e}", file=sys.stderr)
        return None, None, None, None
# --- END NEW FUNCTION ---

# --- Function to run the strategy and get performance ---
def run_backtest_and_get_performance(signal_map, original_config_path, artifacts_data):
    """
    Applies a signal map using pre-loaded BACKTEST states/features and calculates performance.

    Args:
        signal_map (dict): The signal map to evaluate.
        original_config_path (str): Path to the original configuration file (used for fee).
        artifacts_data (tuple): Tuple containing loaded states_bt, states_ft, df_feat_bt, df_feat_ft.

    Returns:
        dict: A dictionary containing performance metrics like 'Sharpe' and 'Total Return [%]'.
              Returns {'Sharpe': -math.inf, 'Total Return [%]': -math.inf} if execution fails.
    """
    states_bt, states_ft, df_feat_bt, df_feat_ft = artifacts_data
    error_return = {'Sharpe': -math.inf, 'Total Return (%)': -math.inf}

    if states_bt is None or df_feat_bt is None: # Check if artifacts loaded
        print("Error: Artifacts not loaded, cannot evaluate signal map.", file=sys.stderr)
        return error_return

    try:
        # 1. Load config just to get the fee
        config_data = load_config(original_config_path)
        if config_data is None:
            print(f"Error: Could not load config {original_config_path} to get fee.", file=sys.stderr)
            return error_return
        trading_fee = config_data.get('trading_fee_percent', 0.0006)

        # 2. Prepare Backtest Data
        if 'state' in df_feat_bt.columns: df_feat_bt = df_feat_bt.drop(columns=['state'])
        df_feat_bt_with_state = df_feat_bt.assign(state=states_bt)

        # 3. Generate Signals (Backtest)
        df_signals_bt = generate_signals(df_feat_bt_with_state, signal_map)

        # 4. Run Backtest Simulation
        df_results_bt = run_backtest(df_signals_bt, trading_fee)

        # 5. Calculate Performance (Backtest)
        performance_summary_bt = calculate_performance(df_results_bt, trading_fee)

        # 6. Extract the desired scores <<< MODIFIED
        sharpe_ratio = performance_summary_bt.get('Annualized Sharpe Ratio', -math.inf)
        total_return = performance_summary_bt.get('Total Return (%)', -math.inf)

        # Handle potential non-numeric values or None for Sharpe
        if not isinstance(sharpe_ratio, (int, float)) or sharpe_ratio is None or np.isnan(sharpe_ratio):
            print(f"Warning: Sharpe Ratio is not a valid number ({sharpe_ratio}). Setting to -inf.", file=sys.stderr)
            sharpe_ratio = -math.inf

        # Handle potential non-numeric values or None for Total Return
        if not isinstance(total_return, (int, float)) or total_return is None or np.isnan(total_return):
            print(f"Warning: Total Return is not a valid number ({total_return}). Setting to -inf.", file=sys.stderr)
            total_return = -math.inf

        # print(f"  -> Evaluated Performance (Sharpe Ratio): {performance_score:.4f}") # Reduce verbosity inside objective
        # <<< MODIFIED RETURN KEY: Use 'Total Return (%)' to match objective_function expectation
        return {'Sharpe': float(sharpe_ratio), 'Total Return (%)': float(total_return)}

    except Exception as e:
        print(f"An unexpected error occurred during signal evaluation: {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc()
        return error_return # <<< Return dict with correct keys

# --- Function to evaluate the strategy on FORWARD TEST data --- <<< NEW FUNCTION
def evaluate_forward_test(signal_map, original_config_path, artifacts_data):
    """
    Applies a signal map using pre-loaded FORWARD TEST states/features and calculates performance.

    Args:
        signal_map (dict): The signal map to evaluate.
        original_config_path (str): Path to the original configuration file (used for fee).
        artifacts_data (tuple): Tuple containing loaded states_bt, states_ft, df_feat_bt, df_feat_ft.

    Returns:
        dict: A dictionary containing performance metrics like 'Sharpe' and 'Total Return (%)'.
              Returns {'Sharpe': -math.inf, 'Total Return (%)': -math.inf} if execution fails.
    """
    _states_bt, states_ft, _df_feat_bt, df_feat_ft = artifacts_data # Unpack, focusing on FT
    error_return = {'Sharpe': -math.inf, 'Total Return (%)': -math.inf}

    if states_ft is None or df_feat_ft is None: # Check if FT artifacts loaded
        print("Error: Forward test artifacts not loaded, cannot evaluate.", file=sys.stderr)
        return error_return

    try:
        # 1. Load config just to get the fee
        config_data = load_config(original_config_path)
        if config_data is None:
            print(f"Error: Could not load config {original_config_path} to get fee.", file=sys.stderr)
            return error_return
        trading_fee = config_data.get('trading_fee_percent', 0.0006)

        # 2. Prepare Forward Test Data
        if 'state' in df_feat_ft.columns: df_feat_ft = df_feat_ft.drop(columns=['state'])
        df_feat_ft_with_state = df_feat_ft.assign(state=states_ft)

        # 3. Generate Signals (Forward Test)
        df_signals_ft = generate_signals(df_feat_ft_with_state, signal_map)

        # 4. Run Forward Test Simulation
        df_results_ft = run_backtest(df_signals_ft, trading_fee)

        # 5. Calculate Performance (Forward Test)
        performance_summary_ft = calculate_performance(df_results_ft, trading_fee)

        # 6. Extract the desired scores
        sharpe_ratio = performance_summary_ft.get('Annualized Sharpe Ratio', -math.inf)
        total_return = performance_summary_ft.get('Total Return (%)', -math.inf)

        # Handle potential non-numeric values or None for Sharpe
        if not isinstance(sharpe_ratio, (int, float)) or sharpe_ratio is None or np.isnan(sharpe_ratio):
            print(f"Warning (FT): Sharpe Ratio is not a valid number ({sharpe_ratio}). Setting to -inf.", file=sys.stderr)
            sharpe_ratio = -math.inf

        # Handle potential non-numeric values or None for Total Return
        if not isinstance(total_return, (int, float)) or total_return is None or np.isnan(total_return):
            print(f"Warning (FT): Total Return is not a valid number ({total_return}). Setting to -inf.", file=sys.stderr)
            total_return = -math.inf

        return {'Sharpe': float(sharpe_ratio), 'Total Return (%)': float(total_return)}

    except Exception as e:
        print(f"An unexpected error occurred during forward test evaluation: {e}", file=sys.stderr)
        return error_return
# --- END NEW FUNCTION ---

# --- Objective Function for Bayesian Optimization --- <<< MODIFIED FUNCTION
# Note: skopt aims to MINIMIZE the objective function.
# Since we want to MAXIMIZE Sharpe Ratio or Total Return, we return the NEGATIVE of the chosen metric.
def objective_function(param_list, state_keys, config_path, artifacts, metric_to_optimize):
    """
    Wrapper function for skopt. Converts parameter list to signal_map,
    runs the backtest, and returns the negative of the chosen performance metric.
    """
    signal_map = dict(zip(state_keys, param_list))
    print(f"Evaluating: {signal_map}") # Keep track of what's being tried
    performance_results = run_backtest_and_get_performance(signal_map, config_path, artifacts)

    # Select the metric to optimize based on the input argument
    if metric_to_optimize == 'sharpe':
        score = performance_results['Sharpe'] # Key is 'Sharpe'
        metric_name = "Sharpe Ratio"
    elif metric_to_optimize == 'total_return':
        score = performance_results['Total Return (%)']
        metric_name = "Total Return (%)"
    else:
        # Should not happen due to argument choices, but good practice
        print(f"Error: Invalid metric_to_optimize '{metric_to_optimize}'", file=sys.stderr)
        return math.inf # Return large positive number for minimization

    # --- MODIFIED CHECK ---
    # Handle the case where the backtest failed (NaN) or returned worst score (-inf)
    # Return positive infinity so skopt avoids this region
    if np.isnan(score): # Check for NaN first - indicates calculation error
        print(f"  -> Evaluation returned invalid ({metric_name}) {score}.")
        return math.inf # Return large positive number for minimization
    elif score == -math.inf: # Check for -inf - indicates worst possible score
        # Note: This might be triggered if calculate_performance returns -inf for negative returns.
        print(f"  -> Evaluation returned worst possible score (-inf) for {metric_name}.")
        return math.inf # Return large positive number for minimization
    # --- END MODIFIED CHECK ---

    print(f"  -> {metric_name}: {score:.4f}")
    return -score # Return negative score for minimization
# --- END MODIFIED FUNCTION ---

# --------------------------------------------------------

# --- Main script logic ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Optimize HMM signal maps using Bayesian Optimization.")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the base configuration file (used for fee).')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to the run output directory containing the artifacts subdirectory.')
    parser.add_argument('--states', type=int, required=True, # <<< Make required
                        help='Number of HMM states used in the artifacts (determines signal map size).')
    parser.add_argument('--n_calls', type=int, default=100,
                        help='Number of optimization iterations (evaluations).')
    parser.add_argument('--patience', type=int, default=None,
                        help='Number of iterations with no improvement to wait before stopping early. Default: None (disabled).')
    parser.add_argument('--optimize_metric', type=str, default='sharpe', # <<< NEW ARGUMENT
                        choices=['sharpe', 'total_return'],
                        help='The performance metric to optimize ("sharpe" or "total_return"). Default: sharpe.')


    args = parser.parse_args()
    # --- END Argument Parser ---

    # Define the keys (state numbers) and possible values
    state_keys = range(args.states)
    values = [-1, 0, 1] # Possible signal values
    original_config_file = args.config
    run_output_dir = args.run_dir
    artifacts_dir = os.path.join(run_output_dir, 'artifacts')
    metric_name_map = {'sharpe': 'Annualized Sharpe Ratio', 'total_return': 'Total Return (%)'} # <<< Map for display


    # --- Load Artifacts ONCE ---
    artifacts_data = load_artifacts(artifacts_dir)
    if artifacts_data[0] is None:
        print("Exiting due to artifact loading failure.", file=sys.stderr)
        exit(1)
    # --- END Load Artifacts ---

    # Check if original config file exists
    if not os.path.exists(original_config_file):
        print(f"Error: Original config file '{original_config_file}' not found.", file=sys.stderr)
        exit(1)

    # --- Define Search Space for skopt ---
    # List of dimensions, one for each state
    search_space = [Integer(low=-1, high=1, name=f'state_{i}') for i in state_keys]
    # --- END Search Space ---

    # --- User Feedback --- <<< MODIFIED
    print(f"Starting Bayesian Optimization for {args.states} states.")
    print(f"Optimizing for: {metric_name_map[args.optimize_metric]}") # <<< Indicate target metric
    print(f"Number of evaluations: {args.n_calls}")
    if args.patience:
        print(f"Early stopping patience: {args.patience}")
    print(f"Using artifacts from: {artifacts_dir}")
    print(f"Base config for fee: {original_config_file}")
    print("---------------------------------------------------------")
    # --- END User Feedback ---

    # --- Prepare the objective function with fixed arguments --- <<< MODIFIED
    # Use functools.partial to pass the fixed arguments (keys, config, artifacts, metric)
    # to the objective function, as skopt only passes the parameters to be optimized.
    obj_func_partial = functools.partial(
        objective_function,
        state_keys=state_keys,
        config_path=original_config_file,
        artifacts=artifacts_data,
        metric_to_optimize=args.optimize_metric # <<< Pass the chosen metric
    )
    # --- END NEW ---

    # --- Prepare Callbacks (Early Stopping) ---
    callbacks = []
    if args.patience is not None and args.patience > 0:
        # Note: EarlyStopper in skopt stops when func calls stop decreasing.
        # Since our objective returns *negative* score, we want it to stop
        # when the negative score stops decreasing (i.e., score stops increasing).
        # The default behavior of EarlyStopper is suitable here.
        # We use a custom class for clearer logging and control.

        class PatienceStopper:
            def __init__(self, patience, metric_name):
                self.patience = patience
                self.metric_name = metric_name
                self.best_fun = math.inf # Store best *negative* score
                self.iters_no_improvement = 0
                print(f"Early stopping enabled for {metric_name} with patience={patience}")

            def __call__(self, res):
                """
                'res' contains the optimization result at the end of each iteration.
                Key attributes: 'fun' (current best objective value), 'x' (location).
                """
                current_best_fun = res.fun # Current best negative score
                if current_best_fun < self.best_fun:
                    self.best_fun = current_best_fun
                    self.iters_no_improvement = 0
                    # print(f"  Callback: New best {self.metric_name} found: {-self.best_fun:.4f}") # Optional debug
                else:
                    self.iters_no_improvement += 1
                    # print(f"  Callback: No improvement iter {self.iters_no_improvement}/{self.patience}") # Optional debug

                if self.iters_no_improvement >= self.patience:
                    print(f"\nEarly stopping triggered for {self.metric_name} after {self.patience} iterations without improvement.")
                    return True # Signal skopt to stop
                else:
                    return False # Signal skopt to continue

        callbacks.append(PatienceStopper(args.patience, metric_name_map[args.optimize_metric]))
    # --- END Callbacks ---

    # --- Run Bayesian Optimization ---
    # gp_minimize performs Bayesian optimization using Gaussian Processes.
    # n_calls: Total number of evaluations.
    # n_initial_points: How many steps of random exploration before starting optimization.
    # acq_func: Acquisition function ('gp_hedge' is often a good default).
    # random_state: For reproducibility.
    result = gp_minimize(
        func=obj_func_partial,
        dimensions=search_space,
        n_calls=args.n_calls,
        n_initial_points=max(10, args.n_calls // 10), # Start with some random exploration
        acq_func="gp_hedge",
        random_state=42,
        callback=callbacks if callbacks else None,
        verbose=True # Print progress from skopt
    )
    # --- END Optimization Run ---

    print("=========================================================")
    print(f"--- Optimization Complete --- ")
    # Report actual evaluations done
    actual_evaluations = len(result.func_vals)
    print(f"Target evaluations: {args.n_calls}")
    print(f"Actual evaluations performed: {actual_evaluations}")
    print(f"Optimized for: {metric_name_map[args.optimize_metric]}") # <<< Remind user of the target

    # Extract best results
    best_params_list = result.x
    best_objective_value = result.fun # This is the minimized negative score

    if best_objective_value == math.inf:
        print("Optimization did not find a valid signal map (all evaluations failed).")
    else:
        best_signal_map = dict(zip(state_keys, best_params_list))
        optimized_score = -best_objective_value # Convert back to positive score

        print("---------------------------------------------------------")
        print("Best Performing Signal Map Found:")
        formatted_map = {k: int(v) for k, v in sorted(best_signal_map.items())} # Convert values to standard int
        print(yaml.dump({'signal_map': formatted_map}, default_flow_style=False, indent=2))
        print(f"Best Backtest Score Achieved ({metric_name_map[args.optimize_metric]}): {optimized_score:.4f}") # Clarify this is Backtest

        # --- Calculate and display BOTH metrics for the best map --- <<< MODIFIED SECTION
        print("\nPerformance Metrics for the Best Map:")
        # Re-run backtest for final numbers (consistent with how score was calculated)
        final_performance_bt = run_backtest_and_get_performance(best_signal_map, original_config_file, artifacts_data)
        final_sharpe_bt = final_performance_bt.get('Sharpe', 'N/A')
        final_return_bt = final_performance_bt.get('Total Return (%)', 'N/A')

        # --- Run Forward Test Evaluation --- <<< NEW
        print("Evaluating best map on Forward Test data...")
        final_performance_ft = evaluate_forward_test(best_signal_map, original_config_file, artifacts_data)
        final_sharpe_ft = final_performance_ft.get('Sharpe', 'N/A')
        final_return_ft = final_performance_ft.get('Total Return (%)', 'N/A')
        # --- END NEW ---

        print("\n  --- Backtest Period ---")
        if isinstance(final_sharpe_bt, float):
            print(f"    - Annualized Sharpe Ratio: {final_sharpe_bt:.4f}")
        else:
            print(f"    - Annualized Sharpe Ratio: {final_sharpe_bt}")
        if isinstance(final_return_bt, float):
             print(f"    - Total Return (%):          {final_return_bt:.4f}")
        else:
             print(f"    - Total Return (%):          {final_return_bt}")

        print("\n  --- Forward Test Period ---") # <<< NEW SECTION
        if isinstance(final_sharpe_ft, float):
            print(f"    - Annualized Sharpe Ratio: {final_sharpe_ft:.4f}")
        else:
            print(f"    - Annualized Sharpe Ratio: {final_sharpe_ft}")
        if isinstance(final_return_ft, float):
             print(f"    - Total Return (%):          {final_return_ft:.4f}")
        else:
             print(f"    - Total Return (%):          {final_return_ft}")
        # --- END NEW SECTION ---

        print("---------------------------------------------------------")
        # --- END MODIFIED SECTION ---


        # Optional: Print info about the optimization process
        # print("Optimization Details:")
        # print(f"Function value at minimum: {-optimized_score:.4f} (Negative {metric_name_map[args.optimize_metric]})")
        # print(f"Location of minimum: {best_params_list}")

    print("=========================================================")

# --- DELETE OLD BRUTE FORCE LOOP ---
#    best_signal_map = None
# ... (rest of the deleted code remains deleted) ...
#    print("=========================================================") 