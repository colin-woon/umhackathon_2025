import pandas as pd
import numpy as np
import yaml
import os
import argparse
from sklearn.preprocessing import StandardScaler
import joblib
from hmmlearn import hmm
import matplotlib.pyplot as plt # For plotting
import seaborn as sns          # For heatmap
from datetime import datetime  # For timestamped output directory
import shutil                 # For copying config file
from scipy.stats import ttest_1samp
from scipy.stats import percentileofscore # For empirical p-value
from itertools import combinations # Needed for comparing chunk pairs

# --- Configuration Loading ---
def load_config(config_path='config.yaml'):
    """Loads configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

# --- Data Loading and Preprocessing (REVISED RENAMING) ---
def load_and_preprocess_data(config, mode='backtest'):
    """Loads data from CSVs, merges, and preprocesses with explicit renaming."""
    data_dir = config['data_directory']
    data_files = config['data_files']
    suffix = f"_{mode}"

    all_data = {}
    base_candle_key = 'candles'
    final_column_names = {} # Keep track of final names for engineer_features

    # --- Load Base Candle Data ---
    try:
        candle_info = data_files.get(base_candle_key, None) # Get the dictionary for candles
        if not isinstance(candle_info, dict) or 'file_base' not in candle_info:
             # Handle cases where 'candles' entry is missing, not a dict, or lacks 'file_base'
             raise ValueError(f"'{base_candle_key}' entry in config['data_files'] is missing or improperly configured (must be a dict with 'file_base').")

        candle_file_base_str = candle_info['file_base'] # Extract the file_base string

        candle_path = os.path.join(data_dir, f"{candle_file_base_str}{suffix}.csv")
        print(f"Loading candles: {candle_path}")
        df_candles = pd.read_csv(candle_path, index_col='timestamp', parse_dates=True)

        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_candles.columns for col in ohlcv_cols):
            ohlc_map = {'t':'start_time', 'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'}
            if all(col in df_candles.columns for col in ohlc_map.keys()):
                df_candles.rename(columns=ohlc_map, inplace=True)
                print("  Renamed candle columns from t,o,h,l,c,v")
            else:
                 # Check if required columns exist even if not named perfectly
                 if not ('close' in df_candles.columns and 'volume' in df_candles.columns):
                    raise ValueError(f"Candle file missing essential 'close' or 'volume' columns")
        # Ensure numeric types for essential columns
        df_candles['close'] = pd.to_numeric(df_candles['close'], errors='coerce')
        df_candles['volume'] = pd.to_numeric(df_candles['volume'], errors='coerce')

        # Store only essential columns with standard names
        merged_df = df_candles[['close', 'volume']].copy()
        final_column_names['close'] = 'close'
        final_column_names['volume'] = 'volume'
        print(f"  Loaded {len(df_candles)} candle records.")

    except FileNotFoundError:
        print(f"Error: Base candle file not found: {candle_path}")
        return None, None
    except Exception as e:
        print(f"Error loading candle data: {e}")
        return None, None

    # --- Load and Merge Other Data Files ---
    for key, file_info in data_files.items():
        if key == base_candle_key:
            continue

        # Ensure file_info is a dictionary (handles the simple 'candles' case if needed)
        if not isinstance(file_info, dict):
            print(f"Warning: Skipping entry '{key}' in data_files as it's not a dictionary.")
            continue

        file_base = file_info.get('file_base')
        original_columns = file_info.get('original_columns', []) # Get list of potential original cols
        target_col_name = file_info.get('target_column')       # Get the desired target name

        if not file_base or not target_col_name:
            print(f"Warning: Skipping entry '{key}' due to missing 'file_base' or 'target_column' in config.")
            continue

        try:
            file_path = os.path.join(data_dir, f"{file_base}{suffix}.csv")
            print(f"Loading {key}: {file_path}")
            df_temp = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

            col_to_use = None
            # Find the first available column from the original_columns list
            for potential_col in original_columns:
                if potential_col in df_temp.columns:
                    col_to_use = potential_col
                    print(f"  Found specified column '{col_to_use}' for key '{key}'.")
                    break # Stop searching once found


            # Fallback if no specific logic or column found
            if col_to_use is None:
                data_cols = [col for col in df_temp.columns if col.lower() not in ['timestamp', 'start_time', 'end_time', 'time', 'date']] # Exclude 'date' too
                if data_cols:
                    col_to_use = data_cols[0]
                    # Use a generic fallback name if target_col_name wasn't set
                    if target_col_name is None: target_col_name = f"{key}_generic"
                    print(f"  Using fallback column '{col_to_use}' for key '{key}', renaming to '{target_col_name}'")
                else:
                    print(f"  Warning: Could not find suitable data column in {key}. Skipping merge.")
                    continue

            # Perform rename and merge
            df_to_merge = df_temp[[col_to_use]].rename(columns={col_to_use: target_col_name})
            # Ensure numeric before merge
            df_to_merge[target_col_name] = pd.to_numeric(df_to_merge[target_col_name], errors='coerce')

            merged_df = pd.merge(merged_df, df_to_merge, left_index=True, right_index=True, how='outer')
            final_column_names[key] = target_col_name # Store the final name used
            print(f"  Loaded and merged {len(df_temp)} records for {key} as '{target_col_name}'.")

        except FileNotFoundError:
            print(f"Warning: Data file not found for {key}: {file_path}. Skipping.")
        except Exception as e:
            print(f"Warning: Error loading or merging data for {key}: {e}. Skipping.")

    # --- Handle Missing Data ---
    print(f"\nColumns before ffill: {merged_df.columns.tolist()}")
    print(f"NaN counts before ffill:\n{merged_df.isna().sum()}")
    # Forward fill NaNs first (handles gaps within series)
    merged_df.ffill(inplace=True)
    # Backward fill NaNs next (handles leading NaNs if first value was missing)
    merged_df.bfill(inplace=True)
    # Drop any rows where essential data (like price) might still be missing
    merged_df.dropna(subset=['close'], inplace=True)
    # Optionally fill remaining NaNs in feature columns with 0 (if bfill wasn't enough)
    merged_df.fillna(0, inplace=True)


    print(f"\nTotal records after merging and NaN handling: {len(merged_df)}")
    print(f"Columns after merge/fill: {merged_df.columns.tolist()}")
    if merged_df.empty:
        print("Error: No data available after loading and merging.")
        return None, None

    # Return both the dataframe and the mapping of config keys to final column names
    return merged_df, final_column_names

# --- Feature Engineering (Revised for Core Robust Features) ---
def engineer_features(df, final_column_names, config_features): # config_features now unused, we use the fixed list
    """Creates a core set of features for HMM, prioritizing stationarity."""
    print("Engineering core features...")
    if df is None or df.empty:
        print("Error: Cannot engineer features on empty DataFrame.")
        return None
    df_feat = df.copy()
    print(f"\nDEBUG: Columns available at start of engineer_features: {df_feat.columns.tolist()}\n")

    # --- Essential Base Columns ---
    close_col = 'close'
    volume_col = 'volume'
    funding_col = final_column_names.get('cryptoquant_funding')
    active_col = final_column_names.get('glassnode_active')
    tx_col = final_column_names.get('glassnode_tx') # Needed for active_addr_tx_ratio calc if used
    inflow_col = final_column_names.get('cryptoquant_inflow')
    oi_col = final_column_names.get('coinglass_oi')
    lsr_col = final_column_names.get('coinglass_lsr') # Assuming this holds the raw L/S ratio

    # --- 1. Price Return ---
    # Hypothesis: Basic market movement.
    if close_col in df_feat.columns:
        df_feat['price_return'] = df_feat[close_col].pct_change()
    else:
        print(f"Warning: Column '{close_col}' not found for price_return.")
        df_feat['price_return'] = 0 # Assign default or handle error

    # --- 2. Price Volatility ---
    # Hypothesis: Market risk regime.
    df_feat['volatility_10d'] = df_feat['price_return'].rolling(window=10).std()

    # --- 3. Funding Rate Change ---
    # Hypothesis: Momentum in derivatives sentiment/cost.
    if funding_col and funding_col in df_feat.columns:
        df_feat['funding_rate_change_1d'] = df_feat[funding_col].diff()
    else:
        print(f"Warning: Column '{funding_col}' not found for funding_rate_change_1d.")
        df_feat['funding_rate_change_1d'] = 0

    # --- 4. Funding Rate Extreme Flag ---
    # Hypothesis: Funding extremes signal potential reversals.
    if funding_col and funding_col in df_feat.columns:
        rolling_window_90d = 90
        # Calculate 5th and 95th percentiles over rolling window
        lower_bound = df_feat[funding_col].rolling(window=rolling_window_90d, min_periods=int(rolling_window_90d*0.8)).quantile(0.05)
        upper_bound = df_feat[funding_col].rolling(window=rolling_window_90d, min_periods=int(rolling_window_90d*0.8)).quantile(0.95)
        df_feat['funding_rate_extreme_flag'] = ((df_feat[funding_col] < lower_bound) | (df_feat[funding_col] > upper_bound)).astype(int)
    else:
        print(f"Warning: Column '{funding_col}' not found for funding_rate_extreme_flag.")
        df_feat['funding_rate_extreme_flag'] = 0

    # --- 5. Active Address Rate of Change (14d) ---
    # Hypothesis: Network adoption/activity momentum.
    if active_col and active_col in df_feat.columns:
        df_feat['active_addr_roc_14d'] = df_feat[active_col].pct_change(periods=14)
    else:
        print(f"Warning: Column '{active_col}' not found for active_addr_roc_14d.")
        df_feat['active_addr_roc_14d'] = 0

    # --- 6. Inflow / Volume Ratio ---
    # Hypothesis: Significance of exchange flow pressure relative to market activity.
    if inflow_col and inflow_col in df_feat.columns and volume_col in df_feat.columns:
         denominator = pd.to_numeric(df_feat[volume_col], errors='coerce').replace(0, np.nan)
         numerator = pd.to_numeric(df_feat[inflow_col], errors='coerce')
         df_feat['inflow_vol_ratio'] = (numerator / denominator)
    else:
         print(f"Warning: Columns needed for inflow_vol_ratio not found ('{inflow_col}', '{volume_col}').")
         df_feat['inflow_vol_ratio'] = 0

    # --- 7. Open Interest Rate of Change (7d) ---
    # Hypothesis: Momentum in leverage and market participation.
    if oi_col and oi_col in df_feat.columns:
        df_feat['oi_roc_7d'] = df_feat[oi_col].pct_change(periods=7)
    else:
        print(f"Warning: Column '{oi_col}' not found for oi_roc_7d.")
        df_feat['oi_roc_7d'] = 0

    # --- 8. & 9. L/S Ratio Z-Score & Extreme Flag ---
    # Hypothesis: Statistically extreme retail sentiment signals potential reversals.
    if lsr_col and lsr_col in df_feat.columns:
        rolling_window_90d_lsr = 90
        lsr_series = pd.to_numeric(df_feat[lsr_col], errors='coerce')
        rolling_mean = lsr_series.rolling(window=rolling_window_90d_lsr, min_periods=int(rolling_window_90d_lsr*0.8)).mean()
        rolling_std = lsr_series.rolling(window=rolling_window_90d_lsr, min_periods=int(rolling_window_90d_lsr*0.8)).std()
        # Calculate Z-score, handle potential division by zero if std is 0
        df_feat['lsr_zscore_90d'] = (lsr_series - rolling_mean) / rolling_std.replace(0, np.nan)
        # Extreme Flag based on Z-score
        z_threshold = 2.0
        df_feat['lsr_extreme_flag'] = ((df_feat['lsr_zscore_90d'] > z_threshold) | (df_feat['lsr_zscore_90d'] < -z_threshold)).astype(int)
    else:
        print(f"Warning: Column '{lsr_col}' not found for lsr_zscore_90d/lsr_extreme_flag.")
        df_feat['lsr_zscore_90d'] = 0
        df_feat['lsr_extreme_flag'] = 0

    # --- 10. OI Change x Price Change Interaction ---
    # Hypothesis: Combined momentum confirms trends or signals divergences.
    # Using 1-day OI change for closer interaction timing, assuming oi_change_1d exists or can be calc'd easily
    if oi_col and oi_col in df_feat.columns: # Recalculate 1d OI change if not present
         if 'oi_change_1d' not in df_feat.columns:
              df_feat['oi_change_1d'] = df_feat[oi_col].pct_change(periods=1)
         df_feat['oi_change_x_price_change'] = df_feat['oi_change_1d'] * df_feat['price_return']
    else:
         print(f"Warning: Columns needed for oi_change_x_price_change not found ('{oi_col}', 'price_return').")
         df_feat['oi_change_x_price_change'] = 0


    # --- Select only the core features for the final DataFrame ---
    core_features_list = [
        "price_return",
        "volatility_10d",
        "funding_rate_change_1d",
        "funding_rate_extreme_flag",
        "active_addr_roc_14d",
        "inflow_vol_ratio",
        "oi_roc_7d",
        "lsr_zscore_90d", # Keep Z-score, it's more informative than just the flag
        # "lsr_extreme_flag", # Can be excluded if Z-score is used, or keep both
        "oi_change_x_price_change"
    ]
    # Also add back 'close' column needed for backtesting calculations downstream
    if close_col in df.columns:
         core_features_list.append(close_col)

    df_final_feat = df_feat[[col for col in core_features_list if col in df_feat.columns]].copy()


    # --- Clean up NaNs/Infs introduced by calculations in the final selection ---
    df_final_feat.fillna(method='ffill', inplace=True)
    df_final_feat.fillna(method='bfill', inplace=True)
    df_final_feat.fillna(0, inplace=True) # Fill any remaining NaNs with 0
    df_final_feat.replace([np.inf, -np.inf], 0, inplace=True)

    print(f"Core features engineered. Data shape: {df_final_feat.shape}")
    print(f"Engineered columns available: {df_final_feat.columns.tolist()}")

    # Ensure no NaNs/Infs remain in the final DataFrame before returning
    if df_final_feat.isnull().values.any() or np.isinf(df_final_feat.select_dtypes(include=np.number)).values.any():
         print("Warning: NaNs or Infs still present after cleaning!")
         # Optional: Implement stricter cleaning or return None
         # df_final_feat = df_final_feat.fillna(0).replace([np.inf, -np.inf], 0) # Re-apply just in case

    return df_final_feat


def generate_full_eda_heatmap(df_raw, final_column_names, full_feature_list_from_config, run_output_dir):
    """
    Calculates as many features as possible from the full config list
    and plots their correlation heatmap for EDA purposes.
    This function is separate from the main pipeline's feature engineering.
    """
    print("\n--- Generating Full EDA Heatmap ---")
    if df_raw is None or df_raw.empty:
        print("Error: Cannot generate EDA heatmap, raw data is empty.")
        return

    df_eda = df_raw.copy() # Start with raw data

    # --- Re-calculate Features from the FULL list ---
    # (This will be a simplified version, add more calcs as needed/possible)
    # You'll need to copy/adapt logic from your research list or engineer_features
    # for ALL the features you want in the EDA heatmap.

    print(f"Base columns for EDA: {df_eda.columns.tolist()}")

    close_col = 'close'
    volume_col = 'volume'
    funding_col = final_column_names.get('cryptoquant_funding')
    active_col = final_column_names.get('glassnode_active')
    tx_col = final_column_names.get('glassnode_tx')
    inflow_col = final_column_names.get('cryptoquant_inflow')
    oi_col = final_column_names.get('coinglass_oi')
    lsr_col = final_column_names.get('coinglass_lsr')
    open_col = 'open' # Assuming 'open' exists in df_raw for body_ratio
    high_col = 'high' # Assuming 'high' exists
    low_col = 'low'   # Assuming 'low' exists

    # --- PRICE ---
    if close_col in df_eda.columns:
        df_eda['price_return'] = df_eda[close_col].pct_change()
        if high_col in df_eda.columns and low_col in df_eda.columns:
             df_eda['price_range_ratio_ohlc'] = (df_eda[high_col] - df_eda[low_col]) / df_eda[close_col].replace(0, np.nan)
        if open_col in df_eda.columns:
             df_eda['price_body_ratio_ohlc'] = (df_eda[close_col] - df_eda[open_col]).abs() / df_eda[close_col].replace(0, np.nan)
    df_eda['volatility_10d'] = df_eda['price_return'].rolling(window=10).std()

    # --- FUNDING ---
    if funding_col and funding_col in df_eda.columns:
        df_eda['cryptoquant_funding_rate'] = df_eda[funding_col] # Keep raw
        df_eda['funding_rate_change_1d'] = df_eda[funding_col].diff()
        df_eda['funding_rate_MA_7d'] = df_eda[funding_col].rolling(window=7).mean()
        # Basic percentile/flag (copy from engineer_features)
        rolling_window_90d = 90
        lower_bound = df_eda[funding_col].rolling(window=rolling_window_90d, min_periods=int(rolling_window_90d*0.8)).quantile(0.05)
        upper_bound = df_eda[funding_col].rolling(window=rolling_window_90d, min_periods=int(rolling_window_90d*0.8)).quantile(0.95)
        df_eda['funding_rate_extreme_flag'] = ((df_eda[funding_col] < lower_bound) | (df_eda[funding_col] > upper_bound)).astype(int)
        # Add more funding features here if desired for EDA plot...

    # --- ACTIVE ADDR ---
    if active_col and active_col in df_eda.columns:
        df_eda['glassnode_active_addr_value'] = df_eda[active_col] # Keep raw
        df_eda['active_addr_roc_14d'] = df_eda[active_col].pct_change(periods=14)
        # Add more active addr features here...

    # --- TX COUNT ---
    if tx_col and tx_col in df_eda.columns:
        df_eda['glassnode_tx_count_value'] = df_eda[tx_col] # Keep raw
        df_eda['tx_count_roc_14d'] = df_eda[tx_col].pct_change(periods=14)

     # --- COMBINED/RATIO ---
    if active_col and tx_col and active_col in df_eda.columns and tx_col in df_eda.columns:
        denominator = pd.to_numeric(df_eda[tx_col], errors='coerce').replace(0, np.nan)
        numerator = pd.to_numeric(df_eda[active_col], errors='coerce')
        df_eda['active_addr_tx_ratio'] = (numerator / denominator)
        # Add inflow_active_addr_ratio_7d etc. if needed

    # --- INFLOW ---
    if inflow_col and inflow_col in df_eda.columns:
        df_eda['cryptoquant_inflow_value'] = df_eda[inflow_col] # Keep raw
        if volume_col in df_eda.columns:
            denominator = pd.to_numeric(df_eda[volume_col], errors='coerce').replace(0, np.nan)
            numerator = pd.to_numeric(df_eda[inflow_col], errors='coerce')
            df_eda['inflow_vol_ratio'] = (numerator / denominator) # Already calculated, maybe keep
        # Add inflow spike flag etc.

    # --- OI ---
    if oi_col and oi_col in df_eda.columns:
        df_eda['coinglass_oi_value'] = df_eda[oi_col] # Keep raw
        df_eda['oi_roc_7d'] = df_eda[oi_col].pct_change(periods=7)
        # Add other OI features...

    # --- LSR ---
    if lsr_col and lsr_col in df_eda.columns:
        df_eda['coinglass_lsr_value'] = df_eda[lsr_col] # Keep raw
        df_eda['lsr_MA_7d'] = df_eda[lsr_col].rolling(window=7).mean()
        # Basic Z-score/flag (copy from engineer_features)
        rolling_window_90d_lsr = 90
        lsr_series = pd.to_numeric(df_eda[lsr_col], errors='coerce')
        rolling_mean = lsr_series.rolling(window=rolling_window_90d_lsr, min_periods=int(rolling_window_90d_lsr*0.8)).mean()
        rolling_std = lsr_series.rolling(window=rolling_window_90d_lsr, min_periods=int(rolling_window_90d_lsr*0.8)).std()
        df_eda['lsr_zscore_90d'] = (lsr_series - rolling_mean) / rolling_std.replace(0, np.nan)
        z_threshold = 2.0
        df_eda['lsr_extreme_flag'] = ((df_eda['lsr_zscore_90d'] > z_threshold) | (df_eda['lsr_zscore_90d'] < -z_threshold)).astype(int)
        # Add more LSR features...

    # --- INTERACTIONS ---
    # Calculate oi_change_1d if needed for interaction
    if oi_col in df_eda.columns and 'oi_change_1d' not in df_eda.columns:
         df_eda['oi_change_1d'] = df_eda[oi_col].pct_change(periods=1)
    if 'oi_change_1d' in df_eda.columns and 'price_return' in df_eda.columns:
         df_eda['oi_change_x_price_change'] = df_eda['oi_change_1d'] * df_eda['price_return']
     # Add funding_rate_x_oi_change if needed


    # --- Final Cleanup for EDA ---
    df_eda.fillna(method='ffill', inplace=True)
    df_eda.fillna(method='bfill', inplace=True)
    df_eda.fillna(0, inplace=True)
    df_eda.replace([np.inf, -np.inf], 0, inplace=True)

    # --- Select ONLY features from the FULL config list that were successfully calculated ---
    eda_features_present = [f for f in full_feature_list_from_config if f in df_eda.columns]
    print(f"Features calculated for EDA heatmap: {eda_features_present}")

    if not eda_features_present:
        print("No features available for EDA heatmap.")
        return

    numeric_eda_df = df_eda[eda_features_present].select_dtypes(include=np.number)
    if numeric_eda_df.empty:
        print("No numeric features available for EDA heatmap.")
        return

    # --- Calculate and Plot ---
    correlation_matrix_eda = numeric_eda_df.corr()
    heatmap_path = os.path.join(run_output_dir, 'correlation_heatmap_FULL_EDA.png') # Use distinct name
    plot_correlation_heatmap(correlation_matrix_eda, numeric_eda_df.columns.tolist(), heatmap_path)

# --- HMM Model (Modified for Scaled Data & Reduced Verbosity) ---
def train_hmm(X_scaled, n_states, covariance_type, n_iter): # Takes X_scaled directly
    """Trains the Gaussian HMM model on pre-scaled data with reduced output."""
    # Note: Changed verbose=False to hide detailed iteration logs
    print(f"Training HMM with {n_states} states on scaled data...")

    if X_scaled is None or len(X_scaled) == 0:
        print("Error: No scaled data available for HMM training.")
        return None

    try:
        model = hmm.GaussianHMM(n_components=n_states,
                                covariance_type=covariance_type,
                                n_iter=n_iter,
                                random_state=42,
                                verbose=False) # <<< Set verbose=False to hide iteration details

        # Although verbose is False, we still need to capture convergence info from the monitor
        # We can temporarily redirect stdout/stderr during fit if needed, but often
        # checking monitor_.converged after fit is sufficient. HMMlearn might still print
        # a small amount, but the iteration table will be gone. Let's rely on the monitor check.

        model.fit(X_scaled) # Fit the model

        # Check convergence status AFTER fitting
        if not model.monitor_.converged:
             print(f"Warning: HMM n_states={n_states} did not converge in {n_iter} iterations!")
        else:
             print("HMM training complete (converged).") # Still useful to know it finished and converged
        return model
    except Exception as e:
        print(f"Error during HMM training: {e}")
        return None

def predict_states(model, X_scaled): # Takes X_scaled directly
    """Predicts hidden states using the trained HMM on pre-scaled data."""
    print("Predicting hidden states on scaled data...")
    # X = df[features].values # REMOVED

    # Removed NaN checks here

    if X_scaled is None or len(X_scaled) == 0:
        print("Error: No scaled data available for state prediction.")
        return None

    try:
        states = model.predict(X_scaled)
        return states
    except Exception as e:
        print(f"Error during state prediction: {e}")
        return None

# --- Signal Generation (Modified) ---
def generate_signals(df_with_state, signal_map): # Takes DataFrame with 'state' column
    """Generates trading signals based on predicted states in the DataFrame."""
    print("Generating trading signals...")
    # if states is None: # Removed - check happens before calling
    #     print("Error: Cannot generate signals, states are None.")
    #     return df
    if 'state' not in df_with_state.columns:
         print("Error: 'state' column not found in DataFrame for signal generation.")
         # Return original df or handle error
         return df_with_state.copy() # Avoid modifying original if error occurs

    df = df_with_state.copy() # Work on a copy
    df['signal'] = df['state'].map(signal_map).fillna(0) # Map states to signals

    df['signal'] = df['signal'].astype(int)

    print("Signals generated.")
    return df

# --- Backtesting (Modified for Cumulative PnL) ---
def run_backtest(df, fee_percent, start_value=1000.0, verbose=True):
    """
    Runs the backtest simulation, calculates PnL, strategy equity,
    Buy & Hold equity, and Cumulative PnL.
    """
    if verbose: print("Running backtest...")
    if 'close' not in df.columns or 'signal' not in df.columns:
         print("Error: DataFrame missing 'close' or 'signal' column for backtest.")
         return None

    df_backtest = df.copy()

    # Strategy Calculation
    df_backtest['position'] = df_backtest['signal'].shift(1).fillna(0)
    df_backtest['price_change_pct'] = df_backtest['close'].pct_change().fillna(0)
    df_backtest['strategy_return_gross'] = df_backtest['position'] * df_backtest['price_change_pct']
    df_backtest['trade'] = df_backtest['position'].diff().fillna(0)
    fees = abs(df_backtest['trade']) * fee_percent
    df_backtest['strategy_return_net'] = df_backtest['strategy_return_gross'] - fees

    # Calculate cumulative returns relative to 1 AND portfolio value
    df_backtest['cumulative_strategy_relative'] = (1 + df_backtest['strategy_return_net']).cumprod()
    df_backtest['strategy_portfolio_value'] = start_value * df_backtest['cumulative_strategy_relative']

    # --- ADD Cumulative PnL Calculation ---
    # This represents the sum of net returns over time, starting from 0
    df_backtest['cumulative_pnl'] = df_backtest['strategy_return_net'].cumsum()
    # --- End Add ---

    # Buy and Hold Calculation
    df_backtest['asset_return'] = df_backtest['close'].pct_change().fillna(0)
    df_backtest['buy_and_hold_relative'] = (1 + df_backtest['asset_return']).cumprod()
    df_backtest['buy_and_hold_portfolio_value'] = start_value * df_backtest['buy_and_hold_relative']

    if verbose: print("Backtest finished.")
    return df_backtest

def calculate_performance(df_backtest, fee_percent, timeframe='daily', start_value=1.0):
    """Calculates performance metrics, adapting annualization for timeframe."""
    print(f"Calculating performance metrics (Timeframe: {timeframe})...")
    results = {}
    if df_backtest is None or df_backtest.empty:
        print("Error: No backtest results DataFrame provided.")
        # Return empty or default dictionary
        return {
            'Total Return (%)': 0, 'Annualized Sharpe Ratio': 0, 'Sortino Ratio (qs)': 'N/A',
            'Maximum Drawdown (%)': 0, 'Trade Frequency (%)': 0, 'Number of Trades': 0,
            'T-statistic (vs 0)': 'N/A', 'P-value (vs 0)': 'N/A'
        }

    # --- Calculate Portfolio Value if missing (based on net returns) ---
    if 'strategy_portfolio_value' not in df_backtest.columns and 'strategy_return_net' in df_backtest.columns:
        print("Calculating portfolio value from net returns...")
        df_backtest['strategy_portfolio_value'] = start_value * (1 + df_backtest['strategy_return_net']).cumprod()
        # Handle initial NaN if pct_change started with NaN
        df_backtest['strategy_portfolio_value'].fillna(start_value, inplace=True)

    # Ensure index is datetime for frequency calculations
    if not isinstance(df_backtest.index, pd.DatetimeIndex):
        print("Warning: DataFrame index is not DatetimeIndex. Converting...")
        try:
            df_backtest.index = pd.to_datetime(df_backtest.index)
        except Exception as e:
            print(f"Error converting index to DatetimeIndex: {e}. Time-based metrics might be inaccurate.")

    # --- Calculate metrics based on PORTFOLIO VALUE and RETURNS ---

    # Total Return - FIX: Use first non-NaN value as starting point
    if 'strategy_portfolio_value' in df_backtest.columns and not df_backtest['strategy_portfolio_value'].empty:
        # Get first valid portfolio value (handling potential NaNs at start)
        start_idx = df_backtest['strategy_portfolio_value'].first_valid_index()
        if start_idx is not None:
            start_value_actual = df_backtest.loc[start_idx, 'strategy_portfolio_value']
            final_value = df_backtest['strategy_portfolio_value'].iloc[-1]

            # Check if values make sense before calculating
            if pd.notnull(final_value) and pd.notnull(start_value_actual) and start_value_actual > 0:
                total_return_pct = ((final_value / start_value_actual) - 1) * 100
                # Sanity check - cap extreme values that are likely calculation errors
                if total_return_pct > 10000:  # Cap at 10,000% which is still extremely high
                    print(f"Warning: Capping unrealistic return of {total_return_pct:.2f}% to 10,000%")
                    total_return_pct = 10000.0
                results['Total Return (%)'] = total_return_pct
            else:
                results['Total Return (%)'] = 0
                print("Warning: Invalid portfolio values detected (NaN or negative). Setting total return to 0%.")
        else:
            results['Total Return (%)'] = 0
            print("Warning: No valid portfolio values found. Setting total return to 0%.")
    else:
        results['Total Return (%)'] = 0
        print("Warning: Could not calculate Total Return ('strategy_portfolio_value' missing or empty).")

    # Sharpe & Sortino based on PERIODIC returns
    if 'strategy_return_net' in df_backtest.columns:
        net_returns = df_backtest['strategy_return_net'].fillna(0)

        # Clean extreme values that might be calculation errors (over 100% in a single period)
        if abs(net_returns).max() > 1.0:
            extreme_count = (abs(net_returns) > 1.0).sum()
            if extreme_count > 0:
                print(f"Warning: Found {extreme_count} extreme return values (>100%). Capping for Sharpe calculation.")
                net_returns = net_returns.clip(-1.0, 1.0)

        # --- Determine Periods Per Year based on timeframe ---
        if timeframe == 'hourly':
            periods_per_year = 365 * 24  # Use calendar hours for crypto
            print(f"Using periods_per_year = {periods_per_year} for hourly data.")
        elif timeframe == 'daily':
            # Try inferring based on median difference, default to 252 if fails
            if isinstance(df_backtest.index, pd.DatetimeIndex) and len(df_backtest.index) > 1:
                time_diff = df_backtest.index.to_series().diff().median()
                # Check if time_diff is valid and close to 1 day
                if pd.notna(time_diff) and time_diff.total_seconds() > 0:
                    if pd.Timedelta(hours=23) <= time_diff <= pd.Timedelta(hours=25):
                        periods_per_year = 365  # Assume calendar days if median diff is daily
                        print(f"Detected daily data (median diff), using periods_per_year = {periods_per_year}.")
                    else:
                        periods_per_year = pd.Timedelta(days=365) / time_diff  # Infer from actual median diff
                        print(f"Inferred periods_per_year = {periods_per_year:.2f} from median time diff.")
                else:
                    periods_per_year = 252  # Fallback for daily
                    print(f"Warning: Could not infer daily frequency. Defaulting periods_per_year to {periods_per_year}.")
            else:
                periods_per_year = 252  # Fallback for daily
                print(f"Warning: No DatetimeIndex or insufficient data. Defaulting periods_per_year to {periods_per_year}.")
        else:  # Handle other timeframes or default
            periods_per_year = 252  # Default if timeframe unspecified or unknown
            print(f"Warning: Unknown timeframe '{timeframe}'. Defaulting periods_per_year to {periods_per_year}.")

        mean_periodic_return = net_returns.mean()
        std_dev_periodic_return = net_returns.std()

        # Annualized Sharpe calculation
        if std_dev_periodic_return > 0 and periods_per_year > 0:
            sharpe_ratio = (mean_periodic_return * np.sqrt(periods_per_year)) / std_dev_periodic_return
        else:
            sharpe_ratio = 0
            print("Warning: Standard deviation is zero or periods_per_year invalid. Sharpe ratio set to 0.")
        results['Annualized Sharpe Ratio'] = sharpe_ratio

        # Optional: Sortino Ratio (using quantstats)
        try:
            # quantstats expects series with datetime index for correct period inference if periods not given
            # Provide the calculated periods_per_year
            results['Sortino Ratio (qs)'] = qs.stats.sortino(net_returns, periods=periods_per_year)
        except NameError:
            results['Sortino Ratio (qs)'] = 'N/A (quantstats not installed)'
        except Exception as e:
            results['Sortino Ratio (qs)'] = f'Error ({type(e).__name__})'
            print(f"Error calculating quantstats Sortino Ratio: {e}")
    else:
        results['Annualized Sharpe Ratio'] = 0
        results['Sortino Ratio (qs)'] = 'N/A'
        print("Warning: 'strategy_return_net' column missing for Sharpe/Sortino.")

    # Maximum Drawdown based on Portfolio Value - FIX: Handle extreme values
    if 'strategy_portfolio_value' in df_backtest.columns and not df_backtest['strategy_portfolio_value'].empty:
        portfolio_value = df_backtest['strategy_portfolio_value'].copy()

        # Clean portfolio values to remove potential errors
        if portfolio_value.min() <= 0:
            print("Warning: Found non-positive portfolio values. Cleaning for drawdown calculation.")
            portfolio_value = portfolio_value.clip(lower=0.001)  # Set minimum to small positive

        running_max = portfolio_value.cummax()
        # Ensure running_max is not zero before dividing
        drawdown = (portfolio_value - running_max) / running_max.replace(0, np.nan)  # Avoid division by zero
        max_drawdown = drawdown.min()

        # Sanity check - drawdown shouldn't be lower than -100%
        if max_drawdown < -1.0:
            print(f"Warning: Fixing unrealistic drawdown of {max_drawdown*100:.2f}% to -100%")
            max_drawdown = -1.0

        results['Maximum Drawdown (%)'] = (max_drawdown * 100) if pd.notna(max_drawdown) else 0
    else:
        results['Maximum Drawdown (%)'] = 0
        print("Warning: Could not calculate Maximum Drawdown ('strategy_portfolio_value' missing or empty).")

    # Trade Frequency (remains the same calculation - per data row)
    if 'trade' in df_backtest.columns:
        num_trades = (df_backtest['trade'] != 0).sum()
        total_rows = len(df_backtest)
        trade_frequency = (num_trades / total_rows) * 100 if total_rows > 0 else 0
        results['Trade Frequency (%)'] = trade_frequency
        results['Number of Trades'] = num_trades
    else:
        results['Trade Frequency (%)'] = 0
        results['Number of Trades'] = 0
        print("Warning: 'trade' column not found for frequency calculation.")

    # T-test on Net Returns (remains the same calculation - tests if mean return is different from zero)
    if 'strategy_return_net' in df_backtest.columns:
        net_returns = df_backtest['strategy_return_net'].fillna(0)
        if len(net_returns) > 1:
            try:
                t_stat, p_value = ttest_1samp(net_returns, 0)
                results['T-statistic (vs 0)'] = t_stat
                results['P-value (vs 0)'] = p_value
            except Exception as e:
                results['T-statistic (vs 0)'] = f'Error ({type(e).__name__})'
                results['P-value (vs 0)'] = f'Error ({type(e).__name__})'
                print(f"Error calculating T-test: {e}")
        else:
            results['T-statistic (vs 0)'] = 'N/A (Too few samples)'
            results['P-value (vs 0)'] = 'N/A (Too few samples)'
    else:
        results['T-statistic (vs 0)'] = 'N/A (No returns column)'
        results['P-value (vs 0)'] = 'N/A (No returns column)'

    print("\n--- Performance Summary ---")
    for key, value in results.items():
        if isinstance(value, (int, float, np.number)):  # Check if numeric
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("---------------------------\n")

    return results

# --- Function to Find Best HMM based on AIC/BIC ---
def find_best_hmm_model(X_scaled, config, feature_list):
    """
    Loops through HMM states, trains models, and selects the best based on BIC/AIC.

    Args:
        X_scaled (np.ndarray): The scaled feature data for training.
        config (dict): The configuration dictionary.
        feature_list (list): List of feature names used (for context in logs).

    Returns:
        tuple: (best_hmm_model, best_n_states, best_bic, best_aic)
               Returns (None, -1, np.inf, np.inf) if no model converges.
    """
    best_n_states = -1
    best_bic = np.inf
    best_aic = np.inf
    best_hmm_model = None

    # Get parameters from config
    hmm_iterations = config['hmm_iterations']
    covariance_type = config['hmm_covariance_type']
    min_states = config['min_hmm_states']
    max_states = config['max_hmm_states']

    print(f"\n--- Searching for best n_states ({min_states} to {max_states}) using BIC/AIC ---")
    print(f"Features being used: {feature_list}")

    for n_states_test in range(min_states, max_states + 1):
        print(f"\nTesting HMM with {n_states_test} states...")
        # Train HMM directly on scaled data
        current_hmm_model = train_hmm(X_scaled,
                                      n_states_test,
                                      covariance_type,
                                      hmm_iterations) # Pass existing train_hmm function

        if current_hmm_model:
            # Check for convergence explicitly if train_hmm doesn't guarantee it
            if hasattr(current_hmm_model, 'monitor_') and not current_hmm_model.monitor_.converged:
                 print(f"  HMM n_states={n_states_test} did not converge. Skipping.")
                 continue # Skip BIC/AIC if not converged

            try:
                current_bic = current_hmm_model.bic(X_scaled)
                current_aic = current_hmm_model.aic(X_scaled)
                print(f"  n_states={n_states_test} -> BIC: {current_bic:.2f}, AIC: {current_aic:.2f}")

                # Check if this model is better (lower BIC is primary criteria)
                if current_bic < best_bic:
                    best_bic = current_bic
                    best_aic = current_aic
                    best_n_states = n_states_test
                    best_hmm_model = current_hmm_model
                    print(f"  *** New best model found with n_states={best_n_states} (BIC: {best_bic:.2f}) ***")
                elif current_bic == best_bic and current_aic < best_aic: # Tie-break with AIC
                    best_aic = current_aic
                    best_n_states = n_states_test
                    best_hmm_model = current_hmm_model
                    print(f"  *** New best model found with n_states={best_n_states} (AIC tie-breaker: {best_aic:.2f}) ***")

            except Exception as e:
                print(f"  Error calculating BIC/AIC for n_states={n_states_test}: {e}")
        else:
             print(f"  HMM training failed for n_states={n_states_test}. Skipping.")

    # --- End of Loop ---
    return best_hmm_model, best_n_states, best_bic, best_aic

# --- Feature Selection (Modified to return correlation matrix) ---
def select_features_by_correlation(df_features_backtest, features_available, corr_threshold):
    """
    Performs feature selection by removing highly correlated features.
    Also returns the correlation matrix and corresponding feature names for plotting.

    Args:
        df_features_backtest (pd.DataFrame): DataFrame containing engineered features for backtesting.
        features_available (list): List of feature names initially available for selection.
        corr_threshold (float): The correlation threshold above which features will be dropped.

    Returns:
        tuple: (final_feature_list, correlation_matrix, numeric_feature_names)
               - final_feature_list (list): Selected feature names.
               - correlation_matrix (pd.DataFrame or None): The calculated correlation matrix.
               - numeric_feature_names (list or None): Names of features used in the matrix.
               Returns (features_available, None, None) if selection cannot be performed.
    """
    print("\n--- Performing Basic Feature Selection (using Backtest data) ---")
    print(f"Features available for selection: {features_available}")
    final_feature_list = features_available[:] # Start with all available features
    correlation_matrix = None
    numeric_feature_names = None

    if len(features_available) <= 1:
        print("Not enough features available for correlation check.")
        return final_feature_list, correlation_matrix, numeric_feature_names

    # Select only numeric features for correlation calculation
    numeric_features_df = df_features_backtest[features_available].select_dtypes(include=np.number)
    numeric_feature_names = numeric_features_df.columns.tolist() # Get names used
    print(f"Numeric features considered for correlation: {numeric_feature_names}")

    if len(numeric_features_df.columns) <= 1:
        print("Not enough numeric features for correlation check.")
        return final_feature_list, correlation_matrix, numeric_feature_names # Return names even if no matrix

    # Calculate correlation matrix
    correlation_matrix = numeric_features_df.corr() # Keep original signs for heatmap
    corr_matrix_abs = correlation_matrix.abs() # Use absolute for dropping
    upper_tri = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

    # Find features to drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]

    if to_drop:
        print(f"Features to drop due to high correlation (>{corr_threshold}): {to_drop}")
        final_feature_list = [f for f in features_available if f not in to_drop]
    else:
        print("No features dropped based on correlation threshold.")

    if not final_feature_list:
        print("Warning: Feature selection removed all features! Returning original list.")
        # Return original list, but still return matrix and names for potential plotting
        return features_available, correlation_matrix, numeric_feature_names

    print(f"\nFeatures selected for HMM: {final_feature_list}")
    # Return list, the calculated matrix (not abs), and the names used for the matrix
    return final_feature_list, correlation_matrix, numeric_feature_names

def perform_within_backtest_stability_check(states_backtest, backtest_index, config):
    """
    Checks HMM state stability by comparing distributions across chunks
    within the backtest period.

    Args:
        states_backtest (np.ndarray): Predicted states for the entire backtest period.
        backtest_index (pd.DatetimeIndex): The datetime index corresponding to states_backtest.
        config (dict): Configuration dictionary containing threshold and chunk settings.

    Returns:
        tuple: (comparison_df, max_diff, model_is_stable)
               - comparison_df (pd.DataFrame or None): DataFrame showing distributions per chunk.
               - max_diff (float): Max difference found between any two chunks.
               - model_is_stable (bool): True if max_diff <= threshold.
               Returns (None, np.inf, False) if check is disabled or fails.
    """
    num_chunks = config['stability_num_backtest_chunks']
    stability_threshold = config.get('state_stability_threshold', 20) # Default threshold
    model_is_stable = False # Default
    max_diff = np.inf       # Default
    comparison_df = None    # Default

    print("\n--- Within-Backtest State Stability Check ---")

    if num_chunks < 2:
        print(f"Check disabled (stability_num_backtest_chunks = {num_chunks} < 2).")
        return comparison_df, max_diff, model_is_stable
    if states_backtest is None or len(states_backtest) == 0:
        print("Could not perform check: Missing backtest state predictions.")
        return comparison_df, max_diff, model_is_stable
    if len(states_backtest) != len(backtest_index):
         print("Could not perform check: Mismatch between states length and index length.")
         return comparison_df, max_diff, model_is_stable
    if len(states_backtest) < num_chunks:
        print(f"Could not perform check: Not enough data points ({len(states_backtest)}) for {num_chunks} chunks.")
        return comparison_df, max_diff, model_is_stable


    print(f"Splitting backtest period into {num_chunks} chunks for comparison.")

    try:
        # Create a Series with datetime index to facilitate splitting if needed,
        # but np.array_split works on the array directly based on position.
        # state_series = pd.Series(states_backtest, index=backtest_index) # Not strictly needed for np.array_split

        # Split the states array into N chunks based on position
        # np.array_split handles uneven divisions reasonably well
        state_chunks = np.array_split(states_backtest, num_chunks)

        chunk_distributions = {}
        all_states = np.unique(states_backtest) # Get all possible state values

        # Calculate distribution for each chunk
        for i, chunk in enumerate(state_chunks):
            if len(chunk) == 0: continue # Skip empty chunks if split resulted in one
            dist = pd.Series(chunk).value_counts(normalize=True) * 100
            # Reindex to include all possible states, filling missing with 0
            chunk_distributions[f'Chunk {i+1} %'] = dist.reindex(all_states, fill_value=0)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(chunk_distributions).sort_index()
        print("\nState Distributions per Backtest Chunk:")
        print(comparison_df)

        # Calculate max difference between all pairs of chunks
        max_diff = 0.0
        if num_chunks >= 2:
            for (col1, col2) in combinations(comparison_df.columns, 2):
                chunk_pair_diff = (comparison_df[col1] - comparison_df[col2]).abs().max()
                max_diff = max(max_diff, chunk_pair_diff)

        print(f"\nMax difference between any two chunks: {max_diff:.2f}%")

        # ANSI escape codes for colors
        # COLOR_GREEN = "\033[92m" # Bright Green
        # COLOR_RED = "\033[91m"   # Bright Red
        # COLOR_RESET = "\033[0m"  # Reset color

        COLOR_GREEN = "" # Bright Green
        COLOR_RED = ""   # Bright Red
        COLOR_RESET = ""  # Reset color

        if max_diff <= stability_threshold:
            print(f"{COLOR_GREEN}State distributions appear reasonably stable within backtest (Max Diff <= {stability_threshold}%).{COLOR_RESET}")
            model_is_stable = True
        else:
            print(f"{COLOR_RED}Warning: State distributions differ significantly within backtest (Max Diff > {stability_threshold}%). Model/features may lack robustness.{COLOR_RESET}")
            model_is_stable = False
        print("--------------------------------------------")

    except Exception as e:
        print(f"Error during within-backtest stability check: {e}")
        max_diff = np.inf
        model_is_stable = False
        comparison_df = None

    return comparison_df, max_diff, model_is_stable

def prepare_feature_arrays(df_features_backtest, df_features_forwardtest, feature_list):
    """
    Selects features, converts to NumPy arrays, and handles NaNs/Infs.

    Args:
        df_features_backtest (pd.DataFrame): Engineered features for backtest.
        df_features_forwardtest (pd.DataFrame): Engineered features for forward test.
        feature_list (list): List of selected feature names to use.

    Returns:
        tuple: (X_backtest, X_forwardtest) NumPy arrays, or (None, None) if errors occur.
    """
    print("\n--- Preparing Data for Model ---")
    # Ensure selected features exist in both dataframes
    if not all(f in df_features_backtest.columns for f in feature_list):
        print("Error: Selected features missing from backtest feature dataframe.")
        return None, None
    if not all(f in df_features_forwardtest.columns for f in feature_list):
        print("Error: Selected features missing from forward test feature dataframe.")
        return None, None

    X_backtest = df_features_backtest[feature_list].values
    X_forwardtest = df_features_forwardtest[feature_list].values

    # Handle NaNs/Infs that might remain or be introduced
    if np.any(np.isnan(X_backtest)) or np.any(np.isinf(X_backtest)):
        print("Warning: Handling NaNs/Infs in backtest feature array.")
        X_backtest = np.nan_to_num(X_backtest)
    if np.any(np.isnan(X_forwardtest)) or np.any(np.isinf(X_forwardtest)):
        print("Warning: Handling NaNs/Infs in forward test feature array.")
        X_forwardtest = np.nan_to_num(X_forwardtest)

    if len(X_backtest) == 0:
        print("Error: No backtest data after NaN handling.")
        return None, None
    if len(X_forwardtest) == 0:
        print("Error: No forward test data after NaN handling.")
        return None, None

    print("Feature arrays prepared.")
    return X_backtest, X_forwardtest

def save_model_and_scaler(model, scaler, directory='models'):
    """Saves the HMM model and scaler to disk."""
    print(f"\n--- Saving Model and Scaler to '{directory}/' ---")
    try:
        os.makedirs(directory, exist_ok=True)
        joblib.dump(scaler, os.path.join(directory, 'scaler.pkl'))
        joblib.dump(model, os.path.join(directory, 'hmm_model.pkl'))
        print("Saved scaler and best HMM model successfully.")
    except Exception as e:
        print(f"Warning: Could not save model/scaler: {e}")

def analyze_hmm_states(hmm_model, feature_list, df_features_backtest, states_backtest):
    """
    Analyzes and prints characteristics of the HMM states based on backtest data.

    Args:
        hmm_model: The trained hmmlearn model instance.
        feature_list (list): List of features used in the model.
        df_features_backtest (pd.DataFrame): Backtest feature dataframe (without 'state').
        states_backtest (np.ndarray): Predicted states for the backtest period.
    """
    if hmm_model is None or states_backtest is None:
        print("Skipping HMM state analysis due to missing model or states.")
        return

    print("\n--- HMM State Analysis (Best Model - Backtest Period) ---")
    best_n_states = hmm_model.n_components
    print(f"Number of States: {best_n_states}")
    print(f"Features used for analysis: {feature_list}")

    # Print Mean Scaled Features
    print("Mean of Scaled Features for each State:")
    try:
        # Check if means_ attribute exists (it should for GaussianHMM)
        if hasattr(hmm_model, 'means_'):
            for i in range(best_n_states):
                print(f"  State {i}:")
                # Ensure feature_list length matches dimensions of means_
                if len(feature_list) == hmm_model.means_.shape[1]:
                    for feature_name, mean_val in zip(feature_list, hmm_model.means_[i]):
                        print(f"    {feature_name}: {mean_val:.4f}")
                else:
                    print(f"    Error: Feature list length ({len(feature_list)}) mismatch with model means dimension ({hmm_model.means_.shape[1]}).")
        else:
             print("    Model does not have 'means_' attribute for analysis.")
    except Exception as e:
        print(f"  Could not print HMM means: {e}")

    # Print Average Raw Return per State
    # Ensure alignment before adding state column for analysis
    if len(states_backtest) == len(df_features_backtest.index):
        # Use assign to create a temporary DataFrame with states without modifying original
        df_analysis = df_features_backtest.assign(state=states_backtest)
        print("\nAverage Raw Price Return per State (Backtest):")
        try:
            if 'price_return' in df_analysis.columns:
                avg_returns = df_analysis.groupby('state')['price_return'].mean()
                print(avg_returns)
            else:
                print("  'price_return' column not found in features for analysis.")
        except Exception as e:
            print(f"  Could not calculate average return per state: {e}")
    else:
        print("  Skipping average return per state analysis due to length mismatch between states and features index.")
    print("--------------------------\n")

def run_and_evaluate_test_period(df_features, states, config, period_label="Test"):
    """
    Adds states, runs signals, backtest simulation, and calculates performance.

    Args:
        df_features (pd.DataFrame): Feature DataFrame for the period.
        states (np.ndarray): Predicted states for the period.
        config (dict): Configuration dictionary.
        period_label (str): Label for printing ('Backtest' or 'Forward Test').

    Returns:
        tuple: (df_results, performance_summary)
               - df_results (pd.DataFrame): DataFrame with backtest results.
               - performance_summary (dict): Dictionary of performance metrics.
               Returns (None, None) if execution fails.
    """
    print(f"\n--- Running Final {period_label} ---")
    if states is None:
        print(f"Skipping final {period_label} due to state prediction error.")
        return None, None
    if len(states) != len(df_features.index):
        print(f"Skipping final {period_label} due to state/index length mismatch.")
        return None, None

    # Safely add 'state' column
    if 'state' in df_features.columns:
        df_features = df_features.drop(columns=['state']) # Drop if re-running cell
    df_features_with_state = df_features.assign(state=states)

    # Run the pipeline
    signal_map_from_config = config.get('signal_map', {})
    # Optional: Add signal map size warning here too if desired, or rely on backtest warning
    print(f"Using signal map from config: {signal_map_from_config}") # Moved print here

    df_signals = generate_signals(df_features_with_state, signal_map_from_config)
    df_results = run_backtest(df_signals, config['trading_fee_percent'])

    print(f"\n--- FINAL {period_label.upper()} PERFORMANCE ---")
    performance_summary = calculate_performance(df_results, config['trading_fee_percent'], timeframe=config['data_timeframe'])

    return df_results, performance_summary

# --- Plotting Functions ---
def plot_correlation_heatmap(corr_matrix, feature_names, output_path):
    """Generates and saves a heatmap of the feature correlation matrix."""
    if corr_matrix is None or corr_matrix.empty:
        print("Skipping correlation heatmap: No correlation matrix provided.")
        return
    print(f"Generating Correlation Heatmap to: {output_path}")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    try:
        plt.savefig(output_path)
        print("Correlation heatmap saved.")
    except Exception as e:
        print(f"Error saving correlation heatmap: {e}")
    plt.close() # Close the plot to free memory

# --- Plotting Functions (Enhanced Equity Curve) ---
def plot_equity_curves(results_bt, results_ft, output_path, start_value=1000.0):
    """
    Generates and saves equity curves for backtest, forward test,
    and Buy & Hold benchmark, including trade markers.
    """
    if results_bt is None or results_ft is None:
        print("Skipping equity curve plot: Missing results data.")
        return

    # Check for necessary columns
    required_cols = ['strategy_portfolio_value', 'buy_and_hold_portfolio_value', 'trade', 'position']
    if not all(col in results_bt.columns for col in required_cols) or \
       not all(col in results_ft.columns for col in required_cols):
           print(f"Skipping equity curve plot: Missing one or more required columns: {required_cols}")
           return

    print(f"Generating Enhanced Equity Curves to: {output_path}")
    plt.figure(figsize=(14, 7)) # Slightly wider figure

    # --- Plot Strategy Equity ---
    plt.plot(results_bt.index, results_bt['strategy_portfolio_value'], label='Backtest Strategy Equity', color='tab:blue', lw=1.5)
    plt.plot(results_ft.index, results_ft['strategy_portfolio_value'], label='Forward Test Strategy Equity', color='tab:orange', lw=1.5)

    # --- Plot Buy & Hold Equity ---
    plt.plot(results_bt.index, results_bt['buy_and_hold_portfolio_value'], label='Backtest Buy & Hold', color='tab:gray', linestyle='--', lw=1)
    # Plot forward test B&H starting from the BT end value for continuity if desired, or restart from start_value
    # Simple restart from start_value for FT B&H:
    # Recalculate FT B&H relative performance and scale
    ft_asset_return = results_ft['close'].pct_change().fillna(0)
    ft_bh_relative = (1 + ft_asset_return).cumprod()
    ft_bh_value = start_value * ft_bh_relative
    plt.plot(results_ft.index, ft_bh_value, label='Forward Test Buy & Hold', color='tab:pink', linestyle='--', lw=1)


    # --- Plot Trade Markers ---
    # Combine results temporarily for easier marker plotting across periods
    # Ensure indices are compatible before combining if necessary
    all_results = pd.concat([results_bt, results_ft]) # Assumes indices are continuous or comparable

    # Identify trade execution points (where position changed)
    trades = all_results[all_results['trade'] != 0].copy() # Use copy to avoid SettingWithCopyWarning

    # Separate Buys (entering Long, position becomes 1) and Sells (entering Short, position becomes -1)
    # Note: This captures entries. Closing trades (position -> 0) aren't marked separately here.
    buy_entries = trades[trades['position'] == 1] # The row where the position *is* 1 (executed based on prev signal)
    sell_entries = trades[trades['position'] == -1] # The row where the position *is* -1

    # Plot markers using portfolio value at the time of the trade
    if not buy_entries.empty:
        plt.scatter(buy_entries.index, buy_entries['strategy_portfolio_value'],
                    label='Buy Entry', marker='^', color='green', s=50, alpha=0.7, zorder=5) # zorder puts markers on top
    if not sell_entries.empty:
        plt.scatter(sell_entries.index, sell_entries['strategy_portfolio_value'],
                    label='Sell Entry', marker='v', color='red', s=50, alpha=0.7, zorder=5)

    # --- Formatting ---
    plt.title(f'Strategy Equity Curve vs Buy & Hold (Start Value: {start_value})')
    plt.xlabel('Date')
    plt.ylabel(f'Portfolio Value (Starting from {start_value})')
    plt.yscale('log') # Use log scale if values vary widely, optional
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5) # Grid for both major and minor ticks if log scale
    plt.tight_layout()

    try:
        plt.savefig(output_path)
        print("Enhanced equity curve plot saved.")
    except Exception as e:
        print(f"Error saving enhanced equity curve plot: {e}")
    plt.close() # Close the plot to free memory

# --- Need to modify save_run_summary function signature ---
def save_run_summary(config, performance_bt, performance_ft, stability_results, output_dir,
                      perm_summary_bt=None, perm_summary_ft=None): # Renamed optional args
    """Saves config and performance summaries to a YAML file in the output dir."""
    print(f"\n--- Saving Run Summary to '{output_dir}/' ---")
    summary_data = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'configuration': config,
        'performance_backtest': performance_bt,
        'performance_forwardtest': performance_ft,
        'state_stability': stability_results
    }
    if perm_summary_bt:
        summary_data['permutation_test_backtest'] = perm_summary_bt
    if perm_summary_ft:
        summary_data['permutation_test_forwardtest'] = perm_summary_ft

    summary_path = os.path.join(output_dir, 'run_summary.yaml')
    # ... (rest of saving logic: write YAML, copy config) ...
    try:
        with open(summary_path, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False, sort_keys=False)
        print(f"Run summary saved to {summary_path}")
        config_source_path = config.get('__source_path__', 'config.yaml')
        if os.path.exists(config_source_path):
             shutil.copy2(config_source_path, os.path.join(output_dir, 'config_used.yaml'))
             print(f"Copied original config '{config_source_path}' to output directory.")
        else:
             print(f"Warning: Could not find original config at '{config_source_path}' to copy.")
    except Exception as e:
        print(f"Error saving run summary: {e}")

# Helper function to create output directory
def create_output_directory(base_dir="outputs"):
    """Creates a timestamped directory for the current run's outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(base_dir, f"run_{timestamp}")
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        print(f"Created output directory: {run_output_dir}")
        return run_output_dir
    except Exception as e:
        print(f"Error creating output directory '{run_output_dir}': {e}")
        return None

# --- New Plotting Function for PnL vs Price ---
def plot_pnl_vs_price(results_bt, results_ft, output_path):
    """
    Generates and saves a plot showing Cumulative PnL vs Close Price
    for backtest and forward test periods.
    """
    if results_bt is None or results_ft is None:
        print("Skipping PnL vs Price plot: Missing results data.")
        return

    required_cols = ['cumulative_pnl', 'close']
    if not all(col in results_bt.columns for col in required_cols) or \
       not all(col in results_ft.columns for col in required_cols):
           print(f"Skipping PnL vs Price plot: Missing one or more required columns: {required_cols}")
           return

    print(f"Generating Cumulative PnL vs Price plot to: {output_path}")

    # Create figure and primary axes for PnL
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Backtest PnL on ax1
    color_bt_pnl = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative PnL', color=color_bt_pnl)
    ax1.plot(results_bt.index, results_bt['cumulative_pnl'], color=color_bt_pnl, label='Backtest Cum. PnL', lw=1.5)
    ax1.tick_params(axis='y', labelcolor=color_bt_pnl)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7) # Grid for PnL axis

    # Plot Forward Test PnL on ax1
    color_ft_pnl = 'tab:orange'
    # Make sure forward test PnL continues from backtest end PnL for visual continuity
    bt_end_pnl = results_bt['cumulative_pnl'].iloc[-1] if not results_bt.empty else 0
    ft_pnl_adjusted = results_ft['cumulative_pnl'] - (results_ft['cumulative_pnl'].iloc[0] if not results_ft.empty else 0) + bt_end_pnl
    ax1.plot(results_ft.index, ft_pnl_adjusted, color=color_ft_pnl, label='Forward Test Cum. PnL', lw=1.5)
    # ax1 limits might need adjustment based on PnL range


    # Create secondary axes for Price, sharing the x-axis
    ax2 = ax1.twinx()
    color_price = 'tab:red'
    ax2.set_ylabel('Close Price', color=color_price)
    ax2.plot(results_bt.index, results_bt['close'], color=color_price, label='Backtest Close Price', linestyle=':', lw=1, alpha=0.8)
    ax2.plot(results_ft.index, results_ft['close'], color='tab:pink', label='Forward Test Close Price', linestyle=':', lw=1, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_price)
    # ax2.set_yscale('log') # Optional: Log scale for price if needed

    # --- Add Trade Markers (Optional, on PnL curve) ---
    all_results = pd.concat([results_bt, results_ft]) # Combine for easier plotting
     # Adjust forward test PnL in the combined df for plotting markers correctly
    if not results_ft.empty and not results_bt.empty:
         all_results.loc[results_ft.index, 'cumulative_pnl'] = ft_pnl_adjusted

    trades = all_results[all_results['trade'] != 0].copy()
    buy_entries = trades[trades['position'] == 1]
    sell_entries = trades[trades['position'] == -1]

    if not buy_entries.empty:
        ax1.scatter(buy_entries.index, buy_entries['cumulative_pnl'],
                    label='Buy Entry', marker='^', color='green', s=50, alpha=0.7, zorder=5)
    if not sell_entries.empty:
        ax1.scatter(sell_entries.index, sell_entries['cumulative_pnl'],
                    label='Sell Entry', marker='v', color='red', s=50, alpha=0.7, zorder=5)
    # --- End Trade Markers ---


    # --- Final Formatting ---
    fig.suptitle('Cumulative PnL vs Close Price (Backtest & Forward Test)') # Use suptitle for overall title
    ax1.set_title('Strategy Performance Analysis', fontsize=10) # Subtitle if needed
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Add scatter labels if they exist
    scatter_handles = [h for h in ax1.collections] # Get scatter handles if any were plotted
    scatter_labels = [h.get_label() for h in scatter_handles]
    ax2.legend(lines + lines2 + scatter_handles, labels + labels2 + scatter_labels, loc='upper left')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap

    try:
        plt.savefig(output_path)
        print("Cumulative PnL vs Price plot saved.")
    except Exception as e:
        print(f"Error saving Cumulative PnL vs Price plot: {e}")
    plt.close(fig) # Close the figure

def run_permutation_test(df_signals_original, config, period_label="Permutation"):
    """
    Runs multiple backtests with randomly shuffled signals to generate a
    null distribution of performance metrics.

    Args:
        df_signals_original (pd.DataFrame): DataFrame containing 'close', 'signal', etc.
                                           from the *actual* strategy run.
        config (dict): Configuration dictionary.
        period_label (str): Label for printing output (e.g., "Backtest Permutation").

    Returns:
        dict: Summary statistics of the permutation test results (mean, std, percentiles, p-value),
              or None if the test is disabled or fails.
    """
    num_runs = config['permutation_test_runs']
    if num_runs <= 0:
        print(f"\n--- Permutation Test ({period_label}) Disabled (runs <= 0) ---")
        return None

    print(f"\n--- Running Permutation Test ({period_label}, {num_runs} runs) ---")
    if df_signals_original is None or 'signal' not in df_signals_original.columns:
        print(f"Cannot run permutation test: Invalid input DataFrame.")
        return None

    original_signals = df_signals_original['signal'].values
    shuffled_results_list = [] # To store key metrics from each run

    for i in range(num_runs):
        if (i + 1) % 10 == 0: # Print progress indicator
             print(f"  Running permutation {i+1}/{num_runs}...")

        df_shuffled = df_signals_original.copy()
        # Shuffle the signals randomly for this iteration
        df_shuffled['signal'] = np.random.permutation(original_signals)

        # Run backtest and calculate performance on shuffled signals
        results_shuffled = run_backtest(df_shuffled, config['trading_fee_percent'], verbose=False)
        if results_shuffled is None:
            print(f"Warning: Backtest failed for permutation {i+1}. Skipping.")
            continue

        # Use a minimal version of calculate_performance or extract needed metrics
        # to avoid excessive printing inside the loop
        net_returns_shuffled = results_shuffled['strategy_return_net'].fillna(0)

        # Calculate Sharpe (Example - add others if needed)
        if isinstance(results_shuffled.index, pd.DatetimeIndex) and len(results_shuffled.index) > 1:
             time_diff = results_shuffled.index.to_series().diff().median()
             periods_per_year = pd.Timedelta(days=365) / time_diff if time_diff and time_diff.total_seconds() > 0 else 252
        else: periods_per_year = 252
        mean_ret = net_returns_shuffled.mean()
        std_dev = net_returns_shuffled.std()
        sharpe = (mean_ret * periods_per_year) / (std_dev * np.sqrt(periods_per_year)) if std_dev != 0 and periods_per_year > 0 else 0

        # Store relevant metrics
        shuffled_results_list.append({'Sharpe': sharpe}) # Add 'Total Return %', etc. if needed

    if not shuffled_results_list:
        print("Permutation test failed: No successful runs.")
        return None

    # Process the collected results
    df_shuffled_summary = pd.DataFrame(shuffled_results_list)
    summary_stats = {
        'num_runs': num_runs,
        'mean_sharpe': df_shuffled_summary['Sharpe'].mean(),
        'median_sharpe': df_shuffled_summary['Sharpe'].median(),
        'std_dev_sharpe': df_shuffled_summary['Sharpe'].std(),
        'sharpe_percentiles': {
            '5th': df_shuffled_summary['Sharpe'].quantile(0.05),
            '25th': df_shuffled_summary['Sharpe'].quantile(0.25),
            '75th': df_shuffled_summary['Sharpe'].quantile(0.75),
            '95th': df_shuffled_summary['Sharpe'].quantile(0.95)
        }
        # Add stats for other metrics if collected
    }

    print(f"--- Permutation Test ({period_label}) Summary ---")
    print(f"  Mean Shuffled Sharpe: {summary_stats['mean_sharpe']:.4f}")
    print(f"  Median Shuffled Sharpe: {summary_stats['median_sharpe']:.4f}")
    print(f"  Std Dev Shuffled Sharpe: {summary_stats['std_dev_sharpe']:.4f}")
    print(f"  95th Percentile Shuffled Sharpe: {summary_stats['sharpe_percentiles']['95th']:.4f}")
    print("---------------------------------------------")

    # Optional: Save full distribution for analysis
    # perm_dist_path = os.path.join(os.path.dirname(config.get('__source_path__', '.')), # Save near config or in output dir
    #                              f'{period_label.lower()}_permutation_sharpe_dist.csv')
    # df_shuffled_summary.to_csv(perm_dist_path, index=False)
    # print(f"Saved full permutation Sharpe distribution to {perm_dist_path}")


    return summary_stats

# Helper function to save data artifacts safely
def save_data_artifact(data, filename, directory):
    """Saves data using joblib, creating the directory if needed."""
    if data is None:
        print(f"Warning: Data is None, skipping saving {filename}")
        return
    try:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)
        joblib.dump(data, filepath)
        print(f"Saved data artifact: {filepath}")
    except Exception as e:
        print(f"Warning: Could not save data artifact {filename}: {e}")

# --- Main Execution (Including Plotting and Saving) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HMM Trading Strategy Backtest.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    # --- Create Output Directory for this Run ---
    run_output_dir = create_output_directory(base_dir="outputs")
    if not run_output_dir: exit("Error: Could not create output directory.")

    # --- Configuration ---
    config_path = args.config
    config = load_config(config_path)
    if not config: exit(1)
    config['__source_path__'] = config_path # Store path for later copying

    # --- Data Loading ---
    # ... (keep existing data loading logic) ...
    df_raw_backtest, final_names_bt = load_and_preprocess_data(config, mode='backtest')
    df_raw_forwardtest, final_names_ft = load_and_preprocess_data(config, mode='forwardtest')
    final_column_names = final_names_bt if final_names_bt else {}
    if df_raw_backtest is None or df_raw_backtest.empty: exit("Error: Could not load backtest data.")
    if df_raw_forwardtest is None or df_raw_forwardtest.empty: exit("Error: Could not load forward test data.")

    # --- Generate FULL EDA Heatmap (using raw data and full config list) ---
    # <<<< NEW CALL >>>>
    full_feature_list_for_eda = config.get('features', []) # Get the full list from config
    generate_full_eda_heatmap(
        df_raw_backtest, # Use raw backtest data as base
        final_column_names,
        full_feature_list_for_eda, # Pass the full list
        run_output_dir
    )
    # <<<< END NEW CALL >>>>

    # --- Feature Engineering ---
    # ... (keep existing feature engineering logic) ...
    feature_list_from_config = config.get('features', [])
    if not feature_list_from_config: exit("Error: 'features' list in config is empty.")
    print("\n--- Engineering Features ---")
    df_features_backtest = engineer_features(df_raw_backtest, final_column_names, feature_list_from_config)
    df_features_forwardtest = engineer_features(df_raw_forwardtest, final_column_names, feature_list_from_config)
    if df_features_backtest is None or df_features_backtest.empty: exit("Error: Backtest feature engineering failed.")
    if df_features_forwardtest is None or df_features_forwardtest.empty: exit("Error: Forward test feature engineering failed.")


    # --- Feature Selection & Correlation Plot ---
    features_available = [f for f in feature_list_from_config if f in df_features_backtest.columns]
    corr_threshold = config['feature_corr_threshold']
    # ---- MODIFIED CALL ----
    feature_list, corr_matrix, corr_feature_names = select_features_by_correlation(
        df_features_backtest, features_available, corr_threshold
    )
    if not feature_list: exit("Error: Feature selection removed all features.")

    # --- Prepare Feature Arrays ---
    X_backtest, X_forwardtest = prepare_feature_arrays(df_features_backtest, df_features_forwardtest, feature_list)
    if X_backtest is None or X_forwardtest is None: exit("Error: Failed to prepare feature arrays.")


    # --- Scale Data ---
    # ... (keep existing scaling logic) ...
    print("\n--- Scaling Data ---")
    scaler = StandardScaler()
    X_scaled_backtest = scaler.fit_transform(X_backtest)
    X_scaled_forwardtest = scaler.transform(X_forwardtest)
    print("Features scaled.")


    # --- Find Best HMM Model ---
    # ... (keep existing HMM finding logic) ...
    hmm_model, best_n_states, _, _ = find_best_hmm_model(X_scaled_backtest, config, feature_list)
    if hmm_model is None: exit("Error: Failed to find suitable HMM model.")
    print(f"\n--- Best model selected: n_states = {best_n_states} ---")


    # --- Save Model Artifacts ---
    # <<<< CALL NEW FUNCTION >>>>
    artifacts_dir = os.path.join(run_output_dir, 'artifacts') # <<< Define artifacts subdir
    save_model_and_scaler(hmm_model, scaler, directory=artifacts_dir) # <<< Save to subdir


    # --- Predict States ---
    # ... (keep existing state prediction logic) ...
    print("\n--- Predicting States ---")
    states_backtest = predict_states(hmm_model, X_scaled_backtest)
    states_forwardtest = predict_states(hmm_model, X_scaled_forwardtest)

# --- Save States and Features Data --- # <<< NEW SECTION >>>
    print("\n--- Saving States and Feature Data ---")
    save_data_artifact(states_backtest, 'states_backtest.pkl', artifacts_dir)
    save_data_artifact(states_forwardtest, 'states_forwardtest.pkl', artifacts_dir)
    # Save the dataframes used for HMM input (needed for signal gen/backtest)
    save_data_artifact(df_features_backtest, 'df_features_backtest.pkl', artifacts_dir)
    save_data_artifact(df_features_forwardtest, 'df_features_forwardtest.pkl', artifacts_dir)
    # Also save the selected feature list used by the model
    save_data_artifact(feature_list, 'selected_features.pkl', artifacts_dir)

    # --- Within-Backtest State Stability Check --- ## <<<< MODIFIED CALL >>>>
    # Pass states, the corresponding index, and the config object
    stability_comparison_df, max_diff, model_is_stable = perform_within_backtest_stability_check(
        states_backtest, df_features_backtest.index, config # Pass index and config
    )
    # Store stability results for summary
    stability_results = {
        'check_type': 'Within Backtest',
        'num_chunks': config['stability_num_backtest_chunks'],
        'max_difference_pct': max_diff if max_diff != np.inf else None, # Store None if error/disabled
        'is_stable': model_is_stable,
        'stability_threshold_pct': config.get('state_stability_threshold', 20),
        'distribution_comparison': stability_comparison_df.to_dict() if stability_comparison_df is not None else None
    }
    # <<<< END MODIFIED CALL >>>>


    # --- Analyze HMM States (Backtest) ---
    # <<<< CALL NEW FUNCTION >>>>
    analyze_hmm_states(hmm_model, feature_list, df_features_backtest, states_backtest)


    # --- Run Backtest Period ---
    # <<<< CALL NEW FUNCTION >>>>
    results_bt, performance_bt = run_and_evaluate_test_period(
        df_features_backtest, states_backtest, config, period_label="Backtest"
    )
    # Handle potential failure
    if performance_bt is None: performance_bt = {"Error": "Backtest execution failed"}


    # --- Run Forward Test Period ---
    # <<<< CALL NEW FUNCTION >>>>
    results_ft, performance_ft = run_and_evaluate_test_period(
        df_features_forwardtest, states_forwardtest, config, period_label="Forward Test"
    )
    # Handle potential failure
    if performance_ft is None: performance_ft = {"Error": "Forward test execution failed"}

    # --- Run PERMUTATION Tests (Controlled by Config) ---
    # We need the dataframes containing the original 'signal' column
    # Let's assume run_and_evaluate_test_period returns df_results which contains signals
    perm_summary_bt = None
    perm_summary_ft = None
    if results_bt is not None:
         perm_summary_bt = run_permutation_test(results_bt, config, period_label="Backtest")
    if results_ft is not None:
         perm_summary_ft = run_permutation_test(results_ft, config, period_label="Forward Test")
    # ---- End Permutation ----

    # --- Plot Cumulative PnL vs Price --- ## <<<< MODIFIED CALL >>>>
    pnl_plot_path = os.path.join(run_output_dir, 'cumulative_pnl_vs_price.png')
    plot_pnl_vs_price(results_bt, results_ft, pnl_plot_path)

    # --- Plot Equity Curves ---
    # <<<< CALL NEW FUNCTION >>>>
    equity_curve_path = os.path.join(run_output_dir, 'equity_curve.png')
    # Pass results_bt and results_ft which are returned by run_and_evaluate_test_period
    plot_equity_curves(results_bt, results_ft, equity_curve_path)


    # --- Save Run Summary ---
    # <<<< CALL NEW FUNCTION >>>>
    save_run_summary(config, performance_bt, performance_ft, stability_results, run_output_dir,
                      perm_summary_bt, perm_summary_ft)

    print(f"\n--- Strategy execution finished ---")
    print(f"Outputs saved in: {run_output_dir}")
