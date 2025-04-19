import requests
import pandas as pd
import os
import time
import argparse
from datetime import datetime
import yaml # requires PyYAML
import re
from urllib.parse import urlencode # Needed for robust param handling if manual construction was used
from dotenv import load_dotenv
import os

# Load environment variables from .env.local
load_dotenv(dotenv_path=".env.local")

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

# --- API Interaction ---
def make_safe_filename(source, path, suffix):
    """Creates a safe filename from source and path."""
    # Replace special characters in path with underscores
    safe_path = re.sub(r'[/:?=&]', '_', path)
    # Remove consecutive underscores
    safe_path = re.sub(r'_+', '_', safe_path).strip('_')
    return f"{source}_{safe_path}{suffix}.csv"

# --- Re-Re-Corrected API Interaction Function ---
def fetch_data_from_api(api_key, base_url, source, path, source_params, start_time_ms, end_time_ms, page_limit=1000):
    """Fetches data using start_time and limit, stopping based on end_time, with robust post-processing."""
    headers = {"X-API-KEY": api_key}
    endpoint_url = f"{base_url}/{source}/{path}"

    base_query_params = {}
    if source_params:
        base_query_params.update(source_params)

    print(f"Fetching: {source}/{path} with params {source_params} starting from {datetime.fromtimestamp(start_time_ms/1000)} until approx {datetime.fromtimestamp(end_time_ms/1000)}...")

    all_data = []
    current_start_time = start_time_ms

    while current_start_time < end_time_ms:
        page_query_params = base_query_params.copy()
        page_query_params["start_time"] = current_start_time
        page_query_params["limit"] = min(page_limit, 100000) # Always include limit

        request_url_display = endpoint_url + "?" + urlencode(page_query_params)
        print(f"  Paging Request URL (approx): {request_url_display}")

        try:
            response = requests.get(endpoint_url, headers=headers, params=page_query_params)
            response.raise_for_status()
            data = response.json()

            if not data or 'data' not in data or not data['data']:
                print(f"  No more data received for {source}/{path} starting {datetime.fromtimestamp(current_start_time/1000)}. Stopping fetch.")
                break

            page_data = data.get('data', [])
            # Filter page data immediately to prevent adding out-of-range data
            page_data = [d for d in page_data if 'start_time' in d and d['start_time'] < end_time_ms]

            if not page_data: # Check if filtering removed all data
                print(f"  All data on page was >= end_time or invalid. Stopping fetch.")
                break

            all_data.extend(page_data)
            last_record_time_ms = page_data[-1].get('start_time')

            print(f"  Fetched {len(page_data)} records ending approx {datetime.fromtimestamp(last_record_time_ms/1000) if last_record_time_ms else 'N/A'}")

            if last_record_time_ms is None:
                 print("  Stopping pagination due to missing timestamp in last record.")
                 break

            # Check if the API returned fewer records than the limit asked for.
            if len(page_data) < page_query_params["limit"]:
                 print("  Received fewer records than limit, assuming end of available data for this range.")
                 # No need to break here explicitly, the outer loop condition current_start_time < end_time_ms handles it.

            # Advance start time for the next iteration
            current_start_time = last_record_time_ms + 1

            time.sleep(0.2) # Rate Limiting

        # --- Error Handling Block (Keep as before) ---
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {source}/{path}: {e}")
            if e.response is not None:
                print(f"  Response Status Code: {e.response.status_code}")
                try:
                    print(f"  Response Body: {e.response.json()}")
                except requests.exceptions.JSONDecodeError:
                    print(f"  Response Body (non-JSON): {e.response.text}")
            all_data = []
            print(f"  Discarding potentially partial data for {source}/{path} due to error.")
            break
        except Exception as e:
            print(f"  An unexpected error occurred during pagination: {e}")
            all_data = []
            print(f"  Discarding potentially partial data for {source}/{path} due to error.")
            break
    # --- End of While Loop ---

    # --- Post-Processing and Filtering ---
    if not all_data: # Check if any data was successfully collected
         print(f"  Warning: No data collected in loop for {source}/{path}")
         return None

    try:
        df = pd.DataFrame(all_data)
        if 'start_time' not in df.columns:
             print(f"  Error: 'start_time' column missing after collecting data for {source}/{path}.")
             return None
        if df.empty:
             print(f"  Warning: DataFrame is empty after creation for {source}/{path}")
             return None

        # Convert to datetime and set index
        df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
        df = df.set_index('timestamp')

        # --- START ADDED RENAMING LOGIC ---
        # Standardize known differing column names from specific endpoints before further processing
        rename_map = {}
        if source == 'cryptoquant':
            if path == 'btc/exchange-flows/inflow' and 'inflow_mean' in df.columns:
                rename_map['inflow_mean'] = 'value'
                print(f"  Standardizing column name: 'inflow_mean' -> 'value'")
            elif path == 'btc/market-data/funding-rates' and 'funding_rates' in df.columns:
                # Prefer renaming to 'fundingRate' as it's more specific, but 'value' is also used in config
                rename_map['funding_rates'] = 'fundingRate'
                print(f"  Standardizing column name: 'funding_rates' -> 'fundingRate'")
            # Add more elif blocks here if other endpoints have known inconsistencies

        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        # --- END ADDED RENAMING LOGIC ---

        # Remove duplicates based on index (important after pagination)
        df = df.loc[~df.index.duplicated(keep='first')]

        # Sort index
        df = df.sort_index()

        # Check index type BEFORE final filtering
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"  Error: Index is not DatetimeIndex after processing for {source}/{path}. Index type: {type(df.index)}")
            return None

        # ---- Perform Final Time Range Filtering ----
        # Ensure start/end times are Timestamps for comparison
        start_ts = pd.to_datetime(start_time_ms, unit='ms')
        end_ts = pd.to_datetime(end_time_ms, unit='ms')

        # Apply the filter - This is where the TypeError previously occurred
        df_filtered = df[(df.index >= start_ts) & (df.index < end_ts)]

        # Check if dataframe is empty after filtering
        if df_filtered.empty:
            print(f"  Warning: DataFrame is empty after final filtering for time range for {source}/{path}")
            return None

        print(f"  Successfully processed {len(df_filtered)} records for {source}/{path}")
        return df_filtered

    except TypeError as te:
        # Catch the specific TypeError if it still happens
        print(f"  TypeError during post-processing/filtering for {source}/{path}: {te}")
        print(f"  DataFrame info before error:\n{df.info() if 'df' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"  Unexpected error during post-processing for {source}/{path}: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from Cybotrade API.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)

    api_key = os.getenv("CYBOTRADE_API_KEY")
    if not api_key:
        print("Error: CYBOTRADE_API_KEY environment variable not set.")
        exit(1)

    output_dir = config['data_directory']
    os.makedirs(output_dir, exist_ok=True)
    print(f"Data will be saved in: {output_dir}")

    base_url = "https://api.datasource.cybotrade.rs"

    download_tasks = []
    metrics_config = config.get('metrics_to_download', {})

    # Backtest Period
    bt_start = config['backtest_start_time_ms']
    bt_end = config['backtest_end_time_ms']
    for source, metrics in metrics_config.items():
        for metric_details in metrics:
             # Handle potential candle source override/structure
            actual_source = metric_details.get('source_override', source) # Use override if present
            path = metric_details['path']
            params = metric_details.get('params', {}) # Get params dict
            download_tasks.append({
                "source": actual_source, "path": path, "params": params,
                "start": bt_start, "end": bt_end, "suffix": "_backtest"
            })

    # Forward Test Period
    ft_start = config['forwardtest_start_time_ms']
    ft_end = config['forwardtest_end_time_ms']
    for source, metrics in metrics_config.items():
        for metric_details in metrics:
            actual_source = metric_details.get('source_override', source)
            path = metric_details['path']
            params = metric_details.get('params', {})
            download_tasks.append({
                "source": actual_source, "path": path, "params": params,
                "start": ft_start, "end": ft_end, "suffix": "_forwardtest"
            })

    total_tasks = len(download_tasks)
    print(f"\nStarting {total_tasks} download tasks...")

    for i, task in enumerate(download_tasks):
        print(f"\n--- Task {i+1}/{total_tasks} ---")

        df_data = fetch_data_from_api(
            api_key, base_url, task['source'], task['path'], task['params'],
            task['start'], task['end'], config['page_limit']
        )

        if df_data is not None and not df_data.empty:
            filename = make_safe_filename(task['source'], task['path'], task['suffix'])
            filepath = os.path.join(output_dir, filename)
            try:
                df_data.to_csv(filepath)
                print(f"  Successfully saved data to {filepath}")
            except Exception as e:
                print(f"  Error saving data to {filepath}: {e}")
        else:
            print(f"  Skipping save for {task['source']}/{task['path']} due to empty or error state.")

    print("\n--- Data download process finished. ---")
