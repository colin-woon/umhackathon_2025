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

# --- Re-Corrected API Interaction Function ---
def fetch_data_from_api(api_key, base_url, source, path, source_params, start_time_ms, end_time_ms, page_limit=1000): # page_limit is now unused when start/end are primary
    """Fetches data using start_time and end_time, strictly adhering to API parameter combination rules."""
    headers = {"X-API-KEY": api_key}
    endpoint_url = f"{base_url}/{source}/{path}"

    # Base query params only include source-specific ones now
    base_query_params = {}
    if source_params:
        base_query_params.update(source_params)

    print(f"Fetching: {source}/{path} with params {source_params} from {datetime.fromtimestamp(start_time_ms/1000)} to {datetime.fromtimestamp(end_time_ms/1000)}...")

    all_data = []
    current_start_time = start_time_ms

    while current_start_time < end_time_ms:
        # --- FIX #2: Create query params for page request using ONLY start_time, end_time, and source_params ---
        page_query_params = base_query_params.copy() # Start with source-specific params
        page_query_params["start_time"] = current_start_time # Update start time for pagination
        page_query_params["end_time"] = end_time_ms   # Keep original end time

        # --- REMOVED THE LINE ADDING 'limit' HERE ---
        # page_query_params["limit"] = min(page_limit, 100000) # REMOVED!

        # Display the URL being requested (now without the explicit limit)
        request_url_display = endpoint_url + "?" + urlencode(page_query_params)
        print(f"  Paging Request URL (approx): {request_url_display}")

        try:
            # Pass the corrected params (start_time, end_time, source_params only)
            response = requests.get(endpoint_url, headers=headers, params=page_query_params)
            response.raise_for_status()
            data = response.json()

            if not data or 'data' not in data or not data['data']:
                print(f"  No more data received for {source}/{path} starting {datetime.fromtimestamp(current_start_time/1000)}. Stopping fetch.")
                break

            page_data = data.get('data', [])
            all_data.extend(page_data)
            last_record_time_ms = page_data[-1].get('start_time') if page_data and 'start_time' in page_data[-1] else None

            print(f"  Fetched {len(page_data)} records ending approx {datetime.fromtimestamp(last_record_time_ms/1000) if last_record_time_ms else 'N/A'}")

            if not page_data or last_record_time_ms is None:
                 print("  Stopping pagination due to empty page or missing timestamp.")
                 break

            # --- Pagination Logic Check ---
            # If the API returns data right up to end_time without needing a limit,
            # the last record might be >= end_time, or the loop condition will handle it.
            # If the API *does* have an internal page size limit even when 'limit' is not sent,
            # this loop *should* still work by advancing start_time.
            # We might need to add a check if len(page_data) == 0 or if last_record_time stops advancing.

            # Advance start time for the next iteration
            current_start_time = last_record_time_ms + 1

            time.sleep(0.2) # Rate Limiting

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {source}/{path}: {e}")
            if e.response is not None:
                print(f"  Response Status Code: {e.response.status_code}")
                try:
                    # Try printing error again - crucial for seeing if it changes
                    print(f"  Response Body: {e.response.json()}")
                except requests.exceptions.JSONDecodeError:
                    print(f"  Response Body (non-JSON): {e.response.text}")
            break # Exit loop on error
        except Exception as e:
            print(f"  An unexpected error occurred: {e}")
            break # Exit loop on other errors

    if not all_data:
      print(f"  Warning: No data fetched overall for {source}/{path}")
      return None

    df = pd.DataFrame(all_data)
    if 'start_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
        df = df.loc[~df.index.duplicated(keep='first')]
        df = df.set_index('timestamp').sort_index()
    else:
        print(f"  Warning: 'start_time' column not found for {source}/{path}. Cannot set time index.")

    return df

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

    output_dir = config.get('data_directory', 'data')
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
            task['start'], task['end']
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
