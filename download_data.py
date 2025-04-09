import requests
import pandas as pd
import os
import time
import argparse
from datetime import datetime
import yaml # requires PyYAML
import re
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
def make_safe_filename(topic_string, prefix, suffix):
    """Creates a safe filename from a topic string."""
    # Remove source prefix like 'cryptoquant|'
    cleaned_topic = topic_string.split('|', 1)[-1]
    # Replace special characters with underscores
    safe_name = re.sub(r'[/:?=&]', '_', cleaned_topic)
    # Remove consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name).strip('_')
    return f"{prefix}_{safe_name}{suffix}.csv"

def fetch_data_from_api(api_key, base_url, source, topic, start_time_ms, end_time_ms, limit=1000):
    """Fetches data for a single topic from the cybotrade.rs API."""
    headers = {"X-API-KEY": api_key}
    endpoint_url = f"{base_url}/{source}/{topic}"
    params = {
        "start_time": start_time_ms,
        "end_time": end_time_ms,
        "limit": limit
    }

    print(f"Fetching: {source}|{topic} from {datetime.fromtimestamp(start_time_ms/1000)} to {datetime.fromtimestamp(end_time_ms/1000)}...")

    all_data = []
    current_start_time = start_time_ms

    while current_start_time < end_time_ms:
        params["start_time"] = current_start_time
        params["limit"] = min(limit, 100000) # Adhere to potential API max limit if known

        try:
            response = requests.get(endpoint_url, headers=headers, params=params)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()

            if not data or 'data' not in data or not data['data']:
                print(f"  No more data received for {source}|{topic} starting {datetime.fromtimestamp(current_start_time/1000)}. Stopping fetch for this topic.")
                break # Exit loop if no data is returned

            page_data = data.get('data', [])
            all_data.extend(page_data)
            print(f"  Fetched {len(page_data)} records ending at {datetime.fromtimestamp(page_data[-1]['start_time']/1000) if page_data else 'N/A'}")

            # --- Pagination Logic ---
            # Move to the next page. The API seems to use 'start_time' of the *last* record
            # fetched + 1ms as the start for the next request to avoid overlap.
            # Adjust if API uses a different pagination method (like next_page tokens).
            last_record_time = page_data[-1]['start_time']
            current_start_time = last_record_time + 1 # Move to next millisecond


            # --- Rate Limiting ---
            time.sleep(0.2) # Add a small delay to respect rate limits (adjust as needed)


        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {source}|{topic}: {e}")
            # Decide how to handle errors: break, retry, log, etc.
            # For simplicity, we break here. Add retries if needed.
            break
        except Exception as e:
            print(f"  An unexpected error occurred: {e}")
            break # Exit on other unexpected errors

    if not all_data:
      print(f"  Warning: No data fetched for {source}|{topic}")
      return None

    df = pd.DataFrame(all_data)
    # Convert start_time (assuming it's the primary timestamp)
    if 'start_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['start_time'], unit='ms')
        df = df.set_index('timestamp').sort_index()
    else:
        print(f"  Warning: 'start_time' column not found for {source}|{topic}. Cannot set time index.")

    return df


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data from Cybotrade API.")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)

    # --- Get API Key ---
    api_key = os.getenv("CYBOTRADE_API_KEY")
    if not api_key:
        print("Error: CYBOTRADE_API_KEY environment variable not set.")
        exit(1)

    # --- Prepare Output Directory ---
    output_dir = config.get('data_directory', 'data')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Data will be saved in: {output_dir}")

    # --- API Base URL ---
    base_url = "https://api.datasource.cybotrade.rs" # Hardcoded as per docs

    # --- Define Download Tasks ---
    download_tasks = []
    metrics_config = config.get('metrics_to_download', {})

    # Backtest Period
    bt_start = config['backtest_start_time_ms']
    bt_end = config['backtest_end_time_ms']
    for source, topics in metrics_config.items():
        for topic in topics:
            download_tasks.append({
                "source": source, "topic": topic, "start": bt_start, "end": bt_end, "suffix": "_backtest"
            })

    # Forward Test Period
    ft_start = config['forwardtest_start_time_ms']
    ft_end = config['forwardtest_end_time_ms']
    for source, topics in metrics_config.items():
        for topic in topics:
            download_tasks.append({
                "source": source, "topic": topic, "start": ft_start, "end": ft_end, "suffix": "_forwardtest"
            })

    # --- Execute Downloads ---
    total_tasks = len(download_tasks)
    print(f"\nStarting {total_tasks} download tasks...")

    for i, task in enumerate(download_tasks):
        print(f"\n--- Task {i+1}/{total_tasks} ---")
        source = task['source']
        # Handle candle source slightly differently if needed based on API structure
        topic_str = task['topic']
        if source == 'candles':
             source_for_url = topic_str.split('|', 1)[0] # e.g., bybit-linear
             endpoint_path = topic_str.split('|', 1)[1] # e.g., candle?symbol=...
             source = source_for_url # Adjust source for URL construction
             topic_to_fetch = endpoint_path
        else:
            topic_to_fetch = topic_str # Use the topic directly for other sources


        df_data = fetch_data_from_api(
            api_key, base_url, source, topic_to_fetch,
            task['start'], task['end']
        )

        if df_data is not None and not df_data.empty:
            # Use original topic_str for filename generation
            filename_prefix = source if source != 'candles' else topic_str.split('|', 1)[0]
            filename = make_safe_filename(topic_str, filename_prefix, task['suffix'])
            filepath = os.path.join(output_dir, filename)
            try:
                df_data.to_csv(filepath)
                print(f"  Successfully saved data to {filepath}")
            except Exception as e:
                print(f"  Error saving data to {filepath}: {e}")
        else:
            print(f"  Skipping save for {source}|{topic_str} due to empty or error state.")

    print("\n--- Data download process finished. ---")
