import json
import os

notebook_path = '/Users/aryan/datathon/resili-net/notebooks/train_gnn.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

new_source = [
    "# Cell 2: Load Historical Data\n",
    "\n",
    "print(\"=\" * 60)\n",
    "print(\"LOADING HISTORICAL DATA\")\n",
    "print(\"=\" * 60)\n",
    "import os\n",
    "\n",
    "# Define date range with buffer for feature engineering\n",
    "TARGET_START = pd.Timestamp('2012-11-08')\n",
    "END_DATE = pd.Timestamp('2022-08-08')\n",
    "BUFFER_DAYS = 90  # Buffer for rolling windows (volatility, etc.)\n",
    "\n",
    "# Fetch extra data prior to target start\n",
    "fetch_start = TARGET_START - pd.Timedelta(days=BUFFER_DAYS)\n",
    "print(f\"Fetching data from {fetch_start.date()} to {END_DATE.date()} (Buffer: {BUFFER_DAYS} days)\")\n",
    "\n",
    "# Force cache clear if needed to ensure we get early data\n",
    "# Note: In production, handle cache invalidation more gracefully\n",
    "cache_path = '../data/historical_market_data_INDIA.pkl'\n",
    "if os.path.exists(cache_path):\n",
    "    # check if cache covers the needed start\n",
    "    try:\n",
    "        df_check = pd.read_pickle(cache_path)\n",
    "        if 'Date' in df_check.columns:\n",
    "            min_date = pd.to_datetime(df_check['Date']).min()\n",
    "        else:\n",
    "            min_date = df_check.index.min()\n",
    "        \n",
    "        if min_date > fetch_start:\n",
    "            print(f\"Cache starts at {min_date.date()}, but we need {fetch_start.date()}. Invalidating cache.\")\n",
    "            os.remove(cache_path)\n",
    "    except:\n",
    "        pass\n",
    "\n",
    "raw_data = fetch_and_cache_market_data(\n",
    "    start_date=str(fetch_start.date()),\n",
    "    end_date=str(END_DATE.date())\n",
    ")\n",
    "\n",
    "processed_data = engineer_features(raw_data)\n",
    "\n",
    "# Filter back to the target range for training\n",
    "processed_data = processed_data[processed_data.index >= TARGET_START]\n",
    "\n",
    "print(f\"\\n📊 Data Summary (Training Range):\")\n",
    "print(f\"   Shape: {processed_data.shape}\")\n",
    "print(f\"   Date range: {processed_data.index.min()} to {processed_data.index.max()}\")\n",
    "print(f\"   Tickers: {processed_data['Ticker'].nunique()}\")\n",
    "print(f\"   Columns: {processed_data.columns.tolist()}\")"
]

# Find the cell to replace
found = False
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "fetch_and_cache_market_data" in source and "engineer_features(raw_data)" in source:
            cell['source'] = new_source
            found = True
            break

if found:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print("Successfully updated notebook.")
else:
    print("Could not find the target cell to update.")
