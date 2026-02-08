#!/usr/bin/env python3
"""
Script to update date ranges in Resili-Net notebooks to use 10-year historical data.
New range: 2012-08-08 to 2022-08-08 (~2500 trading days, well over 2000 required)
"""
import json
import os

# New date range - 10 years
NEW_START_DATE = "2005-08-08"
NEW_END_DATE = "2022-08-08"

# Target start date for training (after feature engineering buffer)
# Buffer of 90 days from NEW_START_DATE means training starts from ~2012-11-06
TARGET_START = "2005-11-08"

def update_test_phase_1():
    """Update test_phase_1.ipynb with new date range."""
    notebook_path = '/Users/aryan/datathon/resili-net/notebooks/test_phase_1.ipynb'
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "start_date = '2022-01-01'" in source_str or "start_date = '2022-06-15'" in source_str:
                new_source = []
                for line in cell['source']:
                    if "start_date = '2022" in line:
                        new_source.append(f"start_date = '{NEW_START_DATE}'\n")
                    elif "end_date = '2023" in line or "end_date = '2022" in line:
                        new_source.append(f"end_date = '{NEW_END_DATE}'\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                updated = True
                break
    
    if updated:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=2)
        print(f"✅ Updated test_phase_1.ipynb: {NEW_START_DATE} to {NEW_END_DATE}")
    else:
        print("⚠️ Could not find date range cell in test_phase_1.ipynb")

def update_test_phase_2():
    """Update test_phase_2.ipynb with new date range."""
    notebook_path = '/Users/aryan/datathon/resili-net/notebooks/test_phase_2.ipynb'
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "fetch_and_cache_market_data" in source_str and "start_date=" in source_str:
                new_source = []
                for line in cell['source']:
                    if "fetch_and_cache_market_data" in line and "start_date=" in line:
                        new_source.append(f"raw_data = fetch_and_cache_market_data(start_date='{NEW_START_DATE}', end_date='{NEW_END_DATE}')\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                updated = True
                break
    
    if updated:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=2)
        print(f"✅ Updated test_phase_2.ipynb: {NEW_START_DATE} to {NEW_END_DATE}")
    else:
        print("⚠️ Could not find date range cell in test_phase_2.ipynb")

def update_train_gnn():
    """Update train_gnn.ipynb with new date range."""
    notebook_path = '/Users/aryan/datathon/resili-net/notebooks/train_gnn.ipynb'
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell['source'])
            if "TARGET_START = pd.Timestamp" in source_str and "END_DATE = pd.Timestamp" in source_str:
                new_source = []
                for line in cell['source']:
                    if "TARGET_START = pd.Timestamp" in line:
                        new_source.append(f"TARGET_START = pd.Timestamp('{TARGET_START}')\n")
                    elif "END_DATE = pd.Timestamp" in line:
                        new_source.append(f"END_DATE = pd.Timestamp('{NEW_END_DATE}')\n")
                    else:
                        new_source.append(line)
                cell['source'] = new_source
                updated = True
                break
    
    if updated:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=2)
        print(f"✅ Updated train_gnn.ipynb: TARGET_START={TARGET_START}, END_DATE={NEW_END_DATE}")
    else:
        print("⚠️ Could not find date range cell in train_gnn.ipynb")

def update_update_notebook_script():
    """Update update_notebook.py script with new date range."""
    script_path = '/Users/aryan/datathon/resili-net/notebooks/update_notebook.py'
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace date constants
    content = content.replace("TARGET_START = pd.Timestamp('2022-06-15')", f"TARGET_START = pd.Timestamp('{TARGET_START}')")
    content = content.replace("END_DATE = pd.Timestamp('2022-08-08')", f"END_DATE = pd.Timestamp('{NEW_END_DATE}')")
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated update_notebook.py: TARGET_START={TARGET_START}, END_DATE={NEW_END_DATE}")

def clear_cache():
    """Delete the cached historical data file."""
    cache_path = '/Users/aryan/datathon/resili-net/data/historical_market_data_INDIA.pkl'
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"✅ Deleted cached data: {cache_path}")
    else:
        print(f"ℹ️ Cache file not found: {cache_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("Updating Resili-Net to use 10-year historical data range")
    print(f"New range: {NEW_START_DATE} to {NEW_END_DATE}")
    print("=" * 60)
    
    update_test_phase_1()
    update_test_phase_2()
    update_train_gnn()
    update_update_notebook_script()
    clear_cache()
    
    print("\n" + "=" * 60)
    print("All updates complete!")
    print("Please restart Jupyter kernels and run test_phase_1.ipynb first.")
    print("=" * 60)
