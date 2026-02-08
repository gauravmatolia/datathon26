# src/data_loader.py

"""
This module is responsible for fetching, caching, and processing all financial data
needed for the simulation using the yfinance API.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
from typing import Dict, Any
from src.config import (
    ALL_TICKERS, GLOBAL_INDEX_TICKERS, DATA_CACHE_DIR, 
    HISTORICAL_DATA_FILE, ROLLING_WINDOW_DAYS, ANNUAL_TRADING_DAYS
)

def fetch_and_cache_market_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical market data (OHLCV) for all specified tickers and caches it.
    Filters data to match requesting date range.
    """
    if os.path.exists(HISTORICAL_DATA_FILE):
        print(f"Loading historical data from cache: {HISTORICAL_DATA_FILE}")
        cached_data = pd.read_pickle(HISTORICAL_DATA_FILE)
        
        # Filter by requested date range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        if 'Date' in cached_data.columns:
            cached_data = cached_data[(cached_data['Date'] >= start_ts) & (cached_data['Date'] <= end_ts)]
        else:
            cached_data = cached_data[(cached_data.index >= start_ts) & (cached_data.index <= end_ts)]
        
        return cached_data
    
    print(f"Cache not found. Fetching historical data for {len(ALL_TICKERS)} tickers...")
    
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    
    data = yf.download(ALL_TICKERS, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

    if data.empty:
        raise ValueError("No data downloaded. Check your internet connection or ticker list.")

    try:
        data = data.stack(level=0, future_stack=True)
    except TypeError:
        data = data.stack(level=0)
        
    data = data.rename_axis(['Date', 'Ticker']).reset_index(level=1)
    
    data.to_pickle(HISTORICAL_DATA_FILE)
    print(f"Historical data cached successfully at {HISTORICAL_DATA_FILE}")
    
    return data

def engineer_features(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers all required features for the simulation.
    """
    print("Starting feature engineering...")
    
    # Ensure Date is a column for consistent processing
    if 'Date' not in historical_data.columns:
        data = historical_data.reset_index()
    else:
        data = historical_data.copy()
    
    data = data.sort_values(by=['Ticker', 'Date'])
    
    # --- 1. Basic Features ---
    # fill_method=None is required for pandas 2.0+
    data['daily_return'] = data.groupby('Ticker')['Close'].pct_change(fill_method=None)
    
    # --- 2. Visible Risk Features ---
    data['realized_volatility'] = data.groupby('Ticker')['daily_return'].transform(
        lambda x: x.rolling(window=ROLLING_WINDOW_DAYS).std() * np.sqrt(ANNUAL_TRADING_DAYS)
    )
    rolling_mean = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=ROLLING_WINDOW_DAYS).mean())
    data['price_momentum'] = (data['Close'] - rolling_mean) / rolling_mean
    data['dollar_volume'] = data['Volume'] * data['Close']
    
    # --- 3. Systemic Risk Features (Beta) ---
    # First, create the 'market_return' column by mapping the NIFTY returns to each day
    market_ticker = GLOBAL_INDEX_TICKERS['market_index']
    market_data = data[data['Ticker'] == market_ticker][['Date', 'daily_return']].copy()
    market_data = market_data.rename(columns={'daily_return': 'market_return'})
    data = data.merge(market_data[['Date', 'market_return']], on='Date', how='left')

    # Second, calculate beta for each group explicitly to avoid the ValueError
    beta_list = []
    for ticker, group in data.groupby('Ticker', group_keys=False):
        # Calculate rolling covariance of the stock's return with the market's return
        rolling_cov = group['daily_return'].rolling(window=ROLLING_WINDOW_DAYS).cov(group['market_return'])
        
        # Calculate rolling variance of the market's return
        rolling_var = group['market_return'].rolling(window=ROLLING_WINDOW_DAYS).var()
        
        # Beta is the ratio
        group_beta = rolling_cov / rolling_var
        beta_list.append(group_beta)

    # Concatenate the list of Series into a single Series.
    if beta_list:
        final_beta_series = pd.concat(beta_list)
        data['beta_market'] = final_beta_series.values
    else:
        data['beta_market'] = np.nan

    # --- 4. Shadow Risk Proxy ---
    vix_ticker = GLOBAL_INDEX_TICKERS['market_fear']
    vix_data = data[data['Ticker'] == vix_ticker][['Date', 'Close']].copy()
    vix_data = vix_data.rename(columns={'Close': 'vix'})
    data = data.merge(vix_data[['Date', 'vix']], on='Date', how='left')
    data['vix'] = data['vix'].ffill().fillna(20.0)
    
    data['implied_volatility'] = data['realized_volatility'] + (data['vix'] / 100.0) * 0.5
    data['shadow_risk_gap'] = data['implied_volatility'] - data['realized_volatility']
    
    # Cleanup
    data = data.dropna()
    
    # Set Date as index for downstream usage
    data = data.set_index('Date')
    
    print("Feature engineering complete.")
    return data

def get_market_data_for_timestep(processed_data: pd.DataFrame, timestamp: pd.Timestamp) -> Dict[str, Any]:
    """
    Extracts data for a single point in time.
    """
    date_str = timestamp.strftime('%Y-%m-%d')
    
    df_indexed = processed_data.reset_index().set_index('Date')

    try:
        day_data = df_indexed.loc[date_str]
    except KeyError:
        return {} # Date not in index
        
    if day_data.empty:
        return {}
        
    data_dict = day_data.set_index('Ticker').to_dict('index')
    
    timestep_data = {ticker: data for ticker, data in data_dict.items() if ticker not in GLOBAL_INDEX_TICKERS.values()}
    
    def get_global_val(ticker, field, default):
        try: return data_dict[ticker][field]
        except KeyError: return default

    timestep_data['GLOBAL'] = {
        'vix': get_global_val(GLOBAL_INDEX_TICKERS['market_fear'], 'Close', 20.0),
        'risk_free_rate': get_global_val(GLOBAL_INDEX_TICKERS['risk_free_rate'], 'Close', 4.0) / 100.0
    }
    
    return timestep_data