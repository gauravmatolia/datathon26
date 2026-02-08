# src/graph_builder.py

"""
This module is responsible for constructing the dynamic financial network graph
for each timestep of the simulation using NetworkX.
"""

import networkx as nx
import pandas as pd
from typing import Dict, Any

from src.config import BANK_TICKERS, EXCHANGE_TICKERS, CORRELATION_THRESHOLD, ROLLING_WINDOW_DAYS
from src.agents import BankAgent, ExchangeNode # Import the class definitions from Phase 0

def build_graph_for_timestep(timestamp: pd.Timestamp, historical_data: pd.DataFrame, processed_day_data: Dict[str, Any]) -> nx.Graph:
    """
    Constructs the financial network graph for a single timestep.

    Args:
        timestamp (pd.Timestamp): The current date of the simulation.
        historical_data (pd.DataFrame): The raw historical data to calculate correlations.
        processed_day_data (Dict[str, Any]): The feature-engineered data for the current day.

    Returns:
        nx.Graph: A NetworkX graph object representing the state of the financial system.
    """
    G = nx.Graph()
    
    # --- 1. Add Nodes with Features ---
    # Add Bank nodes
    for ticker in BANK_TICKERS:
        if ticker in processed_day_data:
            node_data = processed_day_data[ticker]
            # Here, we can initialize the BankAgent with some defaults.
            # The simulation will manage its cash and portfolio.
            agent = BankAgent(ticker=ticker, initial_cash=1_000_000_000, initial_portfolio={})
            
            G.add_node(ticker, agent=agent, type='bank', **node_data)
    
    # Add Exchange nodes
    for ticker in EXCHANGE_TICKERS:
        if ticker in processed_day_data:
            node_data = processed_day_data[ticker]
            exchange = ExchangeNode(ticker=ticker)
            G.add_node(ticker, agent=exchange, type='exchange', **node_data)

    # --- 2. Add Edges based on Rolling Correlation ---
    # Calculate the correlation matrix for bank returns over the last N days
    start_date = timestamp - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
    
    # Ensure we have a 'Date' column to filter on (reset index if needed)
    if 'Date' not in historical_data.columns:
        hist_data = historical_data.reset_index()
    else:
        hist_data = historical_data
    
    # Filter historical data for the rolling window and pivot to get returns matrix
    window_data = hist_data[
        (hist_data['Date'] >= start_date) & 
        (hist_data['Date'] <= timestamp) &
        (hist_data['Ticker'].isin(BANK_TICKERS))
    ]
    returns_matrix = window_data.pivot(index='Date', columns='Ticker', values='daily_return')
    
    # Handle cases with insufficient data for correlation
    if len(returns_matrix) < 2:
        return G # Return graph with no edges if not enough data
        
    correlation_matrix = returns_matrix.corr()

    # Iterate through the correlation matrix to add edges
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            ticker_i = correlation_matrix.columns[i]
            ticker_j = correlation_matrix.columns[j]
            
            correlation_value = correlation_matrix.iloc[i, j]
            
            # Add an edge if the correlation exceeds the threshold
            if abs(correlation_value) > CORRELATION_THRESHOLD:
                G.add_edge(ticker_i, ticker_j, weight=correlation_value, type='correlation')
                
    return G