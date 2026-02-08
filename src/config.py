# src/config.py
"""
Central configuration file for the Resili-Net simulation.
Defines the universe of tickers, constants for calculations, and file paths.
THIS VERSION IS CONFIGURED FOR THE INDIAN MARKET (NSE).
"""

# --- Universe of Tickers (Indian Market) ---

# Banks: A mix of Public Sector Banks (PSBs) and Private Banks.
BANK_TICKERS = [
    'HDFCBANK.NS',   # HDFC Bank
    'ICICIBANK.NS',  # ICICI Bank
    'SBIN.NS',       # State Bank of India
    'KOTAKBANK.NS',  # Kotak Mahindra Bank
    'AXISBANK.NS',   # Axis Bank
    'INDUSINDBK.NS'  # IndusInd Bank
]

# Exchanges: Listed exchange operators in India.
EXCHANGE_TICKERS = [
    'BSE.NS',        # Bombay Stock Exchange Ltd.
    'MCX.NS'         # Multi Commodity Exchange of India Ltd.
]

# Global/Market Indices
GLOBAL_INDEX_TICKERS = {
    # India VIX is the best proxy for market fear.
    'market_fear': '^INDIAVIX',
    
    # REPLACEMENT: ^NSXIID is dead. We use ^TNX (US 10Y Yield) as a reliable
    # proxy for the "global cost of capital" to ensure data availability.
    'risk_free_rate': '^TNX',
    
    # NIFTY 50 is the primary market index for Beta calculations.
    'market_index': '^NSEI'
}

# All tickers to be downloaded
ALL_TICKERS = BANK_TICKERS + EXCHANGE_TICKERS + list(GLOBAL_INDEX_TICKERS.values())

# --- Data Caching Configuration ---
DATA_CACHE_DIR = "data"
HISTORICAL_DATA_FILE = f"{DATA_CACHE_DIR}/historical_market_data_INDIA.pkl"
OPTIONS_DATA_DIR = f"{DATA_CACHE_DIR}/options_INDIA"

# --- Feature Engineering & Graph Constants ---
ROLLING_WINDOW_DAYS = 30
ANNUAL_TRADING_DAYS = 252
CORRELATION_THRESHOLD = 0.6

# --- Phase 2: Simulation Constants ---
VIX_HALT_THRESHOLD = 40.0           # VIX level at which exchanges halt trading
CONTAGION_LOSS_FACTOR = 0.1         # Proportion of defaulted bank's loss applied to neighbors
DEFAULT_INITIAL_CASH = 1_000_000_000  # ₹100 crore initial cash per bank
DEFAULT_MARGIN_RATE = 0.05          # 5% default margin requirement
LOW_CASH_THRESHOLD = 100_000_000    # ₹10 crore - below this, bank may sell assets
SIMULATION_RANDOM_SEED = 42         # For reproducibility