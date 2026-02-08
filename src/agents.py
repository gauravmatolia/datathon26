# src/agents.py

"""
This module defines the core actors (agents) in our financial simulation:
- BankAgent: Represents a financial institution (e.g., a bank) that makes strategic decisions.
- ExchangeNode: Represents a trading venue (e.g., NYSE) that facilitates trading and can halt markets.
- CCP_Controller: Represents the Central Counterparty (CCP) that manages systemic risk.
"""

from typing import Dict, Any, List

# Define a type alias for clarity in portfolio definitions
Portfolio = Dict[str, float]  # e.g., {'AAPL': 1000.0, 'GOOG': 500.0}

class BankAgent:
    """
    Represents a single bank in the financial network.
    Each bank has its own state and can perform actions based on market conditions.
    """
    def __init__(self, ticker: str, initial_cash: float, initial_portfolio: Portfolio):
        """
        Initializes a BankAgent.

        Args:
            ticker (str): The stock ticker symbol for this bank (e.g., 'JPM').
            initial_cash (float): The initial amount of available liquid cash.
            initial_portfolio (Portfolio): A dictionary representing the bank's initial asset holdings.
        """
        self.ticker: str = ticker
        self.initial_cash: float = initial_cash  # Store for reset
        self.initial_portfolio: Portfolio = initial_portfolio.copy()  # Store for reset
        self.available_cash: float = initial_cash
        self.portfolio: Portfolio = initial_portfolio.copy()
        
        # These attributes will be populated with real data in Phase 1
        self.market_cap: float = 0.0
        self.leverage_ratio: float = 0.0
        
        # This will be dynamically calculated by the CCP's GNN model
        self.risk_score: float = 0.0
        self.is_defaulted: bool = False
        
        # Phase 2: Track pending trades and margin
        self.pending_trade: Dict[str, Any] = {}
        self.current_margin_requirement: float = 0.05  # Default 5%

    def __repr__(self) -> str:
        status = "DEFAULTED" if self.is_defaulted else "Active"
        return f"BankAgent({self.ticker}, cash=₹{self.available_cash:,.0f}, status={status})"

    def reset(self):
        """Reset the bank to its initial state for a new simulation run."""
        self.available_cash = self.initial_cash
        self.portfolio = self.initial_portfolio.copy()
        self.is_defaulted = False
        self.pending_trade = {}
        self.risk_score = 0.0

    def update_state(self, market_data: Dict[str, Any]):
        """
        Updates the bank's internal state based on new market data.
        This will be called at each step of the simulation.
        
        Args:
            market_data (Dict[str, Any]): A dictionary containing the latest market prices and info for this bank.
        """
        if market_data:
            self.market_cap = market_data.get('Close', 0.0) * 1_000_000  # Simplified market cap
            # Update portfolio values based on current prices
            self.risk_score = market_data.get('shadow_risk_gap', 0.0)

    def decide_trade_action(self, margin_requirement: float, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        The core decision-making logic of the bank.
        Phase 2: Simple logic - sell if low cash, else hold.
        Phase 4: Will use full game-theoretic utility function.

        Args:
            margin_requirement (float): The percentage of collateral required by the CCP.
            market_conditions (Dict[str, Any]): Global market state (e.g., VIX, interest rates).

        Returns:
            Dict[str, Any]: A dictionary representing the action.
        """
        from src.config import LOW_CASH_THRESHOLD
        
        self.current_margin_requirement = margin_requirement
        
        # Phase 2: Simple decision logic
        if self.is_defaulted:
            return {'action': 'DEFAULTED', 'ticker': self.ticker}
        
        # If cash is low, try to sell some portfolio
        if self.available_cash < LOW_CASH_THRESHOLD and self.portfolio:
            # Sell the first asset in portfolio
            asset_to_sell = list(self.portfolio.keys())[0] if self.portfolio else None
            if asset_to_sell and self.portfolio[asset_to_sell] > 0:
                sell_amount = min(self.portfolio[asset_to_sell] * 0.2, self.portfolio[asset_to_sell])  # Sell 20%
                self.pending_trade = {
                    'action': 'SELL',
                    'asset': asset_to_sell,
                    'quantity': sell_amount,
                    'ticker': self.ticker
                }
                return self.pending_trade
        
        # Default: hold position
        self.pending_trade = {'action': 'HOLD', 'ticker': self.ticker}
        return self.pending_trade

    def execute_trade(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute the pending trade and update cash/portfolio.
        
        Args:
            current_prices (Dict[str, float]): Current prices for assets.
            
        Returns:
            Dict[str, Any]: Trade execution result.
        """
        if not self.pending_trade or self.pending_trade.get('action') in ['HOLD', 'DEFAULTED']:
            return {'executed': False, 'reason': 'No trade to execute'}
        
        action = self.pending_trade.get('action')
        asset = self.pending_trade.get('asset')
        quantity = self.pending_trade.get('quantity', 0)
        
        # Get price (default to 100 if not found)
        price = current_prices.get(asset, 100.0)
        
        if action == 'SELL':
            if asset in self.portfolio and self.portfolio[asset] >= quantity:
                self.portfolio[asset] -= quantity
                proceeds = quantity * price
                self.available_cash += proceeds
                self.pending_trade = {}
                return {'executed': True, 'action': 'SELL', 'asset': asset, 
                        'quantity': quantity, 'proceeds': proceeds}
            else:
                return {'executed': False, 'reason': 'Insufficient holdings'}
                
        elif action == 'BUY':
            cost = quantity * price
            if self.available_cash >= cost:
                self.available_cash -= cost
                self.portfolio[asset] = self.portfolio.get(asset, 0) + quantity
                self.pending_trade = {}
                return {'executed': True, 'action': 'BUY', 'asset': asset,
                        'quantity': quantity, 'cost': cost}
            else:
                return {'executed': False, 'reason': 'Insufficient cash'}
        
        return {'executed': False, 'reason': 'Unknown action'}

    def apply_loss(self, loss_amount: float) -> float:
        """
        Apply a loss to this bank (from contagion or other events).
        
        Args:
            loss_amount (float): The amount of loss to apply.
            
        Returns:
            float: The actual loss applied.
        """
        actual_loss = min(loss_amount, self.available_cash + 1_000_000_000)  # Cap at available + buffer
        self.available_cash -= actual_loss
        return actual_loss

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total value of bank (cash + portfolio).
        
        Args:
            current_prices (Dict[str, float]): Current prices for assets.
            
        Returns:
            float: Total value.
        """
        portfolio_value = sum(
            qty * current_prices.get(asset, 100.0) 
            for asset, qty in self.portfolio.items()
        )
        return self.available_cash + portfolio_value
        
    def calculate_utility(self, margin_requirement: float, market_conditions: Dict[str, Any]) -> float:
        """
        Calculates the bank's payoff (utility) under the current conditions.
        
        PHASE 4 IMPLEMENTATION DETAIL:
        This will implement the formal utility function defined in the project blueprint.
        Utility = (Expected Return) - (Cost of Margin) - (Penalty for Shadow Risk).

        Returns:
            float: The calculated utility score.
        """
        # --- Logic to be implemented in Phase 4 ---
        return 0.0


class ExchangeNode:
    """
    Represents a trading exchange. It primarily manages liquidity and market stability.
    """
    def __init__(self, ticker: str):
        """
        Initializes an ExchangeNode.

        Args:
            ticker (str): The ticker symbol for the exchange operator (e.g., 'NDAQ').
        """
        self.ticker: str = ticker
        self.is_open: bool = True
        self.liquidity_depth: float = 0.0  # Will be proxied by trading volume
        self.halt_reason: str = ""
        self.trades_blocked: int = 0  # Count of blocked trades during halt

    def __repr__(self) -> str:
        status = 'HALTED' if not self.is_open else 'Open'
        return f"ExchangeNode({self.ticker}, status={status})"

    def check_volatility_halt(self, market_vix: float, halt_threshold: float = 40.0) -> bool:
        """
        Simulates a circuit breaker. Halts trading if market volatility exceeds a threshold.

        Args:
            market_vix (float): The current value of the VIX index.
            halt_threshold (float): The VIX level at which the exchange halts trading.
            
        Returns:
            bool: True if exchange is now halted, False otherwise.
        """
        if market_vix > halt_threshold:
            self.is_open = False
            self.halt_reason = f"VIX exceeded threshold ({market_vix:.1f} > {halt_threshold})"
            return True
        else:
            self.is_open = True
            self.halt_reason = ""
            return False

    def update_liquidity(self, dollar_volume: float):
        """
        Update the exchange's liquidity depth based on trading volume.
        
        Args:
            dollar_volume (float): The total dollar volume traded.
        """
        self.liquidity_depth = dollar_volume

    def reset(self):
        """Reset exchange to initial state."""
        self.is_open = True
        self.halt_reason = ""
        self.trades_blocked = 0
        self.liquidity_depth = 0.0


class CCP_Controller:
    """
    Represents the Central Counterparty. It observes the entire system and sets risk parameters.
    """
    def __init__(self, initial_default_fund: float, risk_aversion_lambda: float):
        """
        Initializes the CCP Controller.

        Args:
            initial_default_fund (float): The starting capital buffer to absorb losses.
            risk_aversion_lambda (float): A parameter that weights the penalty for systemic risk in the utility function.
        """
        self.initial_default_fund: float = initial_default_fund
        self.default_fund: float = initial_default_fund
        self.risk_aversion_lambda: float = risk_aversion_lambda
        
        # Phase 2: Store current system state
        self.current_graph = None
        self.current_market_conditions: Dict[str, Any] = {}
        self.total_losses_absorbed: float = 0.0

    def __repr__(self) -> str:
        return f"CCP_Controller(default_fund=₹{self.default_fund:,.0f}, losses_absorbed=₹{self.total_losses_absorbed:,.0f})"

    def reset(self):
        """Reset CCP to initial state."""
        self.default_fund = self.initial_default_fund
        self.current_graph = None
        self.current_market_conditions = {}
        self.total_losses_absorbed = 0.0

    def observe_system_state(self, system_graph: Any, market_conditions: Dict[str, Any]):
        """
        Takes a snapshot of the entire financial system.
        
        Args:
            system_graph (Any): The NetworkX graph object representing the current state.
            market_conditions (Dict[str, Any]): Global market state.
        """
        self.current_graph = system_graph
        self.current_market_conditions = market_conditions

    def absorb_loss(self, loss_amount: float) -> float:
        """
        Absorb losses from defaulted banks using the default fund.
        
        Args:
            loss_amount (float): The amount of loss to absorb.
            
        Returns:
            float: The amount actually absorbed (may be less if fund is depleted).
        """
        absorbed = min(loss_amount, self.default_fund)
        self.default_fund -= absorbed
        self.total_losses_absorbed += absorbed
        return absorbed

    # In agents.py, update CCP_Controller class

    def calculate_margin_requirements(self, banks: List['BankAgent'], system_graph: Any) -> Dict[str, float]:
        """Use GNN predictions for margin calculation."""
        from src.models import load_model
        from src.gnn_utils import networkx_to_pyg, NODE_FEATURES
        
        margins = {}
        
        # Try to use GNN model
        try:
            model = load_model('models/gnn_risk_model.pt', input_dim=len(NODE_FEATURES))
            pyg_data, ticker_to_idx = networkx_to_pyg(system_graph)
            
            if pyg_data is not None:
                risk_scores = model.predict(pyg_data)
                
                for bank in banks:
                    idx = ticker_to_idx.get(bank.ticker)
                    if idx is not None:
                        gnn_risk = risk_scores[idx].item()
                        # Higher risk → higher margin (5% base + up to 15% extra)
                        margins[bank.ticker] = 0.05 + gnn_risk * 0.15
                    else:
                        margins[bank.ticker] = 0.05
            else:
                margins = {b.ticker: 0.05 for b in banks}
                
        except Exception as e:
            # Fallback to simple margin if model fails
            margins = {b.ticker: 0.05 for b in banks}
        
        return margins

    def calculate_utility(self, total_market_volume: float, potential_loss: float) -> float:
        """
        Calculates the CCP's utility based on its dual mandate of market health and stability.
        
        PHASE 4 IMPLEMENTATION DETAIL:
        Utility = (Log of Market Volume) - (Lambda * Systemic Breach Risk).

        Args:
            total_market_volume (float): The total dollar volume traded in the last step.
            potential_loss (float): The estimated loss in a stress scenario.

        Returns:
            float: The calculated utility score for the CCP.
        """
        # --- Logic to be implemented in Phase 4 ---
        return 0.0