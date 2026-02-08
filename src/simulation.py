# src/simulation.py

"""
Phase 2: Core Simulation Engine

This module implements the FinancialSystem class that orchestrates
all agent interactions following fundamental financial rules:
- Rule 1: Liquidity Flow (trade settlement)
- Rule 2: Default & Contagion (solvency checks, loss propagation)
- Rule 3: Market Halts (VIX-based circuit breakers)
"""

import pandas as pd
import networkx as nx
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.agents import BankAgent, ExchangeNode, CCP_Controller
from src.config import (
    BANK_TICKERS, EXCHANGE_TICKERS, GLOBAL_INDEX_TICKERS,
    VIX_HALT_THRESHOLD, CONTAGION_LOSS_FACTOR, DEFAULT_INITIAL_CASH,
    DEFAULT_MARGIN_RATE
)
from src.data_loader import get_market_data_for_timestep
from src.graph_builder import build_graph_for_timestep


class SimulationEvent:
    """Represents a single event in the simulation log."""
    def __init__(self, timestamp: pd.Timestamp, event_type: str, details: Dict[str, Any]):
        self.timestamp = timestamp
        self.event_type = event_type
        self.details = details
        self.created_at = datetime.now()
    
    def __repr__(self) -> str:
        return f"[{self.timestamp.date()}] {self.event_type}: {self.details}"


class FinancialSystem:
    """
    The core simulation engine that orchestrates all agent interactions.
    
    This class manages:
    - Banks (BankAgent instances)
    - Exchanges (ExchangeNode instances)  
    - CCP (CCP_Controller instance)
    - The financial network graph
    - Simulation state and logging
    """
    
    def __init__(
        self,
        processed_data: pd.DataFrame,
        initial_bank_cash: float = DEFAULT_INITIAL_CASH,
        ccp_default_fund: float = 10_000_000_000,  # ₹1000 crore
        ccp_risk_aversion: float = 0.5
    ):
        """
        Initialize the Financial System.
        
        Args:
            processed_data: DataFrame with engineered features from Phase 1
            initial_bank_cash: Starting cash for each bank
            ccp_default_fund: Initial CCP default fund size
            ccp_risk_aversion: Lambda parameter for CCP utility function
        """
        self.processed_data = processed_data
        self.initial_bank_cash = initial_bank_cash
        
        # Initialize agents
        self.banks: Dict[str, BankAgent] = {}
        self.exchanges: Dict[str, ExchangeNode] = {}
        self.ccp: CCP_Controller = CCP_Controller(ccp_default_fund, ccp_risk_aversion)
        
        # Initialize banks with empty portfolios (Phase 2 simplification)
        for ticker in BANK_TICKERS:
            self.banks[ticker] = BankAgent(
                ticker=ticker,
                initial_cash=initial_bank_cash,
                initial_portfolio={}  # Start with cash only
            )
        
        # Initialize exchanges
        for ticker in EXCHANGE_TICKERS:
            self.exchanges[ticker] = ExchangeNode(ticker=ticker)
        
        # Simulation state
        self.graph: Optional[nx.Graph] = None
        self.timestamp: Optional[pd.Timestamp] = None
        self.current_step: int = 0
        self.simulation_log: List[SimulationEvent] = []
        
        # Get available dates from processed data
        if 'Date' in processed_data.columns:
            self.available_dates = sorted(processed_data['Date'].unique())
        else:
            self.available_dates = sorted(processed_data.index.unique())
        
        print(f"✅ FinancialSystem initialized:")
        print(f"   Banks: {len(self.banks)}")
        print(f"   Exchanges: {len(self.exchanges)}")
        print(f"   CCP Default Fund: ₹{ccp_default_fund:,.0f}")
        print(f"   Available dates: {len(self.available_dates)} trading days")

    def reset(self):
        """Reset the entire system to initial state."""
        for bank in self.banks.values():
            bank.reset()
        for exchange in self.exchanges.values():
            exchange.reset()
        self.ccp.reset()
        self.graph = None
        self.timestamp = None
        self.current_step = 0
        self.simulation_log = []
        print("🔄 System reset to initial state")

    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Add an event to the simulation log."""
        event = SimulationEvent(self.timestamp, event_type, details)
        self.simulation_log.append(event)

    def step(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        Execute a single simulation step. This is the heart of the simulation.
        
        Sequence of operations:
        1. Update timestamp and build graph
        2. CCP observes system state
        3. Update exchange status (check for halts)
        4. CCP sets margin requirements
        5. Banks decide trade actions
        6. Settle trades (if exchanges are open)
        7. Check solvency and propagate defaults
        
        Args:
            timestamp: The date for this simulation step
            
        Returns:
            Dict with step results and statistics
        """
        self.timestamp = timestamp
        self.current_step += 1
        
        step_results = {
            'timestamp': timestamp,
            'step': self.current_step,
            'events': [],
            'defaults': [],
            'halted_exchanges': [],
            'trades_executed': 0,
            'total_contagion_loss': 0.0
        }
        
        # --- 1. Get market data and build graph ---
        market_data = get_market_data_for_timestep(self.processed_data, timestamp)
        
        if not market_data:
            self._log_event("NO_DATA", {'reason': f'No market data for {timestamp.date()}'})
            step_results['events'].append(f"No data for {timestamp.date()}")
            return step_results
        
        # Build the network graph
        self.graph = build_graph_for_timestep(timestamp, self.processed_data, market_data)
        
        # Update bank states with market data
        for ticker, bank in self.banks.items():
            if ticker in market_data:
                bank.update_state(market_data[ticker])
        
        self._log_event("GRAPH_BUILT", {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges()
        })
        
        # --- 2. CCP observes system ---
        global_data = market_data.get('GLOBAL', {})
        self.ccp.observe_system_state(self.graph, global_data)
        self._log_event("CCP_OBSERVED", {'vix': global_data.get('vix', 0)})
        
        # --- 3. Update exchange status (check for market halts) ---
        halted = self.update_exchange_status(global_data)
        step_results['halted_exchanges'] = halted
        
        # --- 4. CCP sets margin requirements ---
        margins = self.ccp.calculate_margin_requirements(
            list(self.banks.values()), 
            self.graph
        )
        self._log_event("MARGINS_SET", margins)
        
        # --- 5. Banks decide trade actions ---
        trade_decisions = {}
        for ticker, bank in self.banks.items():
            if not bank.is_defaulted:
                decision = bank.decide_trade_action(margins.get(ticker, 0.05), global_data)
                trade_decisions[ticker] = decision
        
        self._log_event("TRADE_DECISIONS", {
            'total': len(trade_decisions),
            'sells': sum(1 for d in trade_decisions.values() if d.get('action') == 'SELL'),
            'holds': sum(1 for d in trade_decisions.values() if d.get('action') == 'HOLD')
        })
        
        # --- 6. Settle trades ---
        current_prices = {ticker: market_data.get(ticker, {}).get('Close', 100.0) 
                         for ticker in BANK_TICKERS}
        trades_result = self.settle_trades(current_prices)
        step_results['trades_executed'] = trades_result['executed_count']
        
        # --- 7. Check solvency and propagate defaults ---
        defaults_result = self.check_solvency()
        step_results['defaults'] = defaults_result['newly_defaulted']
        step_results['total_contagion_loss'] = defaults_result['total_contagion_loss']
        
        return step_results

    def update_exchange_status(self, global_data: Dict[str, Any]) -> List[str]:
        """
        Check VIX level and update exchange halt status.
        
        Rule 3: If VIX > threshold, exchanges halt trading.
        
        Args:
            global_data: Dictionary with VIX and other global indicators
            
        Returns:
            List of halted exchange tickers
        """
        vix = global_data.get('vix', 20.0)
        halted = []
        
        for ticker, exchange in self.exchanges.items():
            was_halted = exchange.check_volatility_halt(vix, VIX_HALT_THRESHOLD)
            if was_halted:
                halted.append(ticker)
                self._log_event("MARKET_HALT", {
                    'exchange': ticker,
                    'vix': vix,
                    'threshold': VIX_HALT_THRESHOLD
                })
        
        return halted

    def settle_trades(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Execute pending trades for all banks.
        
        Rule 1: Liquidity Flow - update cash and portfolio based on trades.
        Trades are only executed if exchanges are open.
        
        Args:
            current_prices: Current market prices for assets
            
        Returns:
            Dict with settlement statistics
        """
        # Check if any exchange is open
        any_exchange_open = any(ex.is_open for ex in self.exchanges.values())
        
        result = {
            'executed_count': 0,
            'blocked_count': 0,
            'total_volume': 0.0
        }
        
        if not any_exchange_open:
            # Block all trades if all exchanges are halted
            for exchange in self.exchanges.values():
                exchange.trades_blocked += len([b for b in self.banks.values() 
                                                if b.pending_trade.get('action') not in ['HOLD', 'DEFAULTED', None]])
            self._log_event("ALL_TRADES_BLOCKED", {'reason': 'All exchanges halted'})
            result['blocked_count'] = len(self.banks)
            return result
        
        # Execute trades for each bank
        for ticker, bank in self.banks.items():
            if bank.is_defaulted:
                continue
                
            trade_result = bank.execute_trade(current_prices)
            
            if trade_result.get('executed'):
                result['executed_count'] += 1
                volume = trade_result.get('proceeds', trade_result.get('cost', 0))
                result['total_volume'] += volume
                
                self._log_event("TRADE_EXECUTED", {
                    'bank': ticker,
                    'action': trade_result.get('action'),
                    'asset': trade_result.get('asset'),
                    'amount': volume
                })
        
        # Update exchange liquidity
        for exchange in self.exchanges.values():
            exchange.update_liquidity(result['total_volume'])
        
        return result

    def check_solvency(self) -> Dict[str, Any]:
        """
        Check which banks have negative cash (defaulted) and propagate losses.
        
        Rule 2: Default & Contagion
        - Banks with negative cash are marked as defaulted
        - Losses propagate to connected banks based on edge weights
        
        Returns:
            Dict with default and contagion statistics
        """
        result = {
            'newly_defaulted': [],
            'total_contagion_loss': 0.0,
            'ccp_absorbed': 0.0
        }
        
        # First pass: identify newly defaulted banks
        newly_defaulted = []
        for ticker, bank in self.banks.items():
            if bank.available_cash < 0 and not bank.is_defaulted:
                bank.is_defaulted = True
                newly_defaulted.append(ticker)
                
                self._log_event("BANK_DEFAULT", {
                    'bank': ticker,
                    'cash_shortfall': abs(bank.available_cash)
                })
        
        result['newly_defaulted'] = newly_defaulted
        
        # Second pass: propagate losses to neighbors
        for defaulted_ticker in newly_defaulted:
            contagion_result = self.propagate_default(defaulted_ticker)
            result['total_contagion_loss'] += contagion_result['total_loss_applied']
            result['ccp_absorbed'] += contagion_result['ccp_absorbed']
        
        return result

    def propagate_default(self, defaulted_ticker: str) -> Dict[str, Any]:
        """
        Propagate losses from a defaulted bank to its neighbors.
        
        Loss calculation: loss = |defaulted_bank.cash| * edge_weight * CONTAGION_LOSS_FACTOR
        
        Args:
            defaulted_ticker: Ticker of the bank that defaulted
            
        Returns:
            Dict with contagion statistics
        """
        result = {
            'source': defaulted_ticker,
            'affected_banks': [],
            'total_loss_applied': 0.0,
            'ccp_absorbed': 0.0
        }
        
        if self.graph is None or defaulted_ticker not in self.graph:
            return result
        
        defaulted_bank = self.banks.get(defaulted_ticker)
        if not defaulted_bank:
            return result
        
        # Calculate base loss amount
        base_loss = abs(defaulted_bank.available_cash)
        
        # Get neighbors from the graph
        neighbors = list(self.graph.neighbors(defaulted_ticker))
        
        for neighbor_ticker in neighbors:
            if neighbor_ticker not in self.banks:
                continue  # Skip non-bank nodes
                
            neighbor_bank = self.banks[neighbor_ticker]
            if neighbor_bank.is_defaulted:
                continue  # Already defaulted
            
            # Get edge weight (correlation)
            edge_data = self.graph.get_edge_data(defaulted_ticker, neighbor_ticker)
            edge_weight = abs(edge_data.get('weight', 0.5)) if edge_data else 0.5
            
            # Calculate loss to apply
            loss_amount = base_loss * edge_weight * CONTAGION_LOSS_FACTOR
            
            # Apply loss to neighbor
            actual_loss = neighbor_bank.apply_loss(loss_amount)
            result['affected_banks'].append({
                'bank': neighbor_ticker,
                'loss': actual_loss,
                'edge_weight': edge_weight
            })
            result['total_loss_applied'] += actual_loss
            
            self._log_event("CONTAGION_LOSS", {
                'from': defaulted_ticker,
                'to': neighbor_ticker,
                'loss': actual_loss,
                'edge_weight': edge_weight
            })
        
        # CCP absorbs any remaining loss
        remaining_loss = base_loss * (1 - CONTAGION_LOSS_FACTOR * len(neighbors))
        if remaining_loss > 0:
            absorbed = self.ccp.absorb_loss(remaining_loss)
            result['ccp_absorbed'] = absorbed
            
            self._log_event("CCP_ABSORBED_LOSS", {
                'from': defaulted_ticker,
                'amount': absorbed
            })
        
        return result

    def run_simulation(
        self, 
        start_date: pd.Timestamp, 
        num_steps: int = 10,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run the simulation for multiple steps.
        
        Args:
            start_date: Starting date for simulation
            num_steps: Number of steps to run
            verbose: Whether to print progress
            
        Returns:
            List of step results
        """
        results = []
        
        # Find starting index in available dates
        start_idx = 0
        for i, date in enumerate(self.available_dates):
            if pd.Timestamp(date) >= start_date:
                start_idx = i
                break
        
        if verbose:
            print(f"\n🚀 Starting simulation from {self.available_dates[start_idx]}")
            print(f"   Running {num_steps} steps...")
            print("-" * 50)
        
        for i in range(num_steps):
            if start_idx + i >= len(self.available_dates):
                print(f"⚠️ Reached end of available data at step {i}")
                break
            
            current_date = pd.Timestamp(self.available_dates[start_idx + i])
            step_result = self.step(current_date)
            results.append(step_result)
            
            if verbose:
                self._print_step_summary(step_result)
        
        if verbose:
            print("-" * 50)
            print(f"✅ Simulation complete. {len(results)} steps executed.")
            self._print_final_summary()
        
        return results

    def _print_step_summary(self, step_result: Dict[str, Any]):
        """Print a summary of a single step."""
        print(f"Step {step_result['step']} ({step_result['timestamp'].date()}): "
              f"Trades={step_result['trades_executed']}, "
              f"Defaults={len(step_result['defaults'])}, "
              f"Halts={len(step_result['halted_exchanges'])}")

    def _print_final_summary(self):
        """Print final simulation summary."""
        active_banks = sum(1 for b in self.banks.values() if not b.is_defaulted)
        defaulted_banks = sum(1 for b in self.banks.values() if b.is_defaulted)
        
        print(f"\n📊 Final State:")
        print(f"   Active Banks: {active_banks}/{len(self.banks)}")
        print(f"   Defaulted Banks: {defaulted_banks}")
        print(f"   CCP Default Fund: ₹{self.ccp.default_fund:,.0f}")
        print(f"   Total CCP Losses Absorbed: ₹{self.ccp.total_losses_absorbed:,.0f}")

    def get_system_state(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current system state.
        
        Returns:
            Dict with complete system state
        """
        return {
            'timestamp': self.timestamp,
            'step': self.current_step,
            'banks': {
                ticker: {
                    'cash': bank.available_cash,
                    'is_defaulted': bank.is_defaulted,
                    'risk_score': bank.risk_score,
                    'portfolio_value': bank.get_total_value({})
                }
                for ticker, bank in self.banks.items()
            },
            'exchanges': {
                ticker: {
                    'is_open': ex.is_open,
                    'halt_reason': ex.halt_reason
                }
                for ticker, ex in self.exchanges.items()
            },
            'ccp': {
                'default_fund': self.ccp.default_fund,
                'total_losses_absorbed': self.ccp.total_losses_absorbed
            },
            'graph': {
                'nodes': self.graph.number_of_nodes() if self.graph else 0,
                'edges': self.graph.number_of_edges() if self.graph else 0
            }
        }
