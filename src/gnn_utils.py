# src/gnn_utils.py
"""
Utilities for converting NetworkX graphs to PyTorch Geometric format.
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional

from src.config import BANK_TICKERS


# Define which features to use as node inputs
NODE_FEATURES = [
    'daily_return',
    'realized_volatility', 
    'beta_market',
    'shadow_risk_gap',
    'price_momentum',
    'vix'
]


def networkx_to_pyg(
    G: nx.Graph, 
    node_features: List[str] = None,
    bank_tickers: List[str] = None
) -> Tuple[Data, Dict[str, int]]:
    """
    Convert a NetworkX graph to PyTorch Geometric Data object.
    
    Args:
        G: NetworkX graph from build_graph_for_timestep()
        node_features: List of feature names to extract
        bank_tickers: List of bank tickers (for consistent ordering)
        
    Returns:
        data: PyTorch Geometric Data object
        ticker_to_idx: Mapping from ticker to node index
    """
    if node_features is None:
        node_features = NODE_FEATURES
    if bank_tickers is None:
        bank_tickers = BANK_TICKERS
    
    # Filter to only bank nodes that exist in graph
    bank_nodes = [t for t in bank_tickers if t in G.nodes()]
    
    if len(bank_nodes) == 0:
        return None, {}
    
    # Create ticker to index mapping
    ticker_to_idx = {ticker: i for i, ticker in enumerate(bank_nodes)}
    
    # Extract node features
    feature_matrix = []
    for ticker in bank_nodes:
        node_data = G.nodes[ticker]
        features = []
        for feat_name in node_features:
            val = node_data.get(feat_name, 0.0)
            # Handle NaN values
            if pd.isna(val) or np.isinf(val):
                val = 0.0
            features.append(val)
        feature_matrix.append(features)
    
    x = torch.tensor(feature_matrix, dtype=torch.float)
    
    # Extract edges
    edge_list = []
    edge_weights = []
    
    for u, v, data in G.edges(data=True):
        if u in ticker_to_idx and v in ticker_to_idx:
            i, j = ticker_to_idx[u], ticker_to_idx[v]
            # Add both directions (undirected graph)
            edge_list.append([i, j])
            edge_list.append([j, i])
            weight = data.get('weight', 1.0)
            edge_weights.extend([weight, weight])
    
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    else:
        # No edges - create empty tensors
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(bank_nodes)
    )
    
    return data, ticker_to_idx


def create_target_labels(
    processed_data: pd.DataFrame,
    current_date: pd.Timestamp,
    forward_days: int = 5,
    drop_threshold: float = -0.10,
    bank_tickers: List[str] = None
) -> Dict[str, float]:
    """
    Create binary labels: did this bank's stock drop > threshold% in next N days?
    
    Args:
        processed_data: DataFrame with all processed data
        current_date: The date for which to create labels
        forward_days: Number of days to look ahead
        drop_threshold: Negative return threshold (e.g., -0.10 for 10% drop)
        bank_tickers: List of bank tickers
        
    Returns:
        Dict mapping ticker -> label (1.0 if high risk, 0.0 otherwise)
    """
    if bank_tickers is None:
        bank_tickers = BANK_TICKERS
        
    labels = {}
    
    # Get the date index
    if 'Date' in processed_data.columns:
        dates = processed_data['Date'].unique()
    else:
        dates = processed_data.index.unique()
    
    dates = sorted(dates)
    
    try:
        current_idx = list(dates).index(current_date)
    except ValueError:
        # Date not found
        return {t: 0.0 for t in bank_tickers}
    
    # Check if we have enough future data
    if current_idx + forward_days >= len(dates):
        return {t: 0.0 for t in bank_tickers}
    
    future_date = dates[current_idx + forward_days]
    
    # Calculate forward return for each bank
    for ticker in bank_tickers:
        if 'Date' in processed_data.columns:
            current_price = processed_data[
                (processed_data['Date'] == current_date) & 
                (processed_data['Ticker'] == ticker)
            ]['Close'].values
            
            future_price = processed_data[
                (processed_data['Date'] == future_date) & 
                (processed_data['Ticker'] == ticker)
            ]['Close'].values
        else:
            # Date is index
            try:
                current_price = processed_data.loc[
                    (processed_data.index == current_date) & 
                    (processed_data['Ticker'] == ticker)
                ]['Close'].values
                
                future_price = processed_data.loc[
                    (processed_data.index == future_date) & 
                    (processed_data['Ticker'] == ticker)
                ]['Close'].values
            except:
                labels[ticker] = 0.0
                continue
        
        if len(current_price) > 0 and len(future_price) > 0:
            current_p = current_price[0]
            future_p = future_price[0]
            
            if current_p > 0:
                forward_return = (future_p - current_p) / current_p
                # Label = 1 if price dropped more than threshold
                labels[ticker] = 1.0 if forward_return < drop_threshold else 0.0
            else:
                labels[ticker] = 0.0
        else:
            labels[ticker] = 0.0
    
    return labels


def prepare_training_data(
    processed_data: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    forward_days: int = 5,
    drop_threshold: float = -0.10
) -> List[Tuple[Data, torch.Tensor, pd.Timestamp]]:
    """
    Prepare training data by generating graphs and labels for each day.
    
    Args:
        processed_data: Full processed dataset
        start_date: First date to include
        end_date: Last date to include
        forward_days: Days to look ahead for labels
        drop_threshold: Return threshold for "high risk" label
        
    Returns:
        List of (pyg_data, labels_tensor, date) tuples
    """
    from src.graph_builder import build_graph_for_timestep
    from src.data_loader import get_market_data_for_timestep
    
    # Get unique dates in range
    if 'Date' in processed_data.columns:
        all_dates = sorted(processed_data['Date'].unique())
    else:
        all_dates = sorted(processed_data.index.unique())
    
    dates_in_range = [d for d in all_dates 
                      if pd.Timestamp(d) >= start_date and pd.Timestamp(d) <= end_date]
    
    training_data = []
    
    for date in dates_in_range:
        date_ts = pd.Timestamp(date)
        
        # Get market data for this day
        market_data = get_market_data_for_timestep(processed_data, date_ts)
        if not market_data:
            continue
            
        # Build NetworkX graph
        G = build_graph_for_timestep(date_ts, processed_data, market_data)
        
        if G.number_of_nodes() == 0:
            continue
            
        # Convert to PyTorch Geometric
        pyg_data, ticker_to_idx = networkx_to_pyg(G)
        
        if pyg_data is None:
            continue
            
        # Create labels
        labels_dict = create_target_labels(
            processed_data, date_ts, forward_days, drop_threshold
        )
        
        # Order labels by ticker index
        bank_nodes = [t for t in BANK_TICKERS if t in ticker_to_idx]
        labels_list = [labels_dict.get(t, 0.0) for t in bank_nodes]
        labels_tensor = torch.tensor(labels_list, dtype=torch.float).unsqueeze(1)
        
        training_data.append((pyg_data, labels_tensor, date_ts))
    
    return training_data


def normalize_features(data_list: List[Data]) -> List[Data]:
    """
    Normalize node features across all graphs to zero mean, unit variance.
    """
    if not data_list:
        return data_list
        
    # Stack all features to compute statistics
    all_features = torch.cat([d.x for d in data_list], dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    std[std == 0] = 1  # Avoid division by zero
    
    # Normalize each graph
    normalized = []
    for d in data_list:
        d = d.clone()
        d.x = (d.x - mean) / std
        normalized.append(d)
        
    return normalized, mean, std


# =============================================================================
# SMALL DATA TRAINING UTILITIES
# =============================================================================

import random


def augment_graph(data: Data, noise_scale: float = 0.05, edge_drop_prob: float = 0.1) -> Data:
    """
    Augment graph data by adding noise and randomly dropping edges.
    Useful for training with limited data.
    
    Args:
        data: PyTorch Geometric Data object
        noise_scale: Standard deviation of Gaussian noise to add to features
        edge_drop_prob: Probability of dropping each edge
        
    Returns:
        Augmented copy of the graph
    """
    data = data.clone()
    
    # 1. Add Gaussian noise to node features
    noise = torch.randn_like(data.x) * noise_scale
    data.x = data.x + noise
    
    # 2. Random edge dropout (simulate missing correlations)
    if data.edge_index.size(1) > 0 and random.random() < 0.5:
        num_edges = data.edge_index.size(1)
        keep_mask = torch.rand(num_edges) > edge_drop_prob
        data.edge_index = data.edge_index[:, keep_mask]
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr[keep_mask]
    
    return data


class EarlyStopping:
    """
    Early stopping to prevent overfitting during training.
    
    Usage:
        early_stop = EarlyStopping(patience=10)
        for epoch in range(max_epochs):
            train(...)
            val_loss = validate(...)
            if early_stop(val_loss):
                print("Early stopping triggered!")
                break
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        else:
            self.best_loss = val_loss
            self.counter = 0
            
        return False
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.should_stop = False


def train_with_kfold(
    training_data: List[Tuple[Data, torch.Tensor, pd.Timestamp]],
    input_dim: int,
    n_folds: int = 5,
    epochs: int = 100,
    hidden_dim: int = 64,
    num_heads: int = 4,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    use_augmentation: bool = True,
    pos_weight: float = 5.0,  # Weight for positive class (neg/pos ratio)
    focal_gamma: float = 2.0,
    verbose: bool = True
) -> Dict:
    """
    Train GNN with K-Fold Cross-Validation using Weighted BCE Loss for class imbalance.
    
    Args:
        training_data: List of (graph, labels, date) tuples
        input_dim: Number of input features
        n_folds: Number of CV folds
        epochs: Training epochs per fold
        hidden_dim: Hidden layer dimension
        num_heads: Number of attention heads
        learning_rate: Learning rate
        weight_decay: L2 regularization strength
        use_augmentation: Whether to use data augmentation
        pos_weight: Weight for positive class (set to neg/pos ratio, e.g., 5.0 for 16% pos class)
        focal_gamma: Focal loss gamma (0 = standard BCE, 2 = strong focal effect)
        verbose: Print progress
        
    Returns:
        Dict with training results including per-fold losses and best models
    """
    from sklearn.model_selection import KFold
    from src.models import create_model, save_model, WeightedBCEWithLogitsLoss
    import torch.optim as optim
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize Weighted BCE Loss for class imbalance
    # pos_weight = num_negative / num_positive ≈ 84% / 16% ≈ 5.25
    criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight, gamma=focal_gamma)
    
    results = {
        'fold_val_losses': [],
        'fold_train_losses': [],
        'best_val_loss': float('inf'),
        'best_model_state': None
    }
    
    indices = list(range(len(training_data)))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if verbose:
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        # Split data
        train_graphs = [training_data[i][0] for i in train_idx]
        train_labels = [training_data[i][1] for i in train_idx]
        val_graphs = [training_data[i][0] for i in val_idx]
        val_labels = [training_data[i][1] for i in val_idx]
        
        # Normalize features (compute stats on train only)
        train_graphs_norm, feat_mean, feat_std = normalize_features(train_graphs)
        
        # Create fresh model for each fold
        model = create_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stop = EarlyStopping(patience=15)
        
        best_fold_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []
            
            for graph, labels in zip(train_graphs_norm, train_labels):
                optimizer.zero_grad()
                
                # Original forward pass with Focal Loss
                out = model(graph.x, graph.edge_index, graph.edge_attr)
                loss = criterion(out, labels)
                
                # Augmented forward passes
                if use_augmentation:
                    aug1 = augment_graph(graph)
                    out_aug1 = model(aug1.x, aug1.edge_index, aug1.edge_attr)
                    loss_aug1 = criterion(out_aug1, labels)
                    
                    aug2 = augment_graph(graph)
                    out_aug2 = model(aug2.x, aug2.edge_index, aug2.edge_attr)
                    loss_aug2 = criterion(out_aug2, labels)
                    
                    loss = (loss + loss_aug1 + loss_aug2) / 3
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for graph, labels in zip(val_graphs, val_labels):
                    graph = graph.clone()
                    graph.x = (graph.x - feat_mean) / feat_std
                    graph.x = torch.nan_to_num(graph.x, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    out = model(graph.x, graph.edge_index, graph.edge_attr)
                    val_loss = criterion(out, labels)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = np.mean(val_losses) if val_losses else 0
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_fold_val_loss:
                best_fold_val_loss = avg_val_loss
                if avg_val_loss < results['best_val_loss']:
                    results['best_val_loss'] = avg_val_loss
                    results['best_model_state'] = model.state_dict().copy()
            
            # Early stopping
            if early_stop(avg_val_loss):
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break
        
        results['fold_val_losses'].append(best_fold_val_loss)
        results['fold_train_losses'].append(np.mean(train_losses))
        
        if verbose:
            print(f"  Best Val Loss: {best_fold_val_loss:.4f}")
    
    if verbose:
        print(f"\n📊 Cross-Validation Results:")
        print(f"   Mean Val Loss: {np.mean(results['fold_val_losses']):.4f} ± {np.std(results['fold_val_losses']):.4f}")
        print(f"   Best Val Loss: {results['best_val_loss']:.4f}")
    
    return results