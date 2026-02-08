# src/models.py
"""
Phase 3: Graph Neural Network Model for Systemic Risk Prediction

This module defines:
- BankRiskGNN: A Graph Attention Network (GAT) that predicts risk scores
- WeightedBCEWithLogitsLoss: Loss function to handle class imbalance
- The model takes graph structure + node features and outputs risk logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted Binary Cross Entropy with Logits Loss for class imbalance.
    
    Combines:
    - pos_weight: Multiplicative weight for positive class samples
    - Focal loss style focusing (optional)
    
    For dataset with ~16% high risk and ~84% low risk:
    - pos_weight = 84/16 = 5.25 gives balanced gradient contributions
    """
    
    def __init__(self, pos_weight: float = 5.0, gamma: float = 0.0, reduction: str = 'mean'):
        """
        Args:
            pos_weight: Weight for positive class (high risk). Set to neg/pos ratio.
            gamma: Focal loss gamma. 0 = standard BCE, 2 = strong focal effect.
            reduction: 'mean', 'sum', or 'none'
        """
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (before sigmoid) [N, 1]
            targets: Ground truth labels [N, 1]
        """
        # Apply pos_weight: weight positive samples more heavily
        # BCE formula with pos_weight: -[pos_weight * y * log(p) + (1-y) * log(1-p)]
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
            reduction='none'
        )
        
        # Apply focal loss focusing if gamma > 0
        if self.gamma > 0:
            probs = torch.sigmoid(logits)
            p_t = targets * probs + (1 - targets) * (1 - probs)
            focal_weight = (1 - p_t) ** self.gamma
            bce_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        return bce_loss


class BankRiskGNN(nn.Module):
    """
    Enhanced Graph Attention Network for bank systemic risk prediction.
    
    Architecture (updated for larger dataset):
    - 3 GAT layers with multi-head attention and residual connections
    - Layer normalization for training stability
    - Dropout for regularization
    - Deeper MLP with LeakyReLU for final classification
    - Outputs LOGITS (not sigmoid) for use with BCEWithLogitsLoss
    
    Why GAT?
    - Graph Attention allows the model to learn which neighboring banks
      matter most (weighted correlations become learned attention)
    - Multi-head attention captures different relationship patterns
    """
    
    def __init__(
        self,
        input_dim: int,           # Number of node features
        hidden_dim: int = 128,     # Hidden layer dimension
        num_heads: int = 8,       # Number of attention heads
        dropout: float = 0.4,     # Dropout rate
        output_dim: int = 1       # 1 for binary classification (risk/no-risk)
    ):
        super(BankRiskGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input projection for residual connections
        self.input_proj = nn.Linear(input_dim, hidden_dim * num_heads)
        
        # GAT Layer 1: input_dim → hidden_dim * num_heads
        self.gat1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # GAT Layer 2: (hidden_dim * num_heads) → hidden_dim * num_heads
        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # GAT Layer 3: (hidden_dim * num_heads) → hidden_dim
        self.gat3 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            dropout=dropout,
            concat=False
        )
        
        # Layer Normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim * num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim * num_heads)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Deeper MLP for risk classification (outputs LOGITS, not probabilities)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
            # NO sigmoid here - use BCEWithLogitsLoss during training
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the GNN with residual connections.
        Returns LOGITS (apply sigmoid for probabilities).
        """
        # Project input for residual connection
        x_res = self.input_proj(x)
        
        # GAT Layer 1 + Residual
        x1 = self.gat1(x, edge_index)
        x1 = self.ln1(x1)
        x1 = F.leaky_relu(x1, 0.2)
        x1 = x1 + x_res  # Residual connection
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        # GAT Layer 2 + Residual
        x2 = self.gat2(x1, edge_index)
        x2 = self.ln2(x2)
        x2 = F.leaky_relu(x2, 0.2)
        x2 = x2 + x1  # Residual connection
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        # GAT Layer 3
        x3 = self.gat3(x2, edge_index)
        x3 = self.ln3(x3)
        x3 = F.leaky_relu(x3, 0.2)
        
        # MLP for final logits
        logits = self.mlp(x3)
        
        return logits
    
    def predict_proba(self, data: Data) -> torch.Tensor:
        """
        Convenience method for inference - returns probabilities.
        """
        self.eval()
        with torch.no_grad():
            logits = self(data.x, data.edge_index, data.edge_attr, data.batch)
            return torch.sigmoid(logits)
    
    def predict(self, data: Data, threshold: float = 0.3) -> torch.Tensor:
        """
        Convenience method for inference - returns binary predictions.
        
        Args:
            data: PyTorch Geometric Data object
            threshold: Classification threshold (lower = more high-risk predictions)
        """
        probs = self.predict_proba(data)
        return (probs >= threshold).float()


class RiskDataset:
    """
    Container for training data: list of (graph, labels) pairs.
    """
    def __init__(self):
        self.graphs = []
        self.labels = []
        
    def add(self, graph: Data, labels: torch.Tensor):
        self.graphs.append(graph)
        self.labels.append(labels)
        
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]
    
    def to_batch(self) -> Tuple[Batch, torch.Tensor]:
        """Convert all graphs to a single batch for training."""
        batch = Batch.from_data_list(self.graphs)
        all_labels = torch.cat(self.labels, dim=0)
        return batch, all_labels


def create_model(input_dim: int, hidden_dim: int = 64, num_heads: int = 4) -> BankRiskGNN:
    """Factory function to create a new model instance."""
    return BankRiskGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.3,
        output_dim=1
    )


def load_model(path: str, input_dim: int, hidden_dim: int = 64, num_heads: int = 4) -> BankRiskGNN:
    """Load a trained model from disk."""
    model = create_model(input_dim, hidden_dim, num_heads)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def save_model(model: BankRiskGNN, path: str):
    """Save model weights to disk."""
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")