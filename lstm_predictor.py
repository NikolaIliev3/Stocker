"""
LSTM Neural Network for Stock Prediction
Captures temporal patterns in price sequences for improved prediction accuracy.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)

# Try to import PyTorch
HAS_PYTORCH = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_PYTORCH = True
except ImportError:
    logger.warning("PyTorch not available. Install with: pip install torch")


class LSTMModel(nn.Module):
    """LSTM Neural Network for sequence prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.3, num_classes: int = 2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # LSTM forward pass
        # x shape: (batch, sequence_length, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class LSTMPredictor:
    """
    LSTM-based stock predictor that learns temporal patterns.
    
    Features:
    - Processes sequences of daily data (e.g., last 20-60 days)
    - Captures patterns like: "RSI was low 5 days ago, now rising = BUY"
    - Outputs probability of price moving up/down
    """
    
    def __init__(self, data_dir: Path, sequence_length: int = 30):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_params = None
        self.feature_names = None
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"LSTM using device: {self.device}")
        
        # Try to load existing model
        self._load_model()
    
    def _get_model_path(self) -> Path:
        return self.data_dir / 'lstm_model.pt'
    
    def _get_params_path(self) -> Path:
        return self.data_dir / 'lstm_params.pkl'
    
    def _load_model(self):
        """Load trained model if exists"""
        model_path = self._get_model_path()
        params_path = self._get_params_path()
        
        if model_path.exists() and params_path.exists():
            try:
                with open(params_path, 'rb') as f:
                    params = pickle.load(f)
                
                self.scaler_params = params['scaler_params']
                self.feature_names = params['feature_names']
                input_size = params['input_size']
                
                self.model = LSTMModel(input_size=input_size)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.is_trained = True
                
                logger.info(f"Loaded LSTM model with {input_size} features")
            except Exception as e:
                logger.warning(f"Could not load LSTM model: {e}")
    
    def _save_model(self):
        """Save trained model"""
        if self.model is None:
            return
            
        model_path = self._get_model_path()
        params_path = self._get_params_path()
        
        torch.save(self.model.state_dict(), model_path)
        
        params = {
            'scaler_params': self.scaler_params,
            'feature_names': self.feature_names,
            'input_size': len(self.feature_names),
            'sequence_length': self.sequence_length
        }
        with open(params_path, 'wb') as f:
            pickle.dump(params, f)
            
        logger.info(f"Saved LSTM model to {model_path}")
    
    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using z-score"""
        if fit or self.scaler_params is None:
            self.scaler_params = {
                'mean': np.mean(X, axis=(0, 1)),
                'std': np.std(X, axis=(0, 1)) + 1e-8
            }
        
        return (X - self.scaler_params['mean']) / self.scaler_params['std']
    
    def prepare_sequences(self, features_list: List[Dict], labels: List[int] = None) -> Tuple:
        """
        Convert list of daily features to sequences for LSTM.
        
        Args:
            features_list: List of feature dicts (one per day)
            labels: Optional list of labels (UP=1, DOWN=0)
            
        Returns:
            Tuple of (X_sequences, y_labels) if labels provided, else just X
        """
        if len(features_list) < self.sequence_length:
            return None, None
        
        # Get feature names from first sample
        if self.feature_names is None:
            self.feature_names = sorted([k for k in features_list[0].keys() 
                                        if isinstance(features_list[0][k], (int, float))])
        
        # Convert to numpy array
        n_samples = len(features_list)
        n_features = len(self.feature_names)
        
        data = np.zeros((n_samples, n_features))
        for i, feat_dict in enumerate(features_list):
            for j, name in enumerate(self.feature_names):
                data[i, j] = feat_dict.get(name, 0)
        
        # Handle NaN/Inf
        data = np.nan_to_num(data, nan=0, posinf=1e6, neginf=-1e6)
        
        # Create sequences
        X_sequences = []
        y_labels = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            X_sequences.append(seq)
            
            if labels is not None:
                y_labels.append(labels[i + self.sequence_length - 1])
        
        X = np.array(X_sequences)
        
        if labels is not None:
            y = np.array(y_labels)
            return X, y
        return X, None
    
    def train(self, training_data: List[Dict], epochs: int = 50, 
              batch_size: int = 32, learning_rate: float = 0.001,
              validation_split: float = 0.2) -> Dict:
        """
        Train LSTM model on historical data.
        
        Args:
            training_data: List of dicts with 'features' and 'label' keys
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            validation_split: Fraction for validation
            
        Returns:
            Dict with training metrics
        """
        if not HAS_PYTORCH:
            logger.error("PyTorch not available. Cannot train LSTM.")
            return {'error': 'PyTorch not installed'}
        
        # Extract features and labels
        features_list = [d['features'] for d in training_data]
        labels = [1 if d['label'] == 'BUY' else 0 for d in training_data]
        
        # Prepare sequences
        X, y = self.prepare_sequences(features_list, labels)
        
        if X is None or len(X) < 100:
            logger.warning(f"Not enough data for LSTM training: {len(X) if X is not None else 0}")
            return {'error': 'Insufficient training data'}
        
        # Normalize
        X = self._normalize(X, fit=True)
        
        # Train/validation split
        n_val = int(len(X) * validation_split)
        indices = np.random.permutation(len(X))
        
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMModel(input_size=input_size)
        self.model.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training loop
        best_val_acc = 0
        best_model_state = None
        
        logger.info(f"Training LSTM: {len(X_train)} train, {len(X_val)} val samples")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                X_val_dev = X_val_t.to(self.device)
                y_val_dev = y_val_t.to(self.device)
                
                val_outputs = self.model(X_val_dev)
                val_loss = criterion(val_outputs, y_val_dev)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val_dev).float().mean().item()
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, "
                           f"Val Acc: {val_acc:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_trained = True
        self._save_model()
        
        logger.info(f"LSTM training complete. Best validation accuracy: {best_val_acc:.4f}")
        
        return {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'best_val_accuracy': best_val_acc,
            'epochs': epochs
        }
    
    def predict(self, features_sequence: List[Dict]) -> Dict:
        """
        Predict using trained LSTM.
        
        Args:
            features_sequence: List of feature dicts for last N days
            
        Returns:
            Dict with prediction and confidence
        """
        if not self.is_trained or self.model is None:
            return {
                'action': 'HOLD',
                'confidence': 50.0,
                'probabilities': {'BUY': 50.0, 'SELL': 50.0}
            }
        
        if len(features_sequence) < self.sequence_length:
            logger.debug(f"Not enough sequence data: {len(features_sequence)} < {self.sequence_length}")
            return {
                'action': 'HOLD',
                'confidence': 50.0,
                'probabilities': {'BUY': 50.0, 'SELL': 50.0}
            }
        
        try:
            # Prepare sequence
            X, _ = self.prepare_sequences(features_sequence[-self.sequence_length - 1:])
            
            if X is None or len(X) == 0:
                return {
                    'action': 'HOLD',
                    'confidence': 50.0,
                    'probabilities': {'BUY': 50.0, 'SELL': 50.0}
                }
            
            # Normalize
            X = self._normalize(X, fit=False)
            
            # Get last sequence
            X_last = torch.FloatTensor(X[-1:]).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_last)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # probs: [P(DOWN), P(UP)]
            prob_up = float(probs[1])
            prob_down = float(probs[0])
            
            if prob_up > 0.55:
                action = 'BUY'
                confidence = prob_up * 100
            elif prob_down > 0.55:
                action = 'SELL'
                confidence = prob_down * 100
            else:
                action = 'HOLD'
                confidence = max(prob_up, prob_down) * 100
            
            return {
                'action': action,
                'confidence': confidence,
                'probabilities': {
                    'BUY': prob_up * 100,
                    'SELL': prob_down * 100
                }
            }
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return {
                'action': 'HOLD',
                'confidence': 50.0,
                'probabilities': {'BUY': 50.0, 'SELL': 50.0}
            }


# Factory function
def create_lstm_predictor(data_dir: Path, sequence_length: int = 30) -> Optional[LSTMPredictor]:
    """Create LSTM predictor if PyTorch is available"""
    if not HAS_PYTORCH:
        logger.warning("PyTorch not available. LSTM predictor disabled.")
        return None
    return LSTMPredictor(data_dir, sequence_length)
