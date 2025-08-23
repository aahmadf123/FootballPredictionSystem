"""Play-by-play Transformer for sequence modeling of football games."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import math
from loguru import logger

from grid.config import GridConfig


class PlaySequenceDataset(Dataset):
    """Dataset for play-by-play sequences."""
    
    def __init__(self, sequences: List[Dict], max_seq_len: int = 200):
        self.sequences = sequences
        self.max_seq_len = max_seq_len
        
        # Vocabularies for categorical features
        self.play_type_vocab = self._build_vocab([s['play_types'] for s in sequences])
        self.down_vocab = {1: 1, 2: 2, 3: 3, 4: 4, 0: 0}  # 0 for special cases
        
        self.vocab_sizes = {
            'play_type': len(self.play_type_vocab),
            'down': len(self.down_vocab),
            'quarter': 5,  # 1-4 + overtime
            'yardline': 101,  # 0-100
            'distance': 51  # 0-50+ (capped)
        }
    
    def _build_vocab(self, sequences: List[List[str]]) -> Dict[str, int]:
        """Build vocabulary from sequences."""
        vocab = {'<PAD>': 0, '<UNK>': 1}
        idx = 2
        
        for seq in sequences:
            for item in seq:
                if item not in vocab:
                    vocab[item] = idx
                    idx += 1
        
        return vocab
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Encode sequence
        encoded = self._encode_sequence(sequence)
        
        # Pad or truncate to max_seq_len
        for key in encoded:
            if len(encoded[key]) > self.max_seq_len:
                encoded[key] = encoded[key][:self.max_seq_len]
            else:
                pad_len = self.max_seq_len - len(encoded[key])
                if key in ['play_type', 'down', 'quarter']:
                    encoded[key].extend([0] * pad_len)  # Pad with 0
                else:
                    encoded[key].extend([0.0] * pad_len)
        
        # Convert to tensors
        result = {}
        for key, values in encoded.items():
            if key in ['play_type', 'down', 'quarter', 'yardline', 'distance']:
                result[key] = torch.LongTensor(values)
            else:
                result[key] = torch.FloatTensor(values)
        
        # Add attention mask
        actual_length = min(len(sequence['play_types']), self.max_seq_len)
        mask = [1] * actual_length + [0] * (self.max_seq_len - actual_length)
        result['attention_mask'] = torch.BoolTensor(mask)
        
        return result
    
    def _encode_sequence(self, sequence: Dict) -> Dict[str, List]:
        """Encode a single sequence."""
        encoded = {
            'play_type': [],
            'down': [],
            'quarter': [],
            'yardline': [],
            'distance': [],
            'score_diff': [],
            'time_left': [],
            'epa': [],
            'success': []
        }
        
        for i in range(len(sequence['play_types'])):
            # Categorical features
            play_type = sequence['play_types'][i]
            encoded['play_type'].append(
                self.play_type_vocab.get(play_type, self.play_type_vocab['<UNK>'])
            )
            
            encoded['down'].append(
                self.down_vocab.get(sequence.get('downs', [1])[i], 0)
            )
            
            encoded['quarter'].append(
                min(sequence.get('quarters', [1])[i], 4)
            )
            
            # Numeric features
            encoded['yardline'].append(
                max(0, min(100, sequence.get('yardlines', [50])[i]))
            )
            
            encoded['distance'].append(
                max(0, min(50, sequence.get('distances', [10])[i]))
            )
            
            # Continuous features
            encoded['score_diff'].append(
                sequence.get('score_diffs', [0.0])[i]
            )
            
            encoded['time_left'].append(
                sequence.get('time_lefts', [900.0])[i] / 3600.0  # Normalize to [0,1]
            )
            
            encoded['epa'].append(
                sequence.get('epas', [0.0])[i]
            )
            
            encoded['success'].append(
                float(sequence.get('successes', [0])[i])
            )
        
        return encoded


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class PlayByPlayTransformer(nn.Module):
    """Transformer model for play-by-play sequences."""
    
    def __init__(self, config: GridConfig):
        super().__init__()
        
        # Model configuration
        self.d_model = config.models.pbp_transformer.d_model
        self.nhead = config.models.pbp_transformer.heads
        self.num_layers = config.models.pbp_transformer.layers
        self.max_seq_len = config.models.pbp_transformer.seq_len
        
        # Embedding dimensions
        self.vocab_sizes = {
            'play_type': 50,  # Will be updated from dataset
            'down': 5,
            'quarter': 5,
            'yardline': 101,
            'distance': 51
        }
        
        # Embedding layers
        self.embeddings = nn.ModuleDict()
        for name, vocab_size in self.vocab_sizes.items():
            self.embeddings[name] = nn.Embedding(vocab_size, self.d_model // 8)
        
        # Continuous feature projection
        self.continuous_projection = nn.Linear(4, self.d_model // 2)  # score_diff, time_left, epa, success
        
        # Input projection to d_model
        total_embed_dim = len(self.vocab_sizes) * (self.d_model // 8) + (self.d_model // 2)
        self.input_projection = nn.Linear(total_embed_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output heads for different tasks
        self.next_play_head = nn.Linear(self.d_model, self.vocab_sizes['play_type'])
        self.success_head = nn.Linear(self.d_model, 1)
        self.epa_head = nn.Linear(self.d_model, 1)
        
        # Game-level aggregation
        self.game_aggregation = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.nhead,
            batch_first=True
        )
        
        # Game-level features
        self.game_feature_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, 64)
        )
    
    def forward(self, batch: Dict[str, torch.Tensor], 
                return_game_features: bool = False) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = batch['play_type'].shape
        
        # Create embeddings
        embeddings = []
        
        # Categorical embeddings
        for name in ['play_type', 'down', 'quarter', 'yardline', 'distance']:
            if name in batch:
                emb = self.embeddings[name](batch[name])
                embeddings.append(emb.view(batch_size, seq_len, -1))
        
        # Continuous features
        continuous_features = torch.stack([
            batch.get('score_diff', torch.zeros_like(batch['play_type'], dtype=torch.float)),
            batch.get('time_left', torch.zeros_like(batch['play_type'], dtype=torch.float)),
            batch.get('epa', torch.zeros_like(batch['play_type'], dtype=torch.float)),
            batch.get('success', torch.zeros_like(batch['play_type'], dtype=torch.float))
        ], dim=-1)
        
        continuous_emb = self.continuous_projection(continuous_features)
        embeddings.append(continuous_emb)
        
        # Concatenate all embeddings
        combined_emb = torch.cat(embeddings, dim=-1)
        
        # Project to d_model
        x = self.input_projection(combined_emb)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Create attention mask
        if 'attention_mask' in batch:
            src_key_padding_mask = ~batch['attention_mask']
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Output predictions
        outputs = {
            'next_play_logits': self.next_play_head(encoded),
            'success_pred': torch.sigmoid(self.success_head(encoded).squeeze(-1)),
            'epa_pred': self.epa_head(encoded).squeeze(-1),
            'sequence_encodings': encoded
        }
        
        # Game-level features if requested
        if return_game_features:
            # Attention pooling for game representation
            game_repr, _ = self.game_aggregation(encoded, encoded, encoded, 
                                               key_padding_mask=src_key_padding_mask)
            
            # Average over sequence length (masked)
            if src_key_padding_mask is not None:
                mask = (~src_key_padding_mask).float().unsqueeze(-1)
                game_repr = (game_repr * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                game_repr = game_repr.mean(dim=1)
            
            game_features = self.game_feature_head(game_repr)
            outputs['game_features'] = game_features
        
        return outputs


class PBPTransformerTrainer:
    """Trainer for play-by-play transformer."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 50
        
        # Loss weights
        self.loss_weights = {
            'next_play': 1.0,
            'success': 0.5,
            'epa': 0.5
        }
    
    def prepare_data(self, pbp_df: pd.DataFrame) -> List[Dict]:
        """Prepare play-by-play data for transformer training."""
        sequences = []
        
        # Group by game
        for game_id, game_data in pbp_df.groupby('game_id'):
            # Sort by play order
            game_data = game_data.sort_values(['quarter', 'play_id'])
            
            if len(game_data) < 5:  # Skip games with too few plays
                continue
            
            sequence = {
                'game_id': game_id,
                'play_types': game_data['play_type'].fillna('unknown').tolist(),
                'downs': game_data['down'].fillna(1).astype(int).tolist(),
                'quarters': game_data['quarter'].fillna(1).astype(int).tolist(),
                'yardlines': game_data['yardline_100'].fillna(50).astype(int).tolist(),
                'distances': game_data['dist'].fillna(10).astype(int).tolist(),
                'score_diffs': [0.0] * len(game_data),  # Would calculate from actual scores
                'time_lefts': game_data['sec_left'].fillna(900).astype(float).tolist(),
                'epas': game_data['epa'].fillna(0.0).astype(float).tolist(),
                'successes': (game_data['epa'] > 0).astype(int).tolist()
            }
            
            sequences.append(sequence)
        
        logger.info(f"Prepared {len(sequences)} game sequences for training")
        return sequences
    
    def create_data_loaders(self, sequences: List[Dict], 
                          train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        # Split sequences
        split_idx = int(len(sequences) * train_split)
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        # Create datasets
        train_dataset = PlaySequenceDataset(train_sequences, self.config.models.pbp_transformer.seq_len)
        val_dataset = PlaySequenceDataset(val_sequences, self.config.models.pbp_transformer.seq_len)
        
        # Update model vocab sizes
        if self.model:
            self.model.vocab_sizes.update(train_dataset.vocab_sizes)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, 
            collate_fn=self._collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=self._collate_fn
        )
        
        return train_loader, val_loader
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for data loader."""
        collated = {}
        
        for key in batch[0].keys():
            collated[key] = torch.stack([item[key] for item in batch])
        
        return collated
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss."""
        total_loss = 0.0
        
        # Next play prediction loss
        if 'next_play_logits' in outputs and 'play_type' in batch:
            # Shift targets for next-play prediction
            next_play_targets = batch['play_type'][:, 1:].contiguous()
            next_play_logits = outputs['next_play_logits'][:, :-1].contiguous()
            
            # Flatten for cross entropy
            next_play_logits = next_play_logits.view(-1, next_play_logits.size(-1))
            next_play_targets = next_play_targets.view(-1)
            
            # Mask padding tokens
            mask = next_play_targets != 0
            if mask.any():
                next_play_loss = nn.CrossEntropyLoss(ignore_index=0)(
                    next_play_logits, next_play_targets
                )
                total_loss += self.loss_weights['next_play'] * next_play_loss
        
        # Success prediction loss
        if 'success_pred' in outputs and 'success' in batch:
            success_targets = batch['success']
            success_preds = outputs['success_pred']
            
            success_loss = nn.BCELoss()(success_preds, success_targets)
            total_loss += self.loss_weights['success'] * success_loss
        
        # EPA prediction loss
        if 'epa_pred' in outputs and 'epa' in batch:
            epa_targets = batch['epa']
            epa_preds = outputs['epa_pred']
            
            epa_loss = nn.MSELoss()(epa_preds, epa_targets)
            total_loss += self.loss_weights['epa'] * epa_loss
        
        return total_loss
    
    def train(self, pbp_df: pd.DataFrame) -> Dict[str, List[float]]:
        """Train the transformer model."""
        logger.info("Training PBP Transformer...")
        
        # Prepare data
        sequences = self.prepare_data(pbp_df)
        train_loader, val_loader = self.create_data_loaders(sequences)
        
        # Create model
        self.model = PlayByPlayTransformer(self.config).to(self.device)
        
        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                
                outputs = self.model(batch)
                loss = self.compute_loss(outputs, batch)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    outputs = self.model(batch)
                    loss = self.compute_loss(outputs, batch)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_pbp_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_pbp_model.pth'))
        
        logger.info(f"PBP Transformer training completed. Best val loss: {best_val_loss:.4f}")
        return history
    
    def extract_game_features(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Extract game-level features from trained transformer."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        self.model.eval()
        
        sequences = self.prepare_data(pbp_df)
        dataset = PlaySequenceDataset(sequences, self.config.models.pbp_transformer.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self._collate_fn)
        
        game_features = []
        game_ids = []
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(batch, return_game_features=True)
                features = outputs['game_features'].cpu().numpy()
                
                # Get corresponding game IDs
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(sequences))
                batch_game_ids = [sequences[j]['game_id'] for j in range(start_idx, end_idx)]
                
                game_features.extend(features)
                game_ids.extend(batch_game_ids)
        
        # Create DataFrame
        feature_df = pd.DataFrame(game_features)
        feature_df.columns = [f'pbp_transformer_feat_{i}' for i in range(feature_df.shape[1])]
        feature_df['game_id'] = game_ids
        
        return feature_df
