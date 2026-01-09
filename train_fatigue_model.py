"""
T-SEEDS Fatigue Detection LSTM Training
Trains LSTM model on eye closure sequences for drowsiness prediction
Predicts fatigue 30 seconds ahead using dlib eye landmarks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Fatigue-Training")


class FatigueDataset(Dataset):
    """Dataset for fatigue detection sequences"""

    def __init__(self, sequences, labels, seq_len=30):
        """
        Args:
            sequences: Eye closure sequences (EAR values)
            labels: Fatigue labels (0=alert, 1=drowsy)
            seq_len: Sequence length (30 = 1 second at 30 FPS)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class FatigueLSTM(nn.Module):
    """LSTM model for fatigue prediction"""

    def __init__(self, input_size=6, hidden_size=128, num_layers=2, dropout=0.3):
        """
        Args:
            input_size: Features per timestep (left_ear, right_ear, avg_ear, head_pitch, head_yaw, head_roll)
            hidden_size: LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(FatigueLSTM, self).__init__()

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

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input sequences (batch_size, seq_len, input_size)
        Returns:
            Fatigue probability (batch_size, 1)
        """
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Attention weights
        attention_weights = self.attention(lstm_out)

        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected
        output = self.fc(context)

        return output


def load_fatigue_dataset(dataset_path):
    """
    Load fatigue dataset from CSV files
    Expected format:
    timestamp, left_ear, right_ear, avg_ear, head_pitch, head_yaw, head_roll, fatigue_label
    """
    dataset_dir = Path(dataset_path)

    # Load all CSV files
    all_data = []
    for csv_file in dataset_dir.glob('*.csv'):
        df = pd.read_csv(csv_file)
        all_data.append(df)

    if not all_data:
        logger.error(f"No CSV files found in {dataset_path}")
        return None, None

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} samples from {len(all_data)} files")

    return combined_df


def create_sequences(data, seq_len=30, step=10):
    """
    Create overlapping sequences from time series data
    Args:
        data: DataFrame with features
        seq_len: Sequence length (30 frames = 1 second at 30 FPS)
        step: Step size for sliding window
    """
    features = ['left_ear', 'right_ear', 'avg_ear', 'head_pitch', 'head_yaw', 'head_roll']

    sequences = []
    labels = []

    for i in range(0, len(data) - seq_len, step):
        seq = data[features].iloc[i:i+seq_len].values
        # Label is fatigue state 30 seconds ahead (900 frames)
        future_idx = min(i + seq_len + 900, len(data) - 1)
        label = data['fatigue_label'].iloc[future_idx]

        sequences.append(seq)
        labels.append(label)

    logger.info(f"Created {len(sequences)} sequences of length {seq_len}")

    return np.array(sequences), np.array(labels)


def generate_synthetic_data(num_samples=10000, seq_len=30):
    """
    Generate synthetic fatigue data for demonstration
    Real deployment should use actual driver video data
    """
    logger.info(f"Generating {num_samples} synthetic samples...")

    sequences = []
    labels = []

    for i in range(num_samples):
        # Generate alert or drowsy sequence
        is_drowsy = np.random.rand() > 0.5

        if is_drowsy:
            # Drowsy: decreasing EAR, increasing closure duration
            base_ear = np.random.uniform(0.15, 0.20)
            ear_sequence = base_ear + np.random.normal(0, 0.02, seq_len) - np.linspace(0, 0.05, seq_len)
            head_pitch = np.random.normal(10, 5, seq_len)  # Head drooping
            head_yaw = np.random.normal(0, 2, seq_len)
            head_roll = np.random.normal(0, 2, seq_len)
            label = 1
        else:
            # Alert: normal EAR, stable head pose
            base_ear = np.random.uniform(0.25, 0.35)
            ear_sequence = base_ear + np.random.normal(0, 0.03, seq_len)
            head_pitch = np.random.normal(0, 3, seq_len)  # Head upright
            head_yaw = np.random.normal(0, 5, seq_len)
            head_roll = np.random.normal(0, 3, seq_len)
            label = 0

        # Clip EAR values
        ear_sequence = np.clip(ear_sequence, 0, 0.4)

        # Create feature vector
        sequence = np.column_stack([
            ear_sequence,  # left_ear
            ear_sequence + np.random.normal(0, 0.01, seq_len),  # right_ear
            ear_sequence,  # avg_ear
            head_pitch,
            head_yaw,
            head_roll
        ])

        sequences.append(sequence)
        labels.append(label)

    return np.array(sequences), np.array(labels)


def train_fatigue_model(args):
    """Train LSTM fatigue detection model"""
    logger.info("=" * 60)
    logger.info("TNT T-SEEDS Fatigue Detection Training")
    logger.info("=" * 60)

    # Load or generate dataset
    if Path(args.dataset).exists():
        logger.info(f"Loading dataset from {args.dataset}")
        data = load_fatigue_dataset(args.dataset)
        if data is not None:
            sequences, labels = create_sequences(data, args.seq_len)
        else:
            logger.warning("Failed to load dataset, using synthetic data")
            sequences, labels = generate_synthetic_data(args.samples, args.seq_len)
    else:
        logger.info("Dataset not found, generating synthetic data")
        sequences, labels = generate_synthetic_data(args.samples, args.seq_len)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    logger.info(f"Alert samples: {np.sum(y_train==0)}, Drowsy samples: {np.sum(y_train==1)}")

    # Create datasets and dataloaders
    train_dataset = FatigueDataset(X_train, y_train, args.seq_len)
    test_dataset = FatigueDataset(X_test, y_test, args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    model = FatigueLSTM(
        input_size=6,
        hidden_size=args.hidden_size,
        num_layers=2,
        dropout=0.3
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)

        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)
                outputs = model(sequences)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)

        # Learning rate scheduling
        scheduler.step(test_loss)

        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{args.epochs}] "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                       f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            os.makedirs('models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'accuracy': test_acc
            }, 'models/fatigue_lstm.pth')
            logger.info(f"Saved best model with test loss: {test_loss:.4f}")

    # Final evaluation
    logger.info("=" * 60)
    logger.info("Training Results:")
    logger.info(f"  Best Test Loss: {best_loss:.4f}")
    logger.info(f"  Final Test Accuracy: {test_acc:.2f}%")
    logger.info("=" * 60)

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('T-SEEDS Fatigue LSTM Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/fatigue_training_curves.png')
    logger.info("Training curves saved to models/fatigue_training_curves.png")

    # Save metadata
    metadata = {
        'model_name': 'fatigue_lstm',
        'version': '1.0.0',
        'training_date': datetime.now().isoformat(),
        'epochs': args.epochs,
        'hidden_size': args.hidden_size,
        'seq_len': args.seq_len,
        'best_test_loss': float(best_loss),
        'final_test_accuracy': float(test_acc),
        'features': ['left_ear', 'right_ear', 'avg_ear', 'head_pitch', 'head_yaw', 'head_roll'],
        'prediction_horizon': '30 seconds ahead'
    }

    with open('models/fatigue_lstm_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("Training complete! Model ready for deployment.")

    return 'models/fatigue_lstm.pth', metadata


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Train T-SEEDS fatigue LSTM")
    parser.add_argument('--dataset', type=str, default='datasets/fatigue_dataset',
                       help='Path to fatigue dataset')
    parser.add_argument('--seq-len', type=int, default=30,
                       help='Sequence length (frames)')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--samples', type=int, default=10000,
                       help='Number of synthetic samples if no dataset')

    args = parser.parse_args()

    model_path, metadata = train_fatigue_model(args)

    logger.info("=" * 60)
    logger.info("T-SEEDS Fatigue LSTM training complete!")
    logger.info(f"Model: {model_path}")
    logger.info(f"Accuracy: {metadata['final_test_accuracy']:.2f}%")
    logger.info("=" * 60)

