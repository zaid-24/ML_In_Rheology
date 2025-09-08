import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================
# PART 1: DATA PREPARATION
# ============================================

class RheologyDataset(Dataset):
    """Custom dataset for rheological time series data"""
    
    def __init__(self, df, sequence_length=50, stride=10):
        """
        Args:
            df: DataFrame with columns [time, strain, strain_rate, stress]
            sequence_length: Length of input sequences
            stride: Stride for creating overlapping sequences
        """
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Prepare data
        self.features = df[['strain', 'strain_rate']].values
        self.targets = df['stress'].values
        self.time = df['time'].values
        
        # Normalize
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.features = self.scaler_X.fit_transform(self.features)
        self.targets = self.scaler_y.fit_transform(self.targets.reshape(-1, 1)).flatten()
        
        # Create sequences
        self.sequences = []
        self.labels = []
        
        for i in range(0, len(self.features) - sequence_length, stride):
            self.sequences.append(self.features[i:i+sequence_length])
            self.labels.append(self.targets[i:i+sequence_length])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.sequences[idx]), 
                torch.FloatTensor(self.labels[idx]))

# ============================================
# PART 2: NEURAL NETWORK ARCHITECTURES
# ============================================

class SimpleNN(nn.Module):
    """Simple feedforward neural network"""
    
    def __init__(self, input_size=2, hidden_size=64, output_size=1, 
                 activation='relu', num_layers=3):
        super(SimpleNN, self).__init__()
        
        # Select activation function
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
            'softplus': nn.Softplus()  # Smooth, good for physical systems
        }
        self.activation = activations.get(activation, nn.ReLU())
        
        # Build layers
        layers = []
        in_features = input_size
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self.activation)
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size * seq_len, -1)
        out = self.network(x)
        return out.reshape(batch_size, seq_len, -1).squeeze(-1)

class RNNModel(nn.Module):
    """Standard RNN model"""
    
    def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out.squeeze(-1)

class GRUModel(nn.Module):
    """GRU model"""
    
    def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return out.squeeze(-1)

class LSTMModel(nn.Module):
    """LSTM model"""
    
    def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(-1)

class PhysicsInformedNN(nn.Module):
    """Physics-informed neural network with Maxwell constraints"""
    
    def __init__(self, input_size=2, hidden_size=64, output_size=1):
        super(PhysicsInformedNN, self).__init__()
        
        # Neural network for learning residuals
        self.nn = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size),  # +1 for previous stress
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Learnable Maxwell parameters
        self.log_E = nn.Parameter(torch.tensor([0.0]))  # log(E) for stability
        self.log_eta = nn.Parameter(torch.tensor([0.0]))  # log(eta)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        E = torch.exp(self.log_E)
        eta = torch.exp(self.log_eta)
        tau = eta / E
        
        stress_pred = torch.zeros(batch_size, seq_len, device=x.device)
        stress_prev = torch.zeros(batch_size, device=x.device)
        
        for t in range(seq_len):
            strain = x[:, t, 0]
            strain_rate = x[:, t, 1]
            
            # Maxwell model prediction
            stress_maxwell = E * strain + eta * strain_rate
            
            # Neural network correction
            nn_input = torch.cat([x[:, t], stress_prev.unsqueeze(1)], dim=1)
            correction = self.nn(nn_input).squeeze()
            
            # Combined prediction
            stress_pred[:, t] = stress_maxwell + correction
            stress_prev = stress_pred[:, t].detach()
        
        return stress_pred

# ============================================
# PART 3: TRAINING AND EVALUATION
# ============================================

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, 
                device='cpu', model_name='Model'):
    """Train a model and return training history"""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'{model_name} - Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return model, train_losses, val_losses

def evaluate_models(models_dict, test_loader, device='cpu'):
    """Evaluate all models on test set"""
    
    results = {}
    criterion = nn.MSELoss()
    
    for name, model in models_dict.items():
        model.eval()
        test_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
                
                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader)
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Calculate additional metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        r2 = 1 - (np.sum((targets - predictions) ** 2) / 
                  np.sum((targets - targets.mean()) ** 2))
        
        results[name] = {
            'test_loss': test_loss,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }
    
    return results

# ============================================
# PART 4: EXPERIMENT 1 RUNNER
# ============================================

class Experiment1Runner:
    """Main class to run Experiment 1 - Architecture Comparison"""
    
    def __init__(self, df, sequence_length=50, batch_size=32):
        self.df = df
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Prepare datasets
        self.prepare_data()
    
    def prepare_data(self):
        """Prepare train, validation, and test datasets"""
        
        # Create dataset
        dataset = RheologyDataset(self.df, self.sequence_length)
        
        # Split data
        n_samples = len(dataset)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        test_size = n_samples - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
    
    def run_experiment(self):
        """
        Experiment 1: Compare different neural network architectures
        """
        print("\n" + "="*60)
        print("EXPERIMENT 1: Architecture Comparison")
        print("="*60)
        
        models = {
            'SimpleNN_ReLU': SimpleNN(activation='relu'),
            'SimpleNN_Tanh': SimpleNN(activation='tanh'),
            'SimpleNN_Softplus': SimpleNN(activation='softplus'),
            'RNN': RNNModel(),
            'GRU': GRUModel(),
            'LSTM': LSTMModel(),
            'PhysicsInformed': PhysicsInformedNN()
        }
        
        trained_models = {}
        training_histories = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use lower learning rate for Physics-Informed model
            lr = 0.0001 if name == 'PhysicsInformed' else 0.001
            
            trained_model, train_losses, val_losses = train_model(
                model, self.train_loader, self.val_loader,
                epochs=50, lr=lr, device=self.device, model_name=name
            )
            trained_models[name] = trained_model
            training_histories[name] = (train_losses, val_losses)
        
        # Evaluate all models
        print("\n" + "="*40)
        print("Model Evaluation Results:")
        print("="*40)
        
        results = evaluate_models(trained_models, self.test_loader, self.device)
        
        # Print comparison table
        print(f"\n{'Model':<20} {'MSE':<12} {'MAE':<12} {'R²':<12}")
        print("-" * 56)
        
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['mse']:<12.6f} {metrics['mae']:<12.6f} {metrics['r2']:<12.6f}")
        
        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['mse'])[0]
        print(f"\nBest performing model: {best_model}")
        
        return trained_models, results, training_histories
    
    def visualize_results(self, models, results):
        """Visualize model predictions vs actual"""
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, (name, metrics) in enumerate(list(results.items())[:7]):
            ax = axes[idx]
            
            # Plot first 200 predictions
            pred_sample = metrics['predictions'].flatten()[:200]
            target_sample = metrics['targets'].flatten()[:200]
            
            ax.plot(target_sample, label='Actual', alpha=0.7)
            ax.plot(pred_sample, label='Predicted', alpha=0.7)
            ax.set_title(f'{name}\nR² = {metrics["r2"]:.4f}')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Stress')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(results) < 8:
            axes[-1].remove()
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_training_histories(self, histories):
        """Plot training histories for all models"""
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, (name, (train_losses, val_losses)) in enumerate(list(histories.items())[:7]):
            ax = axes[idx]
            ax.plot(train_losses, label='Train', alpha=0.7)
            ax.plot(val_losses, label='Validation', alpha=0.7)
            ax.set_title(name)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplot
        if len(histories) < 8:
            axes[-1].remove()
        
        plt.tight_layout()
        plt.show()
        
        return fig

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    # Load your data
    print("Loading data...")
    df = pd.read_csv('dataset.csv')
    
    # Initialize experiment runner
    runner = Experiment1Runner(df, sequence_length=50, batch_size=32)
    
    # Run Experiment 1
    trained_models, results, histories = runner.run_experiment()
    
    # Visualize results
    fig1 = runner.visualize_results(trained_models, results)
    
    # Plot training histories
    fig2 = runner.plot_training_histories(histories)
    
    return trained_models, results, histories

if __name__ == "__main__":
    trained_models, results, histories = main()
