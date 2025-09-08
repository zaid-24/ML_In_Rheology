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
# PART 3: CUSTOM MAXWELL NEURON
# ============================================

class MaxwellNeuron(nn.Module):
    """
    A single Maxwell element as a neural network component
    This can be stacked to create generalized Maxwell models
    """
    
    def __init__(self, init_E=0, init_eta=0, learnable=True):
        super(MaxwellNeuron, self).__init__()
        
        if learnable:
            self.log_E = nn.Parameter(torch.log(torch.tensor([init_E])))
            self.log_eta = nn.Parameter(torch.log(torch.tensor([init_eta])))
        else:
            self.log_E = torch.log(torch.tensor([init_E]))
            self.log_eta = torch.log(torch.tensor([init_eta]))
    
    def forward(self, strain, strain_rate, dt=0.1):
        """
        Forward pass through Maxwell element
        Args:
            strain: (batch, seq_len)
            strain_rate: (batch, seq_len)
            dt: time step
        """
        E = torch.exp(self.log_E)
        eta = torch.exp(self.log_eta)
        tau = eta / E
        
        batch_size, seq_len = strain.shape
        stress = torch.zeros_like(strain)
        
        # Create a new tensor for each time step instead of in-place modification
        stress_values = []
        
        for t in range(seq_len):
            if t == 0:
                stress_t = E * strain[:, t] + eta * strain_rate[:, t]
            else:
                stress_prev = stress_values[t-1]
                stress_t = stress_prev * torch.exp(-dt/tau) + \
                        E * (strain[:, t] - strain[:, t-1])
            stress_values.append(stress_t)
        
        # Reconstruct the full stress tensor
        for t in range(seq_len):
            stress = stress.clone()  # Clone to avoid in-place operation
            stress[:, t] = stress_values[t]
        
        return stress
    
    def get_parameters(self):
        """Return physical parameters"""
        E = torch.exp(self.log_E).item()
        eta = torch.exp(self.log_eta).item()
        return {'E': E, 'eta': eta, 'tau': eta/E}

class GeneralizedMaxwellNetwork(nn.Module):
    """
    Network of Maxwell elements in parallel
    This represents a generalized Maxwell model
    """
    
    def __init__(self, n_elements=10, input_size=2):
        super(GeneralizedMaxwellNetwork, self).__init__()
        
        # Create multiple Maxwell elements
        self.maxwell_elements = nn.ModuleList([
            MaxwellNeuron(
                init_E=10**(np.random.uniform(5, 7)),
                init_eta=10**(np.random.uniform(7, 9))
            ) for _ in range(n_elements)
        ])
        
        # Learnable weights for combining elements
        self.element_weights = nn.Parameter(torch.ones(n_elements) / n_elements)
        
        # Optional: Gating mechanism to sparsify the network
        self.gate = nn.Sequential(
            nn.Linear(input_size, n_elements),
            nn.Sigmoid()
        )
    
    def forward(self, x, dt=0.1):
        """
        x: (batch, seq_len, 2) - strain and strain_rate
        """
        batch_size, seq_len, _ = x.shape
        strain = x[:, :, 0]
        strain_rate = x[:, :, 1]
        
        # Compute gating values
        gate_values = self.gate(x.mean(dim=1))  # (batch, n_elements)
        
        # Compute stress from each Maxwell element
        stresses = []
        for i, element in enumerate(self.maxwell_elements):
            stress = element(strain, strain_rate, dt)
            # Apply gating and weighting
            weighted_stress = stress * self.element_weights[i] * gate_values[:, i].unsqueeze(1)
            stresses.append(weighted_stress)
        
        # Combine all stresses
        total_stress = torch.stack(stresses).sum(dim=0)
        
        return total_stress
    
    def get_element_contributions(self, x, dt=0.1):
        """Get contribution of each Maxwell element"""
        batch_size, seq_len, _ = x.shape
        strain = x[:, :, 0]
        strain_rate = x[:, :, 1]
        
        gate_values = self.gate(x.mean(dim=1))
        
        contributions = []
        parameters = []
        
        for i, element in enumerate(self.maxwell_elements):
            stress = element(strain, strain_rate, dt)
            weight = self.element_weights[i] * gate_values[:, i].mean()
            
            contributions.append({
                'element_id': i,
                'weight': weight.item(),
                'avg_stress': stress.mean().item(),
                'contribution': (stress * weight).mean().item()
            })
            
            params = element.get_parameters()
            params['element_id'] = i
            params['weight'] = weight.item()
            parameters.append(params)
        
        return contributions, parameters
    
    def prune_network(self, threshold=0.01):
        """Remove elements with low contribution"""
        with torch.no_grad():
            weights = torch.abs(self.element_weights)
            mask = weights > threshold
            self.element_weights.data *= mask.float()
        
        active_elements = mask.sum().item()
        print(f"Active elements after pruning: {active_elements}/{len(self.maxwell_elements)}")
        
        return mask

# ============================================
# PART 4: TRAINING AND EVALUATION
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
# PART 5: EXPERIMENT RUNNER
# ============================================

class RheologyExperimentRunner:
    """Main class to run experiments"""
    
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
    
    def run_experiment_1(self):
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
            'PhysicsInformed': PhysicsInformedNN(),
            'MaxwellNetwork': GeneralizedMaxwellNetwork(n_elements=5)
        }
        
        trained_models = {}
        training_histories = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            trained_model, train_losses, val_losses = train_model(
                model, self.train_loader, self.val_loader,
                epochs=100, lr=0.001, device=self.device, model_name=name
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
    
    def run_experiment_2(self, n_elements=10):
        """
        Experiment 2: Train Generalized Maxwell Network and analyze components
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Generalized Maxwell Network Analysis")
        print("="*60)
        
        # Create and train the network
        model = GeneralizedMaxwellNetwork(n_elements=n_elements)
        
        print(f"\nTraining Generalized Maxwell Network with {n_elements} elements...")
        trained_model, train_losses, val_losses = train_model(
            model, self.train_loader, self.val_loader,
            epochs=100, lr=0.001, device=self.device,
            model_name='GeneralizedMaxwell'
        )
        
        # Analyze element contributions
        print("\n" + "="*40)
        print("Element Analysis:")
        print("="*40)
        
        # Get a sample batch for analysis
        sample_batch = next(iter(self.test_loader))
        sample_x, sample_y = sample_batch
        sample_x = sample_x.to(self.device)
        
        contributions, parameters = trained_model.get_element_contributions(sample_x)
        
        # Sort by contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        print(f"\n{'Element':<10} {'Weight':<12} {'E (Pa)':<12} {'η (Pa·s)':<12} {'τ (s)':<12} {'Contribution':<12}")
        print("-" * 70)
        
        for i, (contrib, params) in enumerate(zip(contributions[:10], parameters[:10])):
            if contrib['weight'] > 0.01:  # Only show significant elements
                print(f"{params['element_id']:<10} {params['weight']:<12.4f} "
                      f"{params['E']:<12.2e} {params['eta']:<12.2e} "
                      f"{params['tau']:<12.4f} {contrib['contribution']:<12.6f}")
        
        # Prune network
        print("\n" + "="*40)
        print("Network Pruning:")
        print("="*40)
        
        mask = trained_model.prune_network(threshold=0.05)
        
        # Re-evaluate after pruning
        trained_model.eval()
        test_loss = 0
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = trained_model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
        
        test_loss /= len(self.test_loader)
        print(f"Test loss after pruning: {test_loss:.6f}")
        
        return trained_model, contributions, parameters
    
    def visualize_results(self, models, results):
        """Visualize model predictions vs actual"""
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, (name, metrics) in enumerate(list(results.items())[:8]):
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
        
        plt.tight_layout()
        plt.show()
        
        return fig

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    # Load your data
    print("Loading data...")
    # Assuming you have your cleaned ramp experiment data
    # df = pd.read_csv('your_cleaned_ramp_data.csv')
    
    # For demonstration, let's create sample data
    # You should replace this with your actual data loading
    
    df = pd.read_csv('dataset.csv')
    
    # Initialize experiment runner
    runner = RheologyExperimentRunner(df, sequence_length=50, batch_size=32)
    
    # Run Experiment 1
    trained_models, results, histories = runner.run_experiment_1()
    
    # Visualize results
    fig = runner.visualize_results(trained_models, results)
    
    # Run Experiment 2
    maxwell_model, contributions, parameters = runner.run_experiment_2(n_elements=10)
    
    # Plot training histories
    fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, (train_losses, val_losses)) in enumerate(list(histories.items())[:8]):
        ax = axes[idx]
        ax.plot(train_losses, label='Train', alpha=0.7)
        ax.plot(val_losses, label='Validation', alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return trained_models, results, maxwell_model

if __name__ == "__main__":
    # Note: Import the data generator from the previous artifact
    # or load your actual cleaned data
    
    trained_models, results, maxwell_model = main()