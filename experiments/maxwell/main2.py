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
# PART 2: PHYSICS-INFORMED NEURAL NETWORK COMPONENTS
# ============================================

class PINNNeuron(nn.Module):
    """A single Physics-Informed Neural Network neuron/component"""
    
    def __init__(self, input_size=2, hidden_size=32, output_size=1, neuron_id=0):
        super(PINNNeuron, self).__init__()
        
        self.neuron_id = neuron_id
        
        # Neural network for learning residuals (smaller for individual neurons)
        self.nn = nn.Sequential(
            nn.Linear(input_size + 1, hidden_size),  # +1 for previous stress
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, output_size)
        )
        
        # Learnable Maxwell parameters (initialized with different values)
        init_E = np.random.uniform(-1, 1)  # log space
        init_eta = np.random.uniform(-1, 1)
        self.log_E = nn.Parameter(torch.tensor([init_E]))
        self.log_eta = nn.Parameter(torch.tensor([init_eta]))
        
        # Importance weight for this neuron
        self.importance = nn.Parameter(torch.tensor([1.0]))
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        E = torch.exp(self.log_E)
        eta = torch.exp(self.log_eta)
        
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
            
            # Combined prediction weighted by importance
            importance_weight = torch.sigmoid(self.importance).to(x.device)
            stress_pred[:, t] = (stress_maxwell + correction) * importance_weight
            stress_prev = stress_pred[:, t].detach()
        
        return stress_pred
    
    def get_physics_parameters(self):
        """Return learned physical parameters"""
        E = torch.exp(self.log_E).item()
        eta = torch.exp(self.log_eta).item()
        tau = eta / E if E > 1e-10 else float('inf')
        importance = torch.sigmoid(self.importance).item()
        return {
            'neuron_id': self.neuron_id,
            'E': E, 
            'eta': eta, 
            'tau': tau,
            'importance': importance
        }
    
    def physics_loss(self, x, y_pred, y_true):
        """Compute physics-informed loss for this neuron"""
        # MSE loss
        mse_loss = nn.MSELoss()(y_pred, y_true)
        
        # Physics constraint: Maxwell differential equation
        batch_size, seq_len = y_pred.shape
        strain = x[:, :, 0]
        strain_rate = x[:, :, 1]
        
        E = torch.exp(self.log_E)
        eta = torch.exp(self.log_eta)
        
        physics_residual = 0.0
        for t in range(1, seq_len):
            dt = 1.0
            stress_rate = (y_pred[:, t] - y_pred[:, t-1]) / dt
            
            # Maxwell equation residual
            maxwell_lhs = y_pred[:, t] + (eta/E) * stress_rate
            maxwell_rhs = eta * strain_rate[:, t] + E * strain[:, t]
            
            residual = torch.mean((maxwell_lhs - maxwell_rhs)**2)
            physics_residual += residual
        
        physics_residual /= (seq_len - 1)
        
        return mse_loss, physics_residual

class MultiPINNNetwork(nn.Module):
    """Multi-layer network of PINN neurons with pruning capability"""
    
    def __init__(self, neurons_per_layer=4, num_layers=4, input_size=2):
        super(MultiPINNNetwork, self).__init__()
        
        self.neurons_per_layer = neurons_per_layer
        self.num_layers = num_layers
        self.total_neurons = neurons_per_layer * num_layers
        
        # Create PINN neurons organized in layers
        self.layers = nn.ModuleList()
        neuron_id = 0
        
        for layer_idx in range(num_layers):
            layer_neurons = nn.ModuleList()
            for neuron_idx in range(neurons_per_layer):
                neuron = PINNNeuron(input_size=input_size, neuron_id=neuron_id)
                layer_neurons.append(neuron)
                neuron_id += 1
            self.layers.append(layer_neurons)
        
        # Layer combination weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Final combination layer
        self.final_combination = nn.Sequential(
            nn.Linear(num_layers, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        layer_outputs = []
        
        # Process each layer
        for layer_idx, layer in enumerate(self.layers):
            neuron_outputs = []
            
            # Process each neuron in the layer
            for neuron in layer:
                neuron_output = neuron(x)
                neuron_outputs.append(neuron_output)
            
            # Combine neurons in this layer (weighted average)
            if neuron_outputs:
                stacked_outputs = torch.stack(neuron_outputs, dim=0)  # [neurons, batch, seq_len]
                
                # Get importance weights for neurons in this layer
                importances = torch.stack([torch.sigmoid(neuron.importance).to(x.device) for neuron in layer])
                importances = importances / (importances.sum() + 1e-8)  # Normalize
                
                # Weighted combination
                layer_output = torch.sum(stacked_outputs * importances.unsqueeze(1).unsqueeze(2), dim=0)
                layer_outputs.append(layer_output)
        
        # Combine layers
        if layer_outputs:
            layer_stack = torch.stack(layer_outputs, dim=-1)  # [batch, seq_len, num_layers]
            
            # Apply layer weights
            layer_weights_norm = torch.softmax(self.layer_weights, dim=0)
            weighted_layers = layer_stack * layer_weights_norm.unsqueeze(0).unsqueeze(0)
            
            # Final combination through a small network
            final_output = self.final_combination(weighted_layers).squeeze(-1)
            
            return final_output
        
        return torch.zeros(batch_size, seq_len, device=x.device)
    
    def get_all_neuron_parameters(self):
        """Get parameters from all neurons"""
        all_params = []
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer):
                params = neuron.get_physics_parameters()
                params['layer'] = layer_idx
                params['position'] = neuron_idx
                all_params.append(params)
        return all_params
    
    def prune_neurons(self, importance_threshold=0.7):
        """Prune neurons with low importance"""
        pruned_count = 0
        active_neurons = []
        
        print(f"\n{'Layer':<8} {'Neuron':<8} {'ID':<5} {'Importance':<12} {'E':<12} {'η':<12} {'τ':<12} {'Status':<10}")
        print("-" * 85)
        
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer):
                params = neuron.get_physics_parameters()
                importance = params['importance']
                
                status = "ACTIVE" if importance > importance_threshold else "PRUNED"
                if importance <= importance_threshold:
                    # Set importance to zero (soft pruning)
                    with torch.no_grad():
                        neuron.importance.data = torch.tensor([-10.0])  # Very low importance
                    pruned_count += 1
                else:
                    active_neurons.append(params)
                
                print(f"{layer_idx:<8} {neuron_idx:<8} {params['neuron_id']:<5} "
                      f"{importance:<12.4f} {params['E']:<12.2e} {params['eta']:<12.2e} "
                      f"{params['tau']:<12.4e} {status:<10}")
        
        active_count = self.total_neurons - pruned_count
        print(f"\nPruning Summary:")
        print(f"Total neurons: {self.total_neurons}")
        print(f"Active neurons: {active_count}")
        print(f"Pruned neurons: {pruned_count}")
        print(f"Pruning ratio: {pruned_count/self.total_neurons:.2%}")
        
        return active_neurons, pruned_count, active_count
    
    def physics_loss(self, x, y_pred, y_true):
        """Compute combined physics loss from active neurons"""
        total_mse = nn.MSELoss()(y_pred, y_true)
        total_physics = torch.tensor(0.0, device=x.device, requires_grad=True)
        active_count = 0
        
        for layer in self.layers:
            for neuron in layer:
                importance = torch.sigmoid(neuron.importance).to(x.device)
                if importance > 0.7:  # Only consider important neurons
                    neuron_pred = neuron(x)
                    _, physics_loss = neuron.physics_loss(x, neuron_pred, y_true)
                    total_physics = total_physics + importance * physics_loss
                    active_count += 1
        
        if active_count > 0:
            total_physics = total_physics / active_count
        
        return total_mse, total_physics

# ============================================
# PART 3: TRAINING AND EVALUATION
# ============================================

def train_multi_pinn_model(model, train_loader, val_loader, epochs=40, lr=0.001, 
                           device='cpu', model_name='MultiPINN', physics_weight=0.1):
    """Train a Multi-PINN Network and return training history"""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.7)
    
    train_losses = []
    val_losses = []
    physics_losses = []
    data_losses = []
    importance_history = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        physics_loss_epoch = 0
        data_loss_epoch = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            # Compute physics-informed loss
            mse_loss, physics_residual = model.physics_loss(batch_x, outputs, batch_y)
            
            # Combined loss
            total_loss = mse_loss + physics_weight * physics_residual
            
            total_loss.backward()
            
            # Add gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += total_loss.item()
            physics_loss_epoch += physics_residual.item() if hasattr(physics_residual, 'item') else physics_residual
            data_loss_epoch += mse_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                mse_loss, physics_residual = model.physics_loss(batch_x, outputs, batch_y)
                total_loss = mse_loss + physics_weight * physics_residual
                val_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        physics_loss_epoch /= len(train_loader)
        data_loss_epoch /= len(train_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        physics_losses.append(physics_loss_epoch)
        data_losses.append(data_loss_epoch)
        
        # Track importance evolution
        if (epoch + 1) % 10 == 0:
            all_params = model.get_all_neuron_parameters()
            avg_importance = np.mean([p['importance'] for p in all_params])
            importance_history.append(avg_importance)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            active_neurons = sum(1 for p in model.get_all_neuron_parameters() if p['importance'] > 0.7)
            print(f'{model_name} - Epoch [{epoch+1}/{epochs}], '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
                  f'Physics Loss: {physics_loss_epoch:.6f}, Data Loss: {data_loss_epoch:.6f}, '
                  f'Active Neurons: {active_neurons}/{model.total_neurons}')
    
    return model, train_losses, val_losses, physics_losses, data_losses, importance_history

# ============================================
# PART 4: EXPERIMENT 2 RUNNER
# ============================================

class Experiment2Runner:
    """Main class to run Experiment 2 - Physics-Informed Neural Network Analysis"""
    
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
    
    def run_experiment(self, neurons_per_layer=4, num_layers=4, physics_weight=0.5):
        """
        Experiment 2: Train Multi-PINN Network with pruning analysis
        """
        print("\n" + "="*60)
        print("EXPERIMENT 2: Multi-PINN Network Analysis")
        print("="*60)
        print(f"Architecture: {num_layers} layers × {neurons_per_layer} PINN neurons = {num_layers * neurons_per_layer} total neurons")
        
        # Create the Multi-PINN Network
        model = MultiPINNNetwork(
            neurons_per_layer=neurons_per_layer,
            num_layers=num_layers,
            input_size=2
        )
        
        print(f"\nTraining Multi-PINN Network...")
        print(f"Physics Weight: {physics_weight}")
        
        # Train the model
        trained_model, train_losses, val_losses, physics_losses, data_losses, importance_history = train_multi_pinn_model(
            model, self.train_loader, self.val_loader,
            epochs=40, lr=0.001, device=self.device,
            model_name='MultiPINN', physics_weight=physics_weight
        )
        
        # Analyze all neurons before pruning
        print("\n" + "="*60)
        print("NEURON ANALYSIS BEFORE PRUNING")
        print("="*60)
        
        all_params = trained_model.get_all_neuron_parameters()
        
        # Sort by importance
        all_params_sorted = sorted(all_params, key=lambda x: x['importance'], reverse=True)
        
        print(f"\n{'Layer':<8} {'Neuron':<8} {'ID':<5} {'Importance':<12} {'E':<12} {'η':<12} {'τ':<12}")
        print("-" * 75)
        
        for params in all_params_sorted[:10]:  # Show top 10
            print(f"{params['layer']:<8} {params['position']:<8} {params['neuron_id']:<5} "
                  f"{params['importance']:<12.4f} {params['E']:<12.2e} {params['eta']:<12.2e} "
                  f"{params['tau']:<12.4e}")
        
        # Evaluate before pruning
        trained_model.eval()
        test_loss_before = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = trained_model(batch_x)
                mse_loss, physics_residual = trained_model.physics_loss(batch_x, outputs, batch_y)
                total_loss = mse_loss + physics_weight * physics_residual
                test_loss_before += total_loss.item()
        test_loss_before /= len(self.test_loader)
        
        print(f"\nTest Loss Before Pruning: {test_loss_before:.6f}")
        
        # Prune the network
        print("\n" + "="*60)
        print("PRUNING ANALYSIS")
        print("="*60)
        
        active_neurons, pruned_count, active_count = trained_model.prune_neurons(importance_threshold=0.7)
        
        # Evaluate after pruning
        trained_model.eval()
        test_loss_after = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = trained_model(batch_x)
                mse_loss, physics_residual = trained_model.physics_loss(batch_x, outputs, batch_y)
                total_loss = mse_loss + physics_weight * physics_residual
                test_loss_after += total_loss.item()
        test_loss_after /= len(self.test_loader)
        
        print(f"\nTest Loss After Pruning: {test_loss_after:.6f}")
        print(f"Performance Change: {((test_loss_after - test_loss_before) / test_loss_before) * 100:+.2f}%")
        
        # Analyze active neurons
        print("\n" + "="*60)
        print("ACTIVE NEURONS ANALYSIS")
        print("="*60)
        
        # Get all neurons with importance > threshold (whether pruned or not)
        all_active_neurons = [p for p in all_params if p['importance'] > 0.7]
        
        if all_active_neurons:
            # Group by layer
            layer_counts = {}
            for neuron in all_active_neurons:
                layer = neuron['layer']
                if layer not in layer_counts:
                    layer_counts[layer] = 0
                layer_counts[layer] += 1
            
            print("\nActive neurons per layer:")
            for layer in range(num_layers):
                count = layer_counts.get(layer, 0)
                print(f"Layer {layer}: {count}/{neurons_per_layer} neurons active ({count/neurons_per_layer:.1%})")
            
            # Show most important active neurons
            active_sorted = sorted(all_active_neurons, key=lambda x: x['importance'], reverse=True)
            print(f"\nTop {min(5, len(active_sorted))} Most Important Active Neurons:")
            print(f"{'Layer':<8} {'Neuron':<8} {'ID':<5} {'Importance':<12} {'E':<12} {'η':<12} {'τ':<12}")
            print("-" * 75)
            
            for params in active_sorted[:5]:
                print(f"{params['layer']:<8} {params['position']:<8} {params['neuron_id']:<5} "
                      f"{params['importance']:<12.4f} {params['E']:<12.2e} {params['eta']:<12.2e} "
                      f"{params['tau']:<12.4e}")
        else:
            print("\nNo active neurons found (all below importance threshold).")
        
        results = {
            'model': trained_model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'physics_losses': physics_losses,
            'data_losses': data_losses,
            'importance_history': importance_history,
            'all_params': all_params,
            'active_neurons': all_active_neurons,  # Use the corrected active neurons list
            'pruned_count': pruned_count,
            'active_count': active_count,
            'test_loss_before': test_loss_before,
            'test_loss_after': test_loss_after,
            'total_neurons': trained_model.total_neurons
        }
        
        return results
    
    def plot_neuron_analysis(self, results):
        """Plot neuron importance and parameter analysis"""
        
        all_params = results['all_params']
        active_neurons = results['active_neurons']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Plot 1: Importance distribution
        importances = [p['importance'] for p in all_params]
        axes[0, 0].hist(importances, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(0.1, color='red', linestyle='--', label='Pruning Threshold')
        axes[0, 0].set_xlabel('Importance')
        axes[0, 0].set_ylabel('Number of Neurons')
        axes[0, 0].set_title('Importance Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: E parameter distribution (log scale)
        E_vals = [p['E'] for p in all_params]
        axes[0, 1].hist(np.log10(E_vals), bins=15, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('log₁₀(E) [Pa]')
        axes[0, 1].set_ylabel('Number of Neurons')
        axes[0, 1].set_title('Elastic Modulus Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: η parameter distribution (log scale)
        eta_vals = [p['eta'] for p in all_params]
        axes[0, 2].hist(np.log10(eta_vals), bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('log₁₀(η) [Pa·s]')
        axes[0, 2].set_ylabel('Number of Neurons')
        axes[0, 2].set_title('Viscosity Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Neuron importance by layer
        layers = [p['layer'] for p in all_params]
        layer_importances = {}
        for p in all_params:
            layer = p['layer']
            if layer not in layer_importances:
                layer_importances[layer] = []
            layer_importances[layer].append(p['importance'])
        
        layer_nums = list(layer_importances.keys())
        layer_means = [np.mean(layer_importances[l]) for l in layer_nums]
        layer_stds = [np.std(layer_importances[l]) for l in layer_nums]
        
        axes[1, 0].bar(layer_nums, layer_means, yerr=layer_stds, capsize=5, 
                      alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Average Importance')
        axes[1, 0].set_title('Average Importance by Layer')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Active vs Pruned neurons per layer
        layer_counts = {i: {'active': 0, 'pruned': 0} for i in range(results['model'].num_layers)}
        for p in all_params:
            layer = p['layer']
            if p['importance'] > 0.7:
                layer_counts[layer]['active'] += 1
            else:
                layer_counts[layer]['pruned'] += 1
        
        layers = list(layer_counts.keys())
        active_counts = [layer_counts[l]['active'] for l in layers]
        pruned_counts = [layer_counts[l]['pruned'] for l in layers]
        
        width = 0.35
        axes[1, 1].bar([l - width/2 for l in layers], active_counts, width, 
                      label='Active', alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].bar([l + width/2 for l in layers], pruned_counts, width, 
                      label='Pruned', alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Number of Neurons')
        axes[1, 1].set_title('Active vs Pruned Neurons by Layer')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: E vs η scatter plot (colored by importance)
        if len(active_neurons) > 0:
            E_active = [p['E'] for p in active_neurons]
            eta_active = [p['eta'] for p in active_neurons]
            imp_active = [p['importance'] for p in active_neurons]
            
            scatter = axes[1, 2].scatter(E_active, eta_active, c=imp_active, 
                                       cmap='viridis', alpha=0.7, s=60, edgecolors='black')
            axes[1, 2].set_xlabel('E (Pa)')
            axes[1, 2].set_ylabel('η (Pa·s)')
            axes[1, 2].set_title('E vs η (Active Neurons)')
            axes[1, 2].set_xscale('log')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 2], label='Importance')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def plot_training_analysis(self, results):
        """Plot training history and importance evolution"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Loss components over time
        train_losses = results['train_losses']
        val_losses = results['val_losses']
        physics_losses = results['physics_losses']
        data_losses = results['data_losses']
        
        epochs = range(len(train_losses))
        
        axes[0, 0].plot(epochs, train_losses, label='Total Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, label='Total Val Loss', linewidth=2)
        axes[0, 0].plot(epochs, physics_losses, label='Physics Loss', alpha=0.7, linestyle='--')
        axes[0, 0].plot(epochs, data_losses, label='Data Loss', alpha=0.7, linestyle='--')
        
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training History')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Active neuron count over time
        if 'importance_history' in results:
            importance_epochs = range(0, len(train_losses), 10)[:len(results['importance_history'])]
            avg_importance = results['importance_history']
            
            axes[0, 1].plot(importance_epochs, avg_importance, 'b-', linewidth=2, marker='o')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Average Importance')
            axes[0, 1].set_title('Average Neuron Importance Evolution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance comparison
        test_before = results['test_loss_before']
        test_after = results['test_loss_after']
        
        categories = ['Before Pruning', 'After Pruning']
        losses = [test_before, test_after]
        colors = ['blue', 'red']
        
        bars = axes[1, 0].bar(categories, losses, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_ylabel('Test Loss')
        axes[1, 0].set_title('Performance Before/After Pruning')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, loss in zip(bars, losses):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{loss:.4f}', ha='center', va='bottom')
        
        # Plot 4: Neuron count summary
        total_neurons = results['total_neurons']
        active_count = results['active_count']
        pruned_count = results['pruned_count']
        
        categories = ['Total', 'Active', 'Pruned']
        counts = [total_neurons, active_count, pruned_count]
        colors = ['gray', 'green', 'red']
        
        bars = axes[1, 1].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Number of Neurons')
        axes[1, 1].set_title('Neuron Count Summary')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return fig

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    # Load your data
    print("Loading data...")
    df = pd.read_csv('parallel_maxwell_ramp.csv')
    
    # Initialize experiment runner
    runner = Experiment2Runner(df, sequence_length=50, batch_size=32)
    
    # Run Experiment 2 - Multi-PINN Network with Pruning Analysis
    results = runner.run_experiment(
        neurons_per_layer=4, 
        num_layers=4, 
        physics_weight=0.5
    )
    
    # Plot neuron analysis
    fig1 = runner.plot_neuron_analysis(results)
    
    # Plot training analysis
    fig2 = runner.plot_training_analysis(results)
    
    return results

if __name__ == "__main__":
    results = main()
