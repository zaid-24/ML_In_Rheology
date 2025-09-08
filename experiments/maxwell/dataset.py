import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class MaxwellModelDataGenerator:
    """
    Generate synthetic rheological data for a Maxwell model
    Maxwell model: Spring (E) and Dashpot (η) in series
    """
    
    def __init__(self, E=1e6, eta=1e8, seed=42):
        """
        Initialize Maxwell model parameters
        
        Parameters:
        -----------
        E : float
            Elastic modulus (Pa)
        eta : float
            Viscosity (Pa·s)
        seed : int
            Random seed for reproducibility
        """
        self.E = E
        self.eta = eta
        self.tau = eta / E  # Relaxation time
        np.random.seed(seed)
        
    def maxwell_ode(self, state, t, stress_func, dstress_dt_func):
        """
        Maxwell model differential equation
        dε/dt = (1/E)·(dσ/dt) + σ/η
        
        Parameters:
        -----------
        state : array
            Current strain value
        t : float
            Time
        stress_func : function
            Function returning stress at time t
        dstress_dt_func : function
            Function returning stress derivative at time t
        """
        strain = state[0]
        stress = stress_func(t)
        dstress_dt = dstress_dt_func(t)
        
        # Maxwell equation
        dstrain_dt = dstress_dt / self.E + stress / self.eta
        
        return [dstrain_dt]
    
    def generate_creep_test(self, stress_level=1000, duration=100, n_points=500):
        """
        Generate data for creep test (constant stress)
        """
        t = np.linspace(0, duration, n_points)
        
        # Step stress function
        def stress_func(time):
            return stress_level if time > 0 else 0
        
        def dstress_dt_func(time):
            return 0  # Constant stress after step
        
        # Initial condition
        strain0 = [0]
        
        # Solve ODE
        solution = odeint(self.maxwell_ode, strain0, t, 
                         args=(stress_func, dstress_dt_func))
        
        strain = solution[:, 0]
        strain_rate = np.gradient(strain, t)
        stress = np.array([stress_func(ti) for ti in t])
        
        return t, strain, strain_rate, stress
    
    def generate_stress_relaxation(self, strain_level=0.01, duration=100, n_points=500):
        """
        Generate data for stress relaxation test (constant strain)
        """
        t = np.linspace(0, duration, n_points)
        
        # Constant strain
        strain = np.ones(n_points) * strain_level
        strain_rate = np.zeros(n_points)
        
        # Initial stress (elastic response)
        stress0 = self.E * strain_level
        
        # Stress relaxation: σ(t) = σ0 * exp(-t/τ)
        stress = stress0 * np.exp(-t / self.tau)
        
        return t, strain, strain_rate, stress
    
    def generate_oscillatory_test(self, amplitude=0.001, frequency=0.1, 
                                 duration=100, n_points=1000):
        """
        Generate data for oscillatory loading
        """
        t = np.linspace(0, duration, n_points)
        omega = 2 * np.pi * frequency
        
        # Sinusoidal strain
        strain = amplitude * np.sin(omega * t)
        strain_rate = amplitude * omega * np.cos(omega * t)
        
        # Complex modulus for Maxwell model
        G_storage = self.E * (omega * self.tau)**2 / (1 + (omega * self.tau)**2)
        G_loss = self.E * omega * self.tau / (1 + (omega * self.tau)**2)
        
        # Stress response
        stress = G_storage * strain + (G_loss / omega) * strain_rate
        
        return t, strain, strain_rate, stress
    
    def generate_ramp_loading(self, max_strain=0.02, ramp_rate=0.0001, 
                             hold_time=50, n_points=1000):
        """
        Generate data for ramp loading followed by hold
        """
        ramp_time = max_strain / ramp_rate
        total_time = ramp_time + hold_time
        t = np.linspace(0, total_time, n_points)
        
        # Strain profile
        strain = np.where(t <= ramp_time, ramp_rate * t, max_strain)
        strain_rate = np.where(t <= ramp_time, ramp_rate, 0)
        
        # Stress calculation
        stress = np.zeros(n_points)
        for i in range(1, n_points):
            dt = t[i] - t[i-1]
            # During ramp: elastic + viscous response
            if t[i] <= ramp_time:
                stress[i] = self.E * strain[i] * (1 - np.exp(-dt/self.tau)) + \
                           stress[i-1] * np.exp(-dt/self.tau) + \
                           self.eta * strain_rate[i]
            # During hold: stress relaxation
            else:
                stress[i] = stress[i-1] * np.exp(-dt/self.tau)
        
        return t, strain, strain_rate, stress
    
    def generate_random_loading(self, duration=200, n_points=2000, 
                               max_strain_rate=0.001):
        """
        Generate data with random loading conditions
        """
        t = np.linspace(0, duration, n_points)
        dt = t[1] - t[0]
        
        # Generate random strain rate with smoothing
        strain_rate = np.random.randn(n_points) * max_strain_rate
        strain_rate = np.convolve(strain_rate, np.ones(10)/10, mode='same')
        
        # Integrate to get strain
        strain = np.cumsum(strain_rate) * dt
        
        # Calculate stress using Maxwell model
        stress = np.zeros(n_points)
        for i in range(1, n_points):
            dstrain = strain[i] - strain[i-1]
            stress[i] = stress[i-1] * np.exp(-dt/self.tau) + \
                       self.E * dstrain
        
        return t, strain, strain_rate, stress
    
    def add_noise(self, data, noise_level=0.02):
        """
        Add Gaussian noise to simulate measurement errors
        """
        noise = np.random.randn(len(data)) * np.std(data) * noise_level
        return data + noise
    
    def generate_comprehensive_dataset(self, add_noise_flag=True, noise_level=0.01):
        """
        Generate a comprehensive dataset with multiple test types
        """
        all_data = []
        """
        # 1. Multiple creep tests
        for stress in [500, 1000, 1500, 2000]:
            t, strain, strain_rate, stress_arr = self.generate_creep_test(
                stress_level=stress, duration=100, n_points=300
            )
            test_data = pd.DataFrame({
                'time': t,
                'strain': strain,
                'strain_rate': strain_rate,
                'stress': stress_arr,
                'test_type': 'creep',
                'test_id': f'creep_{stress}Pa'
            })
            all_data.append(test_data)
        
        # 2. Multiple stress relaxation tests
        for strain in [0.005, 0.01, 0.015, 0.02]:
            t, strain_arr, strain_rate, stress = self.generate_stress_relaxation(
                strain_level=strain, duration=100, n_points=300
            )
            test_data = pd.DataFrame({
                'time': t,
                'strain': strain_arr,
                'strain_rate': strain_rate,
                'stress': stress,
                'test_type': 'relaxation',
                'test_id': f'relax_{strain:.3f}'
            })
            all_data.append(test_data)
        
        # 3. Oscillatory tests at different frequencies
        for freq in [0.01, 0.05, 0.1, 0.5, 1.0]:
            t, strain, strain_rate, stress = self.generate_oscillatory_test(
                frequency=freq, duration=50/freq, n_points=500
            )
            test_data = pd.DataFrame({
                'time': t,
                'strain': strain,
                'strain_rate': strain_rate,
                'stress': stress,
                'test_type': 'oscillatory',
                'test_id': f'osc_{freq}Hz'
            })
            all_data.append(test_data)

        # 5. Random loading tests
        for i in range(5):
            t, strain, strain_rate, stress = self.generate_random_loading(
                duration=200, n_points=1000
            )
            test_data = pd.DataFrame({
                'time': t,
                'strain': strain,
                'strain_rate': strain_rate,
                'stress': stress,
                'test_type': 'random',
                'test_id': f'random_{i}'
            })
            all_data.append(test_data)
        """
        # 4. Ramp tests with different rates
        for rate in [0.00005, 0.0001, 0.0002, 0.0005]:
            t, strain, strain_rate, stress = self.generate_ramp_loading(
                ramp_rate=rate, n_points=500
            )
            test_data = pd.DataFrame({
                'time': t,
                'strain': strain,
                'strain_rate': strain_rate,
                'stress': stress,
                'test_type': 'ramp',
                'test_id': f'ramp_{rate:.5f}'
            })
            all_data.append(test_data)
        
        
        
        # Combine all data
        df = pd.concat(all_data, ignore_index=True)
        
        # Add noise if requested
        if add_noise_flag:
            df['strain'] = self.add_noise(df['strain'].values, noise_level)
            df['strain_rate'] = self.add_noise(df['strain_rate'].values, noise_level)
            df['stress'] = self.add_noise(df['stress'].values, noise_level)
        
        # Add material parameters as features
        df['E_true'] = self.E
        df['eta_true'] = self.eta
        df['tau_true'] = self.tau
        
        return df
    
    def plot_sample_data(self, df, n_samples=4):
        """
        Plot sample data from the dataset
        """
        fig, axes = plt.subplots(3, n_samples, figsize=(15, 10))
        
        test_types = df['test_type'].unique()[:n_samples]
        
        for i, test_type in enumerate(test_types):
            test_data = df[df['test_type'] == test_type].iloc[:500]
            
            axes[0, i].plot(test_data['time'], test_data['strain'])
            axes[0, i].set_title(f'{test_type.capitalize()} Test')
            axes[0, i].set_ylabel('Strain')
            axes[0, i].grid(True, alpha=0.3)
            
            axes[1, i].plot(test_data['time'], test_data['strain_rate'])
            axes[1, i].set_ylabel('Strain Rate (1/s)')
            axes[1, i].grid(True, alpha=0.3)
            
            axes[2, i].plot(test_data['time'], test_data['stress'])
            axes[2, i].set_xlabel('Time (s)')
            axes[2, i].set_ylabel('Stress (Pa)')
            axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# Example usage
if __name__ == "__main__":
    # Create generator with specific Maxwell parameters
    generator = MaxwellModelDataGenerator(
        E=1e6,      # Elastic modulus: 1 MPa
        eta=1e8,    # Viscosity: 100 MPa·s
        seed=42
    )
    
    # Generate comprehensive dataset
    print("Generating comprehensive dataset...")
    df = generator.generate_comprehensive_dataset(
        add_noise_flag=True,
        noise_level=0.01
    )
    
    # Display dataset info
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nTest types included:")
    print(df.groupby('test_type').size())
    
    # Display first few rows
    print("\nFirst 10 rows of the dataset:")
    print(df.head(10))
    
    # Save to CSV
    output_file = 'maxwell_rheological_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to '{output_file}'")
    
    # Plot sample data
    print("\nGenerating sample plots...")
    fig = generator.plot_sample_data(df, n_samples=4)
    
    # Generate dataset with different parameters
    print("\n" + "="*50)
    print("Generating multi-parameter dataset for ML training...")
    
    multi_param_data = []
    
    # Vary E and eta to create diverse dataset
    E_values = np.logspace(5, 7, 5)  # 0.1 to 10 MPa
    eta_values = np.logspace(7, 9, 5)  # 10 to 1000 MPa·s
    
    for E in E_values:
        for eta in eta_values:
            gen = MaxwellModelDataGenerator(E=E, eta=eta)
            df_temp = gen.generate_comprehensive_dataset(
                add_noise_flag=True,
                noise_level=0.02
            )
            multi_param_data.append(df_temp)
    
    # Combine all parameter variations
    df_multi = pd.concat(multi_param_data, ignore_index=True)
    
    print(f"Multi-parameter dataset shape: {df_multi.shape}")
    print(f"Parameter ranges:")
    print(f"  E: {df_multi['E_true'].min():.2e} to {df_multi['E_true'].max():.2e} Pa")
    print(f"  η: {df_multi['eta_true'].min():.2e} to {df_multi['eta_true'].max():.2e} Pa·s")
    
    # Save multi-parameter dataset
    output_file_multi = 'maxwell_rheological_dataset_multi_param.csv'
    df_multi.to_csv(output_file_multi, index=False)
    print(f"\nMulti-parameter dataset saved to '{output_file_multi}'")
    
    # Statistical summary
    print("\n" + "="*50)
    print("Dataset Statistics:")
    print(df_multi[['strain', 'strain_rate', 'stress']].describe())