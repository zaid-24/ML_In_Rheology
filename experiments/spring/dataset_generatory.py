import numpy as np
import pandas as pd
import math

# Parameters
E = 200.0   # Spring constant (Young's modulus)
T = 1000    # total time (seconds)
dt = 1   # time step
t = np.arange(0, T, dt)

# Generate synthetic strain (sinusoidal + some noise)
strain = abs(0.02 * np.sin(2 * np.pi * 1.0 * t) + 0.005 * np.random.randn(len(t)))*100

# Strain rate (numerical derivative)
strain_rate = np.gradient(strain, dt)

# Stress from Hooke’s law
stress = E * strain

# Store in DataFrame
df = pd.DataFrame({
    "time": t,
    "strain": strain,
    "strain_rate": strain_rate,
    "stress": stress
})

# Save to CSV
df.to_csv("spring_data.csv", index=False)

print("Synthetic spring data generated and saved to spring_data.csv")
print(df.head())
