import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
E = 200.0   # Young's modulus
T = 10.0    # total time
dt = 0.01   # time step
t = np.arange(0, T, dt)

# Define stress history (choose any loading condition)
# Example 1: sinusoidal stress
stress = 50 * np.sin(2 * np.pi * 0.5 * t)  

# Example 2: uncomment for ramp stress
# stress = 5 * t  

# Example 3: uncomment for step stress
# stress = np.where(t < 5, 0, 100)

# Compute strain from Hooke’s law
strain = stress / E

# Strain rate (numerical derivative)
strain_rate = np.gradient(strain, dt)

# Store in DataFrame
df = pd.DataFrame({
    "time": t,
    "strain": strain,
    "strain_rate": strain_rate,
    "stress": stress
})

# Save to CSV
df.to_csv("spring_constitutive_data.csv", index=False)

print("Data generated using spring constitutive equation and saved to spring_constitutive_data.csv")

# Plot stress-strain curve
plt.figure(figsize=(6,4))
plt.plot(strain, stress, label="Spring σ=Eε")
plt.xlabel("Strain")
plt.ylabel("Stress")
plt.title("Stress-Strain Curve for Spring")
plt.legend()
plt.show()
