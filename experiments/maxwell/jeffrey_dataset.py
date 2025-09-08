import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class ParallelDashpotMaxwellGenerator:
    """
    Generator for the rheology architecture:
       parallel branch 1: dashpot eta1 (stress = eta1 * strain_rate)
       parallel branch 2: Maxwell branch: spring C2 in series with dashpot eta2
           -> Maxwell eqn for branch stress sigma2:
              d(sigma2)/dt + (C2/eta2) * sigma2 = C2 * strain_rate
    Total stress = sigma1 + sigma2
    """

    def __init__(self, C2=1e6, eta1=5e7, eta2=1e8, seed=42):
        self.C2 = float(C2)
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)
        self.tau2 = self.eta2 / self.C2
        np.random.seed(seed)

    @staticmethod
    def _strain_profile_ramp_and_hold(max_strain, ramp_rate, hold_time, n_points):
        ramp_time = max_strain / ramp_rate
        total_time = ramp_time + hold_time
        t = np.linspace(0.0, total_time, int(n_points))
        dt = t[1] - t[0]
        strain = np.where(t <= ramp_time, ramp_rate * t, max_strain)
        # compute a smooth strain_rate (use gradient to avoid discontinuity at the hold point)
        strain_rate = np.gradient(strain, dt)
        return t, strain, strain_rate

    def _sigma2_ode_rhs(self, t, sigma2, strain_rate_func):
        """
        Right-hand side of Maxwell-branch ODE:
            dσ2/dt = C2 * strain_rate(t) - (C2/eta2) * σ2
        """
        eps_dot = strain_rate_func(t)
        return self.C2 * eps_dot - (self.C2 / self.eta2) * sigma2

    def generate_ramp_dataset(self,
                              max_strain=0.02,
                              ramp_rate=1e-4,
                              hold_time=50.0,
                              n_points=2000,
                              noise_level=0.0,
                              save_csv="rheology_parallel_maxwell.csv"):
        """
        Generate dataset for ramp-and-hold strain input.

        Returns:
            df : pandas.DataFrame with columns
                time, strain, strain_rate, stress, stress_eta1, stress_maxwell_branch
        """
        t, strain, strain_rate = self._strain_profile_ramp_and_hold(
            max_strain, ramp_rate, hold_time, n_points
        )

        # create callable interpolation for strain_rate for the ODE solver
        # use linear interpolation (np.interp) inside a lambda
        def strain_rate_func(tt):
            return np.interp(tt, t, strain_rate)

        # initial condition: sigma2(0) = 0 (Maxwell spring/dashpot initially unstressed)
        sigma2_initial = 0.0

        # integrate the sigma2 ODE over the time span
        sol = solve_ivp(fun=lambda tt, s: self._sigma2_ode_rhs(tt, s, strain_rate_func),
                        t_span=(t[0], t[-1]),
                        y0=[sigma2_initial],
                        t_eval=t,
                        method="RK45",  # good general purpose; switch to 'BDF' for stiff systems
                        atol=1e-9, rtol=1e-7)

        sigma2 = sol.y[0]

        # dashpot branch stress
        sigma_eta1 = self.eta1 * strain_rate

        # total stress
        stress = sigma_eta1 + sigma2

        # Add optional noise to mimic measurements (applied to stress only here)
        if noise_level and noise_level > 0.0:
            noise = np.random.randn(len(stress)) * (np.std(stress) * noise_level + 1e-12)
            stress_noisy = stress + noise
        else:
            stress_noisy = stress.copy()

        df = pd.DataFrame({
            "time": t,
            "strain": strain,
            "strain_rate": strain_rate,
            "stress": stress_noisy,
            "stress_eta1": sigma_eta1,
            "stress_maxwell_branch": sigma2
        })

        if save_csv:
            df.to_csv(save_csv, index=False)

        return df

# --------------------------
# Example usage and quick plot
# --------------------------
if __name__ == "__main__":
    gen = ParallelDashpotMaxwellGenerator(C2=1e6, eta1=5e7, eta2=1e8, seed=123)

    df = gen.generate_ramp_dataset(
        max_strain=0.02,
        ramp_rate=1e-4,
        hold_time=50.0,
        n_points=2000,
        noise_level=0.01,           # small measurement noise
        save_csv="parallel_maxwell_ramp.csv"
    )

    print(df.head())

    # quick diagnostic plots
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(df.time, df.strain, label="strain (input)")
    plt.plot(df.time, df.stress, label="total stress (measured)")
    plt.plot(df.time, df.stress_maxwell_branch, '--', label="sigma2 (Maxwell branch)")
    plt.plot(df.time, df.stress_eta1, ':', label="sigma1 (dashpot eta1)")
    plt.xlabel("time (s)"); plt.legend(); plt.title("Time domain")

    plt.subplot(1,2,2)
    plt.plot(df.strain, df.stress, label="σ vs ε (total)")
    plt.xlabel("strain"); plt.ylabel("stress"); plt.title("Stress vs Strain")
    plt.legend()

    plt.tight_layout()
    plt.show()
