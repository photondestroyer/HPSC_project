import numpy as np
import matplotlib.pyplot as plt

def green_ampt_infiltration(K_s, psi, theta_e, theta_i, duration_hours, dt_min=1):
    """
    Calculates Green-Ampt infiltration rate and cumulative infiltration.
    
    Args:
        K_s (float): Saturated hydraulic conductivity (mm/hr)
        psi (float): Wetting front suction head (mm)
        theta_e (float): Effective porosity
        theta_i (float): Initial moisture content
        duration_hours (float): Duration of simulation in hours
        dt_min (float): Time step in minutes
        
    Returns:
        dict: Time, Infiltration Rate (f), Cumulative Infiltration (F)
    """
    dt = dt_min / 60.0 # Convert time step to hours
    steps = int(duration_hours / dt)
    
    # Moisture deficit
    d_theta = theta_e - theta_i
    
    # Initialize arrays
    F = np.zeros(steps) # Cumulative infiltration (mm)
    f = np.zeros(steps) # Infiltration rate (mm/hr)
    time = np.zeros(steps)
    
    # Initial guess for F at first time step (assume close to K_s * dt)
    # Using implicit solution F = K*t + psi*d_theta * ln(1 + F/(psi*d_theta))
    # We solve iteratively for F at each time step.
    
    F_current = 0.0
    
    for t in range(1, steps):
        current_time = t * dt
        time[t] = current_time
        
        # Implicit solution for Green-Ampt Cumulative Infiltration F(t):
        # t = (F - psi * d_theta * ln(1 + F / (psi * d_theta))) / K_s
        # We need to find F such that the equation holds for current_time.
        # Using Newton-Raphson for root finding.
        
        # Function g(F) = F - psi*d_theta*ln(1 + F/(psi*d_theta)) - K_s*current_time = 0
        # Derivative g'(F) = 1 - 1/(1 + F/(psi*d_theta)) = F / (F + psi*d_theta)
        
        # Initial guess for F (previous F + K_s * dt)
        F_guess = F_current + K_s * dt 
        
        for _ in range(10): # 10 iterations usually sufficient
            g_val = F_guess - psi * d_theta * np.log(1 + F_guess / (psi * d_theta)) - K_s * current_time
            g_prime = F_guess / (F_guess + psi * d_theta)
            
            if abs(g_val) < 1e-5:
                break
                
            F_guess = F_guess - g_val / g_prime
            
        F[t] = F_guess
        F_current = F_guess
        
        # Calculate rate f = K_s * (1 + (psi * d_theta) / F)
        if F_current > 0:
            f[t] = K_s * (1 + (psi * d_theta) / F_current)
        else:
            f[t] = K_s * 100 # Theoretically infinity at t=0
            
    return {"time": time, "rate": f, "cumulative": F}

# --- Example Usage for Silty Clay Loam ---

# Parameters (Rawls et al. 1983 for Silty Clay Loam)
Ks_val = 1.5      # mm/hr (Conservative estimate)
psi_val = 273.0   # mm
porosity = 0.432  # effective porosity
init_moisture = 0.25 # assume soil is partially dry

results = green_ampt_infiltration(Ks_val, psi_val, porosity, init_moisture, duration_hours=5)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(results["time"], results["rate"], label='Infiltration Rate (mm/hr)', color='blue', linewidth=2)
plt.axhline(y=Ks_val, color='red', linestyle='--', label=f'Ks (Steady State): {Ks_val} mm/hr')
plt.title(f'Green-Ampt Infiltration Curve: Silty Clay Loam\n(Ks={Ks_val} mm/hr, Psi={psi_val} mm)', fontsize=14)
plt.xlabel('Time (hours)', fontsize=12)
plt.ylabel('Infiltration Rate (mm/hr)', fontsize=12)
plt.ylim(0, 50) # Limit y-axis to see the decay clearly
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.legend()
plt.show()

