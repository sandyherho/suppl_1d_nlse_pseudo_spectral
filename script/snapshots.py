#!/usr/bin/env python
"""
Evolution Snapshots
Displays wave intensity profiles |ψ|² at key time points for all scenarios
EPS-compatible version with uniform axis limits
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.signal import find_peaks
import os
from pathlib import Path

# Set publication quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5

# Create output directories
Path('../figs').mkdir(parents=True, exist_ok=True)
Path('../stats').mkdir(parents=True, exist_ok=True)

# Load data from all scenarios
scenarios = {
    'single_soliton': 'Single Soliton',
    'two_soliton_collision': 'Two-Soliton',
    'breather_solution': 'Breather',
    'modulation_instability': 'Modulation'
}

# Time points to display (fractions of total time)
time_fractions = [0, 1/3, 2/3, 1]
time_labels = ['t = 0', 't = T/3', 't = 2T/3', 't = T']

# Color scheme for time evolution (solid colors, no transparency for EPS)
time_colors = ['#2E4057', '#048A81', '#54C6EB', '#ED6A5A']

# Storage for finding global limits
all_x_data = []
all_intensity_data = []

# First pass: load all data to find global limits
print("Loading data for uniform scaling...")
for scenario_file, scenario_name in scenarios.items():
    nc_path = f'../outputs/{scenario_file}.nc'
    if os.path.exists(nc_path):
        nc = Dataset(nc_path, 'r')
        x = nc.variables['x'][:]
        t = nc.variables['t'][:]
        psi_abs = nc.variables['psi_abs'][:]
        nc.close()
        
        # Convert to intensity |ψ|²
        intensity = psi_abs ** 2
        
        all_x_data.append(x)
        all_intensity_data.append(intensity)

# Determine uniform axis limits
x_min = min(x.min() for x in all_x_data)
x_max = max(x.max() for x in all_x_data)
y_max = max(intensity.max() for intensity in all_intensity_data) * 1.1

print(f"Uniform limits: x=[{x_min:.2f}, {x_max:.2f}], y=[0, {y_max:.3f}]")

# Initialize statistics
stats_text = []
stats_text.append("="*70)
stats_text.append("Evolution Snapshots Analysis")
stats_text.append("Physical Quantity: |ψ|² (Intensity/Power Density)")
stats_text.append("="*70)

# Create 4x4 grid figure
fig, axes = plt.subplots(4, 4, figsize=(14, 12))

# Second pass: create plots with uniform limits
for row_idx, (scenario_file, scenario_name) in enumerate(scenarios.items()):
    nc_path = f'../outputs/{scenario_file}.nc'
    if os.path.exists(nc_path):
        nc = Dataset(nc_path, 'r')
        x = nc.variables['x'][:]
        t = nc.variables['t'][:]
        psi_abs = nc.variables['psi_abs'][:]
        psi_real = nc.variables['psi_real'][:]
        psi_imag = nc.variables['psi_imag'][:]
        nc.close()
        
        # Convert to intensity |ψ|²
        intensity = psi_abs ** 2
        
        # Calculate time indices
        n_times = len(t)
        time_indices = [int(frac * (n_times - 1)) for frac in time_fractions]
        
        # Store statistics for this scenario
        stats_text.append(f"\n{scenario_name}:")
        
        # Plot snapshots at different times
        for col_idx, (t_idx, t_label, color) in enumerate(zip(time_indices, time_labels, time_colors)):
            ax = axes[row_idx, col_idx]
            
            # Plot |ψ|² profile (no transparency for EPS compatibility)
            ax.plot(x, intensity[t_idx], color=color, linewidth=2)
            
            # Optional: Add very light fill without alpha (using lighter color)
            # Skip fill_between for cleanest EPS output
            
            # Set uniform limits
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([0, y_max])
            
            # Grid (dotted style instead of alpha for EPS)
            ax.grid(True, linestyle=':', linewidth=0.5)
            
            # Labels
            if row_idx == 0:
                ax.set_title(t_label, fontsize=14, pad=5, fontweight='bold')
            if row_idx == 3:
                ax.set_xlabel('Position $x$', fontsize=14, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{scenario_name}\n' + r'$|\psi|^2$', fontsize=14, fontweight='bold')
            
            # Calculate and record statistics for this snapshot
            if col_idx == 0:
                stats_text.append(f"  Time t={t[t_idx]:.3f} ({t_label}):")
            else:
                stats_text.append(f"  Time t={t[t_idx]:.3f} ({t_label}):")
            
            # Wave packet properties (using intensity)
            max_intensity = np.max(intensity[t_idx])
            mean_intensity = np.mean(intensity[t_idx])
            total_power = np.trapz(intensity[t_idx], x)  # ∫|ψ|²dx
            
            # Find peaks in intensity profile
            peaks, properties = find_peaks(intensity[t_idx], height=0.1*max_intensity)
            n_peaks = len(peaks)
            
            # Peak positions and widths
            if n_peaks > 0:
                peak_positions = x[peaks]
                peak_intensities = intensity[t_idx][peaks]
                # Estimate width at half maximum for main peak
                main_peak_idx = peaks[np.argmax(peak_intensities)]
                half_max = intensity[t_idx][main_peak_idx] / 2
                # Find width
                left_idx = main_peak_idx
                right_idx = main_peak_idx
                while left_idx > 0 and intensity[t_idx][left_idx] > half_max:
                    left_idx -= 1
                while right_idx < len(x)-1 and intensity[t_idx][right_idx] > half_max:
                    right_idx += 1
                fwhm = x[right_idx] - x[left_idx]
            else:
                fwhm = 0
            
            # Energy components (Hamiltonian terms)
            psi_complex = psi_real[t_idx] + 1j * psi_imag[t_idx]
            dx = x[1] - x[0]
            
            # Kinetic energy: ∫|∂ψ/∂x|²dx
            psi_x = np.gradient(psi_complex, dx)
            kinetic = np.trapz(np.abs(psi_x)**2, x)
            
            # Potential energy (nonlinear term): ∫|ψ|⁴dx = ∫(|ψ|²)²dx
            potential = np.trapz(intensity[t_idx]**2, x)
            
            # Total energy (conserved quantity)
            total_energy = kinetic - 0.5 * potential  # For focusing NLSE
            
            stats_text.append(f"    Max intensity: {max_intensity:.3f}")
            stats_text.append(f"    Mean intensity: {mean_intensity:.3f}")
            stats_text.append(f"    Total power (∫|ψ|²dx): {total_power:.3f}")
            stats_text.append(f"    Number of peaks: {n_peaks}")
            if n_peaks > 0:
                stats_text.append(f"    Main peak FWHM: {fwhm:.3f}")
            stats_text.append(f"    Kinetic energy: {kinetic:.3f}")
            stats_text.append(f"    Potential energy: {potential:.3f}")
            stats_text.append(f"    Total energy: {total_energy:.3f}")

# Adjust layout
plt.tight_layout()

# Save figures in multiple formats (EPS-friendly)
print("Saving figures...")
plt.savefig('../figs/evolution_snapshots.png', dpi=300, bbox_inches='tight')
plt.savefig('../figs/evolution_snapshots.pdf', bbox_inches='tight')
plt.savefig('../figs/evolution_snapshots.eps', format='eps', bbox_inches='tight')
plt.close()

# Add interpretation
stats_text.append("\n" + "="*70)
stats_text.append("Interpretation of Intensity Evolution:")
stats_text.append("="*70)
stats_text.append("""
Intensity (|ψ|²) Evolution Analysis:

PHYSICAL SIGNIFICANCE:
- |ψ|² represents power density in nonlinear optics
- |ψ|² is the observable quantity (what photodetectors measure)
- Total power ∫|ψ|²dx is conserved in lossless systems
- Peak intensity determines nonlinear effects strength

1. Single Soliton:
   - Constant peak intensity (fundamental soliton property)
   - Shape preservation: sech² intensity profile maintained
   - Center of mass motion at constant velocity
   - FWHM remains constant (no dispersion)
   - Kinetic-potential energy balance preserved

2. Two-Soliton Collision:
   - Initial: Two separated intensity peaks
   - T/3: Peaks approaching (pre-collision)
   - 2T/3: Collision event - temporary intensity redistribution
   - Final: Elastic collision - solitons emerge unchanged
   - Total power conserved throughout collision
   - Phase shift occurs but intensity profiles preserved

3. Breather Solution:
   - Periodic intensity oscillation (breathing mode)
   - Peak intensity varies between maximum and minimum
   - Width anti-correlated with peak intensity
   - Total power conserved despite local variations
   - Energy periodically exchanged between center and wings
   - FWHM modulation indicates pulsation

4. Modulation Instability:
   - Initial: Nearly uniform intensity with seed perturbations
   - T/3: Exponential growth of intensity modulations
   - 2T/3: Nonlinear saturation and pattern formation
   - Final: Train of intensity peaks (soliton train)
   - Benjamin-Feir instability mechanism
   - Energy flows from continuous wave to sidebands

CONSERVATION LAWS:
- Power: ∫|ψ|²dx = constant
- Energy: E = ∫[|∂ψ/∂x|² - ½|ψ|⁴]dx = constant
- Momentum: P = -i∫ψ*∂ψ/∂x dx = constant

NONLINEAR DYNAMICS:
- Balance: Kerr nonlinearity ∝ |ψ|² balances dispersion
- Threshold: Critical power for self-focusing
- Stability: Solitons are attractors in phase space
- Interactions: Phase-sensitive but intensity-preserving

EXPERIMENTAL RELEVANCE:
- Intensity directly measurable by photodetectors
- Peak intensity determines damage threshold
- FWHM relates to pulse duration (time domain)
- Pattern formation visible in beam cross-section
""")

# Save statistics to file
with open('../stats/evolution_snapshots.txt', 'w') as f:
    f.write('\n'.join(stats_text))

print("Evolution snapshots saved to ../figs/")
print("Statistics saved to ../stats/evolution_snapshots.txt")
print(f"Uniform axis limits applied: x=[{x_min:.2f}, {x_max:.2f}], y_max={y_max:.3f}")
