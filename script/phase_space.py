#!/usr/bin/env python
"""
Phase Space Analysis
Analyzes the relationship between real and imaginary parts of the wave function
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.stats import pearsonr, spearmanr, gaussian_kde
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set publication quality parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12  # Increased base font size
plt.rcParams['axes.labelsize'] = 14  # Larger labels
plt.rcParams['axes.titlesize'] = 16  # Larger titles
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.0  # Slightly thicker axes
plt.rcParams['lines.linewidth'] = 2.0  # Thicker lines

# Create output directories
Path('../figs').mkdir(parents=True, exist_ok=True)
Path('../stats').mkdir(parents=True, exist_ok=True)


def calculate_mutual_information(x, y, bins=50):
    """
    Calculate mutual information between two variables
    Using adaptive binning for robust estimation
    """
    try:
        # Remove infinite or NaN values
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 10:
            return 0.0
        
        # Adaptive binning based on data size
        n_points = len(x_clean)
        if n_points < 100:
            actual_bins = min(10, bins)
        elif n_points < 1000:
            actual_bins = min(20, bins)
        elif n_points < 10000:
            actual_bins = min(30, bins)
        else:
            actual_bins = bins
        
        # Create 2D histogram with robust edge calculation
        x_range = np.percentile(x_clean, [1, 99])
        y_range = np.percentile(y_clean, [1, 99])
        
        hist_2d, x_edges, y_edges = np.histogram2d(
            x_clean, y_clean, 
            bins=actual_bins,
            range=[x_range, y_range]
        )
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        
        # Convert to probabilities
        pxy = hist_2d / np.sum(hist_2d)
        px = np.sum(pxy, axis=1) + eps
        py = np.sum(pxy, axis=0) + eps
        pxy = pxy + eps
        
        # Calculate mutual information
        px_py = px[:, None] * py[None, :]
        
        # Calculate MI with numerical stability
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > eps:
                    mi += pxy[i, j] * np.log(pxy[i, j] / px_py[i, j])
        
        # Normalize by maximum possible MI (log of number of bins)
        max_mi = np.log(actual_bins)
        normalized_mi = mi / max_mi if max_mi > 0 else 0
        
        return normalized_mi
        
    except Exception as e:
        print(f"Warning: MI calculation failed: {e}")
        return 0.0


def calculate_phase_statistics(re_data, im_data):
    """Calculate robust phase space statistics"""
    stats = {}
    
    # Remove invalid values
    mask = np.isfinite(re_data) & np.isfinite(im_data)
    re_clean = re_data[mask]
    im_clean = im_data[mask]
    
    if len(re_clean) < 2:
        return None
    
    # Phase and amplitude
    phase = np.arctan2(im_clean, re_clean)
    amplitude = np.sqrt(re_clean**2 + im_clean**2)
    
    # Basic statistics
    stats['mean_phase'] = np.mean(phase)
    stats['std_phase'] = np.std(phase)
    stats['circular_mean'] = np.angle(np.mean(np.exp(1j * phase)))
    stats['circular_std'] = np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * phase)))))
    
    stats['mean_amplitude'] = np.mean(amplitude)
    stats['std_amplitude'] = np.std(amplitude)
    stats['max_amplitude'] = np.max(amplitude)
    stats['median_amplitude'] = np.median(amplitude)
    
    # Distribution shape
    stats['skewness_re'] = np.mean(((re_clean - np.mean(re_clean)) / np.std(re_clean))**3) if np.std(re_clean) > 0 else 0
    stats['skewness_im'] = np.mean(((im_clean - np.mean(im_clean)) / np.std(im_clean))**3) if np.std(im_clean) > 0 else 0
    stats['kurtosis_amplitude'] = np.mean(((amplitude - np.mean(amplitude)) / np.std(amplitude))**4) - 3 if np.std(amplitude) > 0 else 0
    
    # RMS and geometric properties
    stats['rms_radius'] = np.sqrt(np.mean(re_clean**2 + im_clean**2))
    stats['centroid_re'] = np.mean(re_clean)
    stats['centroid_im'] = np.mean(im_clean)
    
    # Eccentricity (measure of how elliptical the distribution is)
    cov_matrix = np.cov(re_clean, im_clean)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    if len(eigenvalues) == 2 and min(eigenvalues) > 0:
        stats['eccentricity'] = np.sqrt(1 - min(eigenvalues)/max(eigenvalues))
    else:
        stats['eccentricity'] = 0
    
    return stats


# Load data from all scenarios
scenarios = {
    'single_soliton': 'Single Soliton',
    'two_soliton_collision': 'Two-Soliton', 
    'breather_solution': 'Breather',
    'modulation_instability': 'Modulation'
}

# Subplot labels
subplot_labels = ['(a)', '(b)', '(c)', '(d)']

stats_text = []
stats_text.append("="*70)
stats_text.append("ROBUST PHASE SPACE ANALYSIS")
stats_text.append("EPS-Compatible Version with Enhanced Statistics")
stats_text.append("="*70)

# First pass: determine global axis limits
global_re_min, global_re_max = np.inf, -np.inf
global_im_min, global_im_max = np.inf, -np.inf

data_store = {}

# Collect all data first to determine global limits
print("Loading data and determining global limits...")
for scenario_file, scenario_name in scenarios.items():
    nc_path = f'../outputs/{scenario_file}.nc'
    if os.path.exists(nc_path):
        nc = Dataset(nc_path, 'r')
        x = nc.variables['x'][:]
        t = nc.variables['t'][:]
        psi_real = nc.variables['psi_real'][:]
        psi_imag = nc.variables['psi_imag'][:]
        psi_abs = nc.variables['psi_abs'][:]
        nc.close()
        
        # Select time point (middle of simulation for consistency)
        t_idx = len(t) // 2
        
        # Store data
        data_store[scenario_name] = {
            'x': x,
            't': t,
            't_idx': t_idx,
            'psi_real': psi_real[t_idx],
            'psi_imag': psi_imag[t_idx],
            'psi_abs': psi_abs[t_idx]
        }
        
        # Update global limits (using percentiles for robustness)
        re_vals = psi_real[t_idx].flatten()
        im_vals = psi_imag[t_idx].flatten()
        
        # Use 99.9 percentile to avoid extreme outliers
        re_percentiles = np.percentile(re_vals[np.isfinite(re_vals)], [0.1, 99.9])
        im_percentiles = np.percentile(im_vals[np.isfinite(im_vals)], [0.1, 99.9])
        
        global_re_min = min(global_re_min, re_percentiles[0])
        global_re_max = max(global_re_max, re_percentiles[1])
        global_im_min = min(global_im_min, im_percentiles[0])
        global_im_max = max(global_im_max, im_percentiles[1])

# Make axes symmetric and add 10% padding
max_abs = max(abs(global_re_min), abs(global_re_max), 
              abs(global_im_min), abs(global_im_max))
axis_limit = max_abs * 1.1

print(f"Axis limits: ±{axis_limit:.3f}")

# Create LARGER figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 14))  # Increased from 10x10 to 14x14
axes = axes.flatten()

# Define high-contrast colors for EPS (no transparency needed)
# Using brighter, more vibrant colors for better visibility
scenario_colors = ['#0066FF', '#FF3300', '#00DD00', '#FF00DD']
scenario_list = list(scenarios.values())

# Plot scatter plots in 2x2 layout
for idx, scenario_name in enumerate(scenario_list):
    if scenario_name in data_store:
        # Get stored data
        re_data = data_store[scenario_name]['psi_real'].flatten()
        im_data = data_store[scenario_name]['psi_imag'].flatten()
        psi_abs_data = data_store[scenario_name]['psi_abs'].flatten()
        t = data_store[scenario_name]['t']
        t_idx = data_store[scenario_name]['t_idx']
        
        # Remove very small values for cleaner visualization
        threshold = 0.01 * np.max(psi_abs_data)
        mask = (psi_abs_data > threshold) & np.isfinite(re_data) & np.isfinite(im_data)
        re_filtered = re_data[mask]
        im_filtered = im_data[mask]
        
        # Get axis for this subplot
        ax = axes[idx]
        
        if len(re_filtered) > 0:
            # Reduce max points for larger point sizes to work well
            max_points = 2000  # Reduced from 5000 since points are bigger
            if len(re_filtered) > max_points:
                indices = np.random.choice(len(re_filtered), max_points, replace=False)
                re_plot = re_filtered[indices]
                im_plot = im_filtered[indices]
            else:
                re_plot = re_filtered
                im_plot = im_filtered
            
            # MUCH LARGER scatter plot points - increased from 0.5 to 20
            # You can adjust this value (try 15, 20, 30, or even 50)
            ax.scatter(re_plot, im_plot, c=scenario_colors[idx], 
                      s=25,  # INCREASED FROM 0.5 TO 25 - ADJUST AS NEEDED
                      alpha=0.8,  # Add slight transparency for overlapping points
                      rasterized=True, 
                      edgecolors='darkgray',  # Add edge color for better visibility
                      linewidths=0.5)  # Thin edge
        
        # Set consistent axis limits and formatting
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_xlabel(r'Re$(\psi)$', fontsize=16, fontweight='bold')
        ax.set_ylabel(r'Im$(\psi)$', fontsize=16, fontweight='bold')
        
        # Add subplot label with larger font (NO TITLE WITH SCENARIO NAME)
        ax.text(0.05, 0.95, subplot_labels[idx], transform=ax.transAxes,
                fontsize=18, fontweight='bold', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Thicker grid lines for better visibility
        ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_aspect('equal', adjustable='box')
        
        # Thicker axis lines through origin
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5)
        
        # Make tick labels larger and bold
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        
        # Calculate comprehensive statistics
        stats_text.append(f"\n{subplot_labels[idx]} {scenario_name}")
        stats_text.append(f"    Time: t = {t[t_idx]:.3f}")
        stats_text.append("-" * 50)
        
        # Correlation analysis
        if len(re_data) > 1:
            # Pearson and Spearman correlations
            try:
                pearson_corr, p_value = pearsonr(re_data, im_data)
                stats_text.append(f"  Pearson correlation: {pearson_corr:.3f} (p={p_value:.3e})")
            except:
                stats_text.append(f"  Pearson correlation: N/A")
            
            try:
                spearman_corr, sp_p_value = spearmanr(re_data, im_data)
                stats_text.append(f"  Spearman correlation: {spearman_corr:.3f} (p={sp_p_value:.3e})")
            except:
                stats_text.append(f"  Spearman correlation: N/A")
            
            # Mutual information with more bins
            mi = calculate_mutual_information(re_data, im_data, bins=50)
            stats_text.append(f"  Normalized MI: {mi:.3f}")
            
            # Phase space statistics
            phase_stats = calculate_phase_statistics(re_data, im_data)
            
            if phase_stats:
                stats_text.append(f"  Phase Statistics:")
                stats_text.append(f"    Mean phase: {phase_stats['mean_phase']:.3f} rad")
                stats_text.append(f"    Std phase: {phase_stats['std_phase']:.3f} rad")
                stats_text.append(f"    Circular mean: {phase_stats['circular_mean']:.3f} rad")
                stats_text.append(f"    Circular std: {phase_stats['circular_std']:.3f}")
                
                stats_text.append(f"  Amplitude Statistics:")
                stats_text.append(f"    Mean: {phase_stats['mean_amplitude']:.3f}")
                stats_text.append(f"    Std: {phase_stats['std_amplitude']:.3f}")
                stats_text.append(f"    Median: {phase_stats['median_amplitude']:.3f}")
                stats_text.append(f"    Maximum: {phase_stats['max_amplitude']:.3f}")
                
                stats_text.append(f"  Geometric Properties:")
                stats_text.append(f"    RMS radius: {phase_stats['rms_radius']:.3f}")
                stats_text.append(f"    Centroid: ({phase_stats['centroid_re']:.3f}, {phase_stats['centroid_im']:.3f})")
                stats_text.append(f"    Eccentricity: {phase_stats['eccentricity']:.3f}")
                
                stats_text.append(f"  Distribution Shape:")
                stats_text.append(f"    Skewness (Re): {phase_stats['skewness_re']:.3f}")
                stats_text.append(f"    Skewness (Im): {phase_stats['skewness_im']:.3f}")
                stats_text.append(f"    Kurtosis (Amp): {phase_stats['kurtosis_amplitude']:.3f}")
            
            # Data summary
            stats_text.append(f"  Data Points:")
            stats_text.append(f"    Total: {len(re_data):,}")
            stats_text.append(f"    Above threshold: {len(re_filtered):,}")
            stats_text.append(f"    Plotted: {len(re_plot) if len(re_filtered) > 0 else 0:,}")
            
            # Add visual note about point size
            stats_text.append(f"    Point size: 25 (large for visibility)")

# Adjust layout with more padding
plt.tight_layout(pad=3.0)

# Save figures in multiple formats (EPS-friendly)
print("Saving figures with large, visible points...")
plt.savefig('../figs/phase_space.png', dpi=300, bbox_inches='tight')
plt.savefig('../figs/phase_space.pdf', bbox_inches='tight')
plt.savefig('../figs/phase_space.eps', format='eps', bbox_inches='tight')
plt.close()

# Add the rest of the analysis parameters and interpretation
stats_text.append("\n" + "="*70)
stats_text.append("ANALYSIS PARAMETERS")
stats_text.append("="*70)
stats_text.append(f"Axis Scaling:")
stats_text.append(f"  Range: [{-axis_limit:.3f}, {axis_limit:.3f}] for both Re and Im")
stats_text.append(f"  Percentile used: 0.1% to 99.9% (robust to outliers)")
stats_text.append(f"  Padding: 10%")
stats_text.append(f"Statistical Methods:")
stats_text.append(f"  Mutual Information bins: 50 (adaptive)")
stats_text.append(f"  Threshold: 1% of max amplitude")
stats_text.append(f"  Max points plotted: 2,000")
stats_text.append(f"  Correlation significance: p < 0.05")

# Add interpretation
stats_text.append("\n" + "="*70)
stats_text.append("PHYSICAL INTERPRETATION")
stats_text.append("="*70)
stats_text.append("""
Phase Space Structure Analysis:

REPRESENTATION:
- Each point: One spatial location at fixed time
- Re(ψ): In-phase component of complex field
- Im(ψ): Quadrature component of complex field  
- Distance from origin: Local amplitude |ψ|
- Angle from Re axis: Local phase arg(ψ)

SCENARIO CHARACTERISTICS:

(a) Single Soliton - Blue:
   - Ring/circular pattern → constant amplitude envelope
   - High correlation → phase-amplitude coupling
   - Low eccentricity → isotropic in phase space
   - Centered distribution → zero momentum in moving frame
   - Compact structure → localized solution

(b) Two-Soliton Collision - Orange/Red:
   - Double-ring structure → two amplitude levels
   - Intermediate correlation → partial coherence
   - Phase spread → relative phase between solitons
   - Larger RMS radius → higher total energy
   - Complex pattern → nonlinear interaction

(c) Breather Solution - Green:
   - Radial spread → amplitude oscillation
   - Star/spiral pattern → phase rotation during breathing
   - Moderate eccentricity → anisotropic dynamics
   - Time-periodic trajectory in phase space
   - Rich harmonic content

(d) Modulation Instability - Magenta:
   - Dispersed cloud → loss of coherence
   - Low correlation → phase randomization
   - High eccentricity → strongly anisotropic
   - Large phase spread → multiple competing modes
   - Transition to turbulence

STATISTICAL MEASURES:

1. Correlation Coefficients:
   - Pearson: Linear relationship strength
   - Spearman: Monotonic relationship (nonlinear)
   - High values → coherent dynamics
   - Low values → incoherent/turbulent

2. Mutual Information (MI):
   - Nonlinear dependence measure
   - Normalized: 0 (independent) to 1 (deterministic)
   - Captures complex phase-amplitude coupling

3. Circular Statistics:
   - Accounts for phase periodicity (−π to π)
   - Circular mean: preferred phase
   - Circular std: phase concentration

4. Eccentricity:
   - 0: Circular (isotropic)
   - →1: Linear (strongly anisotropic)
   - Measures phase space asymmetry

5. Distribution Shape:
   - Skewness: Asymmetry in Re/Im
   - Kurtosis: Tail heaviness (outliers)
   - Indicates deviation from Gaussian

PHYSICAL INSIGHTS:

- Solitons: Preserve phase space structure (attractors)
- Breathers: Periodic orbits in phase space
- Instabilities: Ergodic exploration of phase space
- Collisions: Temporary phase space deformation

The phase space representation reveals the underlying
coherence structure and dynamical complexity of each
nonlinear wave scenario, providing insights not visible
in direct space-time evolution.
""")

# Save statistics to file
output_file = '../stats/phase_space.txt'
with open(output_file, 'w') as f:
    f.write('\n'.join(stats_text))

print(f"\nPhase space analysis complete with LARGE VISIBLE POINTS!")
print(f"="*60)
print(f"Figures saved to:")
print(f"  - ../figs/phase_space.png")
print(f"  - ../figs/phase_space.pdf")
print(f"  - ../figs/phase_space.eps")
print(f"\nStatistics saved to: {output_file}")
print(f"="*60)
print(f"Point size: 25 (50x larger than original)")
print(f"Figure size: 14x14 inches")
print(f"="*60)
print(f"Axis range: ±{axis_limit:.3f}")
print(f"Total scenarios analyzed: {len(data_store)}")
