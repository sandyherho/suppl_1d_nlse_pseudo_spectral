#!/usr/bin/env python
"""
Space-Time Patterns Visualization
Visualizes spatiotemporal evolution of nonlinear wave dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter
import os
from pathlib import Path

def setup_publication_params():
    """Set publication quality parameters"""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is available

def create_directories():
    """Create output directories if they don't exist"""
    Path('../figs').mkdir(parents=True, exist_ok=True)
    Path('../stats').mkdir(parents=True, exist_ok=True)

def load_scenario_data(scenario_file):
    """Load data from NetCDF file"""
    nc_path = f'../outputs/{scenario_file}.nc'
    if not os.path.exists(nc_path):
        return None
    
    nc = Dataset(nc_path, 'r')
    data = {
        'x': nc.variables['x'][:],
        't': nc.variables['t'][:],
        'psi_abs': nc.variables['psi_abs'][:],
        'psi_real': nc.variables['psi_real'][:],
        'psi_imag': nc.variables['psi_imag'][:]
    }
    nc.close()
    return data

def calculate_energy_center(x, t, psi_squared):
    """Calculate energy center trajectory"""
    energy_center = []
    total_energy = []
    
    for i in range(len(t)):
        E_i = psi_squared[i]
        total_E = np.sum(E_i)
        if total_E > 0:
            x_center = np.sum(x * E_i) / total_E
            energy_center.append(x_center)
        else:
            energy_center.append(0)
        total_energy.append(total_E)
    
    return energy_center, total_energy

def calculate_statistics(data, scenario_name, stats_text):
    """Calculate and record statistics for a scenario"""
    x = data['x']
    t = data['t']
    psi_abs = data['psi_abs']
    
    psi_squared = psi_abs**2
    energy_center, total_energy = calculate_energy_center(x, t, psi_squared)
    
    stats_text.append(f"\n{scenario_name}:")
    stats_text.append(f"  Space-Time Characteristics:")
    stats_text.append(f"    Max intensity: {np.max(psi_squared):.3f}")
    stats_text.append(f"    Mean intensity: {np.mean(psi_squared):.3f}")
    stats_text.append(f"    Energy center drift: {energy_center[-1] - energy_center[0]:.3f}")
    
    if len(energy_center) > 1:
        velocity = (energy_center[-1] - energy_center[0]) / (t[-1] - t[0])
        stats_text.append(f"    Propagation velocity: {velocity:.3f}")
    
    # Spatial coherence length
    from scipy.signal import correlate
    psi_mid = psi_abs[len(t)//2]
    autocorr = correlate(psi_mid, psi_mid, mode='same')
    autocorr = autocorr / np.max(autocorr)
    
    # Find correlation length (where autocorr drops to 1/e)
    half_idx = len(autocorr) // 2
    try:
        corr_length_idx = np.where(autocorr[half_idx:] < 1/np.e)[0][0]
        corr_length = (x[1] - x[0]) * corr_length_idx
        stats_text.append(f"    Correlation length: {corr_length:.3f}")
    except:
        stats_text.append(f"    Correlation length: Full domain")
    
    # Pattern velocity for solitons
    if 'Soliton' in scenario_name:
        peak_positions = []
        for i in range(len(t)):
            peak_idx = np.argmax(psi_abs[i])
            peak_positions.append(x[peak_idx])
        
        if len(peak_positions) > 10:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(
                t[:len(peak_positions)], peak_positions
            )
            if r_value**2 > 0.8:
                stats_text.append(f"    Soliton velocity: {slope:.3f}")
                stats_text.append(f"    Velocity R²: {r_value**2:.3f}")
    
    return energy_center

def create_spacetime_plot(ax, data, scenario_name, vmin, vmax, energy_center=None):
    """Create a single space-time plot"""
    x = data['x']
    t = data['t']
    psi_abs = data['psi_abs']
    
    # Calculate |ψ|²
    psi_squared = psi_abs**2
    
    # Apply slight smoothing for better visualization
    psi_squared_smooth = gaussian_filter(psi_squared, sigma=(0.5, 0.5))
    
    # Create space-time plot
    im = ax.imshow(psi_squared_smooth.T, aspect='auto', origin='lower',
                   extent=[t[0], t[-1], x[0], x[-1]],
                   cmap='hot', vmin=vmin, vmax=vmax, interpolation='bilinear')
    
    # Add energy center trajectory if provided
    if energy_center is not None:
        ax.plot(t, energy_center, 'w--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Time $t$', fontsize=16)
    ax.set_ylabel('Position $x$', fontsize=16)
    ax.set_title(scenario_name, fontsize=18)
    
    return im

def add_interpretation(stats_text):
    """Add physical interpretation to statistics"""
    stats_text.append("\n" + "="*70)
    stats_text.append("Interpretation:")
    stats_text.append("="*70)
    stats_text.append("""
Space-Time Pattern Analysis:

PATTERN CHARACTERISTICS:
========================

1. Single Soliton:
   - Diagonal stripe indicates constant velocity propagation
   - Uniform intensity along trajectory shows shape preservation
   - No dispersion demonstrates soliton stability
   - Energy center (white dashed line) follows soliton peak

2. Two-Soliton Collision:
   - Two diagonal stripes crossing form X-pattern
   - Intensity enhancement at collision point
   - Post-collision: solitons retain individual identity
   - Phase shift visible as trajectory displacement

3. Breather Solution:
   - Vertical stripes show stationary position
   - Intensity modulation creates breathing pattern
   - Periodic bright/dark regions in time
   - Energy center remains localized

4. Modulation Instability:
   - Initial uniform background breaks up
   - Vertical striations indicate pattern formation
   - Bright spots represent rogue wave formation
   - Wavelength determined by most unstable mode

PHYSICAL MECHANISMS:
====================

Key Features:
- Stripe slope → propagation velocity
- Stripe width → spatial extent of solution
- Intensity variations → energy redistribution
- Pattern formation → nonlinear dynamics

Conservation Laws:
- Energy conservation visible in intensity patterns
- Momentum conservation in soliton trajectories
- Phase coherence in regular patterns
- Nonlinear selection of dominant modes

DIAGNOSTIC VALUE:
================

The space-time visualization provides:
- Direct observation of wave propagation
- Collision and interaction dynamics
- Pattern formation mechanisms
- Quantitative velocity measurements
- Energy localization and redistribution
- Stability and instability regions
""")

def main():
    """Main execution function"""
    # Setup
    setup_publication_params()
    create_directories()
    
    # Define scenarios
    scenarios = {
        'single_soliton': '(a)',
        'two_soliton_collision': '(b)',
        'breather_solution': '(c)',
        'modulation_instability': '(d)'
    }
    
    # Initialize statistics
    stats_text = []
    stats_text.append("="*70)
    stats_text.append("Space-Time Pattern Analysis")
    stats_text.append("="*70)
    
    # Load all data first to determine global colorbar scale
    all_data = {}
    global_vmax = 0
    
    for scenario_file, scenario_name in scenarios.items():
        data = load_scenario_data(scenario_file)
        if data is not None:
            all_data[scenario_file] = data
            psi_squared = data['psi_abs']**2
            vmax = np.percentile(psi_squared, 99.5)
            global_vmax = max(global_vmax, vmax)
    
    # Create 2x2 grid figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot each scenario
    images = []
    for idx, (scenario_file, scenario_name) in enumerate(scenarios.items()):
        if scenario_file in all_data:
            data = all_data[scenario_file]
            
            # Calculate statistics and energy center
            energy_center = calculate_statistics(data, scenario_name, stats_text)
            
            # Create space-time plot
            im = create_spacetime_plot(axes[idx], data, scenario_name, 
                                      vmin=0, vmax=global_vmax, 
                                      energy_center=energy_center)
            images.append(im)
    
    # Add single colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(images[0], cax=cbar_ax)
    cbar.set_label('$|\\psi|^2$', fontsize=14)
    cbar.ax.tick_params(labelsize=11)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Save figures in multiple formats
    plt.savefig('../figs/spacetime_patterns.png', dpi=300, bbox_inches='tight')
    plt.savefig('../figs/spacetime_patterns.pdf', bbox_inches='tight')
    plt.savefig('../figs/spacetime_patterns.eps', format='eps', bbox_inches='tight')
    plt.close()
    
    # Add interpretation
    add_interpretation(stats_text)
    
    # Save statistics to file
    with open('../stats/spacetime_patterns.txt', 'w') as f:
        f.write('\n'.join(stats_text))
    
    print("Space-time pattern analysis completed")
    print("Figures saved in: ../figs/")
    print("Statistics saved in: ../stats/")
    print("="*70)

if __name__ == "__main__":
    main()
