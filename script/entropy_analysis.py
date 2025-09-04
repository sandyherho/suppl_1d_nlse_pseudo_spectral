#!/usr/bin/env python
"""
Entropy Analysis for 1D NLSE Dynamics
Text-only output with detailed entropy calculations
"""

import numpy as np
from netCDF4 import Dataset
from scipy import stats
import warnings
import os
from pathlib import Path
from datetime import datetime

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Create output directory
Path('../stats').mkdir(parents=True, exist_ok=True)

def shannon_entropy(psi_abs):
    """Calculate Shannon entropy from probability distribution"""
    try:
        p = psi_abs**2
        p_sum = np.sum(p)
        if p_sum <= 0:
            return 0.0
        p = p / p_sum
        p = p[p > 1e-15]
        if len(p) == 0:
            return 0.0
        S = -np.sum(p * np.log(p))
        return S
    except:
        return 0.0

def spectral_entropy(psi_complex):
    """Calculate entropy in Fourier space"""
    try:
        psi_fft = np.fft.fft(psi_complex)
        power = np.abs(psi_fft)**2
        power_sum = np.sum(power)
        if power_sum <= 0:
            return 0.0
        power = power / power_sum
        power = power[power > 1e-15]
        if len(power) == 0:
            return 0.0
        S_k = -np.sum(power * np.log(power))
        return S_k
    except:
        return 0.0

def renyi_entropy(psi_abs, alpha):
    """Calculate Renyi entropy of order alpha"""
    try:
        p = psi_abs**2
        p_sum = np.sum(p)
        if p_sum <= 0:
            return 0.0
        p = p / p_sum
        p = p[p > 1e-15]
        
        if len(p) == 0:
            return 0.0
        
        if np.abs(alpha - 1.0) < 1e-10:
            return -np.sum(p * np.log(p))  # Shannon entropy
        elif alpha == 0:
            return np.log(len(p[p > 0]))  # Hartley entropy
        elif alpha == np.inf:
            return -np.log(np.max(p))  # Min-entropy
        else:
            sum_p_alpha = np.sum(p**alpha)
            if sum_p_alpha <= 0:
                return 0.0
            return np.log(sum_p_alpha) / (1 - alpha)
    except:
        return 0.0

def tsallis_entropy(psi_abs, q):
    """Calculate Tsallis entropy with parameter q"""
    try:
        p = psi_abs**2
        p_sum = np.sum(p)
        if p_sum <= 0:
            return 0.0
        p = p / p_sum
        p = p[p > 1e-15]
        
        if len(p) == 0:
            return 0.0
        
        if np.abs(q - 1.0) < 1e-10:
            return -np.sum(p * np.log(p))  # Shannon entropy
        else:
            sum_p_q = np.sum(p**q)
            return (1 - sum_p_q) / (q - 1)
    except:
        return 0.0

def permutation_entropy(signal, order=3, delay=1):
    """Calculate permutation entropy"""
    try:
        n = len(signal)
        if n < order:
            return 0.0
        
        # Create embedded matrix
        embedded = np.zeros((n - (order - 1) * delay, order))
        for i in range(order):
            embedded[:, i] = signal[i * delay:i * delay + embedded.shape[0]]
        
        # Get permutation patterns
        permutations = {}
        for i in range(embedded.shape[0]):
            sorted_indices = tuple(np.argsort(embedded[i]))
            permutations[sorted_indices] = permutations.get(sorted_indices, 0) + 1
        
        # Calculate probabilities
        total = sum(permutations.values())
        if total == 0:
            return 0.0
        p = np.array(list(permutations.values())) / total
        
        # Permutation entropy
        PE = -np.sum(p * np.log(p))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(np.math.factorial(order))
        if max_entropy > 0:
            PE_normalized = PE / max_entropy
        else:
            PE_normalized = 0.0
        
        return PE_normalized
    except:
        return 0.0

def sample_entropy(signal, m=2, r=None):
    """Calculate sample entropy"""
    try:
        N = len(signal)
        if N < m + 1:
            return 0.0
        
        if r is None:
            std = np.std(signal)
            if std <= 0:
                return 0.0
            r = 0.2 * std
        
        def _maxdist(xi, xj):
            return np.max(np.abs(xi - xj))
        
        def _phi(m):
            patterns = [signal[i:i + m] for i in range(N - m + 1)]
            if len(patterns) < 2:
                return 0.0
            
            C = 0
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    if _maxdist(patterns[i], patterns[j]) <= r:
                        C += 2
            
            if len(patterns) * (len(patterns) - 1) > 0:
                return C / (len(patterns) * (len(patterns) - 1))
            return 0.0
        
        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        
        if phi_m1 == 0 or phi_m == 0:
            return 0.0
        return -np.log(phi_m1 / phi_m)
    except:
        return 0.0

def statistical_complexity(psi_abs):
    """Calculate statistical complexity C = S * D"""
    try:
        p = psi_abs**2
        p_sum = np.sum(p)
        if p_sum <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        p = p / p_sum
        
        # Shannon entropy
        p_nonzero = p[p > 1e-15]
        if len(p_nonzero) == 0:
            return 0.0, 0.0, 0.0, 0.0
        
        S = -np.sum(p_nonzero * np.log(p_nonzero))
        
        # Normalize entropy
        S_max = np.log(len(p))
        S_norm = S / S_max if S_max > 0 else 0
        
        # Disequilibrium
        p_uniform = 1.0 / len(p)
        D = np.sum((p - p_uniform)**2)
        
        # Complexity
        C_SD = S_norm * D * len(p)
        
        return C_SD, S_norm, D, S
    except:
        return 0.0, 0.0, 0.0, 0.0

def differential_entropy(psi_abs, bins=50):
    """Calculate differential entropy using histogram approximation"""
    try:
        if len(psi_abs) == 0:
            return 0.0
        
        # Create histogram
        hist, bin_edges = np.histogram(psi_abs, bins=bins, density=True)
        bin_width = bin_edges[1] - bin_edges[0]
        
        # Calculate differential entropy
        h = 0
        for p in hist:
            if p > 0:
                h -= p * np.log(p) * bin_width
        
        return h + np.log(bin_width)
    except:
        return 0.0

def spectral_richness(psi_complex, threshold=0.01):
    """Calculate spectral richness and bandwidth"""
    try:
        psi_fft = np.fft.fft(psi_complex)
        power = np.abs(psi_fft)**2
        max_power = np.max(power)
        if max_power <= 0:
            return 0, 0.0, 0.0
        
        power = power / max_power
        
        # Count significant modes
        n_modes = np.sum(power > threshold)
        
        # Calculate effective bandwidth
        freqs = np.fft.fftfreq(len(psi_complex))
        power_sum = np.sum(power)
        if power_sum <= 0:
            return int(n_modes), 0.0, 0.0
        
        power_norm = power / power_sum
        mean_freq = np.sum(freqs * power_norm)
        bandwidth = np.sqrt(np.sum((freqs - mean_freq)**2 * power_norm))
        
        return int(n_modes), bandwidth, mean_freq
    except:
        return 0, 0.0, 0.0

# Load and analyze all scenarios
scenarios = {
    'single_soliton': 'Single Soliton',
    'two_soliton_collision': 'Two-Soliton Collision',
    'breather_solution': 'Breather Solution',
    'modulation_instability': 'Modulation Instability'
}

# Initialize output text
output = []
output.append("="*80)
output.append("COMPREHENSIVE ENTROPY ANALYSIS FOR 1D NLSE DYNAMICS")
output.append("="*80)
output.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output.append(f"Number of Scenarios: {len(scenarios)}")
output.append("")

# Process each scenario
all_results = {}

for scenario_file, scenario_name in scenarios.items():
    nc_path = f'../outputs/{scenario_file}.nc'
    
    if not os.path.exists(nc_path):
        output.append(f"Warning: File not found - {nc_path}")
        continue
    
    output.append("="*80)
    output.append(f"SCENARIO: {scenario_name}")
    output.append("="*80)
    
    try:
        # Load data
        nc = Dataset(nc_path, 'r')
        x = nc.variables['x'][:]
        t = nc.variables['t'][:]
        psi_abs = nc.variables['psi_abs'][:]
        psi_real = nc.variables['psi_real'][:]
        psi_imag = nc.variables['psi_imag'][:]
        nc.close()
        
        output.append(f"Grid Points: {len(x)}")
        output.append(f"Time Steps: {len(t)}")
        output.append(f"Time Range: [{t[0]:.3f}, {t[-1]:.3f}]")
        output.append(f"Spatial Range: [{x[0]:.3f}, {x[-1]:.3f}]")
        output.append("")
        
        # Initialize storage for time series
        results = {
            'shannon_spatial': [],
            'shannon_spectral': [],
            'renyi_0': [],
            'renyi_2': [],
            'renyi_inf': [],
            'tsallis_0.5': [],
            'tsallis_2': [],
            'permutation': [],
            'sample': [],
            'complexity_SD': [],
            'differential': [],
            'n_modes': [],
            'bandwidth': [],
            'mean_freq': []
        }
        
        # Analyze each time step
        time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]  # Sample time points
        
        for i in time_indices:
            psi_complex = psi_real[i] + 1j * psi_imag[i]
            
            # Basic entropies
            S_shannon = shannon_entropy(psi_abs[i])
            S_spectral = spectral_entropy(psi_complex)
            
            # Renyi entropies
            R0 = renyi_entropy(psi_abs[i], 0)
            R2 = renyi_entropy(psi_abs[i], 2)
            Rinf = renyi_entropy(psi_abs[i], np.inf)
            
            # Tsallis entropies
            T05 = tsallis_entropy(psi_abs[i], 0.5)
            T2 = tsallis_entropy(psi_abs[i], 2)
            
            # Time series entropies
            PE = permutation_entropy(np.abs(psi_complex))
            SE = sample_entropy(np.abs(psi_complex))
            
            # Complexity measures
            C_SD, H_norm, D, S_raw = statistical_complexity(psi_abs[i])
            
            # Differential entropy
            h_diff = differential_entropy(psi_abs[i])
            
            # Spectral characteristics
            n_modes, bandwidth, mean_freq = spectral_richness(psi_complex)
            
            # Store results
            results['shannon_spatial'].append(S_shannon)
            results['shannon_spectral'].append(S_spectral)
            results['renyi_0'].append(R0)
            results['renyi_2'].append(R2)
            results['renyi_inf'].append(Rinf)
            results['tsallis_0.5'].append(T05)
            results['tsallis_2'].append(T2)
            results['permutation'].append(PE)
            results['sample'].append(SE)
            results['complexity_SD'].append(C_SD)
            results['differential'].append(h_diff)
            results['n_modes'].append(n_modes)
            results['bandwidth'].append(bandwidth)
            results['mean_freq'].append(mean_freq)
            
            # Output for this time point
            output.append(f"\nTime t = {t[i]:.3f}:")
            output.append("-" * 40)
            output.append("Shannon Entropies:")
            output.append(f"  Spatial:  {S_shannon:.3f}")
            output.append(f"  Spectral: {S_spectral:.3f}")
            output.append("")
            output.append("Generalized Entropies:")
            output.append(f"  Renyi (a=0):   {R0:.3f} (Hartley)")
            output.append(f"  Renyi (a=2):   {R2:.3f} (Collision)")
            output.append(f"  Renyi (a=inf): {Rinf:.3f} (Min-entropy)")
            output.append(f"  Tsallis (q=0.5): {T05:.3f}")
            output.append(f"  Tsallis (q=2):   {T2:.3f}")
            output.append("")
            output.append("Time Series Measures:")
            output.append(f"  Permutation:     {PE:.3f}")
            output.append(f"  Sample Entropy:  {SE:.3f}")
            output.append(f"  Differential:    {h_diff:.3f}")
            output.append("")
            output.append("Complexity Measures:")
            output.append(f"  Statistical (S*D): {C_SD:.3f}")
            output.append(f"  Normalized H:      {H_norm:.3f}")
            output.append(f"  Disequilibrium D:  {D:.3f}")
            output.append("")
            output.append("Spectral Properties:")
            output.append(f"  Active Modes:    {n_modes}")
            output.append(f"  Bandwidth:       {bandwidth:.3f}")
            output.append(f"  Mean Frequency:  {mean_freq:.3f}")
        
        # Calculate statistics over time
        output.append("\n" + "="*40)
        output.append("TEMPORAL STATISTICS:")
        output.append("="*40)
        
        for key in results.keys():
            if len(results[key]) > 0 and key != 'n_modes':  # n_modes is integer
                values = np.array(results[key])
                output.append(f"\n{key.replace('_', ' ').title()}:")
                output.append(f"  Initial: {values[0]:.3f}")
                output.append(f"  Final:   {values[-1]:.3f}")
                output.append(f"  Change:  {values[-1] - values[0]:.3f}")
                output.append(f"  Mean:    {np.mean(values):.3f}")
                output.append(f"  Std:     {np.std(values):.3f}")
                output.append(f"  Min:     {np.min(values):.3f}")
                output.append(f"  Max:     {np.max(values):.3f}")
            elif key == 'n_modes':
                values = np.array(results[key])
                output.append(f"\nNumber of Modes:")
                output.append(f"  Initial: {values[0]}")
                output.append(f"  Final:   {values[-1]}")
                output.append(f"  Change:  {values[-1] - values[0]}")
                output.append(f"  Mean:    {np.mean(values):.0f}")
                output.append(f"  Min:     {np.min(values)}")
                output.append(f"  Max:     {np.max(values)}")
        
        # Full temporal analysis
        output.append("\n" + "="*40)
        output.append("FULL TEMPORAL EVOLUTION:")
        output.append("="*40)
        
        # Calculate for all time steps
        S_shannon_all = [shannon_entropy(psi_abs[i]) for i in range(len(t))]
        S_spectral_all = [spectral_entropy(psi_real[i] + 1j * psi_imag[i]) for i in range(len(t))]
        
        # Check for valid data
        if len(S_shannon_all) > 1 and np.std(S_shannon_all) > 0:
            corr, _ = stats.spearmanr(t, S_shannon_all)
            output.append(f"\nShannon Entropy Evolution:")
            output.append(f"  Monotonic Trend: {corr:.3f}")
            output.append(f"  Oscillation Amplitude: {np.max(S_shannon_all) - np.min(S_shannon_all):.3f}")
            output.append(f"  Coefficient of Variation: {np.std(S_shannon_all) / np.mean(S_shannon_all):.3f}")
        else:
            output.append(f"\nShannon Entropy Evolution:")
            output.append(f"  Constant value: {np.mean(S_shannon_all):.3f}")
        
        # Entropy production rate
        if len(t) > 1:
            dt = t[1] - t[0]
            dS_dt = np.gradient(S_shannon_all, dt)
            output.append(f"\nEntropy Production:")
            output.append(f"  Mean Rate: {np.mean(dS_dt):.3f}")
            output.append(f"  Max Rate:  {np.max(np.abs(dS_dt)):.3f}")
            output.append(f"  Total Production: {S_shannon_all[-1] - S_shannon_all[0]:.3f}")
        
        # Store for comparison
        all_results[scenario_name] = results
        
    except Exception as e:
        output.append(f"Error processing {scenario_name}: {str(e)}")
        continue

# Comparative analysis
output.append("\n" + "="*80)
output.append("COMPARATIVE ANALYSIS ACROSS SCENARIOS")
output.append("="*80)

# Create comparison table
output.append("\nInitial Shannon Entropy Comparison:")
for name in all_results.keys():
    if 'shannon_spatial' in all_results[name] and len(all_results[name]['shannon_spatial']) > 0:
        output.append(f"  {name:30s}: {all_results[name]['shannon_spatial'][0]:.3f}")

output.append("\nFinal Shannon Entropy Comparison:")
for name in all_results.keys():
    if 'shannon_spatial' in all_results[name] and len(all_results[name]['shannon_spatial']) > 0:
        output.append(f"  {name:30s}: {all_results[name]['shannon_spatial'][-1]:.3f}")

output.append("\nMean Complexity Comparison:")
for name in all_results.keys():
    if 'complexity_SD' in all_results[name] and len(all_results[name]['complexity_SD']) > 0:
        output.append(f"  {name:30s}: {np.mean(all_results[name]['complexity_SD']):.3f}")

# Physical interpretation
output.append("\n" + "="*80)
output.append("PHYSICAL INTERPRETATION")
output.append("="*80)
output.append("""
1. SHANNON ENTROPY:
   - Measures information content and localization
   - Low values: Localized states (solitons)
   - High values: Delocalized states (modulation instability)
   - Oscillations: Breathing modes or periodic dynamics

2. SPECTRAL ENTROPY:
   - Quantifies frequency distribution complexity
   - Low values: Few dominant modes (coherent)
   - High values: Many modes (turbulent/chaotic)

3. RENYI ENTROPIES:
   - a=0: Counts support size (Hartley)
   - a=2: Sensitive to correlations (Collision)
   - a=inf: Dominated by peak (Min-entropy)
   - Different a values probe different scales

4. TSALLIS ENTROPY:
   - Non-extensive entropy for long-range correlations
   - q<1: Sub-extensive systems
   - q>1: Super-extensive systems

5. PERMUTATION ENTROPY:
   - Captures temporal ordering patterns
   - Robust to noise and outliers
   - Indicates dynamical complexity

6. SAMPLE ENTROPY:
   - Measures time series regularity
   - Low values: Regular, predictable
   - High values: Irregular, complex

7. STATISTICAL COMPLEXITY:
   - Distinguishes between order and randomness
   - Maximum at intermediate entropy
   - Captures emergent structures

8. SPECTRAL CHARACTERISTICS:
   - Active modes: Degree of multimodality
   - Bandwidth: Frequency spread
   - Mean frequency: Dominant oscillation

SCENARIO-SPECIFIC INSIGHTS:
- Single Soliton: Low entropy, high coherence, stable
- Two-Soliton: Moderate entropy, collision dynamics
- Breather: Oscillating entropy, periodic dynamics
- Modulation: Highest initial entropy, pattern formation
""")

# Save to file
output_file = '../stats/comprehensive_entropy_analysis.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print(f"Comprehensive entropy analysis completed")
print(f"Results saved to: {output_file}")
print(f"Total lines written: {len(output)}")
print(f"File size: {os.path.getsize(output_file) / 1024:.1f} KB")

# Summary statistics to console
print("\n" + "="*60)
print("SUMMARY OF KEY FINDINGS:")
print("="*60)
for name in all_results.keys():
    if 'shannon_spatial' in all_results[name] and len(all_results[name]['shannon_spatial']) > 0:
        initial = all_results[name]['shannon_spatial'][0]
        final = all_results[name]['shannon_spatial'][-1]
        change = final - initial
        print(f"\n{name}:")
        print(f"  Shannon Entropy: {initial:.3f} -> {final:.3f} (Change = {change:+.3f})")
        
        if 'complexity_SD' in all_results[name] and len(all_results[name]['complexity_SD']) > 0:
            print(f"  Mean Complexity: {np.mean(all_results[name]['complexity_SD']):.3f}")
        
        if 'n_modes' in all_results[name] and len(all_results[name]['n_modes']) > 0:
            print(f"  Spectral Modes: {int(np.mean(all_results[name]['n_modes']))}")
