# Supplementary Materials: High-Order Pseudo-Spectral Solver for 1D NLSE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)

Comprehensive data analysis and visualization suite accompanying the paper *"High-Order Pseudo-Spectral Solver for the 1D Nonlinear Schrödinger Equation"* by Herho et al.

## Overview

This repository contains the complete computational results, analysis scripts, and visualizations demonstrating the capabilities of the [`simple-idealized-1d-nlse`](https://github.com/samuderasains/simple-idealized-1d-nlse) solver package (v0.0.3). We analyze four canonical NLSE scenarios using advanced statistical and entropy-based methods.

## Repository Structure

```
.
├── figs/           # Publication-ready figures (EPS, PDF, PNG)
├── outputs/        # Raw simulation data (NetCDF) and animations (GIF)
├── script/         # Analysis scripts
├── stats/          # Statistical analysis results
└── LICENSE.txt     # WTFPL License
```

## Key Results

### Simulation Scenarios
- **Single Soliton**: Fundamental soliton propagation (η=2.0, v=1.0)
- **Two-Soliton Collision**: Elastic collision dynamics (η₁=2.0, η₂=1.5)
- **Breather Solution**: Akhmediev breather oscillations (a=0.5)
- **Modulation Instability**: Benjamin-Feir instability evolution

### Analysis Performed
1. **Statistical Analysis** (`stats.py`): Intensity distribution, effect sizes, hypothesis testing
2. **Phase Space Analysis** (`phase_space.py`): Re(ψ) vs Im(ψ) dynamics, correlation metrics
3. **Space-Time Patterns** (`spatio_temporal.py`): Spatiotemporal evolution visualization
4. **Evolution Snapshots** (`snapshots.py`): Intensity profiles at key time points
5. **Entropy Analysis** (`entropy_analysis.py`): Shannon, Rényi, Tsallis, and permutation entropies

## Requirements

```bash
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
netCDF4>=1.5.0
```

## Reproducing Results

1. **Install the solver package**:
```bash
pip install simple-idealized-1d-nlse==0.0.3
```

2. **Generate simulation data**:
```bash
nlse-simulate --all
```

3. **Run analysis scripts**:
```bash
cd script/
python stats.py           # Statistical analysis
python phase_space.py     # Phase space visualization
python spatio_temporal.py # Space-time patterns
python snapshots.py       # Evolution snapshots
python entropy_analysis.py # Entropy measures
```

## Key Findings

- **Conservation Laws**: Mass, momentum, and energy conserved to machine precision (ΔE/E < 10⁻¹²)
- **Spectral Accuracy**: Exponential convergence with 512 spatial modes
- **Stability**: Solitons maintain shape over 20 time units with <0.01% amplitude variation
- **Entropy Analysis**: Modulation instability shows highest Shannon entropy (S=4.9), indicating delocalization

## Data Format

### NetCDF Files
Each `.nc` file contains:
- `x[512]`: Spatial grid (-25 to 25)
- `t[100]`: Time points (0 to 20)
- `psi_real[100,512]`: Real part of ψ
- `psi_imag[100,512]`: Imaginary part of ψ
- `psi_abs[100,512]`: |ψ|

### Figures
All figures available in three formats:
- **EPS**: Vector format for LaTeX
- **PDF**: High-quality printing
- **PNG**: Web display (300 DPI)

## Citation

If you use this data or code, please cite:

```bibtex
@article{herho2025nlse,
  title={{H}igh-{O}rder {P}seudo-{S}pectral {S}olver for the {1D} {N}onlinear {S}chrödinger {E}quation},
  author={Herho, S. H. S. and Anwar, I. P. and Khadami, F. and 
          Riawan, E. and Suwarman, R. and Irawan, D. E.},
  journal={xxxx},
  year={202x},
  note={In preparation}
}
```

## License

WTFPL - Do What The F*ck You Want To Public License

## Contact

Sandy H. S. Herho - sandy.herho@email.ucr.edu
