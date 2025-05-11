# SSH-Hubbard Variational Monte Carlo


This repository contains a JAX-based implementation of Variational Monte Carlo (VMC) methods to study the square lattice SSH-Hubbard model with electron-phonon coupling. It supports optimization of Jastrow-Slater wave functions using Stochastic Reconfiguration (SR), the Slater part contains backflow terms depending upon phonon configurations.

For the 1D version, see:
[![arXiv](https://img.shields.io/badge/arXiv-2407.03046-b31b1b.svg)](https://arxiv.org/abs/2407.03046)

## Overview

This code enables the ground state properties of the 2D square lattice SSH-Hubbard model.

The implementation leverages JAX for efficient computation on CPU, MPI for parallel simulations, and a variational approach that includes electron-phonon coupling.

## Features

- Stochastic Reconfiguration optimization for wave function parameters
- Support for different BOW patterns (e.g., (π,π) and (π,0))
- Calculation of various observables including charge and spin correlations
- Parallel execution via MPI
- Configurable simulation parameters via YAML
- JAX-based implementation for efficient computation and automatic differentiation

## Installation

### Prerequisites

- Python 3.8+
- JAX and JAXlib
- Flax
- MPI4py
- mpi4Jax
- PyYAML


## Usage

### Basic Execution

To run a simulation:

```bash
# Using MPI with 4 processes
mpirun -n 4 python run_vMC.py
```


```bash
# For the non-parallel impelmentation
python run_vMC_no_MPI.py
```

### Configuration

Edit the `config.yaml` file to set simulation parameters:

```yaml
system:
  t_hub: -1.0           # Hopping parameter
  omega: 1.0            # Phonon frequency
  alpha: 0.4            # Electron-phonon coupling
  U_hub: 0.0            # Hubbard repulsion
  Lx: 8                 # Lattice size (x dimension)
  Ly: 8                 # Lattice size (y dimension)
  N_e_up: 32            # Number of up electrons

phonons:
  ALLOW_BOW_X_PHON: true    # Enable X phonons
  ALLOW_BOW_Y_PHON: false   # Enable Y phonons
  Q_x: "pi"                 # Wave vector x component
  Q_y: "pi"                 # Wave vector y component

# ...other parameters
```

## Primary Components

- **Lattice**: Defines the square lattice structure and neighbor tables
- **Wave Function**: Implements backflow-corrected variational ansatz
- **Metropolis**: Monte Carlo sampling algorithms with electron and phonon moves
- **Observables**: Measurement of physical quantities (energy, correlations, etc.)

