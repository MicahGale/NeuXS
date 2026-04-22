# NeuXS 
A new way to test evaluating neutron cross-sections on GPUs.

![CI](https://github.com/MicahGale/NeuXS/actions/workflows/workflow.yml/badge.svg?branch=main)
![CUDA](https://img.shields.io/badge/CUDA-12.0+-green)
![C++](https://img.shields.io/badge/C++-17-blue)

## Overview

NeuXS is a high-performance Nvidia-GPU-accelerated library for evaluating neutron cross-section look up
for Monte Carlo particle transport. It provides a (!) flexible, type-safe abstractions framework
for testing neutron cross-section look strategies optimized for GPU computation.

NeuXS isn't a particle transport code. This started as an HPC class project by Micah Gale and Ebny Walid Ahammed.
It is still under continuous development. 

### Key Features

- **GPU-Accelerated**: Full CUDA support for fast cross-section lookups
- **Multiple Storage Strategies**: Hash based Cross-section, Array of Structs (AoS), Struct of Arrays (SoA), and SLBW resonance representations
- **Material Composition**: Handle complex multi-isotope material mixtures
- **Flexible Interpolation**: Linear, log-log, and custom interpolation methods
- **Validated Data**: Comprehensive consistency checking and validation framework
- **Energy Grid Optimization**: 2D energy grid acceleration for multi-nuclide scenarios


## Dependencies

### Required

- **CUDA Toolkit**: Version 12.0 or higher
    - Download: https://developer.nvidia.com/cuda-downloads
    - Required for GPU compilation and Thrust library

- **C++ Compiler**: C++17 or higher
    - NVIDIA nvcc (included with CUDA Toolkit)
    - GCC 13.3.0 for host compilation

- **CMake**: Version 3.29 or higher
    - Download: https://cmake.org/download/
    - Used for cross-platform build configuration

- **Thrust**: NVIDIA Thrust library (included with CUDA Toolkit)
    - Parallel algorithms and containers for GPU/CPU
    - Version 1.9.0+
- **HDF5**: HDF5 lib
    - Required for reading hdf5 based neutron cross-sections
    - Supports hierarchical datasets and chunked I/O for large-scale nuclear data
    - Download: https://www.hdfgroup.org/solutions/hdf5/

## Build 

### System Requirements

**OS**: Linux (We have only tested on linux!)

**COMPILER**: NVCC and gcc

**GPU**: Any Nvidia GPU that supports CUDA 12.0 (Cause that's what we have)


### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/MicahGale/NeuXS.git
   cd NeuXS
   ```

2. **Create build directory**
   ```bash
   mkdir build && cd build
   ```

3. **Configure with CMake**
   ```bash
   # Default build (Release, all features)
   cmake ..
   ```
   or if you want more control, you can change the build args!

4. **Compile**
   ```bash
   make -j$(nproc)
   ```
   

### Debug guidelines

I hope you never hit that rabbit hole and blow up your leg with this poorly designed software
but if you do, then cuda-gdb could be your (read with air quotes) friend. 

First remove the existing build director and recompile with `Debug` mode.
   ```bash
   rm -rf build && mkdir build 
   cd build 
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make -j $(nproc)
   ```
Then, 
   ```bash
   cuda-gdb ./path_to_executable
   ```
