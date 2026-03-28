# openFoamLPT

A Python library for post-processing Lagrangian Particle Tracking (LPT) data from OpenFOAM and NEK5000 simulations. Originally developed for cough aerosol dispersion research, comparing DNS (Direct Numerical Simulation) and RANS (Reynolds-Averaged Navier–Stokes) results across multiple particle size classes.

---

## Overview

This library provides tools to:

- Read and process NEK5000 binary particle data files (`.3D`, `.gz`)
- Read OpenFOAM Lagrangian VTK output and velocity/position data
- Compute particle cloud statistics (mean position, mean velocity components)
- Build transfer/transition matrices for particle transport analysis
- Generate publication-quality plots comparing DNS and RANS results
- Create video sequences from particle snapshots

---

## Repository Structure

```
openFoamLPT/
├── NEK5000_func_lib.py      # Core library: NEK5000 binary reader + analysis functions
├── openFoamClass.py         # OpenFOAM class: VTK reader, LPT velocity/position extraction
├── readData.py              # DNS vs RANS comparison: position and velocity plots
├── readLagrangianData.py    # Per-particle-size statistics from OpenFOAM VTK output
├── readOFData.py            # Example usage of openFoamClass
├── plot_particle_data.py    # Multi-panel plots: cloud position, Vx, Vz vs DNS
├── DNS_cought_data.py       # DNS centroid extraction and scaling
└── .gitignore
```

---

## Modules

### `NEK5000_func_lib.py`

Core library for reading and analysing NEK5000 Lagrangian particle binary output.

**Key functions:**

| Function | Description |
|---|---|
| `readParticleFile(pfilename)` | Reads NEK5000 binary `.3D` or `.gz` particle file; returns `(time, pdata)` structured array with fields: `batch`, `sp` (size), `xp` (coordinates), `up` (velocity) |
| `particleCoords(path, fileName, particleSize)` | Extracts 3D coordinates for a given particle size class |
| `particleCoordsNew(path, fileName)` | Extracts coordinates for all particle size classes simultaneously |
| `plotParticle(pfilename, time, pdata, particleSize)` | Renders 3D scatter plot of particle positions, saved as PNG |
| `plotVideo(choose, n, Dimension, particleSize, center, radius, plotsmbl)` | Generates 2D or 3D image sequence over all timesteps; supports selection modes: `all`, `random`, `index`, `sphere` |
| `plotTrajectory(...)` | Plots particle trajectories over all timesteps |
| `totalDist(filename, extension, particleSize)` | Computes cumulative displacement for each particle between consecutive timesteps |
| `PDF(filename, extension, particleSize)` | Computes normalised PDF of particle displacement |
| `binner(x, y, z, n)` | Maps 3D coordinates to a 1D bin index for an n×n×n grid over `[-0.5, 0.5]³` |
| `matrix(data_t1, data_t2, n)` | Builds an n³×n³ transfer matrix counting particle transitions between bins from time t1 to t2 |
| `build_matrix(choose, tt1, tt2, n, ...)` | High-level transfer matrix builder; supports `all` and `sphere` selection modes |
| `createVideo(name, file)` | Assembles PNG sequence into MP4 using `ffmpeg` |
| `stuck(case_name, file_ext)` | Identifies and plots particles deposited on domain walls (hot wall, cold wall, ceiling, floor, adiabatic faces) |
| `listfile(folders, file)` | Scans multiple directories for files matching a given extension |
| `fast_scandir(dirname)` | Recursively lists all subdirectories |

**Physical scaling** (cough aerosol case):
- Length: `× 0.02` (non-dimensional → metres)
- Time: `× 0.0002` (non-dimensional → seconds)
- Velocity: `× 4.8` (non-dimensional → m/s)

---

### `openFoamClass.py`

Object-oriented interface for OpenFOAM Lagrangian post-processing via the `openFoam` class.

**Methods:**

| Method | Description |
|---|---|
| `renameLPTData(dirPath, foldName, timeScale, extension)` | Renames raw OpenFOAM `U` files to `U_<time>.ext` with parentheses stripped |
| `readLPTVelocity(dirPath, foldName, timeScale, extension)` | Reads renamed velocity files; returns array `[t, mVx, mVy, mVz]` |
| `readLPTPositions(dirPath, foldName, timeScale)` | Reads `.vtk` position files; returns array `[t, xm, ym, zm]` |
| `readLagrangianVtk(dirPath, file)` | Reads OpenFOAM Lagrangian VTK file using the VTK Python library; returns array `[origId, d, T, Ux, Uy, Uz, x, y, z]` |
| `fast_scandir`, `listfile`, `find_in_list_of_list` | Filesystem utilities (mirrored from `NEK5000_func_lib.py`) |

**Dependencies:** `vtk`, `vtk.util.numpy_support`

---

### `readLagrangianData.py`

Processes OpenFOAM Lagrangian VTK output to extract per-particle-size statistics over time.

- Groups particles by diameter (`dp`)
- Filters particles to the positive-x half of the domain (`x ≥ 0`)
- Corrects near-wall artefacts (`Vz = 0` for `z < −0.495`)
- Outputs one `.dat` file per particle size class: `[t, d, mVx, mVy, mVz, xm, ym, zm]`

---

### `readData.py`

Generates three-panel comparison plots of DNS vs RANS results for three particle diameters (4 µm, 32 µm, 128 µm):

- **Panel 1:** Mean cloud centroid position (x vs z)
- **Panel 2:** Mean streamwise velocity Vx vs time
- **Panel 3:** Mean vertical velocity Vz vs time

Output: `DNS_RANS.png`

---

### `plot_particle_data.py`

Generates multi-panel (7 particle sizes) comparison plots across DNS and multiple RANS configurations:

- **`XZ_distance.png`:** Mean cloud position (x vs z) for diameters 4–256 µm; includes aerosol cloud boundary contours
- **`Vx.png`:** Mean streamwise velocity vs time per size class
- **`Vz.png`:** Mean vertical velocity vs time per size class

Particle sizes: `[4, 8, 16, 32, 64, 128, 256]` µm

RANS configurations compared: baseline (`1kk`), bumped turbulence model (`1kk_bump`), no turbulent dispersion (`1kk_noDisp`), higher turbulence intensity (`10kk`, `17kk`)

---

### `DNS_cought_data.py`

Reads raw DNS centroid data from `DNS_centroid.csv`, applies physical scaling, and saves the cloud boundary to `dns_cloud.dat`.

---

## Dependencies

```
numpy
matplotlib
scipy
vtk
multiprocessing (stdlib)
```

Install with:

```bash
pip install numpy matplotlib scipy vtk
```

`ffmpeg` is required for video generation (`createVideo`).

---

## Data Format

### NEK5000 particle files (`.3D` / `.gz`)

Fortran binary format. Each file contains:
- `time` (float64)
- `nseedParticles`, `nsizes`, `nfields`, `nparticles` (int32 counters)
- Per-particle records: `batch`, `sp` (size), `xp[3]` (coords), `up[3]` (velocity), additional fields

### OpenFOAM LPT output

- Velocity: plain-text `U` files (OpenFOAM internal format, header stripped)
- Positions: VTK PolyData (`.vtk`) written by the `reactingCloud` solver

### Output `.dat` files

Plain-text space-delimited arrays, readable by `numpy.genfromtxt`.

---

## Usage Example

```python
from NEK5000_func_lib import readParticleFile, particleCoords

path = '/path/to/particle/files/'
fileName = 'part00100.3D'

time, pdata = readParticleFile(path + fileName)
t, coords = particleCoords(path, fileName, particleSize=0)

print(f"Time: {time:.4f} s,  Particles: {coords.shape[0]}")
```

```python
from openFoamClass import openFoam

of = openFoam()
velocity = of.readLPTVelocity(
    dirPath='/path/to/case/',
    foldName='reactingCloud1',
    timeScale=0.05,
    extension='.txt'
)
# velocity columns: [t, mVx, mVy, mVz]
```

---

## Context

Developed during PhD research at Universitat Rovira i Virgili (URV), Tarragona, as part of a study on cough aerosol dispersion. The library was used to validate RANS-based Lagrangian Particle Tracking simulations against DNS reference data for respiratory droplets spanning 4–256 µm in diameter in a buoyancy-driven indoor airflow.

---

## Author

**Akim Lavrinenko**  
PhD in Fluid Mechanics, Universitat Rovira i Virgili (2023)  
[github.com/Akimlav](https://github.com/Akimlav)
