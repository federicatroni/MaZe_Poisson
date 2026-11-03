# MaZe Poisson

MaZe Poisson is a research code for molecular dynamics of charged particles. It
combines a Python command-line interface and input/output layer with a C
simulation library. The code can calculate electrostatic interactions with
several Poisson solvers and run particle dynamics with optional short-range
interactions. It also contains charge-smoothing options. The
`eps_electric_field` branch develops Poisson MaZe with spatially and/or
electric-field-dependent permittivity and a Poisson–Boltzmann adaptation.

The project is at a research/pre-alpha stage. Start with the small, self-contained
example below; some older files in `examples/` refer to external input files.

## Quick start

Requirements: Python 3.10 or newer, CMake 3.20 or newer, and a C compiler.
FFTW3 is required for the FFT solver and Gaussian charge smoothing used by
P3MaZe. MPI and OpenMP are optional; the commands below disable them for a
first build. Python dependencies are installed with the package.

### 1. Get the source and create a virtual environment

```bash
git clone https://github.com/federicatroni/MaZe_Poisson.git
cd MaZe_Poisson
python -m venv .venv
source .venv/bin/activate
```

Run the remaining commands from the repository root, with `.venv` active.

### 2. Compile the C library

**Linux:** install CMake, a C compiler, and FFTW3 development files using your
distribution's package manager. Then run:

```bash
cmake -S library -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/maze_poisson" \
  -DMP_USE_MPI=OFF -DMP_USE_OPENMP=OFF
cmake --build build
cmake --install build
```

**macOS:** with Homebrew, install the build tools and FFTW3, then use Homebrew
LLVM as the C compiler:

```bash
brew install cmake fftw llvm
cmake -S library -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/maze_poisson" \
  -DCMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
  -DMP_USE_MPI=OFF -DMP_USE_OPENMP=OFF
cmake --build build
cmake --install build
```

Choose the commands for your operating system; `cmake --install build` puts
`libmaze_poisson.so` in `maze_poisson/`. For an MPI/OpenMP build, install the
corresponding libraries and set `MP_USE_MPI=ON` and/or `MP_USE_OPENMP=ON` when
configuring.

### 3. Install the Python package

```bash
python -m pip install -e .
```

This installs the Python dependencies and the `maze` command in the active
virtual environment. For plotting commands, use
`python -m pip install -e '.[plotting]'` instead.

### 4. Run the first example

```bash
maze --help
maze run md examples/quickstart.yml
```

The last command runs two MD steps for a pair of oppositely charged particles
using Poisson MaZe without charge smoothing.
Look in `examples/quickstart/output/` for `energy.csv`, `temperature.csv`,
`log.txt`, `input.json`, and `timing.json`. Input and output paths in the YAML
are resolved from the **current working directory**, so run this command from
the repository root. The example is a smoke test, not a physical benchmark.

If `maze` is not found, activate `.venv` and rerun `python -m pip install -e .`.
If the run reports that `maze_poisson/libmaze_poisson.so` cannot be loaded,
check that `cmake --install build` completed and that the file exists in the
package directory. For P3MaZe with `GAUSS` smoothing, the CMake configure
output should report FFTW3 as found; install FFTW3 and reconfigure if it does not.

## Choose a simulation

| Case | Input setting | Example |
| --- | --- | --- |
| Poisson MaZe | `method: MAZE-MULTIGRID`, `charge_smoothing: NONE` | `maze run md examples/quickstart.yml` |
| P3MaZe | `method: MAZE-MULTIGRID`, `charge_smoothing: GAUSS` | `maze run md examples/p3maze.yml` |
| Variable dielectric and Poisson–Boltzmann | Spatially and/or field-dependent permittivity; optional ionic screening | Work in progress on branch `eps_electric_field`; see the methods guide. |

For complete first-use cases based on the P3MaZe paper setup, see the
[molten NaCl and SPC/FW water examples](examples/paper/README.md). Each has its
own YAML file and all required CSV data in this repository. The two-particle
examples above are smoke tests, not validated physical systems. The
[methods guide](docs/methods.md) explains each case, its parameters, units, and
the relationship to the paper inputs. The [input guide](docs/configuration.md)
explains charge assignment and non-electrostatic potentials; the
[command reference](docs/commands.md) lists the CLI commands.

## Papers

- Poisson MaZe: Federica Troni, Davide Grassano, Jayashree Narayan, Benoît Roux, and Sara Bonella, [“Mass-zero constrained molecular dynamics for electrostatic interactions,” *Journal of Chemical Physics* **163**, 214106 (2025)](https://pubs.aip.org/aip/jcp/article/163/21/214106/3373572/Mass-zero-constrained-molecular-dynamics-for), doi:10.1063/5.0283356.
- P3MaZe: Federica Troni, Violette Gontran, Davide Grassano, and Sara Bonella, [“P3MaZe: a Mass-Zero constrained-dynamics formulation of particle–mesh electrostatics”](https://pubs.acs.org/jctcce/article/doi/10.1021/acs.jctc.6c01315/5458363/P3MaZe-a-Mass-Zero-Constrained-Dynamics), doi:10.1021/acs.jctc.6c01315.
- OVRVO Langevin integrator: David A. Sivak, John D. Chodera, and Gavin E. Crooks, [“Time Step Rescaling Recovers Continuous-Time Dynamical Properties for Discrete-Time Langevin Integration of Nonequilibrium Systems,” *Journal of Physical Chemistry B* **118**, 6466–6474 (2014)](https://pubs.acs.org/doi/10.1021/jp411770f), doi:10.1021/jp411770f. See the [thermostat settings](docs/configuration.md#thermostat-and-integrator).

## Development and license

`maze_poisson/cli/` defines the commands, `maze_poisson/myio/input/` parses the
YAML/JSON configuration, `maze_poisson/solver.py` connects inputs to the C API,
and `library/` contains the C solver. Start with the self-contained files in
`examples/` and the [input guide](docs/configuration.md) when changing a model.

Run `python -m pytest` after installing `.[tests]` and building the C library.
The code is licensed under [MIT](LICENSE.txt).
