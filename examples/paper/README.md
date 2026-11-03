# Molten NaCl and SPC/FW water examples

These four short runs are adapted from the P3MaZe paper's NaCl and SPC/FW water
setups. Each YAML file refers only to data under `examples/paper/data/` and
writes to its own directory under `examples/paper/output/`. Run commands from
the repository root after installing the Python package and compiling the C
library as described in the [main README](../../README.md).

| System | Method | Mesh `N` | Command |
| --- | --- | ---: | --- |
| Molten NaCl | Poisson MaZe | 120 | `maze run md examples/paper/molten-salt-poisson-maze.yml` |
| Molten NaCl | P3MaZe | 60 | `maze run md examples/paper/molten-salt-p3maze.yml` |
| SPC/FW water | Poisson MaZe | 40 | `maze run md examples/paper/water-poisson-maze.yml` |
| SPC/FW water | P3MaZe | 40 | `maze run md examples/paper/water-p3maze.yml` |

The NaCl cases contain 250 Na/Cl particles, the paper's Tosi–Fumi parameters,
and the `restart_step_01000.csv` coordinates and velocities from the standard
run. Both NaCl methods use this same restart; the paper's Poisson MaZe copy is
identical. The water cases contain 648 O/H particles (216 O–H–H triplets), the
SPC/FW Lennard–Jones parameters, and `fig6_restart_replica_01.csv` coordinates
and velocities. The source data came from
`data_p3m/paper_p3maze/data/review/punto_2/{molten_salt_figures_6_7,water_figures_6_7}`.

The method settings follow the source configs: `SPL_CUBIC` charge assignment,
`MAZE-MULTIGRID`, and `charge_smoothing: NONE` for Poisson MaZe or `GAUSS`
for P3MaZe. NaCl uses `T=1500 K`, `dt_fs=0.25`, and 4.5 Å/1.39 Å Gaussian
settings. Water uses `T=298.15 K`, `dt_fs=0.5`, `iswater: true`, and the source's
unsuffixed Gaussian values `6.0` and `1.87` in atomic units. Water also uses
`neighbor_r_cut_ang: 9` and `lj_force_shift: false`. See the
[methods guide](../../docs/methods.md) for how the Gaussian and neighbor cutoffs
interact in the current implementation.

To keep these examples quick, `N_steps` is 2 instead of the paper setup's 102,
and output is limited to `energy.csv` and `temperature.csv`. They check that
each configuration starts and writes finite values; they do not reproduce the
paper's trajectories or benchmark results. For research runs, revisit mesh
resolution, cutoff, convergence tolerance, run length, and equilibration.
