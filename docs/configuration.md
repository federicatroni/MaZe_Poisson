# Input guide

An input file has three top-level sections: `grid_setting`, `md_variables`, and
`output_settings`. The self-contained [quick start example](../examples/quickstart.yml)
is the best starting template. The parser also accepts JSON with the same keys.
For Poisson MaZe, P3MaZe, variable dielectric, and Poisson–Boltzmann, see the
[methods guide](methods.md).

| Section | Key | Meaning |
| --- | --- | --- |
| `grid_setting` | `N`, `N_p`, `N_typs` | Grid points per dimension, particle count, and species count. |
| `grid_setting` | `L_ang` | Cubic box length in ångströms; `L` instead specifies atomic units. |
| `grid_setting` | `input_file`, `particles_file` | CSV paths for initial coordinates and species definitions. |
| `grid_setting` | `cas` | Charge assignment: `CIC`, `SPL_QUADR`, or `SPL_CUBIC`. |
| `md_variables` | `N_steps`, `T`, `dt_fs` | MD steps, temperature in kelvin, timestep in femtoseconds. |
| `md_variables` | `method` | Poisson solver; choices are listed in the README. |
| `md_variables` | `potential`, `potential_params_file` | Short-range potential and its parameter CSV. |
| `md_variables` | `elec`, `not_elec` | Enable electrostatic and non-electrostatic force calculations. |
| `md_variables` | `integrator`, `thermostat`, `gamma` | MD integrator, thermostat switch, and damping coefficient; see below. |
| `output_settings` | `path`, `stride` | Output directory and sampling interval. |
| `output_settings` | `print_energy`, `print_temperature`, `print_solute`, `print_restart` | Select output CSVs. |

The positions CSV requires `type,x,y,z` (coordinates in ångströms). Optional
`vx,vy,vz` columns provide initial velocities; otherwise the code generates
them from `T`. The species CSV requires at least `type,charge,mass`; add
`radius` for Poisson–Boltzmann. Particle `type` values must match between
coordinates, species, and potential parameters. See the [paper examples](../examples/paper/README.md)
for complete input sets.

## Charge assignment (`cas`)

`cas` specifies how each particle's charge is distributed to nearby grid nodes
before solving the field equation. Use one of these exact values:

| Value | Weight function | Grid nodes with nonzero weight per particle | Typical use |
| --- | --- | ---: | --- |
| `CIC` | Cloud-in-cell, linear | 8 (2 per axis) | Small or inexpensive first checks. |
| `SPL_QUADR` | Quadratic B-spline | Up to 27 (3 per axis) | Smoother charge assignment. |
| `SPL_CUBIC` | Cubic B-spline | Up to 64 (4 per axis) | Used by the NaCl and SPC/FW paper examples. |

For example:

```yaml
grid_setting:
  cas: SPL_CUBIC
```

`SPL_QUADRATIC` is not an accepted spelling; the CLI expects `SPL_QUADR`.
Changing `cas` changes the mesh approximation, so compare energies and forces
only after checking convergence with `N`, smoothing, and the solver tolerance.

## Non-electrostatic potentials

`md_variables.not_elec: true` enables the selected pair potential. Set
`potential` and provide its CSV using `potential_params_file`:

```yaml
md_variables:
  elec: true
  not_elec: true
  potential: TF
  potential_params_file: examples/paper/data/molten-salt/tosi_fumi.csv
```

| `potential` | CSV columns | Example | Notes |
| --- | --- | --- | --- |
| `TF` | `type1,type2,A,B,C,D,sigma` | [Molten NaCl](../examples/paper/data/molten-salt/tosi_fumi.csv) | Tosi–Fumi; one row per unique species pair. |
| `LJ` | `type1,type2,sigma,epsilon` | [SPC/FW water](../examples/paper/data/water/lennard_jones.csv) | Lennard–Jones; one row per unique species pair. |
| `SC` | `nu,d` | One-row CSV | Shared power-law potential; `nu` and `d` must be positive. |

For `TF`, `A` is in kJ/mol, `B` in Å⁻¹, `C` in kJ·Å⁶/mol, `D` in kJ·Å⁸/mol,
and `sigma` in Å. For `LJ`, `sigma` is in Å and `epsilon` in kJ/mol. For `SC`,
`nu` is dimensionless and `d` is in Å; the code derives the remaining
coefficient. With two species, pair tables need exactly three unique rows,
such as Na–Na, Na–Cl, and Cl–Cl. Use the same species names as in `species.csv`.

To select `LJ`, for instance, set `potential: LJ` and point
`potential_params_file` to [the water parameter CSV](../examples/paper/data/water/lennard_jones.csv).
For `SC`, the file has one row shared by all particle pairs, such as this
format-only illustration (the numbers are not a validated parameter set):

```csv
nu,d
6,3.0
```

`neighbor_r_cut_ang` sets the pair cutoff (default: half the box length).
`lj_force_shift` defaults to `true`; set it to `false` for the unshifted
Lennard–Jones form with the tail-energy correction used by the water paper
inputs. `elec` independently enables mesh electrostatic forces. The current
initializer still requires a valid `potential` and parameter CSV even when
`not_elec: false`; in that case no non-electrostatic pair force is added.

## Thermostat and integrator

To use the OVRVO Langevin thermostat, configure the MD section as follows:

```yaml
md_variables:
  integrator: OVRVO
  thermostat: true
  T: 298.15
  gamma: 1.0e-3
  dt_fs: 0.5
```

`T` is the target temperature in kelvin and `dt_fs` is the timestep in
femtoseconds. `gamma` is a positive damping rate in inverse atomic time units;
the default is `1.0e-3`. The OVRVO implementation uses the timestep rescaling
described by Sivak, Chodera, and Crooks in [*J. Phys. Chem. B* **118**,
6466–6474 (2014)](https://pubs.acs.org/doi/10.1021/jp411770f),
doi:10.1021/jp411770f.

`integrator: OVRVO` is the default, but `thermostat: false` (also the default)
disables its stochastic thermostat. The supplied paper examples set
`thermostat: false`. With `integrator: VERLET` and `thermostat: true`, the code
instead rescales velocities to the target temperature; that is a different
procedure from the OVRVO Langevin thermostat.

## Advanced solver options

For `MAZE-MULTIGRID`, the correction field can reuse earlier values:

```yaml
grid_setting:
  y_initial_guess: BASE
```

Choices: `BASE` (reuse), `VERLET` (linear extrapolation), `ORDER2` (second-order
extrapolation), `ZERO` (reset). For `MULTIGRID`, `phi_initial_guess` selects
`BASE`, `VERLET` (default), `ORDER2`, `ORDER3`, or `ORDER4`. At startup the code
uses only the available history.

The real-space fourth-order option is:

```yaml
grid_setting:
  discretization: MEHRSTELLEN4
  force_gradient_order: 4
md_variables:
  method: MAZE-MULTIGRID
```

`MEHRSTELLEN4` uses a 19-point Mehrstellen Laplacian and a multigrid
preconditioned Krylov solver. It is currently limited to `MAZE-MULTIGRID` and
is unavailable for Poisson–Boltzmann. For the standard discretization, set
`mg_krylov: true` to try the same Krylov wrapper.

Charge-assignment window deconvolution can be enabled independently of
smoothing:

```yaml
grid_setting:
  smoothing_deconvolve_window: true
```

For the work in progress on spatially and/or field-dependent dielectric
response and Poisson–Boltzmann, see the [methods guide](methods.md).
