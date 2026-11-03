# Poisson MaZe, P3MaZe, and variable dielectric

The CLI runs molecular dynamics (`maze run md INPUT.yml`); it does not expose a
standalone Poisson solve. These names describe how the electrostatic part of an
MD run is configured. Run the examples from the repository root, after building
the C library as described in the [README](../README.md).

| Case | MaZe field solver | Charge smoothing | Electrostatic short-range correction |
| --- | --- | --- | --- |
| Poisson MaZe | `MAZE-MULTIGRID` or `MAZE-LCG` | `NONE` | None from smoothing |
| P3MaZe | `MAZE-MULTIGRID` or `MAZE-LCG` | `GAUSS` or a supported Wendland/diffusion option | Added by the electrostatic force calculation |

The research inputs in `data_p3m/paper_p3maze` use `MAZE-MULTIGRID` with
`charge_smoothing: NONE` for a Poisson MaZe baseline and `MAZE-MULTIGRID` with
`charge_smoothing: GAUSS` for P3MaZe. The corresponding direct P3M comparison
uses `method: MULTIGRID` with Gaussian smoothing. The examples here are small
checks of the current interface, not reproductions of the paper runs.

## Poisson MaZe

Method reference: Federica Troni, Davide Grassano, Jayashree Narayan, Benoît
Roux, and Sara Bonella, [“Mass-zero constrained molecular dynamics for electrostatic
interactions,” *Journal of Chemical Physics* **163**, 214106 (2025)](https://pubs.aip.org/aip/jcp/article/163/21/214106/3373572/Mass-zero-constrained-molecular-dynamics-for),
doi:10.1063/5.0283356.

Run `maze run md examples/quickstart.yml`. Its key settings are:

```yaml
grid_setting:
  N: 40
  L_ang: 20.0
  cas: CIC
  charge_smoothing: NONE
md_variables:
  method: MAZE-MULTIGRID
  tol: 1.0e-7
  elec: true
```

`N` is the mesh size along each axis; `L_ang` is the cubic box length in
ångströms. `cas` assigns particle charge to the mesh (`CIC`, `SPL_QUADR`, or
`SPL_CUBIC`). `tol` is the field solver tolerance. The examples from the paper
typically use `SPL_CUBIC` and a larger mesh; those choices must be converged
for the physical system. With `MAZE-MULTIGRID`, `y_initial_guess` controls the MaZe correction-field
predictor: `BASE` (default), `VERLET`, `ORDER2`, or `ZERO`. With `MAZE-LCG`,
`precond` selects the preconditioner; for `MAZE-MULTIGRID`, keep `precond: NONE`.

`md_variables.elec: true` enables electrostatic forces. `not_elec` separately
controls the non-electrostatic potential (`LJ`, `TF`, or `SC`). The current input
loader still requires the potential and its parameter CSV when `not_elec: false`.
For full NaCl and SPC/FW water inputs, see the [paper-derived examples](../examples/paper/README.md).

## P3MaZe

Method reference: Federica Troni, Violette Gontran, Davide Grassano, and Sara
Bonella, [“P3MaZe: a Mass-Zero constrained-dynamics formulation of
particle–mesh electrostatics”](https://pubs.acs.org/jctcce/article/doi/10.1021/acs.jctc.6c01315/5458363/P3MaZe-a-Mass-Zero-Constrained-Dynamics),
doi:10.1021/acs.jctc.6c01315.

Run `maze run md examples/p3maze.yml`. This example uses the same MaZe field
solver with Gaussian smoothing and a pair inside the short-range cutoff:

```yaml
grid_setting:
  cas: CIC
  charge_smoothing: GAUSS
  smoothing_rcut_ang: 4.5
  smoothing_sigma_ang: 1.39
  y_initial_guess: BASE
md_variables:
  method: MAZE-MULTIGRID
  elec: true
  neighbor_r_cut_ang: 4.5
```

`smoothing_sigma_ang` is the Gaussian width. The current Gaussian implementation
requires a positive `smoothing_rcut_ang`, but its real-space electrostatic
correction actually uses `md_variables.neighbor_r_cut_ang` as its cutoff. This
is also the particle-neighbor cutoff; when omitted it defaults to `L/2`. In
this example both are set to 4.5 Å, and `energy.csv` has a nonzero
`V_short_range`. Unsuffixed lengths (`L`, `smoothing_rcut`, `smoothing_sigma`,
`neighbor_r_cut`) are in atomic units; use `_ang` to input ångströms.

The paper's NaCl P3MaZe inputs include `SPL_CUBIC`,
`smoothing_rcut_ang: 4.5`, and `smoothing_sigma_ang: 1.39`. They do not set
`neighbor_r_cut_ang`, so the current code uses `L/2` for the real-space
correction cutoff. Some water inputs use unsuffixed `smoothing_rcut: 6.0` and
`smoothing_sigma: 1.87`, so those values are in atomic units. These are
system-specific parameters, not defaults. The NaCl runs also set `not_elec:
true` and `potential: TF`, with a Tosi–Fumi parameter file containing one row
per unique species pair and columns `type1,type2,A,B,C,D,sigma`.
Gaussian smoothing requires FFTW3. `charge_smoothing: NONE` removes the
associated short-range electrostatic correction. `not_elec: true` is a separate
choice for a non-electrostatic potential and does not activate P3MaZe.
The [paper-derived examples](../examples/paper/README.md) include NaCl and
SPC/FW water with `charge_smoothing: GAUSS` and complete input CSVs.

For explicit SPC water, set `iswater: true`, supply `O,H,H` triplets in the
positions CSV, and define `O` and `H` in the species CSV. Explicit SPC water is
separate from the variable-dielectric and Poisson–Boltzmann work below.

## Variable dielectric and Poisson–Boltzmann (work in progress)

Use the `eps_electric_field` branch for this work in progress. It includes
Poisson MaZe with permittivity depending on position and/or the electric field,
plus an adaptation for Poisson–Boltzmann. These cases are separate from the
Poisson MaZe and P3MaZe smoke examples on this branch. Configuration and
validation of the variable-dielectric cases are still evolving.

### Poisson MaZe with variable permittivity

The research branch covers a dielectric profile that varies in space, a
response that depends on the electric field, and combinations of the two. Its
input model includes `eps_s` and `eps_int` for solvent and solute permittivity,
and `eps_map`/`eps_field_alpha` for dielectric-map and field-response choices.
The branch is still in progress, so this guide does not present an unverified
standalone YAML recipe for those combinations.

### Poisson–Boltzmann adaptation

The Poisson–Boltzmann input adds spatial dielectric parameters and ionic
screening. A representative set of settings is:

```yaml
grid_setting:
  N: 32
  eps_s: 80.0
  eps_int: 2.0
  I: 0.1
  w_ang: 2.0
md_variables:
  method: MAZE-MULTIGRID
  poisson_boltzmann: true
  eps_map: TRADITIONAL
  pb_force: PB_ROUX
  elec: true
output_settings:
  print_forces_pb: true
```

`eps_s` and `eps_int` are the relative permittivities of solvent and solute.
`I` is ionic strength in mol/L (M); `T` in kelvin also enters the screening
term. `w_ang` is the dielectric transition width in ångströms. The species CSV
needs `radius` in ångströms; the effective solvent radius also includes
`md_variables.probe_radius_ang` (default 1.4 Å). A 32-point grid worked in a
small local smoke check; a 16-point grid was too small for the multigrid
recursion in that check.

`eps_map` accepts `TRADITIONAL`, `SPHERE`, or `FIELD_DEPENDENT`.
`pb_force` accepts `PB_ROUX` or `STRESS_TENSOR`; the latter requires
`eps_map: SPHERE`. `stress_tensor_bc` selects `DBC` or `PBC` for that force
method. `nonpolar_forces: true` enables the nonpolar contribution; its parameters
are `gamma_np` (kcal/mol/Å²) and `beta_np` (kcal/mol). For a nonzero `gamma_np`,
`eps_s` and `eps_int` must differ. The Poisson–Boltzmann path currently uses
`discretization: STANDARD`; `MEHRSTELLEN4` is not implemented for it.

Set physically relevant radii, dielectric constants, ionic strength, box size,
mesh, and boundary conditions before drawing scientific conclusions.
