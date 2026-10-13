# Slit pore with a uniaxial solvent (dos Santos & Levin setup)

Replica of the slit of dos Santos & Levin (arXiv:2607.23812, Fig. 2): 20 Na+ + 20 Cl- in a 1 nm slit,
L_xy = 4 nm, solvent eps = diag(78.54, 78.54, eps_perp), eps_perp = 4. Ions are either dielectric cavities
(eps_int = 2, R+ = 1.5 A, R- = 3.0 A) or point charges.

## Environment variables (C library)
| variable | meaning |
|---|---|
| `MAZE_EPS_PERP` | eps on z edges (uniaxial solvent); unset = isotropic `eps_s` |
| `MAZE_NO_CAVITY=1` | point ions: uniform eps map, radii only set the stress-tensor integration balls |
| `MAZE_WALL="zc,H,s0_0,s0_1,..."` | soft walls at zc +- H [A], U = kT[(s0/s_top)^48 + (s0/s_bot)^48] on the ion centre, s0 per type [A]. Purely steric (eps unchanged). Only with `potential = SC`. MaZe stops if an ion crosses a plane |
| `MAZE_WALL_EPS_HA` | wall energy scale in Ha (default kT at 298.15 K) |
| `MAZE_SLAB_CORR=1` | Yeh-Berkowitz slab correction 2 pi M_z^2/(eps_perp V), needs `MAZE_WALL` and `MAZE_EPS_PERP`; goes into V_notelec |
| `MAZE_MG_MAXDEPTH`, `MAZE_MG_CORR_SCALE` | multigrid depth / coarse correction scale (already in this branch) |

## Run
```
MAZE_SC_D=hard15 OMP_NUM_THREADS=4 python3 run_slit.py cav|point start.csv outdir N_steps [eps_perp=4] [L=40] [h=0.5] [thermostat=1] [stride=20] [seed]
python3 make_restart.py outdir/solute.csv restart.csv   # last complete frame -> restart (A, velocities a.u.)
```
- `MAZE_SC_D=hard15`: soft-core diameters d = 3.170 / 4.796 / 6.434 A (++, +-, --), beta U_SC(R_i + R_j) = 15 kT, no cavity overlap in practice.
- Cubic box only (L = 40 A, h = 0.5 A -> 80^3); slit centred at L/2.
- `TOL` overrides the solver tolerance (default 1e-7).

## Known limitations / cost
- ~94% of the step is the PB multigrid. With eps_perp = 4 it needs 10-12 V-cycles/step vs 4 for isotropic eps (point smoother + full coarsening, anisotropy 19.6). Semi-coarsening in x,y is the expected fix (not implemented).
- At h = 0.5 point-ion z forces are ~10% low vs exact Ewald (grid); profiles of MaZe-point nevertheless agree with exact-Ewald MC.
- Stress tensor undefined when two cavities are closer than ~2-3h (gap); hard15 soft core keeps them apart.
