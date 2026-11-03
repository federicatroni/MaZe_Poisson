"""Dielectric-slab slit with the exact wall stress-tensor closure (MAZE_ST_WALL=1): MaZe run driver used for the 1.0 / 1.5 nm slit runs.
Solvent eps = diag(eps_par, eps_par, eps_perp) only inside |z - L/2| <= H (H = MAZE_SLIT_H, default 5 A), eps = MAZE_EPS_OUT (default 1) outside.
model = cav (dielectric cavities, eps_int = 2, radii MAZE_RADII, default 1.5,3.0 A) | point (MAZE_NO_CAVITY=1, radii only set the ST balls).
Usage: run_slit_slab.py model start.csv outdir N_steps [eps_perp=2.1] [L=40] [h=0.5] [thermostat=0] [stride=1] [seed]
Env: MAZE_EPS_PAR (default 59.2), MAZE_SLIT_H, MAZE_RADII, MAZE_SC_D (hard15 recommended), MAZE_TOL (default 1e-7), MAZE_DT (fs, default 0.5),
MAZE_OUT_FORMAT (csv | bin, default csv), MAZE_BIN_PRECISION (float32 | float64, default float32).
Walls: MAZE_WALL = L/2,H,0.98833*R+,0.98833*R- (types P,M), purely steric. See README.md for the C-library variables."""
import os, sys, numpy as np
model, start, outdir, nsteps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
eperp = float(sys.argv[5]) if len(sys.argv) > 5 else 2.1
L = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0; h = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
thermo = len(sys.argv) > 8 and sys.argv[8] == '1'; stride = int(sys.argv[9]) if len(sys.argv) > 9 else 1
if len(sys.argv) > 10: np.random.seed(int(sys.argv[10]))
os.makedirs(outdir, exist_ok=True)
os.environ['MAZE_EPS_PERP'] = str(eperp)
RP, RM = map(float, os.environ.get('MAZE_RADII', '1.5,3.0').split(','))
os.environ['MAZE_WALL'] = f"{L/2},{os.environ.get('MAZE_SLIT_H', '5.0')},{0.98833*RP:.4f},{0.98833*RM:.4f}"
os.environ.setdefault('MAZE_SLAB_CORR', '1')   # Yeh-Berkowitz slab correction with the slab eps_z (see README.md)
os.environ.setdefault('MAZE_EPS_OUT', '1')
os.environ.setdefault('MAZE_ST_WALL', '1')     # exact closure of the ST control volume at the dielectric interface
os.environ.setdefault('MAZE_MG_SEMI', '1')     # semi-coarsening multigrid (needed for eps_perp << eps_par)
epar = float(os.environ.get('MAZE_EPS_PAR', '59.2'))
if model == 'point': os.environ['MAZE_NO_CAVITY'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # repo root
from maze_poisson.myio.input.dct import initialize_from_dict
from maze_poisson.solver import SolverMD
with open(f'{outdir}/species.csv', 'w') as f: f.write(f"type,charge,mass,radius\nP,1.0,22.99,{RP}\nM,-1.0,35.45,{RM}\n")
# SC diameters d(++), d(+-), d(--) [A]; 'hard15' = beta U_SC(Ri+Rj) = 15 kT with U = B (d/r)^48, B = 1.74/(6 nu d) Ha
scd = os.environ.get('MAZE_SC_D', 'hard15')
def _d15(s):
    kT = 298.15*3.166811563e-6; return (15*kT*s**48*288/(1.74*0.529177210903))**(1/47)
if scd == 'hard15': scd = '3.170,4.796,6.434' if (RP, RM) == (1.5, 3.0) else ','.join(f'{_d15(x):.3f}' for x in (2*RP, RP+RM, 2*RM))
dpp, dpm, dmm = scd.split(',')
with open(f'{outdir}/sc.csv', 'w') as f: f.write(f"type1,type2,nu,d\nP,P,48,{dpp}\nP,M,48,{dpm}\nM,M,48,{dmm}\n")
n_p = sum(1 for _ in open(start)) - 1
d = {'grid_setting': dict(N=int(round(L/h)), N_p=n_p, L_ang=L, eps_s=epar, eps_int=2.0, N_typs=2, particles_file=f'{outdir}/species.csv',
                          input_file=start, I=0.0, w_ang=0.0),
     'output_settings': dict(path=f'{outdir}/', print_energy=True, print_solute=True, print_temperature=True, stride=stride, flushstride=10,
                             format=os.environ.get('MAZE_OUT_FORMAT', 'csv'), bin_precision=os.environ.get('MAZE_BIN_PRECISION', 'float32')),
     'md_variables': dict(N_steps=nsteps, T=298.15, dt_fs=float(os.environ.get('MAZE_DT', '0.5')), potential='SC', potential_params_file=f'{outdir}/sc.csv',
                          integrator='OVRVO' if thermo else 'VERLET', thermostat=thermo, gamma=1e-3,
                          method='MAZE-MULTIGRID', tol=float(os.environ.get('MAZE_TOL', '1e-7')), poisson_boltzmann=True, eps_map='SPHERE', pb_force='STRESS_TENSOR',
                          stress_tensor_bc='PBC', probe_radius_ang=0.0)}
gs, os_, ms = initialize_from_dict(d)
SolverMD(gs, ms, os_).run()
