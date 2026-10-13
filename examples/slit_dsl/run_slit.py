"""Slit pore (dos Santos & Levin, arXiv:2607.23812) with a uniaxial solvent: MaZe run driver.
model = cav (dielectric cavities R+=1.5, R-=3.0, eps_int=2) | point (MAZE_NO_CAVITY=1, same radii as ST integration radii).
Usage: run_slit.py model start.csv outdir N_steps [eps_perp=4] [L=40] [h=0.5] [thermostat=0] [stride=1] [seed]
Walls: MAZE_WALL = L/2,5,1.4825,2.9641 (types P,M). See README.md for the environment variables."""
import os, sys, numpy as np
model, start, outdir, nsteps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
eperp = float(sys.argv[5]) if len(sys.argv) > 5 else 4.0
L = float(sys.argv[6]) if len(sys.argv) > 6 else 40.0; h = float(sys.argv[7]) if len(sys.argv) > 7 else 0.5
thermo = len(sys.argv) > 8 and sys.argv[8] == '1'; stride = int(sys.argv[9]) if len(sys.argv) > 9 else 1
if len(sys.argv) > 10: np.random.seed(int(sys.argv[10]))
os.makedirs(outdir, exist_ok=True)
os.environ['MAZE_EPS_PERP'] = str(eperp)
os.environ['MAZE_WALL'] = f"{L/2},5.0,1.4825,2.9641"
os.environ.setdefault('MAZE_SLAB_CORR', '1')   # Yeh-Berkowitz (uniaxial) slab correction, see README.md
if model == 'point': os.environ['MAZE_NO_CAVITY'] = '1'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # repo root
from maze_poisson.myio.input.dct import initialize_from_dict
from maze_poisson.solver import SolverMD
with open(f'{outdir}/species.csv', 'w') as f: f.write("type,charge,mass,radius\nP,1.0,22.99,1.5\nM,-1.0,35.45,3.0\n")
# SC diameters d(++), d(+-), d(--) [A]; env MAZE_SC_D overrides. Default = old calibration (beta U(Ri+Rj) ~ 1-2.6 kT, rare small overlaps);
# 'hard15' = beta U_SC(Ri+Rj) = 15 kT with MaZe B(d) = 1.74/(6 nu d) Ha -> no cavity overlap in practice (Fede 2026-10-12)
scd = os.environ.get('MAZE_SC_D', '2.995,4.62,6.18')
if scd == 'hard15': scd = '3.170,4.796,6.434'
dpp, dpm, dmm = scd.split(',')
with open(f'{outdir}/sc.csv', 'w') as f: f.write(f"type1,type2,nu,d\nP,P,48,{dpp}\nP,M,48,{dpm}\nM,M,48,{dmm}\n")
n_p = sum(1 for _ in open(start)) - 1
d = {'grid_setting': dict(N=int(round(L/h)), N_p=n_p, L_ang=L, eps_s=78.54, eps_int=2.0, N_typs=2, particles_file=f'{outdir}/species.csv',
                          input_file=start, I=0.0, w_ang=0.0),
     'output_settings': dict(path=f'{outdir}/', print_energy=True, print_solute=True, print_temperature=True, stride=stride, flushstride=10),
     'md_variables': dict(N_steps=nsteps, T=298.15, dt_fs=0.5, potential='SC', potential_params_file=f'{outdir}/sc.csv',
                          integrator='OVRVO' if thermo else 'VERLET', thermostat=thermo, gamma=1e-3,
                          method='MAZE-MULTIGRID', tol=float(os.environ.get('TOL', '1e-7')), poisson_boltzmann=True, eps_map='SPHERE', pb_force='STRESS_TENSOR',
                          stress_tensor_bc='PBC', probe_radius_ang=0.0)}
gs, os_, ms = initialize_from_dict(d)
SolverMD(gs, ms, os_).run()
