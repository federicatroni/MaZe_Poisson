import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


@pytest.mark.parametrize('integrator', ['OVRVO', 'VERLET'])
def test_fixed_probe_dynamics_and_restart(tmp_path, integrator):
    root = Path(__file__).resolve().parents[1]
    if not (root / 'maze_poisson/libmaze_poisson.so').exists():
        pytest.skip('Build the C library first')
    (tmp_path / 'species.csv').write_text('type,charge,mass\nP,1,23\nM,-1,35\n')
    (tmp_path / 'sc.csv').write_text('nu,d\n12,2\n')
    initial = pd.DataFrame({
        'type': ['P', 'P', 'M', 'M'],
        'x': [5., 9., 12., 15.], 'y': [5., 6., 12., 15.], 'z': [5., 6., 12., 15.],
        'vx': [0.001] * 4, 'vy': [0.002] * 4, 'vz': [0.003] * 4,
        'fixed': [1, 0, 0, 0],
    })
    initial.to_csv(tmp_path / 'start.csv', index=False)
    script = '''
import numpy as np
from maze_poisson.myio.input import GridSetting, MDVariables, OutputSettings
from maze_poisson.solver import SolverMD
from maze_poisson.constants import a0, kB
from maze_poisson.c_api import capi
import sys
g = GridSetting(N=8, N_p=4, L=24/a0, N_typs=2,
                particles_file='species.csv', input_file='start.csv', eps_s=80)
m = MDVariables(N_steps=6, init_steps=3, T=300, dt_fs=0.05,
                potential='SC', potential_params_file='sc.csv',
                integrator=sys.argv[1], thermostat=True, gamma=0.001)
o = OutputSettings(path='out', print_restart=True)
s = SolverMD(g, m, o)
s.run()
pos = np.empty((4, 3)); vel = np.empty((4, 3)); force = np.empty((4, 3))
capi.get_pos(pos); capi.get_vel(vel); capi.get_fcs_tot(force)
np.testing.assert_array_equal(pos[0], np.array([5., 5., 5.])/a0)
np.testing.assert_array_equal(vel[0], np.zeros(3))
assert np.linalg.norm(force[0]) > 0
assert np.linalg.norm(vel[1:]) > 0
assert np.isclose(capi.get_temperature(), 2*capi.get_kinetic_energy()/(9*kB))
'''
    env = dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS='1', OMPI_MCA_btl='self')
    result = subprocess.run([sys.executable, '-c', script, integrator],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    restart = pd.read_csv(tmp_path / 'out/restart.csv')
    np.testing.assert_array_equal(restart.fixed, initial.fixed)
    np.testing.assert_allclose(restart.loc[0, ['x', 'y', 'z']].astype(float), [5, 5, 5])
    assert not np.allclose(restart.loc[1:, ['x', 'y', 'z']], initial.loc[1:, ['x', 'y', 'z']])
