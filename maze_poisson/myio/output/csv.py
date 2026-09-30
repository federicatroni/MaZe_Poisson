from abc import abstractmethod

import numpy as np
import pandas as pd

from ...c_api import capi
from ...constants import a0, conv_mass
from .base_out import BaseOutputFile, OutputFiles


class CSVOutputFile(BaseOutputFile):
    extension = 'csv'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_headers()

    def init_headers(self):
        if not self.enabled:
            return
        pd.DataFrame(columns=self.headers).to_csv(self.buffer, index=False)

    def write_data(self, iter: int, solver = None, mode: str = 'a', mpi_bypass: bool = False):
        if not self.enabled:
            # Needed when the get_data method needs to be called on all ranks to avoid MPI deadlock/desyncs
            if self._enabled and mpi_bypass:
                self.get_data(iter, solver)
            return
        header = False
        if mode == 'w':
            open(self.path, 'w').close()
            self.buffer.truncate(0)
            self.buffer.seek(0)
            header = True
        df = self.get_data(iter, solver)
        df.to_csv(self.buffer, columns=self.headers, header=header, index=False, mode=mode)

    @property
    @abstractmethod
    def headers(self) -> list[str]:
        pass

    @abstractmethod
    def get_data(self, iter: int, solver) -> pd.DataFrame:
        pass

class EnergyCSVOutputFile(CSVOutputFile):
    name = 'energy'
    headers = ['iter', 'K', 'V_notelec', 'V_elec']
    
    def get_data(self, iter: int, solver):
        kin = capi.get_kinetic_energy()
        deltaG_elec = capi.get_energy_elec()  # Needs mpi_bypass=True

        return pd.DataFrame({
            'iter': [iter],
            'K': [kin],
            'V_notelec': [solver.potential_notelec],
            'V_elec': [deltaG_elec],
            # 'DeltaG_nonpolar': [solver.energy_nonpolar],
        })

class MomentumCSVOutputFile(CSVOutputFile):
    name = 'momentum'
    headers = ['iter', 'Px', 'Py', 'Pz']
    def get_data(self, iter: int, solver):
        momentum = np.empty(3, dtype=np.float64)
        capi.get_momentum(momentum)
        return pd.DataFrame({
            'iter': [iter],
            'Px': [momentum[0]],
            'Py': [momentum[1]],
            'Pz': [momentum[2]]
        })

class TotForcesCSVOutputFile(CSVOutputFile):
    name = 'forces_tot'
    headers = ['iter', 'Fx', 'Fy', 'Fz']
    def get_data(self, iter: int, solver):
        forces = np.empty((solver.N_p, 3), dtype=np.float64)
        capi.get_fcs_tot(forces)
        df = pd.DataFrame(forces.sum(axis=0).reshape(1,3), columns=['Fx', 'Fy', 'Fz'])
        df['iter'] = iter
        return df

class ForcesPBoltzCSVOutputFile(CSVOutputFile):
    name = 'forces_pb'
    headers = [
        'iter', 'particle',
        'Fx_RF', 'Fy_RF', 'Fz_RF',
        'Fx_DB', 'Fy_DB', 'Fz_DB', 'Fx_IB', 'Fy_IB', 'Fz_IB', 'Fx_NP', 'Fy_NP', 'Fz_NP'
        ]
    def get_data(self, iter: int, solver):
        df = pd.DataFrame()
        forces = np.empty((solver.N_p, 3), dtype=np.float64)

        capi.get_fcs_elec(forces)
        df[['Fx_RF', 'Fy_RF', 'Fz_RF']] = forces

        capi.get_fcs_db(forces)
        df[['Fx_DB', 'Fy_DB', 'Fz_DB']] = forces

        capi.get_fcs_ib(forces)
        df[['Fx_IB', 'Fy_IB', 'Fz_IB']] = forces

        capi.get_fcs_np(forces)
        df[['Fx_NP', 'Fy_NP', 'Fz_NP']] = forces

        df['iter'] = iter
        df['particle'] = range(solver.N_p)
        return df

class TemperatureCSVOutputFile(CSVOutputFile):
    name = 'temperature'
    headers = ['iter', 'T']
    def get_data(self, iter: int, solver):
        temp = capi.get_temperature()
        return pd.DataFrame({
            'iter': [iter],
            'T': [temp]
        })

class SolutesCSVOutputFile(CSVOutputFile):
    name = 'solute'
    headers = ['charge', 'iter', 'particle', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx_elec', 'fy_elec', 'fz_elec']
    def get_data(self, iter: int, solver):
        tmp = np.empty((solver.N_p, 3), dtype=np.float64)
        df = pd.DataFrame()
        
        capi.get_pos(tmp)
        df[['x', 'y', 'z']] = tmp
        capi.get_vel(tmp)
        df[['vx', 'vy', 'vz']] = tmp
        capi.get_fcs_elec(tmp)
        df[['fx_elec', 'fy_elec', 'fz_elec']] = tmp

        tmp = np.empty(solver.N_p, dtype=np.float64)
        capi.get_charges(tmp)
        df['charge'] = tmp

        df['iter'] = iter
        df['particle'] = range(solver.N_p)
        return df

class PerformanceCSVOutputFile(CSVOutputFile):
    name =  'performance'
    headers = ['iter', 'time', 'n_iters', 'eps_phi_iters']
    def get_data(self, iter: int, solver):
        return pd.DataFrame({
            'iter': [iter],
            'time': [solver.t_iters],
            'n_iters': [solver.n_iters],
            'eps_phi_iters': [solver.eps_phi_iters],
        })

class RestartCSVOutputFile(CSVOutputFile):
    name = 'restart'
    headers = ['type', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    def get_data(self, iter: int, solver):
        df = pd.DataFrame()

        tmp = np.empty((solver.N_p, 3), dtype=np.float64)
        capi.get_pos(tmp)
        df[['x', 'y', 'z']] = tmp * a0
        capi.get_vel(tmp)
        df[['vx', 'vy', 'vz']] = tmp

        tmp = np.empty(solver.N_p, dtype=np.int32)
        capi.get_types(tmp)
        df['type'] = [solver.types_num_to_str[t] for t in tmp]

        return df

class RestartFieldCSVOutputFile(CSVOutputFile):
    name = 'restart_field'
    headers = ['phi_prev', 'phi']
    def get_data(self, iter: int, solver):
        df = pd.DataFrame()
        tmp = np.empty((solver.N, solver.N, solver.N), dtype=np.float64)
        capi.get_field(tmp)
        df['phi'] = tmp.flatten()
        capi.get_field_prev(tmp)
        df['phi_prev'] = tmp.flatten()

        return df

class EpsMapCSVOutputFile(CSVOutputFile):
    name = 'epsilon_map'
    headers = ['iter', 'eps_x', 'eps_y', 'eps_z']
    def get_data(self, iter, solver):
        df = pd.DataFrame()
        tmp_x = np.empty((solver.N, solver.N, solver.N), dtype=np.float64)
        tmp_y = np.empty((solver.N, solver.N, solver.N), dtype=np.float64)
        tmp_z = np.empty((solver.N, solver.N, solver.N), dtype=np.float64)
        capi.get_eps_map(tmp_x, tmp_y, tmp_z)

        df['eps_x'] = tmp_x.flatten()
        df['eps_y'] = tmp_y.flatten()
        df['eps_z'] = tmp_z.flatten()
        
        df['iter'] = iter

        return df
        # np.savez_compressed(
        #     filename,
        #     eps_x=eps_x,
        #     eps_y=eps_y,
        #     eps_z=eps_z,
        #     h=self.h,
        #     L=self.L,
        #     eps_s=self.gset.eps_s,
        #     eps_int=self.gset.eps_int,
        #     iter=iter_idx,
        # )

OutputFiles.register_format(
    'csv',
    {
        'performance': PerformanceCSVOutputFile,
        'energy': EnergyCSVOutputFile,
        'momentum': MomentumCSVOutputFile,
        'temperature': TemperatureCSVOutputFile,
        'solute': SolutesCSVOutputFile,
        'tot_force': TotForcesCSVOutputFile,
        'forces_pb': ForcesPBoltzCSVOutputFile,
        'restart': RestartCSVOutputFile,
        'restart_field': RestartFieldCSVOutputFile,
        'eps_map': EpsMapCSVOutputFile,
    }
)
