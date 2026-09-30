"""Compact binary output (``format: bin`` in the output settings).

Every output is a headerless raw binary file ``<name>.bin`` plus a small JSON sidecar
``<name>.bin.json`` that describes the layout, so no extra dependency is needed to read it
(``read_maze_bin`` below, or plain ``numpy``). Little endian, C order.

Two layouts:

``table``   one fixed-size record per row (structured dtype). Used for the scalar outputs
            (performance, energy, momentum, temperature, tot_force). Scalars are always float64.

``frames``  one record per output step: ``iter`` (int64) followed by a fixed set of arrays
            (per-particle blocks, or N^3 grids). Used for solute, forces_pb and eps_map, which
            dominate the disk usage. The arrays use the precision selected with
            ``bin_precision`` (float32 by default, float64 available).

``restart`` and ``restart_field`` stay CSV because they are read back as simulation input.

TODO(bin format): only the writer and ``read_maze_bin`` exist. These consumers still read the CSV
outputs with ``pd.read_csv`` and have to be adapted to also accept ``<name>.bin`` (in most cases
``read_maze_bin(path)`` returns the same DataFrame, so it is a one-line change, e.g. choosing the reader
from the file extension or from ``output_settings.format``):
  - maze_poisson/cli/analyzers/analyzers.py     reads solute
  - maze_poisson/cli/plotters/plot_T_E_tot.py   reads energy, solute and temperature
  - maze_poisson/cli/plotters/plot_force.py     reads tot_force
  - maze_poisson/cli/plotters/plot_scaling.py   reads performance
  - maze_poisson/cli/utilities/convert_to_xyz.py  reads a (merged) solute file
Note that most of these still use the old file names (e.g. ``solute_N<N>.csv``,
``energy_N<N>_N_p_<N_p>.csv``, ``performance_N<N>_N_p<N_p>.csv``) that the current writers no longer produce
(they write ``solute.csv``, ``energy.csv``, ... inside the run directory), so they need the new names anyway.
Also to do: a pytest that writes both formats from the same run and compares them, and support for
appending to an existing ``.bin`` when a run is restarted in the same directory (today it overwrites).
"""
import io
import json
import os

import numpy as np
import pandas as pd

from ...c_api import capi
from . import csv as csv_out
from .base_out import BaseOutputFile, OutputFiles

VERSION = 1
ITER_DTYPE = '<i8'
PRECISIONS = {'float32': '<f4', 'float64': '<f8'}


class BinaryOutputFile(BaseOutputFile):
    """Base class: bytes buffer, append on flush, JSON sidecar written with the first record."""
    extension = 'bin'
    takes_precision = True
    csv_class = None  # CSV output class whose get_data() is reused to collect the data

    def __init__(self, *args, precision: str = 'float32', **kwargs):
        if precision not in PRECISIONS:
            raise ValueError(f"bin_precision must be one of {list(PRECISIONS)}, got '{precision}'")
        self.precision = precision
        self.float_dtype = PRECISIONS[precision]
        super().__init__(*args, **kwargs)
        self._schema_written = False
        if not self.enabled:
            return
        self.buffer.close()
        self.buffer = io.BytesIO()
        if os.path.exists(self.sidecar_path):
            try:
                os.remove(self.sidecar_path)
            except OSError:
                pass

    @property
    def sidecar_path(self):
        return self.path + '.json'

    def flush(self):
        if not self.enabled:
            return
        data = self.buffer.getvalue()
        if data:
            with open(self.path, 'ab') as f:
                f.write(data)
        self.buffer.truncate(0)
        self.buffer.seek(0)

    def write_data(self, iter: int, solver=None, mode: str = 'a', mpi_bypass: bool = False):
        if not self.enabled:
            # get_data has to run on all ranks when it contains MPI collectives (see the CSV files)
            if self._enabled and mpi_bypass:
                self.csv_class.get_data(self, iter, solver)
            return
        self._write_record(iter, solver)

    def _write_schema(self, schema: dict):
        if self._schema_written:
            return
        schema = {'maze_bin_version': VERSION, 'name': self.name, 'byteorder': 'little', **schema}
        with open(self.sidecar_path, 'w') as f:
            json.dump(schema, f, indent=2)
        self._schema_written = True

    def _write_record(self, iter: int, solver):
        raise NotImplementedError


class TableBinaryOutputFile(BinaryOutputFile):
    """One fixed-size record per row."""
    columns = None  # list of (column name, numpy dtype string)

    def _write_record(self, iter: int, solver):
        df = self.csv_class.get_data(self, iter, solver)
        dtype = np.dtype([(name, dt) for name, dt in self.columns])
        rec = np.empty(len(df), dtype=dtype)
        for name, _ in self.columns:
            rec[name] = df[name].to_numpy()
        self._write_schema({
            'layout': 'table',
            'columns': [{'name': n, 'dtype': d} for n, d in self.columns],
        })
        self.buffer.write(rec.tobytes())


class FrameBinaryOutputFile(BinaryOutputFile):
    """One record per output step: iter (int64) followed by the arrays of ``groups``."""
    # list of (array name, [dataframe columns]) taken from csv_class.get_data()
    groups = None
    particle_column = True  # rebuild a 'particle' column (0..N-1) when converting to a DataFrame

    def get_arrays(self, iter: int, solver) -> dict:
        df = self.csv_class.get_data(self, iter, solver)
        out = {}
        for name, cols in self.groups:
            block = df[cols].to_numpy(dtype=self.float_dtype)
            out[name] = block[:, 0] if len(cols) == 1 else block
        return out

    def dataframe_spec(self):
        """Mapping array -> columns and column order for the DataFrame reconstruction."""
        return {
            'arrays_columns': {name: cols for name, cols in self.groups},
            'particle_column': self.particle_column,
            'csv_headers': list(self.csv_class.headers),
        }

    def _write_record(self, iter: int, solver):
        arrays = self.get_arrays(iter, solver)
        shapes = {name: list(a.shape) for name, a in arrays.items()}
        if not self._schema_written:
            self._shapes = shapes
            self._write_schema({
                'layout': 'frames',
                'iter_dtype': ITER_DTYPE,
                'precision': self.precision,
                'arrays': [{'name': n, 'dtype': self.float_dtype, 'shape': shapes[n]} for n in arrays],
                **self.dataframe_spec(),
            })
        elif shapes != self._shapes:
            raise ValueError(
                f"'{self.name}' output changed shape ({self._shapes} -> {shapes}); "
                "the binary format needs a fixed number of particles/grid points"
            )
        self.buffer.write(np.asarray(iter, dtype=ITER_DTYPE).tobytes())
        for name, a in arrays.items():
            self.buffer.write(np.ascontiguousarray(a, dtype=self.float_dtype).tobytes())


# --------------------------------------------------------------------------------------------- outputs

class PerformanceBinaryOutputFile(TableBinaryOutputFile):
    name = 'performance'
    csv_class = csv_out.PerformanceCSVOutputFile
    columns = [('iter', '<i8'), ('time', '<f8'), ('n_iters', '<i8'), ('eps_phi_iters', '<i8')]


class EnergyBinaryOutputFile(TableBinaryOutputFile):
    name = 'energy'
    csv_class = csv_out.EnergyCSVOutputFile
    columns = [('iter', '<i8'), ('K', '<f8'), ('V_notelec', '<f8'), ('V_elec', '<f8')]


class MomentumBinaryOutputFile(TableBinaryOutputFile):
    name = 'momentum'
    csv_class = csv_out.MomentumCSVOutputFile
    columns = [('iter', '<i8'), ('Px', '<f8'), ('Py', '<f8'), ('Pz', '<f8')]


class TemperatureBinaryOutputFile(TableBinaryOutputFile):
    name = 'temperature'
    csv_class = csv_out.TemperatureCSVOutputFile
    columns = [('iter', '<i8'), ('T', '<f8')]


class TotForcesBinaryOutputFile(TableBinaryOutputFile):
    name = 'forces_tot'
    csv_class = csv_out.TotForcesCSVOutputFile
    columns = [('iter', '<i8'), ('Fx', '<f8'), ('Fy', '<f8'), ('Fz', '<f8')]


class SolutesBinaryOutputFile(FrameBinaryOutputFile):
    name = 'solute'
    csv_class = csv_out.SolutesCSVOutputFile
    groups = [
        ('charge', ['charge']),
        ('data', ['x', 'y', 'z', 'vx', 'vy', 'vz', 'fx_elec', 'fy_elec', 'fz_elec']),
    ]


class ForcesPBoltzBinaryOutputFile(FrameBinaryOutputFile):
    name = 'forces_pb'
    csv_class = csv_out.ForcesPBoltzCSVOutputFile
    groups = [
        ('forces', ['Fx_RF', 'Fy_RF', 'Fz_RF', 'Fx_DB', 'Fy_DB', 'Fz_DB',
                    'Fx_IB', 'Fy_IB', 'Fz_IB', 'Fx_NP', 'Fy_NP', 'Fz_NP']),
    ]


class EpsMapBinaryOutputFile(FrameBinaryOutputFile):
    """Dielectric map: three (N, N, N) grids per frame (no DataFrame form, read as arrays)."""
    name = 'epsilon_map'
    csv_class = csv_out.EpsMapCSVOutputFile

    def get_arrays(self, iter: int, solver) -> dict:
        maps = [np.empty((solver.N, solver.N, solver.N), dtype=np.float64) for _ in range(3)]
        capi.get_eps_map(*maps)
        return {k: m.astype(self.float_dtype) for k, m in zip(('eps_x', 'eps_y', 'eps_z'), maps)}

    def dataframe_spec(self):
        return {'arrays_columns': None, 'particle_column': False, 'csv_headers': None}


OutputFiles.register_format(
    'bin',
    {
        'performance': PerformanceBinaryOutputFile,
        'energy': EnergyBinaryOutputFile,
        'momentum': MomentumBinaryOutputFile,
        'temperature': TemperatureBinaryOutputFile,
        'solute': SolutesBinaryOutputFile,
        'tot_force': TotForcesBinaryOutputFile,
        'forces_pb': ForcesPBoltzBinaryOutputFile,
        'eps_map': EpsMapBinaryOutputFile,
        # Read back as simulation input: keep the CSV writers (and the .csv extension)
        'restart': csv_out.RestartCSVOutputFile,
        'restart_field': csv_out.RestartFieldCSVOutputFile,
    }
)


# ------------------------------------------------------------------------------------------------ reader

def read_maze_bin(path: str, as_dataframe: bool = True, float64: bool = False):
    """Read a ``.bin`` output written with ``format: bin``.

    Returns a DataFrame with the same columns as the CSV output of the same name (``as_dataframe=True``),
    or, for outputs without a DataFrame form (eps_map) or with ``as_dataframe=False``, a dict of arrays:
    ``{'iter': (F,), <array name>: (F, *shape)}``. ``float64=True`` casts float32 columns to float64.
    A partially written last record (interrupted run) is ignored.
    """
    path = str(path)
    with open(path + '.json') as f:
        meta = json.load(f)
    if meta.get('maze_bin_version') != VERSION:
        raise ValueError(f"Unsupported maze bin version {meta.get('maze_bin_version')} in {path}.json")

    if meta['layout'] == 'table':
        dtype = np.dtype([(c['name'], c['dtype']) for c in meta['columns']])
        rec = _read_records(path, dtype)
        df = pd.DataFrame({name: rec[name] for name in dtype.names})
        return _maybe_float64(df, float64) if as_dataframe else {n: rec[n] for n in dtype.names}

    fields = [('iter', meta['iter_dtype'])]
    fields += [(a['name'], a['dtype'], tuple(a['shape'])) for a in meta['arrays']]
    rec = _read_records(path, np.dtype(fields))
    arrays = {'iter': rec['iter']}
    arrays.update({a['name']: rec[a['name']] for a in meta['arrays']})

    cols = meta.get('arrays_columns')
    if not as_dataframe or cols is None:
        return arrays

    n_frames = len(rec)
    n_p = meta['arrays'][0]['shape'][0]
    data = {'iter': np.repeat(rec['iter'], n_p)}
    if meta.get('particle_column'):
        data['particle'] = np.tile(np.arange(n_p), n_frames)
    for name, names in cols.items():
        block = rec[name].reshape(n_frames * n_p, len(names))
        for j, col in enumerate(names):
            data[col] = block[:, j]
    df = pd.DataFrame(data)
    if meta.get('csv_headers'):
        df = df[meta['csv_headers']]
    return _maybe_float64(df, float64)


def _read_records(path: str, dtype: np.dtype):
    n = os.path.getsize(path) // dtype.itemsize  # ignore a partial trailing record
    return np.fromfile(path, dtype=dtype, count=n)


def _maybe_float64(df: pd.DataFrame, float64: bool):
    if not float64:
        return df
    f32 = [c for c in df.columns if df[c].dtype == np.float32]
    return df.astype({c: np.float64 for c in f32})
