import atexit
import json
import os
import time
from abc import ABC, abstractmethod
from io import StringIO

import pandas as pd

from ...myio.loggers import Logger
from .. import get_enabled_io
from ..input import OutputSettings


def save_json(path: str, data: dict, overwrite: bool = True):
    """Save a dictionary to a JSON file."""
    enabled = get_enabled_io()
    if not enabled:
        return
    if os.path.exists(path):
        if overwrite:
            os.remove(path)
        else:
            raise ValueError(f"File {path} already exists")
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

class BaseOutputFile(Logger, ABC):
    name = None
    def __init__(self, *args, path: str, enabled: bool = True, overwrite: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.path = os.path.abspath(path)
        self._enabled = enabled
        self._io_enabled = get_enabled_io()

        if not self.enabled:
            return

        self.logger.info("Saving %s to '%s'", self.name, self.path)

        if os.path.exists(path):
            if overwrite:
                try:  # When running with MPI file could be locked
                    os.remove(path)
                except:
                    pass
            else:
                raise ValueError(f"File {path} already exists")

        self.buffer = StringIO()
        atexit.register(self.close)

    @property
    def enabled(self):
        return self._enabled and self._io_enabled

    @abstractmethod
    def write_data(self, df: pd.DataFrame, mode: str = 'a', mpi_bypass: bool = False):
        pass

    def flush(self):
        if self.enabled:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(self.buffer.getvalue())
            self.buffer.truncate(0)
            self.buffer.seek(0)

    def close(self):
        if self.enabled:
            self.flush()
            self.buffer.close()

class OutputFiles:
    performance = None
    energy = None
    momentum = None
    temperature = None
    solute = None
    tot_force = None
    forces_pb = None
    restart = None
    restart_field = None
    eps_map = None

    files = ['performance', 'energy', 'momentum', 'temperature', 'solute', 'tot_force', 'forces_pb', 'eps_map']
    files_restart = ['restart', 'restart_field']

    with_mpi_bypass = ['energy', 'restart_field']

    format_classes = {}

    last = -1

    def __init__(self, oset: OutputSettings):
        self.oset = oset
        self.base_path = oset.path
        self.fmt = oset.format

        if not os.path.exists(self.base_path) and get_enabled_io():
            try:
                os.makedirs(self.base_path, exist_ok=True)
            except:
                pass

        cnt = 0
        while not os.path.exists(self.base_path):
            cnt += 1
            time.sleep(0.5)
            if cnt > 10:
                raise ValueError(f"Failed to create output directory {self.base_path} after 10 attempts")

        if not os.path.isdir(self.base_path):
            raise ValueError(f"Output path {self.base_path} is not a directory")

        self.out_stride = oset.stride
        self.out_flushstride = (oset.flushstride or 0) * oset.stride
        self.restart_step = oset.restart_step

        self.init_files()

    def init_files(self):
        """Initialize the output files."""
        ptr = self.format_classes[self.fmt]
        for name in self.files + self.files_restart:
            try:
                cls = ptr[name]
            except KeyError:
                raise ValueError(f"Output format '{self.fmt}' does not support '{name}' output")
            # A format may reuse a writer of another format (e.g. restart files stay csv in 'bin')
            ext = getattr(cls, 'extension', None) or self.fmt
            _path = os.path.join(self.base_path, f'{name}.{ext}')
            kwargs = {}
            if getattr(cls, 'takes_precision', False):
                kwargs['precision'] = self.oset.bin_precision
            setattr(self, name, cls(
                path = _path,
                enabled = getattr(self.oset, f'print_{name}'),
                overwrite=True,
                **kwargs
            ))

    @property
    def all_files(self):
        return self.files + self.files_restart

    def flush(self):
        for fname in self.all_files:
            file = getattr(self, fname)
            if file:
                file.flush()

    def output(self, itr: int, solver, force: bool = False):
        """Output the results of the molecular dynamics loop."""
        if force or itr % self.out_stride == 0:
            if self.last != itr:
                self.last = itr
                for name in self.files:
                    file = getattr(self, name)
                    if file:
                        file.write_data(itr, solver, mpi_bypass=(name in self.with_mpi_bypass))
                if force or (self.out_flushstride and itr % self.out_flushstride == 0):
                    self.flush()
        if self.restart_step == itr:
            for name in self.files_restart:
                file = getattr(self, name)
                if file:
                    file.write_data(itr, solver, mode='w', mpi_bypass=(name in self.with_mpi_bypass))
                    file.flush()

    @classmethod
    def register_format(cls, name: str, classes: dict):
        cls.format_classes[name] = classes
