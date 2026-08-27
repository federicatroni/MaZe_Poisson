from dataclasses import asdict, dataclass
from functools import wraps

from ...constants import a0, kB, t_au
from ...myio.loggers import logger


class BaseFileInput:
    @staticmethod
    def normalize_ang_to_au(dct):
        """Normalize Angstrom values to atomic units."""
        normalized = {}
        for key, value in dct.items():
            if key.endswith('_ang'):
                normalized[key.replace('_ang', '')] = value / a0
            else:
                normalized[key] = value
        return normalized

    @classmethod
    def from_dict(cls, dct):
        """Create an instance from a dictionary."""
        dct = cls.normalize_ang_to_au(dct)
        try:
            new = cls(**dct)
        except TypeError as e:
            logger.error(f"Error creating {cls.__name__}: {e}")
            exit(1)
        return new
    
    @classmethod
    def normalize_args(cls, dct):
        """Normalize Angstrom values in the dictionary."""
        dct = cls.normalize_ang_to_au(dct)
        return dct

    def to_dict(self):
        """Convert the instance to a dictionary."""
        return asdict(self)

    def __getattr__(self, key):
        multiplier = 1.0
        if '_ang' in key:
            key = key.replace('_ang', '')
            multiplier = a0
        return super().__getattribute__(key) * multiplier

@dataclass(kw_only=True)
class OutputSettings(BaseFileInput):
    print_solute: bool = False
    print_performance: bool = False
    print_momentum: bool = False
    print_energy: bool = False
    print_temperature: bool = False
    print_tot_force: bool = False
    print_forces_pb: bool = False
    print_restart: bool = False
    print_restart_field: bool = False
    print_eps_map: bool = False

    path: str = 'Outputs/'
    format: str = 'csv'

    stride: int = 50
    flushstride: int = 5

    debug: bool = False
    restart_step: int = None

@dataclass(kw_only=True)
class GridSetting(BaseFileInput):
    N: int
    N_p: int
    L: float
    h: float = None
    eps_s: float = 1.0  # Relative permittivity of the solvent (vacuum by default)
    eps_int: float = 1.0  # Relative permittivity inside the solute

    N_typs: int

    particles_file: str = 'species.csv'  # File containing particle definitions
    input_file: str

    restart_field_file: str = None
    cas: str = 'CIC'

    precond: str = 'NONE'
    smoother: str = 'LCG'

    # Poisson-Boltzmann specific
    I: float = None  # Ionic strength
    w: float = None  # Width of the transition region in Angstroms
            
    def __post_init__(self):
        """Post-initialization to set defaults."""
        if self.h is None:
            self.h = self.L / self.N
        elif self.N * self.h != self.L:
            raise ValueError("N * h must equal L. Check your values.")

@dataclass(kw_only=True)
class MDVariables(BaseFileInput):
    N_steps: int  # Number of steps in the simulation
    T: float  # Temperature in Kelvin
    dt_fs: float  # Timestep in femtoseconds

    init_steps: int = 0  # Initial steps before the main simulation
    # init_steps_thermostat: int = None  # Initial steps before the main simulation
    elec: bool = True # Whether to include electrostatic interactions
    not_elec: bool = True  # Whether to include non-electrostatic interactions

    potential: str = 'TF'  # Type of potential to use
    potential_params_file: str = None  # File containing potential parameters

    # Optional pairwise dielectric-saturation correction (Lenart et al., 2007).
    # MANYBODY preserves the existing electrostatic path unchanged.
    electrostatic_model: str = 'MANYBODY'  # MANYBODY or LENART_PAIRWISE
    electrostatic_params_file: str = None

    integrator: str = 'OVRVO'  # Integrator method
    method: str = 'FFT'  # Method for solving the Poisson equation
    tol: float = 1e-7  # Tolerance for convergence

    thermostat: bool = False  # Whether to use a thermostat
    gamma: float = 1e-3  # Damping coefficient for the thermostat

    rescale: bool = False  # Whether to rescale velocities
    invert_time: bool = False  # Whether to invert the time direction

    # Poisson-Boltzmann specific
    poisson_boltzmann: bool = False  # Whether to use Poisson-Boltzmann method
    nonpolar_forces: bool = False # Whether to use non polar forces or not
    field_dependent_dielectric: bool = False  # Backward-compatible shortcut for eps_map='FIELD_DEPENDENT'
    eps_map: str = None  # TRADITIONAL, SPHERE, or FIELD_DEPENDENT
    pb_force: str = 'PB_ROUX'  # PB_ROUX or STRESS_TENSOR
    stress_tensor_bc: str = 'DBC'  # DBC or PBC
    eps_field_alpha: float = 1.0  # Alpha parameter in eps(E) (Hu & Wei Eq. S2)
    gamma_np: float = 0.0  # Non-polarization gamma in kcal/mol/A^2
    beta_np: float = 0.0  # offset in kcal/mol
    probe_radius: float = 1.4 / a0  # Probe radius in a.u.

    def __post_init__(self):
        """Post-initialization to set defaults."""
        if self.dt_fs <= 0:
            raise ValueError("dt_fs must be a positive value.")
        if self.eps_map is None:
            self.eps_map = 'FIELD_DEPENDENT' if self.field_dependent_dielectric else 'TRADITIONAL'
        self.electrostatic_model = self.electrostatic_model.upper()
        if self.electrostatic_model not in {'MANYBODY', 'LENART_PAIRWISE'}:
            raise ValueError(
                "electrostatic_model must be 'MANYBODY' or 'LENART_PAIRWISE'."
            )
        if self.electrostatic_model == 'LENART_PAIRWISE':
            if not self.elec:
                raise ValueError("LENART_PAIRWISE requires elec=true.")
            if self.poisson_boltzmann:
                raise ValueError(
                    "LENART_PAIRWISE requires poisson_boltzmann=false: the base field must use uniform eps_s."
                )
            if self.electrostatic_params_file is None:
                raise ValueError(
                    "LENART_PAIRWISE requires electrostatic_params_file."
                )

    @property
    def kBT(self):
        return self.T * kB

    @property
    def dt(self):
        return self.dt_fs / t_au * (-1 if self.invert_time else 1)

    @property
    def gamma_np_au(self):
        """Return the non-polarization gamma in atomic units."""
        return self.gamma_np * 0.0065934  # Convert to atomic units (a.u.)

def mpi_file_loader(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # obj = None
        # if MPIBase.master:
        #     obj = func(*args, **kwargs)
        # obj = mpi.comm.bcast(obj, root=0)
        obj = func(*args, **kwargs)
        return obj
    return wrapper
