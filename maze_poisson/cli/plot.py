from .click_wrapper import click
from .maze import maze

filename_argument = click.argument(
    'filename',
    type=click.Path(exists=True),
    required=True,
    metavar='CSV_FILE',
    )

filename_argument_solute = click.argument(
    'filename_solute',
    type=click.Path(exists=True),
    required=True,
    metavar='CSV_FILE',
    )

dt_option = click.option(
    '--dt', 'dt',
    type=float,
    default=0.25,
    help='Timestep for the solute evolution given in fs and converted in a.u.',
    )

therm_option = click.option(
    '--therm', 'therm',
    type=str,
    default='N',
    help='Did you previously run with a Thermostat? (Y/N)',
    )

Np_option = click.option(
    '--np', 'np',
    type=int,
    default=250,
    help='Value of total number of particles (N_p)',
    )

N_option = click.option(
    '--n', 'n',
    type=int,
    default=120,
    help='Value of total number of grid points (N)',
    )

L_option = click.option(
    '--l', 'l',
    type=int,
    default=20.64,
    help='Length of box (L)',
    )

@maze.group()
def plot():
    pass

@plot.command()
@filename_argument
@dt_option
def force(filename, dt):
    from .plotters.plot_force import plot_force
    plot_force(filename, dt)

@plot.command()
@filename_argument
@dt_option
def forcemod(filename, dt):
    from .plotters.plot_force import plot_forcemod
    plot_forcemod(filename, dt)
    
@plot.command()
@filename_argument
@dt_option
def temperature(filename, dt):
    from .plotters.plot_T_E_tot import PlotT
    PlotT(filename, dt)
    
@plot.command()
@filename_argument
@dt_option
# @therm_option
@N_option
@Np_option
@L_option
@filename_argument_solute
def energy(filename, filename_solute, dt, n, np, l):
    from .plotters.plot_T_E_tot import plot_Etot_trp
    plot_Etot_trp(filename, filename_solute, dt, n, np, l)

@plot.command()
@Np_option
def time_vs_N3(np):
    from .plotters.plot_scaling import plot_time_iterNgrid
    plot_time_iterNgrid(np)

@plot.command()
@Np_option
def n_iter_vs_N3(np):
    from .plotters.plot_scaling import plot_convNgrid
    plot_convNgrid(np)
    
@plot.command()
def time_vs_Np():
    from .plotters.plot_scaling import plot_scaling_particles_time_iters
    plot_scaling_particles_time_iters()

@plot.command()
def iter_vs_Np():
    from .plotters.plot_scaling import plot_scaling_particles_conv
    plot_scaling_particles_conv()

@plot.command()
@filename_argument
def visualize(filename):
    from .plotters.plot_T_E_tot import visualize_particles
    visualize_particles(filename)

@plot.command()
def time_vs_thread():
    from .plotters.plot_scaling import time_vs_threads
    time_vs_threads()

@plot.command()
def strong_scaling_vs_thread():
    from .plotters.plot_scaling import strong_scaling_vs_threads
    strong_scaling_vs_threads()

@plot.command()
def weak_scaling_vs_thread():
    from .plotters.plot_scaling import weak_scaling_vs_threads
    weak_scaling_vs_threads()

@plot.command()
def iter_vs_tols():
    from .plotters.plot_scaling import iter_vs_tol
    iter_vs_tol()
