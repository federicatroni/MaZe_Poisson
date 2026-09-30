# TODO(bin format): reads output files with pd.read_csv, adapt to also read .bin (see maze_poisson/myio/output/binary.py)
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...constants import a0, density, m_Cl, m_Na, t_au
from ...myio.loggers import logger
from . import get_N, get_Np, get_Np_input


def PlotT(filename, dt, label='iter', outdir='Outputs'):
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.basename(filename)
    outname = os.path.join(outdir, os.path.splitext(fname)[0] + '.pdf')
    
    df = pd.read_csv(filename)
    N = get_N(filename)
    N_p = get_Np(filename)

    dt /=  t_au

    T = df['T'][1:]
    iter = df[label][1:]
    rel_err = np.std(T)/np.mean(T)
    print(np.shape(T))
    print('\nmean:', np.mean(T))
    print('std:', np.std(T))
    print('relative error:', rel_err)
    plt.figure(figsize=(15, 6))
    plt.plot(iter, T, marker='.', color='red', markersize=5, label='T - mean value = ' + str(np.mean(T)) + '$\\pm$' + str(np.std(T)))
    plt.title('Temperature - dt =' + str(np.round(dt,4)) + ' fs - N =' + str(N) + '; N_p = '+str(N_p), fontsize=22)
    plt.xlabel('iter', fontsize=15)
    plt.ylabel('T (K)', fontsize=15)
    plt.axhline(1550)
    plt.legend()
    plt.grid()
    plt.savefig(outname, format='pdf')
    # logger.info("file saved at "+path_pdf+'T_N' + str(N) + '_dt' + str(np.round(dt,4)) + '_N_p_'+str(N_p)+'.pdf')
    plt.show()

def plot_Etot_trp(filename, filename_solute, dt, N, N_p, L, N_th=0, upper_lim=None, outdir='Outputs'):
    # N = get_N(filename)
    # N_p = get_Np(filename)
    # L = np.round((((N_p*(m_Cl + m_Na)) / (2*density))  **(1/3)) *1.e9, 4) / a0

    # File paths
    os.makedirs(outdir, exist_ok=True)
    work_file = os.path.join(outdir, f'work_trp_N{N}_N_p_{N_p}.csv')
    df_E = pd.read_csv(filename)
    outname = os.path.basename(outname)
    outname = os.path.join(outdir, os.path.splitext(outname)[0] + '.pdf')
    # outdir_pdf = os.path.join(outdir, 'PDFs')
    # os.makedirs(outdir_pdf, exist_ok=True)

    # Energy file columns
    K = df_E['K']
    V_notelec = df_E['V_notelec']
    N_steps_energy = len(df_E)
    recompute_work = False
    iterations = np.max(df_E['iter']) + 1

    # Check if the work file exists and has the correct number of lines
    if os.path.exists(work_file):
        work_df = pd.read_csv(work_file)
        if len(work_df) == N_steps_energy:

            # If the file exists and has the correct number of rows, use the work data
            Ework = work_df['work'].tolist()
            #print(f"Work data loaded from {work_file}")
            logger.info(f"Work data loaded from {work_file}")
        else:
            #print(f"Work file exists but has incorrect number of lines. Recomputing work.")
            logger.info("Work file exists but has incorrect number of lines. Recomputing work.")
            recompute_work = True
    else:
        #logger.error(f"Work file doesnt exist in specified path {work_file}")
        recompute_work = True
        #raise FileNotFoundError(f"Work file doesnt exist in specified path {work_file}")
    
    N_steps = iterations
    df = pd.read_csv(filename_solute)
    if recompute_work:
        Np = N_p
        Ework = np.zeros(N_steps)
        print('Np =', Np)

        gb = df.groupby('particle')
        xyz = gb.apply(lambda x: x[['x', 'y', 'z',]]).values.reshape(Np, N_steps, 3)
        f_xyz = gb.apply(lambda x: x[['fx_elec', 'fy_elec', 'fz_elec',]]).values.reshape(Np, N_steps-1, 3)

        delta = np.diff(xyz, axis=1)
        delta -= np.rint(delta / L) * L

        # Get the work per iteration done by all particles
        # Sum over particles (axis 0) and directions (axis 2)
        work = - np.sum((f_xyz[:, 1:] + f_xyz[:, :-1]) * delta / 2, axis=(0,2))

        # Integrate the work per iteration to get the total work at each iteration
        Ework[2:] = np.cumsum(work)[:-1]

        # Create the work file with header "iter,work" and save the computed work
        work_df = pd.DataFrame({'iter': range(N_steps), 'work': Ework})
        work_df.to_csv(work_file, index=False)
        print(f"Work data saved to {work_file}")

    # Compute total energy
    E_tot = K + V_notelec + Ework
    mean = np.mean(E_tot[:upper_lim])
    std = np.std(E_tot[:upper_lim])
    print(mean, std)
    mean_K = np.mean(K)
    mean_V_notelec = np.mean(V_notelec)
    mean_work = np.mean(Ework)
    iterations = range(N_steps)
    iter_E = df_E['iter']

    # Plotting the distance and kinetic energy in subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 8))

    print('N_th =', N_th, type(N_th))
    print('upper_lim =', upper_lim, type(upper_lim))
    # Plotting kinetic energy
    ax1.plot(iter_E[N_th:upper_lim], K[N_th:upper_lim], marker='.', color='b', markersize=5, label=f'Kinetic energy - $|\\frac{{<K>}}{{<V_{{elec}}>}}| ={np.abs(mean_K/mean_work):.4f}$')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy')
    ax1.set_title('Contributions to the total energy of the system - N = ' + str(N) + ', dt = ' + str(dt) + ' fs'+'; N_p_'+str(N_p))
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plotting potential contributions
    ax2.plot(iterations[N_th:upper_lim], Ework[N_th:upper_lim], marker='.', linestyle='-', color='red', markersize=1, label='Elec')
    ax2.plot(iterations[N_th:upper_lim], V_notelec[N_th:upper_lim], marker='.', linestyle='-', color='mediumturquoise', markersize=1, label=f'Not elec - $|\\frac{{<V_{{not elec}}>}}{{<V_{{elec}}>}}| ={np.abs(mean_V_notelec/mean_work):.4f}$')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Potential')
    ax2.set_title('Potential contributions')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Plotting total energy
    ax3.plot(iterations[N_th:upper_lim], E_tot[N_th:upper_lim], marker='.', color='lightgreen', markersize=5, label=f'Tot energy - rel err = {std/mean:.5f}, $\\Delta E / \\Delta V =$ {std/ np.std(Ework)}')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Tot E')
    ax3.set_title('Total energy over iterations')
    ax3.legend(loc='upper right')
    ax3.grid(True)
    plt.tight_layout()
    plt.savefig(outname, format='pdf')
    # logger.info("file saved at "+path_pdf + 'Energy_analysis_trp_N' + str(N) + '_dt_' + str(dt) + '_N_p_'+str(N_p)+'.pdf')
    plt.show()

def plot_work_trp(filename, path, N_th, L=19.659 / a0, outdir='Outputs'):
    N = get_N(filename)
    N_p = get_Np(filename)
    # Check if the work file already exists
    os.makedirs(outdir, exist_ok=True)
    # work_file = path + 'work_N' + str(N) +'_N_p_'+str(N_p)+ '.csv'
    work_file = os.path.join(path, f'work_N{N}_N_p_{N_p}.csv')
    if os.path.exists(work_file):
        # If the file exists, read the work from it
        work_df = pd.read_csv(work_file)
        Ework = work_df['work'].tolist()
        N_steps = len(Ework)
        print(f"Work data loaded from {work_file}")
    else:
        # If the file doesn't exist, compute the work
        df = pd.read_csv(path + 'solute_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
        Np = int(df['particle'].max() + 1)
        df_list = [df[df['particle'] == p].reset_index(drop=True) for p in range(Np)]
        iterations = df['iter']
        N_steps = int(iterations.max() + 1)
        Ework = np.zeros(N_steps)
        work = np.zeros((Np, N_steps))
        print('N_steps =', N_steps)
        print('Np =', Np)
        # Precompute steps and avoid repeated indexing
        for p in range(Np):
            df_p = df_list[p]
            x = df_p['x'].values
            y = df_p['y'].values
            z = df_p['z'].values
            fx = df_p['fx_elec'].values
            fy = df_p['fy_elec'].values
            fz = df_p['fz_elec'].values
            # Initialize cumulative displacement
            cx = np.zeros(N_steps)
            cy = np.zeros(N_steps)
            cz = np.zeros(N_steps)
            for i in range(1, N_steps):
                # Calculate displacement between consecutive steps
                dx = x[i] - x[i-1]
                dy = y[i] - y[i-1]
                dz = z[i] - z[i-1]
                # Apply minimum image convention to account for PBC
                dx -= np.rint(dx / L) * L
                dy -= np.rint(dy / L) * L
                dz -= np.rint(dz / L) * L
                # Accumulate the corrected displacements
                cx[i] = cx[i-1] + dx
                cy[i] = cy[i-1] + dy
                cz[i] = cz[i-1] + dz
                # Compute work for each particle p at each step i
                work[p][i] = - (np.trapz(fx[:i], x=cx[:i]) +
                                np.trapz(fy[:i], x=cy[:i]) +
                                np.trapz(fz[:i], x=cz[:i]))
        # Sum up the work across all particles
        Ework = np.add.reduce(work, axis=0)
        # Create the work file with header "iter,work" and save the computed work
        work_df = pd.DataFrame({'iter': range(N_steps), 'work': Ework})
        work_df.to_csv(work_file, index=False)
        print(f"Work data saved to {work_file}")
    # Plotting the work
    iterations = range(N_steps - 1)
    plt.figure(figsize=(15, 6))
    plt.plot(iterations[N_th:], Ework[N_th:-1], marker='.', color='red', markersize=5, label='Potential energy')
    plt.title('Total Work', fontsize=22)
    plt.xlabel('iter', fontsize=15)
    plt.ylabel('Work ($E_H$)', fontsize=15)
    out_path = os.path.join(outdir, f'Work_trp_N{N}_N_p_{N_p}.pdf')
    plt.legend(loc='upper left')
    plt.savefig(out_path, format='pdf')
    plt.show()

def store_T_analysys(filename, path, n_runs=10, outdir='Outputs'): 
    N = get_N(filename)
    N_p = get_Np(filename)
    avg_T_list = []
    std_T_list = []
    run_list = list(range(1, n_runs + 1))

    os.makedirs(outdir, exist_ok=True)

    # Open a file to store the results
    with open(path + '/temperature_analysis_N' + str(N)+'_N_p_'+str(N_p) + '.txt', 'w') as file:
        # Write the header of the file
        file.write('Run, Avg_T, Std_T\n')

        # Perform the runs and store the data
        for i in range(1,n_runs + 1):
            df = pd.read_csv(path + 'run_' + str(i) + '/output/temperature_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
            T = df['T']
            avg_T = np.mean(T)
            std_T = np.std(T)

            # Append the values for plotting
            avg_T_list.append(avg_T)
            std_T_list.append(std_T)

            # Write the results of this run to the file
            file.write(f'{i}, {avg_T:.6f}, {std_T:.6f}\n')

    # Plotting
    plt.figure(figsize=(10, 5))
    
    # Plot avg_T
    plt.errorbar(run_list, avg_T_list, std_T_list, linestyle='', marker='o', markersize=5, color='b', label='T', capsize=4)
    plt.axhline(np.mean(np.array(avg_T)), label ='mean T = ' + str(np.mean(np.array(avg_T))), color='r')
    plt.xlabel('# run')
    plt.ylabel('Temperature [K]')
    plt.legend()
    #plt.title('Study of the temperature')
    #plt.grid(True)
    out_path = os.path.join(outdir, f'temperature_study_N{N}_N_p_{N_p}.pdf')
    plt.savefig(out_path, format='pdf')
    # Display the plots
    plt.show()

def plot_vacf(input_path, dt):
    """
    Plot the velocity autocorrelation function (VACF) for Na and Cl species and save the plot.
    
    Args:
    input_path (str): Path to the folder containing the "Na/" and "Cl/" directories.
    dt (float): Time step in femtoseconds.
    
    Returns:
    None
    """
    species = ['Na', 'Cl']
    lim = [4800, 4800]
    for i, sp in enumerate(species):
        # Construct the file path
        file_path = os.path.join(input_path, sp, "AvVCT_slice.csv")
        
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Compute time array in femtoseconds
        time = data['iter'] * dt
        
        # Plot the VACF
        plt.figure(figsize=(8, 6))
        #plt.plot(time, data['value'], label=f'{sp} VACF')
        plt.plot(data['iter'], data['value'], label=f'{sp} VACF')
        #plt.xlabel('Time (fs)')
        plt.xlabel('Iter')
        plt.ylabel('Velocity Autocorrelation Function (a.u.)')
        plt.title(f'Velocity Autocorrelation function for {sp}')
        plt.legend()
        plt.xlim([0,lim[i]])
        plt.grid(True)
        
        # Save the plot in the same directory as the input file
        save_path = os.path.join(input_path, sp, f"{sp}_VACF_plot.pdf")
        plt.savefig(save_path)
        plt.show()
        plt.close()

def plot_vacf_normalized(input_path, dt):
    """
    Plot the normalized velocity autocorrelation function (VACF) for Na and Cl species and save the plot.
    
    Args:
    input_path (str): Path to the folder containing the "Na/" and "Cl/" directories.
    dt (float): Time step in femtoseconds.
    
    Returns:
    None
    """
    species = ['Na', 'Cl']
    color_list = ['b', 'royalblue']
    lim = [1.2, 1.2]
    for i, sp in enumerate(species):
        # Construct the file path
        file_path = os.path.join(input_path, sp, "AvVCT_" + sp + ".csv")
        
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Compute time array in femtoseconds
        time = data['iter'] * dt
        
        # Normalize the VACF
        vacf_normalized = data['value'] / data['value'].iloc[0]
        
        # Plot the normalized VACF
        plt.figure(figsize=(8, 4.5))
        plt.axhline(y=0, color='grey')
        plt.plot(time * 1e-3, vacf_normalized, label=f'{sp}', color=color_list[i])
        plt.xlabel('Time (ps)', fontsize=18)
        #plt.ylabel('Normalized Velocity Autocorrelation Function')
        plt.ylabel(r'$\langle \mathbf{v}(0) \cdot \mathbf{v}(t) \rangle / \langle \mathbf{v}(0)^2 \rangle$', fontsize=18)
        plt.xlim([0,lim[i]])
        #plt.title(f'Velocity Autocorrelation function for {sp}')
       
        plt.legend(fontsize=15)
        #plt.grid(True)
        
        # Save the plot in the same directory as the input file
        save_path = os.path.join(input_path, sp, f"{sp}_VACF_normalized_plot.pdf")
        plt.savefig(save_path)
        plt.show()
        plt.close()

def plot_vacf_normalized2(input_path, dt):
    """
    Plot the normalized velocity autocorrelation function (VACF) for Na and Cl species and save the plot.
    
    Args:
    input_path (str): Path to the folder containing the "Na/" and "Cl/" directories.
    dt (float): Time step in femtoseconds.
    
    Returns:
    None
    """
    species = ['Na', 'Cl']
    color_list = ['b', 'royalblue']
    lim = [0.6, 0.6]
    
    plt.figure(figsize=(15, 4.5))  # Create a figure for the subplots
    
    for i, sp in enumerate(species):
        # Construct the file path
        file_path = os.path.join(input_path, sp, "AvVCT_" + sp + ".csv")
        
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Compute time array in femtoseconds
        time = data['iter'] * dt
        
        # Normalize the VACF
        vacf_normalized = data['value'] / data['value'].iloc[0]
        
        # Create subplot for each species
        plt.subplot(1, 2, i + 1)  # 1 row, 2 columns, subplot index i+1
        plt.axhline(y=0, color='grey')
        plt.plot(time * 1e-3, vacf_normalized, label=f'{sp}', color=color_list[i])
        plt.xlabel('Time (ps)', fontsize=14)
        plt.ylabel(r'$\langle \mathbf{v}(0) \cdot \mathbf{v}(t) \rangle / \langle \mathbf{v}(0)^2 \rangle$', fontsize=14)
        plt.xlim([0, lim[i]])
        #plt.title(f'{sp} VACF', fontsize=16)
        plt.legend(fontsize=12)
        #plt.grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(input_path, 'VACF_normalized_plots.pdf')
    plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_integral_vacf(input_path, dt):
    """
    This function reads the VACF data from CSV files for both Na and Cl species, computes the 
    cumulative integral, and plots the integral as a function of time for each species.

    Parameters:
    - input_path: str, path to the directory containing the species folders.
    - dt: float, time step used in the simulation (in a.u.).

    The plot is saved as a PDF file in the respective species directories.
    """

    species = ['Na', 'Cl']
    
    for sp in species:
        # Construct the file path for the species
        file_path = os.path.join(input_path, sp, "AvVCT_" + sp + ".csv")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load the VACF data from the CSV file
        df = pd.read_csv(file_path)

        # Extract the VACF values
        vacf_values = df['value'].values

        # Compute the cumulative integral of the VACF
        integral_vacf = np.cumsum(vacf_values) * dt

        # Generate the time axis
        time_axis = df['iter'].values * dt * 1e-3 #time in ps

        # Plot the integral of the VACF as a function of time
        plt.figure(figsize=(15, 6))
        plt.plot(time_axis, integral_vacf, label=f'Integral of VACF ({sp})')
        plt.xlabel('Time (ps)')
        plt.ylabel('Integral of VACF')
        plt.xlim([0,1.5])
        plt.title(f'Integral of VACF as a function of time ({sp})')
        plt.legend()
        plt.grid(True)

        # Save the plot in the respective species directory
        output_plot = os.path.join(input_path, sp, f'integral_vacf_plot_{sp}.pdf')
        plt.savefig(output_plot, format='pdf')
        print(f"Plot saved to {output_plot}")
        plt.show()
        plt.close()

def plot_integrals_comparison(input_path, dt_fs, xlim=None, ylim=None, D_show=True):
    """
    This function reads the VACF data from CSV files for both Na and Cl species, computes the 
    cumulative integral, and plots the integral as a function of time for each species on the same plot.

    Parameters:
    - input_path: str, path to the directory containing the species folders.
    - dt_fs: float, time step in femtoseconds used in the simulation.

    The plot is saved as a PDF file in the specified directory.
    """
    
    # Convert dt from femtoseconds to atomic units and picoseconds
    dt_au = dt_fs / (2.4188843265857 * 1e-2)  # Time step in atomic units
    a_to_ps = 2.4188843265857 * 1e-5  # Conversion factor: 1 a.u. of time = 2.4188843265857e-5 ps
    dt_ps = dt_au * a_to_ps  # Time step in picoseconds

    species = ['Na', 'Cl']
    labels = {}
    integrals = {}
    times = {}

    for sp in species:
        # Construct the file path for the species
        file_path = os.path.join(input_path, sp, "AvVCT_" + sp + ".csv")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load the VACF data from the CSV file
        df = pd.read_csv(file_path)

        # Extract the VACF values
        vacf_values = df['value'].values

        # Compute the cumulative integral of the VACF
        integral_vacf = np.cumsum(vacf_values) * dt_au

        # Generate the time axis in ps
        time_axis_ps = df['iter'].values * dt_ps


        # Store results for plotting
        integrals[sp] = integral_vacf
        times[sp] = time_axis_ps
        labels[sp] = f'{sp}'

    # Plot the integrals of VACF as a function of time for both species
    plt.figure(figsize=(10, 6))
    for sp in species:
        plt.plot(times[sp], integrals[sp], label=labels[sp])

    plt.xlabel('Time (ps)')
    plt.ylabel('Integral (a.u.)')
    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    plt.title('Integral of VACF for Na and Cl')
    plt.legend()
    if D_show:
        plt.axhline(0.16, label='$D_{Na} = 0.16$ - Galamba, T=1539 K', color='blue', linestyle=':', linewidth=2)
        #plt.axhline(0.22, label='$D_{Na} = 0.22$ - Coretti, T=1550 K', color='violet', linestyle='-')
        plt.axhline(0.14, label='$D_{Cl} = 0.14$ - Galamba, T=1539 K', color='green', linestyle=':', linewidth=2)
        #plt.axhline(0.18, label='$D_{Cl} = 0.18$ - Coretti, T=1550 K', color='gold', linestyle='-')
    #plt.xlim([0, 1.5])  # Limit x-axis to 1.5 ps for better visibility

    # Save the plot
    output_plot = os.path.join(input_path, 'integral_vacf_comparison_plot_xlim_' + str(xlim) + '.pdf')
    plt.savefig(output_plot, format='pdf')
    print(f"Plot saved to {output_plot}")

    plt.show()
    plt.close()



def plot_integrals_comparison_run(path, input_folders, species, dt_fs):
    """
    This function reads the VACF data from CSV files for a specified species, computes the 
    cumulative integral, and plots the integral as a function of time for the species from multiple folders.

    Parameters:
    - input_folders: list of str, paths to the directories containing the species data.
    - species: str, the species to compare (e.g., 'Na' or 'Cl').
    - dt_fs: float, time step in femtoseconds used in the simulation.

    The plot is saved as a PDF file in the specified directory.
    """

    # Convert dt from femtoseconds to atomic units and picoseconds
    dt_au = dt_fs / (2.4188843265857 * 1e-2)  # Time step in atomic units
    a_to_ps = 2.4188843265857 * 1e-5  # Conversion factor: 1 a.u. of time = 2.4188843265857e-5 ps
    dt_ps = dt_au * a_to_ps  # Time step in picoseconds
    conv = (5.29177210903 * 1e-11)**2 / (2.4188843265857 * 1e-17) * 1e7

    # Initialize storage for data
    integrals = {}
    times = {}
    
    # Initialize the plot
    plt.figure(figsize=(10, 6))
    #label_names = ['7.5 ps', '12.5 ps']

    # Loop over each folder to process the VACF data for the same species
    for i, folder in enumerate(input_folders):
        # Construct the file path for the species in this folder
        file_path = os.path.join(folder, species, "AvVCT_" + species + ".csv")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load the VACF data from the CSV file
        df = pd.read_csv(file_path)

        # Extract the VACF values
        vacf_values = df['value'].values

        # Compute the cumulative integral of the VACF
        integral_vacf = np.cumsum(vacf_values) * dt_au

        # Generate the time axis in ps
        time_axis_ps = df['iter'].values * dt_ps

        # Store results for plotting
        folder_name = os.path.basename(folder)  # Use folder name as label
        integrals[folder_name] = integral_vacf #* conv
        times[folder_name] = time_axis_ps
        

        # Plot each folder's data on the same plot
        #plt.plot(time_axis_ps, integral_vacf, label=label_names[i])
        plt.plot(time_axis_ps, integral_vacf * conv, label='run ' + str(i + 1))
    
    if species == 'Na':
        plt.axhline(0.16, label='$D_{Na} = 0.16$ - Galamba, T=1539 K', color='k', linestyle=':', linewidth=2)
    elif species == 'Cl':
        plt.axhline(0.14, label='$D_{Cl} = 0.14$ - Galamba, T=1539 K', color='k', linestyle=':', linewidth=2)

    # Set up plot labels and title
    plt.xlabel('Time (ps)')
    plt.ylabel('$D_{'+ str(species) + '}$ [($10^{-3}$) cm$^2$ / s]')
    #plt.ylabel('Integral (a.u.)')
    #plt.title(f'Diffusion coefficient for {species}')
    xlim = 0.6
    xmax = np.max(time_axis_ps)
    plt.xlim([0,xlim])
    plt.ylim([0, 0.23])
    plt.legend()  # This will display the folder names in the legend

    # Save the plot
    if xlim < xmax:
        output_plot = os.path.join(path, f'integral_vacf_comparison_{species}_t_{xlim}_ps.pdf')
    else:
        output_plot = os.path.join(path, f'integral_vacf_comparison_{species}_t_{xmax}_ps.pdf')
    plt.savefig(output_plot, format='pdf')
    print(f"Plot saved to {output_plot}")

    # Show the plot
    plt.show()
    plt.close()


def plot_vacf_comparison_run(path, input_folders, species, dt_fs):
    """
    This function reads the VACF data from CSV files for a specified species and plots
    the VACF as a function of time for the species from multiple folders.

    Parameters:
    - input_folders: list of str, paths to the directories containing the species data.
    - species: str, the species to compare (e.g., 'Na' or 'Cl').
    - dt_fs: float, time step in femtoseconds used in the simulation.
    """

    # Convert dt from femtoseconds to picoseconds
    dt_au = dt_fs / (2.4188843265857 * 1e-2)  # Time step in atomic units
    a_to_ps = 2.4188843265857 * 1e-5  # Conversion factor: 1 a.u. of time = 2.4188843265857e-5 ps
    dt_ps = dt_au * a_to_ps  # Time step in picoseconds

    # Initialize the plot
    plt.figure(figsize=(10, 6))
    plt.axhline(y=0, color='grey')
    # Loop over each folder to process the VACF data for the same species
    for i, folder in enumerate(input_folders):
        # Construct the file path for the species in this folder
        file_path = os.path.join(folder, species, "AvVCT_" + species + ".csv")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load the VACF data from the CSV file
        df = pd.read_csv(file_path)

        # Extract the VACF values
        vacf_values = df['value'].values /  df['value'].iloc[0]

        # Generate the time axis in ps
        time_axis_ps = df['iter'].values * dt_ps

        # Plot each folder's VACF data on the same plot
        plt.plot(time_axis_ps, vacf_values, label='run ' + str(i + 1))
    
    # Set up plot labels and title
    plt.xlabel('Time (ps)')
    plt.ylabel('VACF (a.u.)')
    plt.legend()  # This will display the folder names in the legend
    xlim = 0.6
    xmax = np.max(time_axis_ps)
    plt.xlim([0,xlim])
    
    # Save the plot
    if xlim < xmax:
        output_plot = os.path.join(path, f'vacf_comparison_{species}_t_{xlim}_ps.pdf')
    else:
        output_plot = os.path.join(path, f'vacf_comparison_{species}_t_{xmax}_ps.pdf')
    plt.savefig(output_plot, format='pdf')
    print(f"Plot saved to {output_plot}")
    # Show the plot
    plt.show()
    plt.close()

def plot_integrals_comparison_dt(input_path_1, input_path_2, dt_fs_1, dt_fs_2, xlim=None, ylim=None, D_show=False, shift=False):
    def load_vacf_data(input_path, dt_fs, shift):
        # Convert dt from femtoseconds to atomic units and picoseconds
        dt_au = dt_fs / (2.4188843265857 * 1e-2)  # Time step in atomic units
        a_to_ps = 2.4188843265857 * 1e-5  # Conversion factor: 1 a.u. of time = 2.4188843265857e-5 ps
        dt_ps = dt_au * a_to_ps  # Time step in picoseconds
        conv = (5.29177210903 * 1e-11)**2 / (2.4188843265857 * 1e-17) * 1e7
        species = ['Na', 'Cl']
        integrals = {}
        times = {}

        for sp in species:
            # Construct the file path for the species
            if shift:
                file_path = os.path.join(input_path, sp, "AvVCT_" + sp + "_shift.csv")
            else:
                file_path = os.path.join(input_path, sp, "AvVCT_" + sp + ".csv")
            
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # Load the VACF data from the CSV file
            df = pd.read_csv(file_path)

            # Extract the VACF values
            vacf_values = df['value'].values

            # Compute the cumulative integral of the VACF
            integral_vacf = np.cumsum(vacf_values) * dt_au

            # Generate the time axis in ps
            time_axis_ps = df['iter'].values * dt_ps

            # Store results for plotting
            integrals[sp] = integral_vacf * conv
            times[sp] = time_axis_ps

        return times, integrals
    
    # Load data for both simulations
    times_1, integrals_1 = load_vacf_data(input_path_1, dt_fs_1, shift)
    times_2, integrals_2 = load_vacf_data(input_path_2, dt_fs_2, shift)

    # Define plot colors for each species
    colors = {'Na': 'blue', 'Cl': 'green'}
    
    # Plot the integrals of VACF as a function of time for both species and both time steps
    plt.figure(figsize=(8, 4))
    species = ['Na', 'Cl']
    for sp in species:
        if sp in integrals_1 and sp in integrals_2:
            # Plot for the first dt with solid linestyle
            plt.plot(times_1[sp], integrals_1[sp], label=f'{sp}, dt = {dt_fs_1} fs', color=colors[sp], linestyle='-')
            # Plot for the second dt with dashed linestyle
            plt.plot(times_2[sp], integrals_2[sp], label=f'{sp}, dt = {dt_fs_2} fs', color=colors[sp], linestyle='--')

    plt.xlabel('Time [ps]')
    plt.ylabel('D [($10^{-3}$) cm$^2$ / s]')
    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    
    if D_show:
        plt.axhline(0.16, label='$D_{Na} = 0.16$ - Galamba, T=1539 K', color='blue', linestyle=':', linewidth=2)
        #plt.axhline(0.22, label='$D_{Na} = 0.22$ - Coretti, T=1550 K', color='violet', linestyle='-')
        plt.axhline(0.14, label='$D_{Cl} = 0.14$ - Galamba, T=1539 K', color='green', linestyle=':', linewidth=2)
        #plt.axhline(0.18, label='$D_{Cl} = 0.18$ - Coretti, T=1550 K', color='gold', linestyle='-')
    
    #plt.title('Diffusion coefficient for different dt - T = 1550 K', fontsize=15)
    plt.legend(loc='lower right')

    # Save the plot
    if shift:
        output_plot = os.path.join(input_path_1, 'integral_vacf_comparison_dt_' + str(dt_fs_1) + '_vs_' + str(dt_fs_2) + '_xlim_' + str(xlim) + '_shift.pdf')
    else:
        output_plot = os.path.join(input_path_1, 'integral_vacf_comparison_dt_' + str(dt_fs_1) + '_vs_' + str(dt_fs_2) + '_xlim_' + str(xlim) + '.pdf')
    plt.savefig(output_plot, format='pdf')
    print(f"Plot saved to {output_plot}")

    plt.show()
    plt.close()

def plot_integrals_comparison_shift(input_path, dt_fs, xlim=None, ylim=None, D_show=False):
    def load_vacf_data(input_path, dt_fs, shift=False):
        # Convert dt from femtoseconds to atomic units and picoseconds
        dt_au = dt_fs / (2.4188843265857 * 1e-2)  # Time step in atomic units
        a_to_ps = 2.4188843265857 * 1e-5  # Conversion factor: 1 a.u. of time = 2.4188843265857e-5 ps
        dt_ps = dt_au * a_to_ps  # Time step in picoseconds
        conv = (5.29177210903 * 1e-11)**2 / (2.4188843265857 * 1e-17) * 1e7
        species = ['Na', 'Cl']
        integrals = {}
        times = {}

        for sp in species:
            # Construct the file path for the species
            if shift:
                file_path = os.path.join(input_path, sp, "AvVCT_" + sp + "_shift.csv")
                print("opening shifted vacf file")
            else:
                file_path = os.path.join(input_path, sp, "AvVCT_" + sp + ".csv")
                print("opening classical vacf file")

            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            # Load the VACF data from the CSV file
            df = pd.read_csv(file_path)

            # Extract the VACF values
            vacf_values = df['value'].values

            # Compute the cumulative integral of the VACF
            integral_vacf = np.cumsum(vacf_values) * dt_au

            # Generate the time axis in ps
            time_axis_ps = df['iter'].values * dt_ps

            # Store results for plotting
            integrals[sp] = integral_vacf * conv
            times[sp] = time_axis_ps

        return times, integrals
    
    # Load data for both simulations
    times_1, integrals_1 = load_vacf_data(input_path, dt_fs, shift=True)
    times_2, integrals_2 = load_vacf_data(input_path, dt_fs, shift=False)

    # Define plot colors for each species
    colors = {'Na': 'blue', 'Cl': 'green'}
    
    # Plot the integrals of VACF as a function of time for both species and both time steps
    plt.figure(figsize=(8, 4))
    species = ['Na', 'Cl']
    for sp in species:
        if sp in integrals_1 and sp in integrals_2:
            plt.plot(times_1[sp], integrals_1[sp], label= sp + ', shifted', color=colors[sp], linestyle='--')
            plt.plot(times_2[sp], integrals_2[sp], label= sp + ', not shifted', color=colors[sp], linestyle='-')
            print(integrals_1[sp] -  integrals_2[sp])
    plt.xlabel('Time [ps]')
    plt.ylabel('D [($10^{-3}$) cm$^2$ / s]')
    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    
    if D_show:
        plt.axhline(0.16, label='$D_{Na} = 0.16$ - Galamba, T=1539 K', color='blue', linestyle=':', linewidth=2)
        #plt.axhline(0.22, label='$D_{Na} = 0.22$ - Coretti, T=1550 K', color='violet', linestyle='-')
        plt.axhline(0.14, label='$D_{Cl} = 0.14$ - Galamba, T=1539 K', color='green', linestyle=':', linewidth=2)
        #plt.axhline(0.18, label='$D_{Cl} = 0.18$ - Coretti, T=1550 K', color='gold', linestyle='-')
    
    #plt.title('Diffusion coefficient for different dt - T = 1550 K', fontsize=15)
    plt.legend(loc='lower right')

    # Save the plot
    output_plot = os.path.join(input_path, 'integral_vacf_comparison_dt_' + str(dt_fs) + '_xlim_' + str(xlim) + '_shift.pdf')
    plt.savefig(output_plot, format='pdf')
    print(f"Plot saved to {output_plot}")

    plt.show()
    plt.close()

def visualize_particles(filename):
    #N_p = get_Np(filename)
    N_p = get_Np_input(filename)
    L = np.round((((N_p*(m_Cl + m_Na)) / (2*density))  **(1/3)) *1.e9, 4)
    particles_df = pd.read_csv(filename)
    num_particles = N_p

    # Extract particle positions
    x = particles_df['x'][:num_particles]
    y = particles_df['y'][:num_particles]
    z = particles_df['z'][:num_particles]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot particles
    ax.scatter(x, y, z, c=particles_df['charge'][:num_particles], cmap='coolwarm', s=100)

    # Set plot labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)

    # Show plot
    plt.title('Particle Visualization')
    plt.show()

def plot_energy_multiple_dt(path_orig, N, dt_list=[0.025, 0.05, 0.25, 1.25, 2.5], L=19.659 / a0): # move 
    N_p = get_Np(path_orig)
    plt.figure(figsize=(15, 6))
    if len(dt_list) <= 2:
        color_list = ['r', 'b', 'm']
    else: 
        color_list = ['r', 'salmon', 'gold', 'palegreen', 'forestgreen']

    for i, dt in enumerate(dt_list):
        # Define the path for the current dt
        path = path_orig + 'dt_' + str(dt) + '/'+ 'output/'
       
        # Load the solute and energy data
        df = pd.read_csv(path + 'solute_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
        df_E = pd.read_csv(path + 'energy_N' + str(N)+'_N_p_'+str(N_p) + '.csv')

        K = df_E['K']
        V_notelec = df_E['V_notelec']
        Np = int(df['particle'].max() + 1)

        df_list = [df[df['particle'] == p].reset_index(drop=True) for p in range(Np)]

        iterations = df['iter']
        N_steps = int(iterations.max() + 1)

        # Compute the time for each iteration
        time = dt * np.arange(N_steps)

        # Check if the work file exists
        work_file = path + f'work_trp_N{N}.csv'
        
        if os.path.exists(work_file):
            # Read the work from the file
            work_df = pd.read_csv(work_file)
            Ework = work_df['work'].tolist()
            print(f'Loaded work data from {work_file}')
        else:
            # Compute the work if the file doesn't exist
            print(f'{work_file} not found. Computing work...')

                            # If the file doesn't exist, compute the work
            df = pd.read_csv(path + 'solute_N' + str(N)+'_N_p_'+str(N_p) + '.csv')
            Np = int(df['particle'].max() + 1)   

            df_list = [df[df['particle'] == p].reset_index(drop=True) for p in range(Np)]
            iterations = df['iter']
            N_steps = int(iterations.max() + 1)
            Ework = np.zeros(N_steps)
            work = np.zeros((Np, N_steps))
            
            print('N_steps =', N_steps)
            print('Np =', Np)
        
            # Precompute steps and avoid repeated indexing
            for p in range(Np):
                df_p = df_list[p]
                x = df_p['x'].values
                y = df_p['y'].values
                z = df_p['z'].values

                fx = df_p['fx'].values
                fy = df_p['fy'].values
                fz = df_p['fz'].values

                # Initialize cumulative displacement
                cx = np.zeros(N_steps)
                cy = np.zeros(N_steps)
                cz = np.zeros(N_steps)
                
                for j in range(1, N_steps):
                    # Calculate djsplacement between consecutjve steps
                    dx = x[j] - x[j-1]
                    dy = y[j] - y[j-1]
                    dz = z[j] - z[j-1]

                    # Apply mjnjmum jmage conventjon to account for PBC
                    dx -= np.rint(dx / L) * L
                    dy -= np.rint(dy / L) * L
                    dz -= np.rint(dz / L) * L

                    # Accumulate the corrected djsplacements
                    cx[j] = cx[j-1] + dx
                    cy[j] = cy[j-1] + dy
                    cz[j] = cz[j-1] + dz
                    
                    # Compute work for each partjcle p at each step j
                    work[p][j] = - (np.trapz(fx[:j], x=cx[:j]) +
                                    np.trapz(fy[:j], x=cy[:j]) +
                                    np.trapz(fz[:j], x=cz[:j]))

            # Sum up the work across all particles
            Ework = np.add.reduce(work, axis=0)
            
            # Create the work file with header "iter,work" and save the computed work
            work_df = pd.DataFrame({'iter': range(N_steps), 'work': Ework})
            work_df.to_csv(work_file, index=False)
            print(f"Work data saved to {work_file}")

        # Compute total energy
        E_tot = K + V_notelec + Ework

        mean = np.mean(E_tot)
        std_E = np.std(E_tot)
        std_work = np.std(Ework)
        
        mean_half = np.mean(E_tot[:int(len(E_tot)/2)])
        std_E_half = np.std(E_tot[:int(len(E_tot)/2)])
        std_work_half = np.std(Ework[:int(len(E_tot)/2)])

        # Plot total energy for this dt with time on the x-axis
        plt.plot(time[1:-1], E_tot[1:-1], marker='.', markersize=1, color=color_list[i], label=f'dt = {dt} fs - {mean:.4f}$\\pm${std_E:.5f} - RE = {(std_E / mean) * 100 :.4f} % - $\\Delta E / \\Delta V$ = {(std_E / std_work):.4f}')
        #plt.plot(time[:-1], E_tot[:-1], marker='.', markersize=5, color=color_list[i], label=f'dt = {dt} fs - {mean:.4f}$\pm${std:.5f} - rel err = {(std / mean) * 100 :.4f} %')
        RE_half = std_E_half / mean_half * 100
        RE_complete =  std_E / mean * 100
        deltaE_deltaV_half =  std_E_half / std_work_half
        deltaE_deltaV_complete = std_E / std_work
        print('\nRE 1/2:', RE_half,' %\tRE complete:', RE_complete,' %\t ratio = ', RE_complete/RE_half)
        print('deltaE/deltaV 1/2:', deltaE_deltaV_half,' %\tdeltaE/deltaV complete:', deltaE_deltaV_complete,' %\t ratio = ', deltaE_deltaV_complete/deltaE_deltaV_half)

    # Customize the plot
    #plt.title('Total energy for different timesteps', fontsize=22)
    plt.xlabel('Time [fs]', fontsize=15)
    plt.ylabel('Energy [$E_H$]', fontsize=15)
    plt.legend(loc='upper left')
    plt.xlim([0,250])

    # Save the figure as a PDF
    plt.savefig(path_orig + 'energy_trp_comparison_time' + str(N) +'_N_p_'+str(N_p)+ '.pdf', format='pdf')
    plt.show()

def plot_E_trp2(path, second_path, N, dt1, dt2, N_th, L=19.659 / a0, upper_lim=None): # move
    N_p =get_Np(path)
    # First dataset - File paths
    work_file = path + 'work_trp_N' + str(N)+'_N_p_'+str(N_p) + '.csv'
    df_E = pd.read_csv(path + 'energy_N' + str(N)+'_N_p_'+str(N_p) + '.csv')

    # Second dataset - File paths
    second_work_file = second_path + 'work_trp_N' + str(N) +'_N_p_'+str(N_p) + '.csv'
    second_df_E = pd.read_csv(second_path + 'energy_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')

    # Energy file columns for the first dataset
    K = df_E['K']
    V_notelec = df_E['V_notelec']
    N_steps_energy = len(df_E)

    # Energy file columns for the second dataset
    K2 = second_df_E['K']
    V_notelec2 = second_df_E['V_notelec']
    N_steps_energy_2 = len(second_df_E)

    recompute_work = False
    recompute_work2 = False
    # Check if the work file exists and has the correct number of lines for the first dataset
    if os.path.exists(work_file):
        work_df = pd.read_csv(work_file)
        if len(work_df) == N_steps_energy:
            Ework = work_df['work'].tolist()
            print(f"Work data loaded from {work_file}")
        else:
            print(f"Work file exists but has incorrect number of lines. Recomputing work.")
            recompute_work = True
    else:
        recompute_work = True

    # Check if the work file exists for the second dataset
    if os.path.exists(second_work_file):
        second_work_df = pd.read_csv(second_work_file)
        Ework2 = second_work_df['work'].tolist()
        print(f"Work data loaded from {second_work_file}")
    else:
        # If not, recompute or handle this case accordingly
        recompute_work2 = True

    if recompute_work:
        # If the file doesn't exist, compute the work
        df = pd.read_csv(path + 'solute_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
        Np = int(df['particle'].max() + 1)   

        df_list = [df[df['particle'] == p].reset_index(drop=True) for p in range(Np)]
        iterations = df['iter']
        N_steps = int(iterations.max() + 1)
        Ework = np.zeros(N_steps)
        work = np.zeros((Np, N_steps))
        
        print('N_steps =', N_steps)
        print('Np =', Np)
       
        # Precompute steps and avoid repeated indexing
        for p in range(Np):
            df_p = df_list[p]
            x = df_p['x'].values
            y = df_p['y'].values
            z = df_p['z'].values

            fx = df_p['fx'].values
            fy = df_p['fy'].values
            fz = df_p['fz'].values

            # Initialize cumulative displacement
            cx = np.zeros(N_steps)
            cy = np.zeros(N_steps)
            cz = np.zeros(N_steps)
            
            for i in range(1, N_steps):
                # Calculate displacement between consecutive steps
                dx = x[i] - x[i-1]
                dy = y[i] - y[i-1]
                dz = z[i] - z[i-1]

                # Apply minimum image convention to account for PBC
                dx -= np.rint(dx / L) * L
                dy -= np.rint(dy / L) * L
                dz -= np.rint(dz / L) * L

                # Accumulate the corrected displacements
                cx[i] = cx[i-1] + dx
                cy[i] = cy[i-1] + dy
                cz[i] = cz[i-1] + dz
                
                # Compute work for each particle p at each step i
                work[p][i] = - (np.trapz(fx[:i], x=cx[:i]) +
                                np.trapz(fy[:i], x=cy[:i]) +
                                np.trapz(fz[:i], x=cz[:i]))

        # Sum up the work across all particles
        Ework = np.add.reduce(work, axis=0)
        
        # Create the work file with header "iter,work" and save the computed work
        work_df = pd.DataFrame({'iter': range(N_steps), 'work': Ework})
        work_df.to_csv(work_file, index=False)
        print(f"Work data saved to {work_file}")
    
    if recompute_work2:
        # If the second work file doesn't exist, compute the work for the second dataset
        df2 = pd.read_csv(second_path + 'solute_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
        Np2 = int(df2['particle'].max() + 1)

        df_list2 = [df2[df2['particle'] == p].reset_index(drop=True) for p in range(Np2)]
        iterations2 = df2['iter']
        N_steps2 = int(iterations2.max() + 1)
        Ework2 = np.zeros(N_steps2)
        work2 = np.zeros((Np2, N_steps2))

        print('N_steps for second dataset =', N_steps2)
        print('Np for second dataset =', Np2)

        # Precompute steps and avoid repeated indexing for second dataset
        for p in range(Np2):
            df_p2 = df_list2[p]
            x2 = df_p2['x'].values
            y2 = df_p2['y'].values
            z2 = df_p2['z'].values

            fx2 = df_p2['fx'].values
            fy2 = df_p2['fy'].values
            fz2 = df_p2['fz'].values

            # Initialize cumulative displacement for second dataset
            cx2 = np.zeros(N_steps2)
            cy2 = np.zeros(N_steps2)
            cz2 = np.zeros(N_steps2)

            for i in range(1, N_steps2):
                # Calculate displacement between consecutive steps for second dataset
                dx2 = x2[i] - x2[i-1]
                dy2 = y2[i] - y2[i-1]
                dz2 = z2[i] - z2[i-1]

                # Apply minimum image convention to account for PBC for second dataset
                dx2 -= np.rint(dx2 / L) * L
                dy2 -= np.rint(dy2 / L) * L
                dz2 -= np.rint(dz2 / L) * L

                # Accumulate the corrected displacements for second dataset
                cx2[i] = cx2[i-1] + dx2
                cy2[i] = cy2[i-1] + dy2
                cz2[i] = cz2[i-1] + dz2

                # Compute work for each particle p at each step i for second dataset
                work2[p][i] = - (np.trapz(fx2[:i], x=cx2[:i]) +
                                np.trapz(fy2[:i], x=cy2[:i]) +
                                np.trapz(fz2[:i], x=cz2[:i]))

        # Sum up the work across all particles for second dataset
        Ework2 = np.add.reduce(work2, axis=0)

        # Create the work file with header "iter,work" and save the computed work for second dataset
        work_df2 = pd.DataFrame({'iter': range(N_steps2), 'work': Ework2})
        work_df2.to_csv(second_work_file, index=False)
        print(f"Work data saved to {second_work_file}")

    df = pd.read_csv(path + 'solute_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
    iterations = df['iter']
    N_steps = int(iterations.max() + 1)

    df2 = pd.read_csv(path + 'solute_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
    iterations2 = df2['iter']
    N_steps2 = int(iterations2.max() + 1)

    # Compute total energy for both datasets
    E_tot = K + V_notelec + Ework
    E_tot_2 = K2 + V_notelec2 + Ework2
    print(V_notelec2)

    # Mean and standard deviation for the first dataset
    mean = np.mean(E_tot)
    std_E = np.std(E_tot)
    std_work = np.std(Ework)
    
    mean_half = np.mean(E_tot[:int(len(E_tot)/2)])
    std_E_half = np.std(E_tot[:int(len(E_tot)/2)])
    std_work_half = np.std(Ework[:int(len(E_tot)/2)])
    RE_half = std_E_half / mean_half * 100
    RE_complete =  std_E / mean * 100
    deltaE_deltaV_half =  std_E_half / std_work_half
    deltaE_deltaV_complete = std_E / std_work
    print(f'RE 1/2:{RE_half:.4f} %\tRE complete: {RE_complete:.4f} %\t ratio = {RE_complete/RE_half:.4f}')
    print(f'deltaE/deltaV 1/2:{deltaE_deltaV_half:.4f} %\tdeltaE/deltaV complete: {deltaE_deltaV_complete:.4f} %\t ratio = {deltaE_deltaV_complete/deltaE_deltaV_half:.4f}')

    # Mean and standard deviation for the second dataset
    mean2 = np.mean(E_tot_2[:upper_lim])
    std_E2 = np.std(E_tot_2[:upper_lim])
    std_work2 = np.std(Ework2[:upper_lim])
    

    mean_half2 = np.mean(E_tot_2[:int(len(E_tot_2)/2)])
    std_E_half2 = np.std(E_tot_2[:int(len(E_tot_2)/2)])
    std_work_half2 = np.std(Ework2[:int(len(E_tot_2)/2)])
    RE_half2 = std_E_half2 / mean_half2 * 100
    RE_complete2 =  std_E2 / mean2 * 100
    deltaE_deltaV_half2 =  std_E_half2 / std_work_half2
    deltaE_deltaV_complete2 = std_E2 / std_work2
    print(f'RE 1/2:{RE_half2:.4f} %\tRE complete: {RE_complete2:.4f} %\t ratio = {RE_complete2/RE_half2:.4f}')
    print(f'deltaE/deltaV 1/2:{deltaE_deltaV_half2:.4f} %\tdeltaE/deltaV complete: {deltaE_deltaV_complete2:.4f} %\t ratio = {deltaE_deltaV_complete2/deltaE_deltaV_half2:.4f}')

    iterations1 = np.array(range(N_steps))
    iterations2 = np.array(range(N_steps2))

    # Create a figure with two subplots stacked vertically
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    
 
    # First subplot
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.plot(iterations1[N_th:upper_lim] * dt1 / 1000, E_tot[N_th:upper_lim], linestyle='-', color='salmon', markersize=5, 
            label=f'dt = {dt1} fs - $\\Delta E / E$ = {std_E/mean:.6f} - $\\Delta E / \\Delta E^W_{{elec}}$ = {(std_E / std_work):.4f}')
    plt.ylabel('Total energy [$E_h$]')
    plt.legend(loc='upper right')
    #plt.title('Energy Comparison for Different Time Steps')

    # Second subplot
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    plt.plot(iterations2[N_th:upper_lim] * dt2 / 1000, E_tot_2[N_th:upper_lim], linestyle='-', color='green', 
            label=f'dt = {dt2} fs - $\\Delta E / E$ = {std_E2/mean2:.6f} - $\\Delta E / \\Delta E^W_{{elec}}$ = {(std_E2 / std_work2):.4f}')
    plt.xlabel('Time [ps]')
    plt.ylabel('Total energy [$E_h$]')
    #plt.title('Total Energy Over Time')
    plt.legend(loc='upper right')
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(path + '/Energy_trp_N' + str(N) +'_N_p_'+str(N_p)+ '_dt_comparison.pdf', format='pdf')
    plt.show()

def plot_energy_multiple_runs(path_orig, N, dt=0.25, L=19.659 / a0, num_runs=10): # move
    N_p = get_Np(path_orig)
    plt.figure(figsize=(10, 5))
    
    # Define a color palette for the different runs
    color_list = ['r', 'salmon', 'gold', 'palegreen', 'forestgreen', 'cyan', 'dodgerblue', 'blue', 'purple', 'violet']

    for run in range(1, num_runs + 1):
        # Define the path for the current run
        path = path_orig + f'run_{run}/output/'
       
        # Load the solute and energy data
        df = pd.read_csv(path + 'solute_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
        df_E = pd.read_csv(path + 'energy_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')

        K = df_E['K']
        V_notelec = df_E['V_notelec']
        Np = int(df['particle'].max() + 1)

        df_list = [df[df['particle'] == p].reset_index(drop=True) for p in range(Np)]

        iterations = df['iter']
        N_steps = int(iterations.max() + 1)

        # Compute the time for each iteration
        time = dt * np.arange(N_steps)

        # Check if the work file exists
        work_file = path + f'work_trp_N{N}.csv'
        
        if os.path.exists(work_file):
            # Read the work from the file
            work_df = pd.read_csv(work_file)
            Ework = work_df['work'].tolist()
            print(f'Loaded work data from {work_file} for run {run}')
        else:
            # Compute the work if the file doesn't exist
            print(f'{work_file} not found for run {run}. Computing work...')

            # If the file doesn't exist, compute the work
            df = pd.read_csv(path + 'solute_N' + str(N) +'_N_p_'+str(N_p)+ '.csv')
            Np = int(df['particle'].max() + 1)   

            df_list = [df[df['particle'] == p].reset_index(drop=True) for p in range(Np)]
            iterations = df['iter']
            N_steps = int(iterations.max() + 1)
            Ework = np.zeros(N_steps)
            work = np.zeros((Np, N_steps))
            
            print('N_steps =', N_steps)
            print('Np =', Np)
        
            # Precompute steps and avoid repeated indexing
            for p in range(Np):
                df_p = df_list[p]
                x = df_p['x'].values
                y = df_p['y'].values
                z = df_p['z'].values

                fx = df_p['fx'].values
                fy = df_p['fy'].values
                fz = df_p['fz'].values

                # Initialize cumulative displacement
                cx = np.zeros(N_steps)
                cy = np.zeros(N_steps)
                cz = np.zeros(N_steps)
                
                for j in range(1, N_steps):
                    # Calculate displacement between consecutive steps
                    dx = x[j] - x[j-1]
                    dy = y[j] - y[j-1]
                    dz = z[j] - z[j-1]

                    # Apply minimum image convention to account for PBC
                    dx -= np.rint(dx / L) * L
                    dy -= np.rint(dy / L) * L
                    dz -= np.rint(dz / L) * L

                    # Accumulate the corrected displacements
                    cx[j] = cx[j-1] + dx
                    cy[j] = cy[j-1] + dy
                    cz[j] = cz[j-1] + dz
                    
                    # Compute work for each particle p at each step j
                    work[p][j] = - (np.trapz(fx[:j], x=cx[:j]) +
                                    np.trapz(fy[:j], x=cy[:j]) +
                                    np.trapz(fz[:j], x=cz[:j]))

            # Sum up the work across all particles
            Ework = np.add.reduce(work, axis=0)
            
            # Create the work file with header "iter,work" and save the computed work
            work_df = pd.DataFrame({'iter': range(N_steps), 'work': Ework})
            work_df.to_csv(work_file, index=False)
            print(f"Work data saved to {work_file} for run {run}")

        # Compute total energy
        E_tot = K + V_notelec + Ework

        mean = np.mean(E_tot)
        std_E = np.std(E_tot)
        std_work = np.std(Ework)
        
        mean_half = np.mean(E_tot[:int(len(E_tot)/2)])
        std_E_half = np.std(E_tot[:int(len(E_tot)/2)])
        std_work_half = np.std(Ework[:int(len(E_tot)/2)])

        # Plot total energy for this run with time on the x-axis
        plt.plot(time[200:-1], E_tot[200:-1], marker='.', markersize=1, color=color_list[run-1], 
                 label=f'Run {run} - {mean:.4f}$\\pm${std_E:.5f} - RE = {(std_E / mean) * 100 :.4f} % - $\\Delta E / \\Delta V$ = {(std_E / std_work):.4f}')
        
        RE_half = std_E_half / mean_half * 100
        RE_complete =  std_E / mean * 100
        deltaE_deltaV_half =  std_E_half / std_work_half
        deltaE_deltaV_complete = std_E / std_work
        #print(f'\nRE 1/2 for run {run}:', RE_half, ' %\tRE complete:', RE_complete, ' %\t ratio = ', RE_complete / RE_half)
        #print(f'deltaE/deltaV 1/2 for run {run}:', deltaE_deltaV_half, ' %\tdeltaE/deltaV complete:', deltaE_deltaV_complete, ' %\t ratio = ', deltaE_deltaV_complete / deltaE_deltaV_half)

    # Customize the plot
    plt.xlabel('Time [fs]', fontsize=15)
    plt.ylabel('Energy [$E_H$]', fontsize=15)
    plt.legend(loc='lower right', fontsize=7)

    # Save the figure as a PDF
    plt.savefig(path_orig + f'energy_trp_comparison_runs_{N}.pdf', format='pdf')
    plt.show()






