# TODO(bin format): reads output files with pd.read_csv, adapt to also read .bin (see maze_poisson/myio/output/binary.py)
import csv
import numpy as np
from ...constants import a0, t_au
import matplotlib.pyplot as plt
import os
import pandas as pd
from . import get_N

def convert_csv_to_xyz(filename, input_csv_file, output_xyz_file, Np):
    N = get_N(filename)
    # Dictionary to map charges to element symbols
    charge_to_element = {1: 'Na', -1: 'Cl'}
    frame = 0
    with open(input_csv_file, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        with open(output_xyz_file, 'w') as xyz_file:
            num_particles = 0
            for row in csv_reader:
                charge = int(float(row['charge']))
                element = charge_to_element.get(charge, 'X')  # Default to 'X' if charge is neither 1 nor -1
                x, y, z = float(row['x']),float(row['y']),float(row['z']) 
                if num_particles % N == 0:
                    xyz_file.write(str(N) + '\n')
                    xyz_file.write(str(frame) + '\n')
                    frame = frame + 1
                line = f"{element} {np.round(x * a0, decimals=3)} {np.round(y * a0, decimals=3)} {np.round(z * a0, decimals=3)}\n"
                xyz_file.write(line)
                num_particles += 1

            print(f"Conversion completed. {num_particles / N} steps written to {output_xyz_file}")

def get_ion_type(charge):
    if charge > 0:
        return 'Na'
    else:
        return 'Cl'

# Function to process the data for a single run
def process_run(filename, Np, run_num, Q, path, parallel):
    N = get_N(filename)
    # Get ion type based on charge
    ion_type = get_ion_type(Q)
    
    if parallel:
        # Input and output file paths
        input_file_path = path + 'run_' + str(run_num) + '/output/solute_N' + str(N) + '.csv'

        # Create output directories based on ion type
        ion_path = os.path.join(path + 'run_' + str(run_num), ion_type)
        if not os.path.exists(ion_path):
            os.makedirs(ion_path)
    else:
        # Input and output file paths
        input_file_path = path + 'solute_N' + str(N) + '.csv'

        # Create output directories based on ion type
        ion_path = os.path.join(path, ion_type)
        if not os.path.exists(ion_path):
            os.makedirs(ion_path)

    output_file_path = ion_path + '/velocity_' + ion_type + '.out'
    type_element = ion_type
    steps_init = 2

    # Read the input CSV file and write to the output text file
    with open(input_file_path, 'r', newline='') as csvfile, open(output_file_path, 'w') as txtfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row

        num_steps = 0
        current_step = -1
        for row in reader:
            if row:
                ion_type = get_ion_type(int(float(row[0])))
                iter_num = int(row[1])  # Assuming iter number is in the second column
                if ion_type == type_element and iter_num > steps_init:
                    if iter_num != current_step:
                        if iter_num > 0:
                            txtfile.write("\n")
                        txtfile.write(f"# step\t{iter_num - steps_init - 1}\n")
                        txtfile.write(f"# Ion kind  Ion velocity (x y z)\n")
                        current_step = iter_num
                        num_steps += 1
                    velocity = [float(v) for v in row[6:9]]  # Assuming velocity components are in columns 6, 7, and 8
                    txtfile.write("{}\t{:.12e}\t{:.12e}\t{:.12e}\n".format(ion_type, *velocity))

    print(f'End of the conversion for atom {type_element} in run_{run_num}')
    print(f'Converted N_steps={iter_num + 1 - steps_init}')
    print(f'Number of particles={Np}\n')

def process_run1(run_num, N, Q, path, parallel, max_steps=None):
    """
    Process the data for a single run by reading velocities from the input CSV file and writing them to an output file.
    
    Parameters:
    - run_num: int, the run number.
    - N: int, the solute particle number.
    - Q: int, charge of the ion (used to determine ion type).
    - path: str, path to the directory containing the data.
    - parallel: bool, whether the runs are processed in parallel or not.
    - max_steps: int, optional, the maximum step number to read (iter_num). If None, all steps will be read.
    """
    # Get ion type based on charge
    ion_type = get_ion_type(Q)
    
    if parallel:
        # Input and output file paths
        input_file_path = path + 'run_' + str(run_num) + '/output/solute_N' + str(N) + '.csv'

        # Create output directories based on ion type
        ion_path = os.path.join(path + 'run_' + str(run_num), ion_type)
        if not os.path.exists(ion_path):
            os.makedirs(ion_path)
    else:
        # Input and output file paths
        input_file_path = path + 'solute_N' + str(N) + '.csv'
        # Create output directories based on ion type
        ion_path = os.path.join(path, ion_type)
        if not os.path.exists(ion_path):
            os.makedirs(ion_path)

    output_file_path = ion_path + '/velocity_' + ion_type + '.out'
    type_element = ion_type
    steps_init = 2  # Assuming steps start at step 2 as in your original code

    # Read the input CSV file and write to the output text file
    with open(input_file_path, 'r', newline='') as csvfile, open(output_file_path, 'w') as txtfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row

        current_step = -1
        for row in reader:
            if row:
                ion_type = get_ion_type(int(float(row[0])))
                iter_num = int(row[1])  # Assuming iter number is in the second column
  
                # Stop processing if iter_num exceeds max_steps
                if max_steps is not None and iter_num >= max_steps + steps_init:
                    print(iter_num, max_steps)
                    break

                # Only process rows for the current ion type and above the initial step
                if ion_type == type_element and iter_num > steps_init:
                    if iter_num != current_step:
                        if iter_num > 0:
                            txtfile.write("\n")
                        
                        txtfile.write(f"# step\t{iter_num - steps_init - 1}\n")
                        txtfile.write(f"# Ion kind  Ion velocity (x y z)\n")
                        current_step = iter_num

                    velocity = [float(v) for v in row[6:9]]  # Assuming velocity components are in columns 6, 7, and 8
                    txtfile.write("{}\t{:.12e}\t{:.12e}\t{:.12e}\n".format(ion_type, *velocity))

    print(f'End of the conversion for atom {type_element} in run_{run_num}')
    print(f'Processed up to step {iter_num}')
    print(f'Number of particles={N}\n')


# Main processing function
def process_all_runs(N, n_runs, parallel, base_path):
    # If parallel flag is True, process all runs
    if parallel:
        for run_num in range(1, n_runs + 1):
            process_run(run_num, N, 1, base_path, parallel)
            process_run(run_num, N, -1, base_path, parallel)
    else:
        # Process just a single run if parallel is False
        process_run1(1, N, 1, base_path, parallel)
        process_run1(1, N, -1, base_path, parallel)

def compute_msd(particles, L):
    initial_positions = particles.groupby('particle')[['x', 'y', 'z']].first().values
    positions = particles[['x', 'y', 'z']].values.reshape(-1, len(initial_positions), 3)
    displacements = positions - initial_positions 
    displacements = displacements - L * np.rint(displacements / L)

    squared_displacements = np.sum(displacements**2, axis=2) # calcolo quadrato displacement per ogni singola particella at each time step sommando contributi di x,y,z
    msd = np.mean(squared_displacements, axis=1) # average sul numero di particelle at each time step
    return msd


def plot_msd_with_fit(time_intervals, msd, slope, intercept, species):
    plt.figure(figsize=(8, 6))
    plt.plot(time_intervals, msd, 'o', label=f'{species} MSD')
    plt.plot(time_intervals, slope * time_intervals + intercept, '-', label=f'Linear fit: MSD = {slope:.3f}t + {intercept:.3f}')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('MSD (a.u.²)')
    plt.title(f'Mean Square Displacement and Linear Fit for {species}')
    plt.legend()
    plt.show()


def compute_diffusion_coefficient(msd, dt, time_intervals, species_name, path, start_idx=0, end_idx=None, dimensions=3, parallel=False):
    # If end_idx is not provided, use the last index
    if end_idx is None:
        end_idx = len(time_intervals)
        
    # Select the range of data to fit
    selected_time_intervals = time_intervals[start_idx:end_idx]
    selected_msd = msd[start_idx:end_idx]
    print('Diffusion coefficient computed at t = ', dt * end_idx / 1000, 'ps\n')
    # Use linear regression to fit MSD to time and get the slope
    slope, intercept = np.polyfit(selected_time_intervals, selected_msd, 1)
    diffusion_coefficient = slope / (2 * dimensions)
    
   
    # Plot the MSD and the fit
    plt.figure(figsize=(15, 6))
    plt.plot(time_intervals /1000, msd, '.-', label='MSD')
    plt.plot(selected_time_intervals / 1000, slope * selected_time_intervals + intercept, 'r--', label='Linear fit')
    plt.xlabel('Time [ps]')
    plt.ylabel('MSD [a.u.²]')
    plt.legend()
    plt.title(f'MSD for {species_name}')
    
    # Save the plot as an image file
    plot_filename = path + str(species_name) + "_MSD_plot.pdf"
    plt.savefig(plot_filename)
    if parallel == False:
        plt.show()
    plt.close()  

    return diffusion_coefficient

def main(filename, path, output_file, dt, L, end_idx, parallel):
    N = get_N(filename)
    # Load the simulation data
    data = pd.read_csv(path + '/solute_N' + str(N) + '.csv')

    # Extract unique time steps (assuming 'iter' is the time step index)
    time_steps = data['iter'].unique()
    
    # Compute actual time intervals
    time_intervals = time_steps * dt
      
    # Separate data for Na and Cl based on charge
    na_data = data[data['charge'] == 1]
    cl_data = data[data['charge'] == -1]
    
    # Compute MSD for Na and Cl
    na_msd = compute_msd(na_data, L)
    cl_msd = compute_msd(cl_data, L)
    
    conv = a0**2 * 1e2
    # Compute diffusion coefficients (convert to 10^-3 cm^2/s by multiplying by conv)
    na_diffusion_coefficient = compute_diffusion_coefficient(na_msd, time_intervals, end_idx=end_idx, species_name='Na', path=path, parallel=parallel)  * conv
    cl_diffusion_coefficient = compute_diffusion_coefficient(cl_msd, time_intervals, end_idx=end_idx, species_name='Cl', path=path, parallel=parallel)  * conv
    
    # Save the results
    with open(output_file, 'w') as f:
        f.write(f"Na Diffusion Coefficient: {na_diffusion_coefficient:.6f} 10^-3 cm^2/s\n")
        f.write(f"Cl Diffusion Coefficient: {cl_diffusion_coefficient:.6f} 10^-3 cm^2/s\n")
    print(f"Na Diffusion Coefficient: {na_diffusion_coefficient:.6f} 10^-3 cm^2/s\n")
    print(f"Cl Diffusion Coefficient: {cl_diffusion_coefficient:.6f} 10^-3 cm^2/s\n")
    return na_diffusion_coefficient, cl_diffusion_coefficient

