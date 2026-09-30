# TODO(bin format): reads output files with pd.read_csv, adapt to also read .bin (see maze_poisson/myio/output/binary.py)
import csv

import numpy as np
import pandas as pd

from ...constants import a0

n =250 # 64

# def convert_csv_to_xyz(input_csv_file, output_xyz_file):
#     # Dictionary to map charges to element symbols
#     charge_to_element = {1: 'Na', -1: 'Cl'}
#     frame = 0
#     with open(input_csv_file, 'r') as csv_file:
#         csv_reader = csv.DictReader(csv_file)
#         with open(output_xyz_file, 'w') as xyz_file:
#             num_particles = 0
#             for row in csv_reader:
#                 charge = int(float(row['charge']))
#                 element = charge_to_element.get(charge, 'X')  # Default to 'X' if charge is neither 1 nor -1
#                 x, y, z = float(row['x']),float(row['y']),float(row['z']) 
#                 if num_particles % n == 0:
#                     xyz_file.write(str(n) + '\n')
#                     xyz_file.write(str(frame) + '\n')
#                     frame = frame + 1
#                 line = f"{element} {np.round(x * a0, decimals=3)} {np.round(y * a0, decimals=3)} {np.round(z * a0, decimals=3)}\n"
#                 xyz_file.write(line)
#                 num_particles += 1

#             print(f"Conversion completed. {num_particles / n} steps written to {output_xyz_file}")

def convert_csv_to_xyz(input_csv_file, output_xyz_file):
    charge_to_element = {1: 'Na', -1: 'Cl'}

    data = pd.read_csv(input_csv_file)
    frames = len(data['iter'].unique())
    # Get n from unique values of 'iter' column
    n = len(data) // frames
    
    data['element'] = data['charge'].apply(lambda charge: charge_to_element.get(charge, 'X'))
    data['x'] = data['x'] * a0
    data['y'] = data['y'] * a0
    data['z'] = data['z'] * a0

    with open(output_xyz_file, 'w') as xyz_file:
       for frame in range(frames):
            xyz_file.write(f"{n}\n")
            xyz_file.write(f"{frame}\n")
            for _, row in data[data['iter'] == frame].iterrows():
                line = f"{row['element']} {row['x']:.3f} {row['y']:.3f} {row['z']:.3f}\n"
                xyz_file.write(line)
            break
# Example usage:
#path = '../data/dati_parigi/test_gr/50K_steps/dt_10_omega1/'
# path = '../data/paper/diffusion/production/cluster/gamma_1e-3_init_10K/parallel/merged_removing_random/'
# #path = '../data/dati_parigi/gr_paper_sara/equilibration/test_OVRVO/'
# input_csv_file = path + 'merged_solute_N100.csv'
# output_xyz_file = path + 'data_N100.xyz'
# convert_csv_to_xyz(input_csv_file, output_xyz_file)
