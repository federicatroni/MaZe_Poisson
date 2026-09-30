# TODO(bin format): reads output files with pd.read_csv, adapt to also read .bin (see maze_poisson/myio/output/binary.py)
import os

import matplotlib.ticker
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ...myio import logger

label_fontsize=17
legend_fontsize=15

N_vector = [30,40,50,60,70,80,90,100,110,120] # will change from 10 to 100 once the CG works for 10,20
N_vector = [30,40,50,60,70,80,90,100,110,120] # will change from 10 to 100 once the CG works for 10,20
N_vector = np.array(N_vector)
N_p_vector = [128,250,432,686,1024,1458]


def g(x,a,b):
    return a * x**b

def g1(x, a, b):
    return a * x + b

def f(x,a,b):
    return a * x**b * np.log(x**b)

def k(x, a, b):
    return a*x + b*x**2

# time-vs-n3:  plot of time/iter VS N_grid = N^3
def plot_time_iterNgrid(N_p):
    path = 'Outputs/'
    path_pdf = os.path.join(path, 'PDFs/')
    filename_MaZe=path+'performance_N'
    data1 = "time" 
    data2 = 'n_iters'

    os.makedirs(path_pdf, exist_ok=True)
    
    path_all_files = [(filename_MaZe + str(i) + '_N_p'+str(N_p)+'.csv') for i in N_vector]
    isExist = [os.path.exists(i) for i in path_all_files]
    if all(isExist) == False:
        print(isExist)
        print(isExist)
        logger.error(str(len(N_vector))+ " files are needed. The files needed do not exist at "+filename_MaZe)
        raise FileNotFoundError(str(len(N_vector))+ " files are needed. The files needed do not exist at "+filename_MaZe)
    elif all(isExist) == True:
        df_list_MaZe = [pd.read_csv(filename_MaZe + str(i) + '_N_p'+str(N_p)+'.csv') for i in N_vector]

    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][100:]/df[data2][100:]))
        sd1.append(np.std(df[data1][100:]/df[data2][100:]))
        
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_vector
    poptMaZe, _ = curve_fit(g1, x**3, avg1, sigma = sd1, absolute_sigma=True, p0=[0.1, 1.4])
    a_optMaZe, b_optMaZe = poptMaZe

    print(f'Optimized parameters MaZe t: a = {a_optMaZe}, b = {b_optMaZe}')

    plt.figure(figsize=(10, 8))
    plt.errorbar(x**3, avg1,sd1, label = 'MaZe', color='r',marker='o', linestyle='', linewidth=1.5, markersize=6,capsize=4)
    plt.plot(x**3, g1(x**3, a_optMaZe, b_optMaZe), label=f'fit $ax+b$, b = {b_optMaZe:.6f},  a = {a_optMaZe}')
    #plt.ylim(0, 0.02)
    #plt.xlim(20**3, 120**3)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Number of grid points', fontsize=label_fontsize)
    plt.ylabel('Time (s)', fontsize=label_fontsize)
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel('Number of grid points', fontsize=label_fontsize)
    plt.ylabel('Time (s)', fontsize=label_fontsize)
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    title = 'time_per_iter_VS_N3_N_p_'
    plt.grid()
    name =  title +str(N_p)+ ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()


def f(x,a,b):
    return a * x + b

# n-iter-vs-n3: plot of n iterations VS N_grid = N^3
# n-iter-vs-n3: plot of n iterations VS N_grid = N^3
def plot_convNgrid(N_p):
    path = 'Outputs/'
    path_pdf = os.path.join(path, 'PDFs/')
    filename_MaZe=path+'performance_N'
    data1 = "n_iters"

    os.makedirs(path_pdf, exist_ok=True)

    path_all_files = [(filename_MaZe + str(i) + '_N_p'+str(N_p)+'.csv') for i in N_vector]
    isExist = [os.path.exists(i) for i in path_all_files]
    if all(isExist) == False:
        logger.error(str(len(N_vector))+ " files are needed. The files needed do not exist at "+filename_MaZe)
        raise FileNotFoundError(str(len(N_vector))+ " files are needed. The files needed do not exist at "+filename_MaZe)
    elif all(isExist) == True:
        df_list_MaZe = [pd.read_csv(filename_MaZe + str(i) + '_N_p'+str(N_p)+'.csv') for i in N_vector]

    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][100:]))
        sd1.append(np.std(df[data1][100:]))
        
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_vector**3
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(N_vector**3, avg1,sd1, label = 'MaZe', color='r',marker='o', linestyle='', markersize=6,capsize=4)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe),  label=f'fit $ax^b$, b = {b_optMaZe:.2f}  $\\approx 1/3$ a = {a_optMaZe:.2f}')
    plt.xlabel('Number of grid points', fontsize=label_fontsize)
    plt.ylabel('# iterations', fontsize=label_fontsize)
    #plt.ylim(0,200)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    plt.grid()
    title='n_iterations_vs_N_grid_N_p_'
    name =  title +str(N_p)+ ".pdf"
    plt.savefig(path_pdf + name, format='pdf')
    plt.show()


#### PLOT FUNCTION OF Number of particles ####

# time-vs-np: plot time / n iterations VS number of particles
def plot_scaling_particles_time_iters():
    data1 = "time"
    data2 = "n_iters"
    filename_MaZe='performance_N'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    
    os.makedirs(path_pdf, exist_ok=True)
    
    N_vector = [80, 100, 120, 140, 160, 180]
    N_p=N_p_vector
    N_p = np.array(N_p)
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'_'+str(j)+ '.csv') for i,j in zip(N_vector, N_p)]
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'_'+str(j)+ '.csv') for i,j in zip(N_vector, N_p)]
    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][-50:]/df[data2][-50:]))
        sd1.append(np.std(df[data1][-50:]/df[data2][-50:]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_p
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True, p0=[1e-8,1])
    a_optMaZe, b_optMaZe = poptMaZe

    print(f'Optimized parameters MaZe: a = {a_optMaZe}, b = {b_optMaZe}')
    plt.figure(figsize=(10, 8))
   
    plt.errorbar(N_p, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe),  label=f'fit $ax^b$, b = {b_optMaZe:.2f}')
    plt.xlabel('Number of particles', fontsize=label_fontsize)
    plt.ylabel('Time (s)', fontsize=label_fontsize)
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    plt.xlabel('Number of particles', fontsize=label_fontsize)
    plt.ylabel('Time (s)', fontsize=label_fontsize)
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    plt.grid()
    title='time_per_n_iterations_vs_N_p'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()


def f(x,a,b):
    return a * np.log(x)**b


# iter-vs-np: plot n iterations VS number of particles
def plot_scaling_particles_conv():
    filename_MaZe='performance_N'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    
    os.makedirs(path_pdf, exist_ok=True)

    data1 = "n_iters"
    N_vector = [80, 100, 120, 140, 160, 180]
    N_p = np.array(N_p_vector)
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'_'+str(j)+'.csv') for i,j in zip(N_vector,N_p)]
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'_'+str(j)+'.csv') for i,j in zip(N_vector,N_p)]
    avg1 = []
    sd1 = []


    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][-50:]))
        sd1.append(np.std(df[data1][-50:]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = N_p
    poptMaZe, _ = curve_fit(f, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    print(N_p_vector)
    print(sd1)
    plt.errorbar(N_p, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, f(x, a_optMaZe, b_optMaZe), label=f'fit $a\\log{{x}}^b$, b = {b_optMaZe:.2f}') 
    plt.xlabel('Number of particles', fontsize=label_fontsize)
    plt.ylabel('# Iterations', fontsize=label_fontsize)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    plt.xlabel('Number of particles', fontsize=label_fontsize)
    plt.ylabel('# Iterations', fontsize=label_fontsize)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    plt.grid()
    title='iterations_vs_N_p'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()

# iter-vs-threads: plot number of iterations vs. number of threads
# SANITY CHECK GRAPH
def iter_vs_threads():
    filename_MaZe='performance_N100_N_p_250_'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    
    os.makedirs(path_pdf, exist_ok=True)

    data1 = "n_iters"
    threads = np.array([5, 6, 7, 8, 9, 10, 11, 12])
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'.csv') for i in threads]
    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1][-50:]))
        sd1.append(np.std(df[data1][-50:]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = threads
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(x, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe), label=f'fit $ax^b$, b = {b_optMaZe:.2f} a = {a_optMaZe:.2f}') 
    plt.xlabel('Number of threads', fontsize=label_fontsize)
    plt.ylabel('# iterations', fontsize=label_fontsize)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    plt.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    plt.grid()
    title='iterations_vs_threads'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()

# time-vs-threads: plot time vs. number of threads
def time_vs_threads():
    # plotting strong scaling and weak scaling as references
    filename_strong='performance_N100_'
    filename_strong='performance_N100_'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    
    os.makedirs(path_pdf, exist_ok=True)

    data1 = "time"
    threads = np.array([1, 2, 4, 8, 16, 32, 64])
    threads = np.array([1, 2, 4, 8, 16, 32, 64])
    df_list_strong = [pd.read_csv(path+filename_strong + str(i) +'.csv') for i in threads]
    avg1 = []
    sd1 = []

    for df in df_list_strong:
        avg1.append(np.mean(df[data1]))
        sd1.append(np.std(df[data1]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)
    speedup=[]

    for i in range(len(avg1)):
        speedup.append(avg1[0]/avg1[i])
    speedup=[]

    for i in range(len(avg1)):
        speedup.append(avg1[0]/avg1[i])

    x = threads
    poptMaZe, _ = curve_fit(g, x, avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(x, avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(x, g(x, a_optMaZe, b_optMaZe), label=f'fit $ax^b$, b = {b_optMaZe:.2f} a = {a_optMaZe:.2f}') 
    plt.xlabel('Number of threads', fontsize=label_fontsize)
    plt.ylabel('time per iteration (s)', fontsize=label_fontsize)
    plt.xlabel('Number of threads', fontsize=label_fontsize)
    plt.ylabel('time per iteration (s)', fontsize=label_fontsize)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper right', fontsize=legend_fontsize)
    plt.legend(frameon=False, loc='upper right', fontsize=legend_fontsize)
    plt.grid()
    title='time_vs_threads'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()


# strong-scaling-vs-threads: changing threads while keeping N constant
def strong_scaling_vs_threads():
    # plotting strong scaling as references
    filename_strong='performance_N100_'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    
    os.makedirs(path_pdf, exist_ok=True)

    data1 = "time"
    threads = np.array([1, 2, 4, 8, 16, 32, 64])
    df_list_strong = [pd.read_csv(path+filename_strong + str(i) +'.csv') for i in threads]
    avg1 = []
    sd1 = []

    for df in df_list_strong:
        avg1.append(np.mean(df[data1]))
        sd1.append(np.std(df[data1]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)
    speedup=[]

    for i in range(len(avg1)):
        speedup.append(avg1[0]/avg1[i])

    x = threads
    poptMaZe, _ = curve_fit(g1, x, speedup, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(x,x, label= 'Ideal strong scaling')
    ax1.plot(x, speedup, color='red', marker='o', label='MaZe')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.set_yticks([1, 2, 4, 8, 16, 32, 64])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel('Number of threads', fontsize=label_fontsize)
    ax1.set_ylabel('Speedup', fontsize=label_fontsize)
    ax1.legend(frameon=False, loc='upper left', fontsize=legend_fontsize)
    ax1.grid()
    title='speedup_vs_threads'
    name =  title + ".pdf"
    fig1.savefig(path_pdf+name, format='pdf')
    plt.show()

    #plt.plot(x,x, label= 'Ideal strong scaling')
    #plt.plot(x, speedup, color='red', marker='o', label='MaZe')
    '''
    plt.xticks(ticks=x, labels =['1','2','4','8','16','32','64'], minor=True)
    plt.yticks(ticks=x, labels =['1','2','4','8','16','32','64'], minor=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of threads', fontsize=18)
    plt.ylabel('Speedup', fontsize=18)
    plt.legend(frameon=False, loc='upper left', fontsize=15)
    plt.grid()
    title='speedup_vs_threads'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()
    '''

# weak-scaling-vs-threads: changing threads while keeping N constant
def weak_scaling_vs_threads():
    # plotting weak scaling as references
    filename_strong='performance_N'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    
    os.makedirs(path_pdf, exist_ok=True)

    data1 = "time"
    data2 = "n_iters"
    N_arr = [30, 36, 42, 50, 60, 71, 85]
    threads = np.array([1, 2, 4, 8, 16, 32, 64])
    one = [1,1,1,1,1,1,1]

    df_list_strong = [pd.read_csv(path+filename_strong + str(i) +'_'+str(j)+'.csv') for i,j in zip(N_arr, threads)]
    avg1 = []
    sd1 = []

    for df in df_list_strong:
        avg1.append(np.mean(df[data1]))
        sd1.append(np.std(df[data1]))
    
    print(avg1) 
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)
    efficiency=[]

    for i in range(len(avg1)):
        efficiency.append(avg1[0]/avg1[i])
    print(efficiency)

    x = threads
    poptMaZe, _ = curve_fit(g1, x, efficiency, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(x, one, label= 'Ideal weak scaling')
    ax1.plot(x, efficiency, color='red', marker='o', label='MaZe')
    ax1.set_xscale('log')
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel('Number of threads', fontsize=label_fontsize)
    ax1.set_ylabel('Efficiency', fontsize=label_fontsize)
    ax1.legend(frameon=False, loc='lower left', fontsize=legend_fontsize)
    ax1.grid()
    title='weak_scaling_vs_threads'
    name =  title + ".pdf"
    fig1.savefig(path_pdf+name, format='pdf')
    plt.show()

def iter_vs_tol():
    filename_MaZe='performance_tol'
    path = 'Outputs/'
    path_pdf = path + 'PDFs/'
    
    os.makedirs(path_pdf, exist_ok=True)

    data1 = "n_iters"
    tolerance = np.array([1.e-5, 1.e-6, 1.e-7, 1.e-8, 1.e-9, 1.e-10])
    df_list_MaZe = [pd.read_csv(path+filename_MaZe + str(i) +'.csv') for i in tolerance]
    avg1 = []
    sd1 = []

    for df in df_list_MaZe:
        avg1.append(np.mean(df[data1]))
        sd1.append(np.std(df[data1]))
     
    avg1 = np.array(avg1)
    sd1 = np.array(sd1)

    x = tolerance
    poptMaZe, _ = curve_fit(g1, np.log10(x), avg1, sigma = sd1, absolute_sigma=True)
    a_optMaZe, b_optMaZe = poptMaZe

    plt.figure(figsize=(10, 8))
    plt.errorbar(np.log10(x), avg1, yerr=sd1, label = 'MaZe', color='r', marker='o', linestyle='', linewidth=1.5, markersize=6, capsize=5)
    plt.plot(np.log10(x), g1(np.log10(x), a_optMaZe, b_optMaZe), label=f'fit $ax^b$, b = {b_optMaZe:.2f} a = {a_optMaZe:.2f}') 
    plt.xlabel(r'log$_{10}$(Tolerance)', fontsize=label_fontsize)
    plt.ylabel('# iterations', fontsize=label_fontsize)
    #plt.xscale('log')
    plt.legend(frameon=False, loc='upper right', fontsize=legend_fontsize)
    plt.legend(frameon=False, loc='upper right', fontsize=legend_fontsize)
    plt.grid()
    plt.title("N_p=250, N=100, dt=0.25")
    title='iterations_vs_tol'
    name =  title + ".pdf"
    plt.savefig(path_pdf+name, format='pdf')
    plt.show()

