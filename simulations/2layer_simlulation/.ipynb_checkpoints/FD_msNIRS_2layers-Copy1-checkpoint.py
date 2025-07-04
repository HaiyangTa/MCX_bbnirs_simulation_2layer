import pmcx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle
import pandas as pd
import math

# speed of light: 
n = 1.370
c = 2.998e+10
c = c / n # cm/s

# default values: 
a1_default = 22
b1_default = 1.2
a2_default = 22
b1_default = 1.2
#
lambdas_default = [784, 800, 818, 835, 851, 868, 881, 894]
g_default = 0.85
n_default = 1.370
distance_default =  [15, 20, 25, 30]

# return mu_a, mu_s in mm-1. 
def compute_ua_us(hbo, hhb, coef_path, a, b, lambdas, g):
    C_true = np.array([hbo, hhb]) / 1e6
    #extinction_coeffs = pd.read_excel(coef_path)
    extinction_coeffs = coef_path
    extinction_coeffs_filtered = extinction_coeffs[extinction_coeffs['Lambda'].isin(lambdas)]
    E3 = extinction_coeffs_filtered[['HbO2', 'Hb']].values
    E3 = E3 * math.log(10)
    mu_a = np.dot(C_true, E3.T)
    mu_s_prime = np.array([a * (wavelength / 500) ** (-b) for wavelength in lambdas]) / (1 - g)
    return mu_a/10, mu_s_prime/10 # mm-1


# get 2 layers' properties. 
def get_2layer_properties(hbo1, hhb1, hbo2, hhb2, coef_path,a1 = a1_default, b1 = b1_default,a2 = a2_default, b2 = b2_default, lambdas = lambdas_default, g = g_default):
    mu_a_1, mu_s_1 = compute_ua_us(hbo1, hhb1, coef_path, a1, b1, lambdas, g)
    mu_a_2, mu_s_2 = compute_ua_us(hbo2, hhb2, coef_path, a2, b2, lambdas, g)
    return mu_a_1, mu_s_1, mu_a_2, mu_s_2 

def run_mcx(ua1, us1, ua2, us2, l1, g = g_default, n = n_default, distances = distance_default, tend =1e-08, devf = 10000, nphoton = 1e8, source_type='laser'):
    
    # define structure properties
    prop = np.array([
        [0.0, 0.0, 1.0, 1.0], # air
        [ua1, us1, g, n], # first layer
        [ua2, us2, g, n], # second layer
    ])

    # define voxel matrix properties 
    vol = np.ones([100, 100, 100], dtype='uint8')
    vol[:, :, 0:1] = 0
    vol[:, :, 1:l1 + 1] = 1
    vol[:, :, l1 + 1: ] = 2

    # define the boundary:
    vol[:, :, 0] = 0
    vol[:, :, 99] = 0
    vol[0, :, :] = 0
    vol[99, :, :] = 0
    vol[:, 0, :] = 0
    vol[:, 99, :] = 0
    
    detpos = [[50 + d, 50, 1, 2] for d in distances]
    
    cfg = {
          'nphoton': nphoton,
          'vol': vol,
          'tstart': 0, # start time = 0
          'tend': tend, # end time
          'tstep': tend/devf, # step size
          'srcpos': [50, 50, 1],
          'srcdir': [0, 0, 1],  # Pointing toward z=1
          'prop': prop,
          'detpos': detpos, 
          'savedetflag': 'dpxsvmw',  # Save detector ID, exit position, exit direction, partial path lengths
          'unitinmm': 1,
          'autopilot': 1,
          'debuglevel': 'DP',
    }
    
    cfg['issaveref']=1
    cfg['issavedet']=1
    cfg['issrcfrom0']=1
    cfg['maxdetphoton']=nphoton
    cfg['seed']= 1

    # define source type: 
    if source_type == 'iso': 
        cfg['srctype']= 'isotropic'

    # Run the simulation
    res = pmcx.mcxlab(cfg)
    return res, cfg


# sum_dref_per_time = x weights/mm-1/photon
def get_intensity_dynamic(cfg, res):
    # get mask. 
    nx, ny, nz, nt = res['dref'].shape
    x_grid, y_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    
    intensity_all_detectors = []
    
    for det in cfg['detpos']:
        det_x, det_y, det_z, det_r = det
        det_mask = (x_grid - det_x)**2 + (y_grid - det_y)**2 <= det_r**2
    
    
        # get intensity dynamic
        sum_dref_per_time = []
        # loop each t 
        for t in range(nt):
            dref_slice = res['dref'][:, :, 0, t]
            sum_val = np.sum(dref_slice[det_mask])
            sum_dref_per_time.append(sum_val)
            
        intensity_all_detectors.append(sum_dref_per_time)
        
    return intensity_all_detectors


def mcx_simulation(ua1, us1, ua2, us2, l1, g = g_default, n = n_default, distance = distance_default, tend =1e-08, devf = 10000, nphoton = 1e8, source_type = 'laser'):
    res, cfg = run_mcx(ua1, us1, ua2, us2, l1, g, n, distance, tend, devf, nphoton, source_type)
    intensity_d_list = get_intensity_dynamic(cfg, res)
    t = np.linspace(0, tend, devf)
    unit = tend / devf
    return intensity_d_list, unit

# final function:
def mcx_sim_2layers(hbo1, hhb1, hbo2, hhb2, l1, coef_path, a1 = a1_default, b1 = b1_default, a2 = a2_default, b2 = b2_default, lambdas = lambdas_default, g = g_default, distance = distance_default, tend =1e-08, devf = 10000, nphoton = 5e7, source_type = 'laser'):
    
    distance_data = {d: [] for d in distance}
    mu_a_1, mu_s_1, mu_a_2, mu_s_2  = get_2layer_properties(hbo1, hhb1, hbo2, hhb2, coef_path, a1, b1, a2, b2, lambdas, g)
    for sim_idx, (ua1, us1, ua2, us2) in enumerate(zip(mu_a_1, mu_s_1, mu_a_2, mu_s_2)): # 8 wl
        TPSF_list, unit = mcx_simulation(ua1, us1, ua2, us2, l1, g, n, distance, tend, devf, nphoton, source_type)
        #[[x[0] * nphoton * tend ] for x in TPSF_list] # weight/mm2
        for i, d in enumerate(distance): # 4 distances
            distance_data[d].append(TPSF_list[i])
    return distance_data


# return uac, udc and phase from fft results.  
def extract_freq(target_freq, TPSF_list, tend, devf):
    t = np.linspace(0, tend, devf)
    omega = 2 * np.pi * target_freq
    amplitude_list = []
    udc_list = []
    phase_list = []
    phase2_list = []
    
    for TPSF in TPSF_list:
        TPSF = np.array(TPSF)
        tau = np.trapz(t * TPSF, t) / np.trapz(TPSF, t)
        
        I_f = np.trapz(TPSF * np.exp(-1j * omega * t), t)
        amplitude = np.abs(I_f)
        phase = np.angle(I_f, deg=False)
        
        udc = np.trapz(TPSF, t)
         # Alternative phase using tau
        phase2 = -2 * np.pi * target_freq * tau
        
        if phase > 0 and phase2 < 0:
            phase = phase - 2 * np.pi
            
        amplitude_list.append(amplitude)
        udc_list.append(udc)
        phase_list.append(phase)
        phase2_list.append(phase2)
        
    return amplitude_list, udc_list, phase_list

