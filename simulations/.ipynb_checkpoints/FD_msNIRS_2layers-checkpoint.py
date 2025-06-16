import pmcx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pickle

# speed of light: 
n = 1.370
c = 2.998e+10
c = c / n # cm/s


def run_mcx(ua, us, g=0.85, n=1.370, distances = [15, 20, 25, 30], tend =1e-08, devf = 10000, nphoton = 1e8, source_type='laser'):
    
    # define structure properties
    prop = np.array([
        [0.0, 0.0, 1.0, 1.0], # air
        [ua, us, g, n], # first layer
    ])

    # define voxel matrix properties 
    vol = np.ones([100, 100, 100], dtype='uint8')
    vol[:, :, 0:1] = 0
    vol[:, :, 1:] = 1

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
    #cfg['seed']= 999

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




def mcx_simulation(ua, us, g=0.85, n=1.370, distance =  [15, 20, 25, 30], tend =1e-08, devf = 10000, nphoton = 1e8, source_type = 'laser'):
    """
    Wrapper function to run MCX simulation and extract time-resolved intensity.

    Parameters:
        ua (float): Absorption coefficient [1/mm]
        us (float): Scattering coefficient [1/mm]
        g (float): Anisotropy factor (default: 0.85)
        n (float): Refractive index (default: 1.370)
        distance (float): Source-detector separation in mm (default: 15)
        tend (float): Simulation time duration (default: 1e-08 s)
        devf (int): Number of time intervals (default: 10000)
        nphoton (float): Total number of photons (default: 1e8)

    Returns:
        intensity_d (list): Time-resolved detector intensity values
        unit (float): Time step per frame (i.e., temporal resolution)
    """
    res, cfg = run_mcx(ua, us, g, n, distance, tend, devf, nphoton, source_type)
    intensity_d_list = get_intensity_dynamic(cfg, res)
    t = np.linspace(0, tend, devf)
    unit = tend / devf
    return intensity_d_list, unit
    

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
        
    return amplitude_list, udc_list, phase_list, phase2_list





