# import os, sys
# import matplotlib
# import matplotlib.pyplot as plt
# plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
#                 'legend.facecolor': 'white', 'legend.framealpha': 0.8,
#                 'legend.loc': 'upper left', 'lines.linewidth': 4.0})
import scipy.interpolate
import numpy as np
from pdb import set_trace as bp

def set_target_library(n_input_samples=10, target_lib_name="SinCos", target_lib_file=None, target_lib_names_file=None, bump_centers=[0.2, 0.4, 0.6], bump_width=2, n_switch_points=5, n_switches=2, start="both"):
    '''Generate a library of functions to which we will fit or use to measure expressivity.'''
    if target_lib_name=='SinCos':
        x = np.linspace(0, 2*np.pi, n_input_samples)
        target_function_sin = 0.5*(np.sin(x)+1)
        target_function_cos = 0.5*(np.cos(x)+1)

        f_targets = np.vstack((target_function_sin, target_function_cos))
    elif target_lib_name=='bumps_all':
        F = bumps_all(n_switches, n_switch_points, start=start)
        f_targets = interp_target(n_input_samples, F, kind='next')
    elif target_lib_name=='MetaClusters':
        with open(target_lib_file, 'rb') as f:
            F = np.load(f) # n_clusters x discretization
            f_targets = interp_target(n_input_samples, F)
    elif target_lib_name=='bump':
        # this function uses relative bump centers over the input domain and bump width in units of the discretization points.
        # SO, if you change n_input_samples, the bump function will change (e.g. finer discretization => narrower bump).
        # to alleviate this, can make the bump_width wider.
        # in the future, should define bumps in real coordinates, THEN interpolate and determine discretizations afterwards.
        f_targets = bump_targets(bump_centers, bump_width, n_input_samples)
    elif target_lib_name=='2d_jacob':
        F = np.load(target_lib_file) # n_clusters x discretization
        F_names = np.load(target_lib_names_file) # n_clusters x discretization
        f_targets = {F_names[i]: F[i] for i in range(len(F_names))}
    else:
        raise('library name "{}" not recognized. Quitting.'.format(target_lib_name))

    return f_targets

def interp_target(n_input_samples, F, kind='cubic'):
    ''''''
    n_clusters, n_pts = F.shape
    grid = np.linspace(0, 1, n_pts)
    target_grid = np.linspace(0, 1, n_input_samples)
    f_interp = scipy.interpolate.interp1d(grid, F, kind=kind)
    f_targets = f_interp(target_grid)
    return f_targets

def bump_targets(bump_centers, bump_width, n_input_samples):
    n_targets = len(bump_centers)
    f_targets = np.zeros((n_targets, n_input_samples))
    for n in range(n_targets):
        i_mid = int(bump_centers[n]*n_input_samples)
        i_low = max(0, i_mid - bump_width)
        i_high = min(n_input_samples-1, i_mid + bump_width)
        f_targets[n,i_low:i_high] = 1 # assign the bump
    return f_targets

def bump_on(bump_starts, n_input_samples):
    n_targets = len(bump_starts)
    f_targets = np.zeros((n_targets, n_input_samples))
    for n in range(n_targets):
        i_low = int(bump_starts[n]*n_input_samples)
        f_targets[n,i_low:] = 1 # assign the bump
    return f_targets

def bumps_all(n_switches=1, n_switch_points=5, bounds=(0,1), start="both", make_constants=False, low=-3, high=3, inc=1):
    '''low: lowest log10 value taken by a target
       high: highest log10 value taken by a target
    '''
    # note: bounds are arbitrary

    if start=="on":
        start_list = [1]
    elif start=="off":
        start_list = [0]
    elif start=="both":
        start_list = [0,1]

    switch_grid = np.linspace(bounds[0], bounds[1], n_switch_points+2) # always includes bounds[0] and bounds[1] by default (+2 for endpoints)

    for n_lam in range(n_switches+1):
        if n_lam==0:
            if make_constants:
                f_targets = [s*np.ones_like(switch_grid) for s in start_list]
            else:
                f_targets = []
            continue
        elif n_lam==1:
            # switch_points = switch_grid[1:-1].reshape(-1,1)
            switch_points = switch_grid[1:].reshape(-1,1)
        else:
            # get all combinations of switch points
            switch_points = np.array(np.meshgrid(*[switch_grid for _ in range(n_lam)])).T.reshape(-1,n_lam)

            # restrict to subset to ordered pairs
            switch_points = switch_points[np.all(switch_points[:,:-1] < switch_points[:,1:], axis=1)]

            # remove startpoint
            # switch_points = switch_points[(switch_points[:,0]>bounds[0]) & (switch_points[:,-1]<bounds[1])]
            switch_points = switch_points[(switch_points[:,0]>bounds[0])]

        for f0 in start_list: # start off/on
            for sp_vec in switch_points:
                f = np.zeros_like(switch_grid)
                f[:] = f0 # set everything to f0 (on/off)
                # print(sp_vec)
                for j in range(1,len(switch_grid)):
                    if switch_grid[j] in sp_vec:
                        # print('Switching at',j)
                        f[j:] = not(f[j-1])
                for k in range(low,high):
                    for j in range(k+1,high+1):
                        f_k = np.copy(f)
                        f_k[f==0] = 10**k
                        f_k[f==1] = 10**j
                        f_targets.append(f_k)

    # make targets an array
    f_targets = np.array(f_targets)
    return f_targets
