import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from opt_utils import plot_targets

from pdb import set_trace as bp
fname = "/Users/matthewlevine/Downloads/hpc/dimers/BoundedBumps_1.0_m3/fit_summary.pkl"
with open(fname, "rb") as f:
    x = pickle.load(f)

# plot all the fits
inputs = 10**np.linspace(-3, 3, x['targets'].shape[-1])
output_dir = "/Users/matthewlevine/Downloads/hpc/dimers/BoundedBumps_1.0_m3/fits"
plot_targets(output_dir, inputs, 10**x['targets'], 10**x['fits'].transpose(0,2,1), label_fits=True)

def get_details(f, f_fits):

    # targets only
    min = np.min(f)
    max = np.max(f)
    if f[0]==min:
        start = 0
    else:
        start = 1
    range = max-min
    shift_inds = np.where(np.diff(f)!=0)[0]
    shift0 = -1
    shift1 = -1
    shift2 = -1
    if len(shift_inds) > 0:
        shift0 = shift_inds[0]
    if len(shift_inds) > 1:
        shift1 = shift_inds[1]
    if len(shift_inds) > 2:
        shift2 = shift_inds[2]

    # fit quality
    mse = np.mean((f - f_fits.T)**2, axis=1)

    data = {'start': start, 'min': min, 'max': max, 'range': range,
            'shift0': shift0,  'shift1': shift1,  'shift2': shift2,
            'mse0': mse[0], 'mse1': mse[1]}
    return data

# get details about the targets
df = pd.DataFrame()
det_list = []
for n in range(x['targets'].shape[0]):
    f = x['targets'][n]
    f_fits = x['fits'][n]
    details = get_details(f, f_fits)
    det_list.append(details)
df = pd.concat([df,pd.DataFrame.from_dict(det_list)])

df_jitter = df.copy()
rn = np.array(np.max(df_jitter.iloc[:,:-3]) - np.min(df_jitter.iloc[:,:-3]))
noise = np.random.randn(*df_jitter.iloc[:,:-3].shape)
df_jitter.iloc[:,:-3] += 0.1 * rn * noise # noise has variance 10% of range

def plot_all(x, **kwargs):
    if kwargs['label'] == first_label:
        foo = sns.scatterplot(data=kwargs['data'], x=x.name, y=kwargs['var'], alpha=0.5)
        foo.set_ylabel(kwargs['var'])
        foo.set_title(kwargs['var'])

# plot mse per descriptor
vars = df.columns[:-3]
for s in range(2):
    varname = 'mse{}'.format(s)
    first_label = df_jitter[varname][0]
    pplot = sns.pairplot(df_jitter, hue=varname, x_vars=vars, y_vars=vars, diag_kind='scatter', plot_kws=dict(alpha=0.5), corner=True) # don't plot shift2, mse0, mse1
    pplot.map_diag(plot_all, data=df_jitter, var=varname)
    plot_dir = "/Users/matthewlevine/Downloads/hpc/dimers/BoundedBumps_1.0_m3/fit_analyses_run{}.pdf"
    pplot.fig.savefig(plot_dir.format(s), format='pdf')

# TODO: 1) add diagonal plots. 2) use alpha for scatter. 3) think about MSE size ?
bp()
