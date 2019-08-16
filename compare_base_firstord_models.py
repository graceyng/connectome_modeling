import numpy as np
import h5py
import mdl_fxns
import matplotlib.pyplot as plt
from scipy.stats import t, sem, ttest_ind
import collections

__author__ = 'Grace Ng'

#####
# Parameters
#####
conn_types = ['base', 'firstord']
connectome_files = {'base': 'W.hdf5', 'firstord': 'retro_firstord_W.hdf5'}
process_path_data_file = 'process_path_data.hdf5'
output_file = None  # 'test_base_vs_firstord.hdf5'
perf_metric = 'corr' # 'corr' or 'dist'
bootstrap = 1000  # None or integer of the number of bootstrapped samples to generate
alpha = 0.05
n_threshold = 1
perf_eval_dim = 'times' #'times' or 'regions'; the dimension along which performance is evaluated
log_shift = "no shift" #'no shift' or 'shift' or 'no log'
group = 'NTG' # group to consider
seed_region = 'R CPu'
c_params = {'base': {'c range type': 'lin', 'c range': (0.001, 10.), 'num c': 100},
            'firstord': {'c range type': 'log', 'c range': (-5., 1.), 'num c': 100}}
do_linregress = False
verbose = True
plot = True


def run_models(L_out, conn_types, Xo, times, regions, data, total_dims):
    c_values = {conn_type: [] for conn_type in conn_types}
    predicts = {conn_type: [] for conn_type in conn_types}
    perfs = {conn_type: [] for conn_type in conn_types}
    dims = {conn_type: [] for conn_type in conn_types}
    for conn_type in conn_types:
        if bootstrap is None:
            all_c, all_perf, _, this_dims, best_c, _, best_predict, _, linregress_params = \
                mdl_fxns.fit(Xo, L_out[conn_type], times, regions, np.array(data), c_params[conn_type]['c range type'],
                             c_params[conn_type]['c range'], c_params[conn_type]['num c'], perf_metric, perf_eval_dim,
                             log_shift, do_linregress, plot=False)
            c_idx = np.where(all_c == best_c)[0][0]
            c_values[conn_type] = best_c
            predicts[conn_type] = best_predict
            perfs[conn_type] = all_perf[:, c_idx]
            dims[conn_type] = np.array(this_dims)
        else:
            for i in range(bootstrap):
                all_c, all_perf, _, this_dims, best_c, _, best_predict, _, linregress_params = \
                    mdl_fxns.fit(Xo, L_out[conn_type], times, regions, np.array(data)[:, i, :],
                                 c_params[conn_type]['c range type'], c_params[conn_type]['c range'],
                                 c_params[conn_type]['num c'], perf_metric, perf_eval_dim, log_shift, do_linregress,
                                 plot=False)
                c_idx = np.where(all_c == best_c)[0][0]
                c_values[conn_type].append(best_c)
                predicts[conn_type].append(best_predict)
                this_perf = np.full(total_dims.size, np.nan)
                this_perf[[i for i in range(total_dims.size) if total_dims[i] in this_dims]] = all_perf[:, c_idx]
                perfs[conn_type].append(this_perf)
                if i % 10 == 0:
                    print('Computed performance for trial ' + str(i))
            perfs[conn_type] = np.array(perfs[conn_type])

    #### Save results
    if output_file is not None:
        with h5py.File(output_file, 'w') as f:
            if type(total_dims[0]) == str:
                f.attrs['total dims'] = [a.encode('utf8') for a in
                                         list(total_dims)]  # necessary because h5py cannot handle
                # unicode; will need to decode this back to unicode later
            else:
                f.attrs['total dims'] = total_dims
            c_val_group = f.create_group('c values')
            pred_group = f.create_group('predictions')
            perf_group = f.create_group('performance scores')
            dims_group = f.create_group('dims')
            for conn_type in conn_types:
                c_val_group.create_dataset(conn_type, data=c_values[conn_type])
                pred_group.create_dataset(conn_type, data=predicts[conn_type])
                perf_group.create_dataset(conn_type, data=perfs[conn_type])
                dims_group.create_dataset(conn_type, data=[a.encode('utf8') for a in dims[conn_type]])
    return c_values, predicts, perfs, dims


def stats_analysis(perfs, total_dims, output_file):
    plot_params = {}
    type_qual_dim_idxs = {}
    for conn_type in conn_types:
        qual_count_dict = collections.Counter(np.where(np.isfinite(perfs[conn_type]))[1])
        type_qual_dim_idxs[conn_type] = np.array([idx for idx in range(total_dims.size) if qual_count_dict[idx] > 2])
    qual_dim_idxs = np.intersect1d(type_qual_dim_idxs['base'], type_qual_dim_idxs['firstord'])
    for conn_type in conn_types:
        mean = np.nanmean(perfs[conn_type][:, qual_dim_idxs], axis=0)
        pctls = np.percentile(perfs[conn_type][:, qual_dim_idxs],
                              [50 - (1. - alpha) / 2. * 100., 50 + (1. - alpha) / 2. * 100.], interpolation='nearest',
                              axis=0)
        full_mean, low_bound, upp_bound = np.full(total_dims.size, np.nan), np.full(total_dims.size, np.nan), \
                                          np.full(total_dims.size, np.nan)
        full_mean[qual_dim_idxs] = mean
        low_bound[qual_dim_idxs] = pctls[0]
        upp_bound[qual_dim_idxs] = pctls[1]
        plot_params[conn_type] = {'mean': full_mean, 'bounds': (low_bound, upp_bound)}
    diffs = perfs['base'][:, qual_dim_idxs] - perfs['firstord'][:, qual_dim_idxs]
    diff_mean = np.nanmean(diffs, axis=0)
    diff_pctls = np.percentile(diffs, [50 - (1. - alpha) / 2. * 100., 50 + (1. - alpha) / 2. * 100.],
                               interpolation='nearest', axis=0)
    diff_full_mean, diff_low_bound, diff_upp_bound = np.full(total_dims.size, np.nan), np.full(total_dims.size, np.nan), \
                                                     np.full(total_dims.size, np.nan)
    diff_full_mean[qual_dim_idxs] = diff_mean
    diff_low_bound[qual_dim_idxs] = diff_pctls[0]
    diff_upp_bound[qual_dim_idxs] = diff_pctls[1]
    plot_params['diff mean'] = diff_mean
    plot_params['diff bounds'] = (diff_low_bound, diff_upp_bound)
    greater_tail = np.divide(np.sum(perfs['base'][:, qual_dim_idxs] > perfs['firstord'][:, qual_dim_idxs], axis=0),
                             np.sum(np.logical_and(np.isfinite(perfs['base'][:, qual_dim_idxs]),
                                                   np.isfinite(perfs['firstord'][:, qual_dim_idxs])), axis=0))
    lesser_tail = np.divide(np.sum(perfs['base'][:, qual_dim_idxs] <= perfs['firstord'][:, qual_dim_idxs], axis=0),
                            np.sum(np.logical_and(np.isfinite(perfs['base'][:, qual_dim_idxs]),
                                                  np.isfinite(perfs['firstord'][:, qual_dim_idxs])), axis=0))
    p_vals = np.full(total_dims.size, np.nan)
    p_vals[qual_dim_idxs] = 2 * np.minimum(greater_tail, lesser_tail)
    sig_idxs = [qual_dim_idxs[i] for i in np.where(p_vals < alpha)[0]]
    plot_params['p vals'] = p_vals
    plot_params['sig idxs'] = sig_idxs

    if output_file is not None:
        with h5py.File(output_file, 'a') as f:
            plot_params_group = f.create_group('plotting parameters')
            for conn_type in conn_types:
                plot_params_subgrp = plot_params_group.create_group(conn_type)
                plot_params_subgrp.create_dataset('mean', data=plot_params[conn_type]['mean'])
                plot_params_subgrp.create_dataset('low bounds', data=plot_params[conn_type]['bounds'][0])
                plot_params_subgrp.create_dataset('upp bounds', data=plot_params[conn_type]['bounds'][1])
            plot_params.create_dataset('diff mean', data=plot_params['diff mean'])
            plot_params.create_dataset('diff low bounds', data=plot_params['diff bounds'][0])
            plot_params.create_dataset('diff upp bounds', data=plot_params['diff bounds'][1])
            f.create_dataset('sig idxs', data=sig_idxs)
            f.create_dataset('p vals', data=p_vals)
    return plot_params


def load_from_file(save_filename, total_dims):
    c_values = {conn_type: [] for conn_type in conn_types}
    predicts = {conn_type: [] for conn_type in conn_types}
    perfs = {conn_type: [] for conn_type in conn_types}
    plot_params = {conn_type: {} for conn_type in conn_types}
    with h5py.File(save_filename, 'r') as f:
        for conn_type in conn_types:
            c_values[conn_type] = f['c values'][conn_type][()]
            predicts[conn_type] = f['predictions'][conn_type][()]
            perfs[conn_type] = f['performance scores'][conn_type][()]
            if 'plotting parameters' in f.keys():
                mean = f['plotting parameters'][conn_type]['mean'][()]
                low_bounds = f['plotting parameters'][conn_type]['low bounds'][()]
                upp_bounds = f['plotting parameters'][conn_type]['upp bounds'][()]
                plot_params[conn_type] = {'mean': mean, 'bounds': (low_bounds, upp_bounds)}
            else:
                plot_params = None
        if 'plotting parameters' in f.keys():
            diff_mean = f['plotting parameters']['diff mean'][()]
            diff_low_bounds = f['plotting parameters']['diff low bounds'][()]
            diff_upp_bounds = f['plotting parameters']['diff upp bounds'][()]
            plot_params['diff mean'] = diff_mean
            plot_params['diff bounds'] = (diff_low_bounds, diff_upp_bounds)
        if 'p vals' in f.keys():
            p_vals = f['p vals'][()]
        else:
            p_vals = None
        if 'sig idxs' in f.keys():
            sig_idxs = f['sig idxs'][()]
        else:
            sig_idxs = None
    return c_values, predicts, perfs, plot_params, p_vals, sig_idxs


def plot_firstord_vs_base(plot_params):
    # Plot 1st order connectivity vs. base model
    plt.figure()
    plt.scatter(plot_params['base']['mean'], plot_params['firstord']['mean'])
    identity_line = np.linspace(min(min(plot_params['base']['mean']), min(plot_params['firstord']['mean'])),
                                max(max(plot_params['base']['mean']), max(plot_params['firstord']['mean'])))
    if log_shift == 'shift' or log_shift == 'no shift':
        label = ' (log with ' + log_shift + ')'
    else:
        label = ''
    plt.xlabel(perf_metric + ' for base model' + label)
    plt.ylabel(perf_metric + ' for 1st order connectivity model' + label)
    plt.plot(identity_line, identity_line, color="red", linestyle="dashed", linewidth=1.0)
    plt.show()


def plot_perf_across_dim(plot_params, total_dims):
    p_vals = plot_params['p vals']
    qual_idxs = np.where(np.isfinite(p_vals))[0]
    asterisks = np.array(['' for i in range(p_vals.size)])
    asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.05)[0]]] = '*'
    # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.01)[0]]] = '**'
    # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.001)[0]]] = '***'
    # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.0001)[0]]] = '****'
    plt.figure()
    for conn_type in conn_types:
        neg_err = plot_params[conn_type]['mean'] - plot_params[conn_type]['bounds'][0]
        pos_err = plot_params[conn_type]['bounds'][1] - plot_params[conn_type]['mean']
        plt.errorbar(total_dims, plot_params[conn_type]['mean'], yerr=[neg_err, pos_err],
                     label=conn_type + ' model', fmt='o')
    y_max = max(np.max(plot_params['base']['bounds'][1]), np.max(plot_params['firstord']['bounds'][1]))
    y_min = min(np.min(plot_params['base']['bounds'][0]), np.max(plot_params['firstord']['bounds'][0]))
    for i, asterisk in enumerate(asterisks):
        if asterisk:
            plt.text(total_dims[i], y_max + abs(y_max - y_min) * 0.01, asterisk, horizontalalignment='center',
                     verticalalignment='center')
    if log_shift == 'shift' or log_shift == 'no shift':
        label = ' (log with ' + log_shift + ')'
    else:
        label = ''
    plt.xlabel(perf_eval_dim)
    plt.ylabel(perf_metric + label)
    plt.title(perf_metric + label + ' across ' + perf_eval_dim)
    plt.legend()
    plt.show()

    plt.figure()
    neg_err = plot_params['diff mean'] - plot_params['diff bounds'][0]
    pos_err = plot_params['diff bounds'][1] - plot_params['diff mean']
    plt.errorbar(total_dims, plot_params['diff mean'], yerr=[neg_err, pos_err], fmt='o')
    y_max = max(np.max(plot_params['diff bounds'][1]), np.max(plot_params['diff bounds'][1]))
    y_min = min(np.min(plot_params['diff bounds'][0]), np.max(plot_params['diff bounds'][0]))
    for i, asterisk in enumerate(asterisks):
        if asterisk:
            plt.text(total_dims[i], y_max + abs(y_max - y_min) * 0.01, asterisk, horizontalalignment='center',
                     verticalalignment='center')
    if log_shift == 'shift' or log_shift == 'no shift':
        label = ' (log with ' + log_shift + ')'
    else:
        label = ''
    plt.xlabel(perf_eval_dim)
    plt.ylabel(perf_metric + label)
    plt.title(perf_metric + label + ' across ' + perf_eval_dim)
    plt.show()
    return asterisks


#####
# Load Data from Files
#####
process_path_data = {}
with h5py.File(process_path_data_file, 'r') as f:
    for g in f.keys():
        if g == group:
            process_path_data[group] = {}
            for time in f[group].keys():
                process_path_data[group][int(time)] = f[group].get(time)[()]
W = {}
for conn_type in conn_types:
    with h5py.File(connectome_files[conn_type], 'r') as f:
        W[conn_type] = f.get('W')[()]
        ipsi_regions = f['W'].attrs['Ipsi_Regions']
        contra_regions = f['W'].attrs['Contra_Regions']

# create a vector of the names of all the regions of interest -- first, with the right-sided regions, and then the
# left-sided regions
regions = np.concatenate((np.add('R ', ipsi_regions), np.add('L ', contra_regions)))

# compute the mean of the pathology data across mice for each group and time point

data = []
for time in process_path_data[group]:
    if bootstrap is None:
        data.append(np.nanmean(process_path_data[group][time], axis=0))
    else:
        # Need to bootstrap if computing correlation
        n_mice = process_path_data[group][time].shape[0]
        combos = np.random.randint(0, n_mice, (bootstrap, n_mice))
        data.append(np.nanmean(process_path_data[group][time][combos, :], axis=1))

#####
# Fit a Model Using Pathology Data with Seeding from the iCPu Region
#####
Xo = mdl_fxns.make_Xo(seed_region, regions)
L_out = {}
for conn_type in conn_types:
    L_out[conn_type] = mdl_fxns.get_L_out(W[conn_type],
                                          normalize=False)  # generate weighted out-degree Laplacian matrix
    # Note that normalization of the W matrix must be excluded when computing L_out because
    # eigenvalues of the first-order W matrix are 0.

times = np.array(list(process_path_data[group].keys()))
if perf_eval_dim == 'times':
    total_dims = np.array(times)
elif perf_eval_dim == 'regions':
    total_dims = np.array(regions)
else:
    raise Exception('perf_eval_dim must be either "times" or "regions".')

# c_values, predicts, perfs, dims = run_models(L_out, conn_types, Xo, times, regions, data, total_dims)
c_values, predicts, perfs, plot_params, p_vals, sig_idxs = load_from_file('test_base_vs_firstord.hdf5', total_dims)
if bootstrap is not None:
    plot_params = stats_analysis(perfs, total_dims, output_file)
    plot_firstord_vs_base(plot_params)
    asterisks = plot_perf_across_dim(plot_params, total_dims)
