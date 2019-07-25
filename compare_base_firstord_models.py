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
perf_metric = 'corr' # 'corr' or 'dist'
bootstrap = 50 # None or integer of the number of bootstrapped samples to generate
n_threshold = 20
perf_eval_dim = 'regions' #'times' or 'regions'; the dimension along which performance is evaluated
log_shift = "no shift" #'no shift' or 'shift' or 'no log'
group = 'NTG' # list of groups to consider, e.g. ['G20', 'NTG'] or 'all'
seed_region = 'R CPu'
c_params = {'base': {'c range type': 'lin', 'c range': (0.001, 10.), 'num c': 100},
            'firstord': {'c range type': 'log', 'c range': (-5., 1.), 'num c': 100}}
do_linregress = True
verbose = True
plot = True

cluster_analysis = 'check silhouettes' #'check silhouettes', 'plot clusters only', 'plot clusters and timecourses', 'None'
n_clusters = 3 #get this value by finding the cluster number with the maximum silhouette value
cluster_to_analyze = 0

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
        data.append(np.nanmean(process_path_data[group][time][combos,:], axis=1))

#####
# Fit a Model Using Pathology Data with Seeding from the iCPu Region
#####
Xo = mdl_fxns.make_Xo(seed_region, regions)
L_out = {}
for conn_type in conn_types:
    L_out[conn_type] = mdl_fxns.get_L_out(W[conn_type]) # generate weighted out-degree Laplacian matrix

times = np.array(list(process_path_data[group].keys()))
if perf_eval_dim == 'times':
    total_dims = np.array(times)
elif perf_eval_dim == 'regions':
    total_dims = np.array(regions)
else:
    raise Exception('perf_eval_dim must be either "times" or "regions".')
c_values = {conn_type: [] for conn_type in conn_types}
predicts = {conn_type: [] for conn_type in conn_types}
perfs = {conn_type: [] for conn_type in conn_types}
dims = {conn_type: [] for conn_type in conn_types}
plot_params = {conn_type: {}}
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
                mdl_fxns.fit(Xo, L_out[conn_type], times, regions, np.array(data)[:,i,:],
                             c_params[conn_type]['c range type'], c_params[conn_type]['c range'],
                             c_params[conn_type]['num c'], perf_metric, perf_eval_dim, log_shift, do_linregress,
                             plot=False)
            c_idx = np.where(all_c == best_c)[0][0]
            c_values[conn_type].append(best_c)
            predicts[conn_type].append(best_predict)
            this_perf = np.full(total_dims.size, np.nan)
            this_perf[[i for i in range(total_dims.size) if total_dims[i] in this_dims]] = all_perf[:, c_idx]
            perfs[conn_type].append(this_perf)
            if i%10 == 0:
                print('Computed performance for trial ' + str(i))
        perfs[conn_type] = np.array(perfs[conn_type])
        qual_count_dict = collections.Counter(np.where(np.isfinite(perfs[conn_type]))[1])
        qual_dim_idxs = np.array([idx for idx in range(total_dims.size) if qual_count_dict[idx] > 3])
        mean = np.nanmean(perfs[conn_type][:,qual_dim_idxs], axis=0)
        itvl = t.interval(0.95, mean.size - 1, loc=mean, scale=sem(perfs[conn_type][:,qual_dim_idxs], nan_policy='omit',
                                                                   axis=0))
        full_mean, neg_err, pos_err = np.full(total_dims.size, np.nan), np.full(total_dims.size, np.nan), \
                                      np.full(total_dims.size, np.nan)
        full_mean[qual_dim_idxs] = mean
        neg_err[qual_dim_idxs] = mean - itvl[0]
        pos_err[qual_dim_idxs] = itvl[1] - mean
        plot_params[conn_type] = {'mean': full_mean, 'err': (neg_err, pos_err)}
        dims[conn_type].append(np.array(this_dims))

#Run a student's t-test, where...
#   for correlation, Ho: firstord >= base and Ha: firstord < base
#   for distance, Ho: base >= firstord and Ha: base < firstord
all_t_stat = []
all_p_val = []
num_samples = []
for idx in range(total_dims.size):
    qual_idxs = np.intersect1d(np.where(np.isfinite(perfs['base'][:,idx]))[0],
                               np.where(np.isfinite(perfs['firstord'][:,idx]))[0])
    if qual_idxs.size > n_threshold:
        if perf_metric == 'corr':
            orig_t_stat, orig_p_val = ttest_ind(perfs['base'][qual_idxs,idx], perfs['firstord'][qual_idxs,idx],
                                                equal_var=False)
        elif perf_metric == 'dist':
            orig_t_stat, orig_p_val = ttest_ind(perfs['firstord'][qual_idxs,idx], perfs['base'][qual_idxs,idx],
                                                equal_var=False)
        else:
            raise Exception('perf_metric must either be "corr" or "dist".')
        all_t_stat.append(orig_t_stat)
        all_p_val.append(orig_p_val / 2.)
        num_samples.append(qual_idxs.size)
    else:
        all_t_stat.append(np.nan)
        all_p_val.append(np.nan)
        num_samples.append(0)
all_t_stat = np.array(all_t_stat)
all_p_val = np.array(all_p_val)
sig_idxs = np.where(np.isfinite(all_t_stat))[0]
sig_idxs = np.array([sig_idx for sig_idx in sig_idxs if all_t_stat[sig_idx] > 0. and all_p_val[sig_idx] < 0.05])
print('Performance for the base model was significantly better than that of the first-order connectivity model for the '
      'following dimensions:')
for sig_idx in sig_idxs:
    print(perf_eval_dim + ' ' + str(total_dims[sig_idx]) +
          ': t-statistic {:.3f}, p-value {:.3f}'.format(all_t_stat[sig_idx], all_p_val[sig_idx]))

#Plot 1st order connectivity vs. base model
plt.figure()
if bootstrap is None:
    dims_to_plot = set(dims['base']) & set(dims['firstord'])
    plt.scatter(perfs['base'][np.array([i for i in range(dims['base'].size) if dims['base'][i] in dims_to_plot])],
                perfs['firstord'][np.array([i for i in range(dims['firstord'].size) if dims['firstord'][i] in dims_to_plot])])
    identity_line = np.linspace(min(min(perfs['base']), min(perfs['firstord'])),
                                max(max(perfs['base']), max(perfs['firstord'])))
else:
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

#Plot distance across regions/times for 1st order connectivity and base model
fig, (ax1, ax2) = plt.subplots(1, 2)
for conn_config in [('base', ax1), ('firstord', ax2)]:
    if bootstrap is None:
        if perf_eval_dim == 'regions':
            conn_config[1].scatter(np.array([i for i in range(total_dims.size) if total_dims[i] in dims[conn_config[0]]]),
                                   perfs[conn_config[0]])
        elif perf_eval_dim == 'times':
            conn_config[1].scatter(np.array([total_dim for total_dim in total_dims if total_dim in dims[conn_config[0]]]),
                                   perfs[conn_config[0]])
    else:
        # plot the shaded range of the confidence intervals
        conn_config[1].errorbar(range(plot_params[conn_config[0]]['mean'].size), plot_params[conn_config[0]]['mean'],
                                yerr=[plot_params[conn_config[0]]['err'][0],
                                      plot_params[conn_config[0]]['err'][1]], fmt='o')
        """
        conn_config[1].fill_between(range(plot_params[conn_config[0]]['mean'].size),
                                    plot_params[conn_config[0]]['err'][1],
                                    plot_params[conn_config[0]]['err'][0], alpha=.5)
        conn_config[1].scatter(plot_params[conn_config[0]]['mean'])
        """
    conn_config[1].set_xlabel(perf_eval_dim)
    conn_config[1].set_ylabel(perf_metric + label)
    conn_config[1].set_title(perf_metric + label + ' across ' + perf_eval_dim + ' for ' + conn_config[0] + ' model')
plt.show()

plt.figure()
for conn_type in conn_types:
    if bootstrap is None:
        if perf_eval_dim == 'regions':
            plt.scatter(np.array([i for i in range(total_dims.size) if total_dims[i] in dims[conn_type]]),
                        perfs[conn_type], label=conn_type + ' model')
        elif perf_eval_dim == 'times':
            plt.scatter(np.array([total_dim for total_dim in total_dims if total_dim in dims[conn_type]]),
                        perfs[conn_type], label=conn_type + ' model')
    else:
        plt.errorbar(range(plot_params[conn_type]['mean'].size), plot_params[conn_type]['mean'],
                     yerr=[plot_params[conn_type]['err'][0], plot_params[conn_type]['err'][1]],
                     label=conn_type + ' model', fmt='o')
        """
        plt.fill_between(range(plot_params[conn_type]['mean'].size), plot_params[conn_type]['err'][1],
                         plot_params[conn_type]['err'][0], alpha=.5)
        plt.scatter(plot_params[conn_type]['mean'], label=conn_type + ' model')
        """
plt.xlabel(perf_eval_dim)
plt.ylabel(perf_metric + label)
plt.title(perf_metric + label + ' across ' + perf_eval_dim)
plt.legend()
plt.show()
