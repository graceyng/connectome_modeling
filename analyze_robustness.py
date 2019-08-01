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
connectome_file = 'W.hdf5'
process_path_data_file = 'process_path_data.hdf5'
output_file = None
bootstrap = 1000 # number of bootstrapped samples to generate
n_threshold = 50 # number of regions for which perforance metric data must exist for both the base model and the
                    # modified model
log_shift = "no shift" #'no shift' or 'shift' or 'no log'
group = 'NTG' # group to consider
seed_region = 'R CPu'
c_params = {'c range type': 'lin', 'c range': (0.001, 10.), 'num c': 100}
do_linregress = False
verbose = True
plot = True


####
# Constants (do not change)
####
perf_metric = 'corr'
perf_eval_dim = 'times' # the dimension along which performance is evaluated

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
with h5py.File(connectome_file, 'r') as f:
    W = f.get('W')[()]
    ipsi_regions = f['W'].attrs['Ipsi_Regions']
    contra_regions = f['W'].attrs['Contra_Regions']

# create a vector of the names of all the regions of interest -- first, with the right-sided regions, and then the
# left-sided regions
regions = np.concatenate((np.add('R ', ipsi_regions), np.add('L ', contra_regions)))
seed_idx = np.where(regions == seed_region)[0]

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
L_outs = [mdl_fxns.get_L_out(W)]
conn_idxs = np.where(W[:,seed_idx] != 0.)[0]
for conn_idx in conn_idxs:
    this_W = np.copy(W)
    this_W[conn_idx,seed_idx] = 0.
    L_outs.append(mdl_fxns.get_L_out(this_W))

times = np.array(list(process_path_data[group].keys()))
total_dims = times

all_c_values, all_predicts, all_perfs, all_dims, all_plot_params = [], [], [], [], []
for n, L_out in enumerate(L_outs):
    c_values, predicts, perfs = [], [], []
    for i in range(bootstrap):
        all_c, all_perf, _, this_dims, best_c, _, best_predict, _, linregress_params = \
            mdl_fxns.fit(Xo, L_out, times, regions, np.array(data)[:,i,:],
                         c_params['c range type'], c_params['c range'],
                         c_params['num c'], perf_metric, perf_eval_dim, log_shift, do_linregress,
                         plot=False)
        c_idx = np.where(all_c == best_c)[0][0]
        c_values.append(best_c)
        predicts.append(best_predict)
        this_perf = np.full(total_dims.size, np.nan)
        this_perf[[i for i in range(total_dims.size) if total_dims[i] in this_dims]] = all_perf[:, c_idx]
        perfs.append(this_perf)
        if i%100 == 0 and verbose:
            print('Computed performance for matrix ' + str(n) + ', trial ' + str(i))
    perfs = np.array(perfs)
    qual_count_dict = collections.Counter(np.where(np.isfinite(perfs))[1])
    qual_dim_idxs = np.array([idx for idx in range(total_dims.size) if qual_count_dict[idx] > 2])
    mean = np.nanmean(perfs[:,qual_dim_idxs], axis=0)
    itvl = t.interval(0.95, mean.size - 1, loc=mean, scale=sem(perfs[:,qual_dim_idxs], nan_policy='omit',
                                                               axis=0))
    full_mean, neg_err, pos_err = np.full(total_dims.size, np.nan), np.full(total_dims.size, np.nan), \
                                  np.full(total_dims.size, np.nan)
    full_mean[qual_dim_idxs] = mean
    neg_err[qual_dim_idxs] = mean - itvl[0]
    pos_err[qual_dim_idxs] = itvl[1] - mean

    all_perfs.append(perfs)
    all_predicts.append(predicts)
    all_c_values.append(c_values)
    all_dims.append(np.array(this_dims))
    all_plot_params.append({'mean': full_mean, 'err': (neg_err, pos_err)})
all_perfs = np.array(all_perfs)

#### Save results
if output_file is not None:
    with h5py.File(output_file, 'w') as f:
        c_val_data = f.create_dataset('c values', data=all_c_values)
        pred_data = f.create_dataset('predictions', data=all_predicts)
        perf_data = f.create_dataset('performance scores', data=all_perfs)
        dim_data = f.create_dataset('dimensions', data=all_dims)
        plot_params_group = f.create_group('plotting parameters')
        for t in range(total_dims.size):
            plot_params_subgrp = plot_params_group.create_group('time ' + str(total_dims[t]))
            plot_params_subgrp.create_dataset('mean', data=[all_plot_params[i]['mean'][t] for i in range(1, len(L_outs))])
            plot_params_subgrp.create_dataset('neg err',
                                              data=[all_plot_params[i]['err'][0][t] for i in range(1, len(L_outs))])
            plot_params_subgrp.create_dataset('pos err',
                                              data=[all_plot_params[i]['err'][1][t] for i in range(1, len(L_outs))])

all_t_stat, all_p_val, all_sig_idxs = [], [], []
for i in range(1, len(L_outs)):
    #Run a student's t-test, where...
    #   for correlation, Ho: firstord >= base and Ha: firstord < base
    #   for distance, Ho: base >= firstord and Ha: base < firstord
    t_stat, p_val, num_samples = [], [], []
    for idx in range(total_dims.size):
        qual_idxs = np.intersect1d(np.where(np.isfinite(all_perfs[0][:,idx]))[0],
                                   np.where(np.isfinite(all_perfs[i][:,idx]))[0])
        if qual_idxs.size > n_threshold:
            this_t_stat, this_p_val = ttest_ind(all_perfs[0][qual_idxs,idx], all_perfs[i][qual_idxs,idx],
                                                equal_var=False)
            t_stat.append(this_t_stat)
            p_val.append(this_p_val / 2.)
            num_samples.append(qual_idxs.size)
        else:
            t_stat.append(np.nan)
            p_val.append(np.nan)
            num_samples.append(0)
    all_t_stat.append(t_stat)
    all_p_val.append(p_val)
    sig_idxs = np.where(np.isfinite(t_stat))[0]
    all_sig_idxs.append(np.array([sig_idx for sig_idx in sig_idxs if t_stat[sig_idx] > 0. and p_val[sig_idx] < 0.05]))
all_t_stat = np.array(all_t_stat)
all_p_val = np.array(all_p_val)
all_sig_idxs = np.array(all_sig_idxs)

if output_file is not None:
    with h5py.File(output_file, 'a') as f:
        f.create_dataset('t stats', data=all_t_stat)
        f.create_dataset('p vals', data=all_p_val)
        f.create_group('sig dim idxs')
        for i, sig_idxs in enumerate(all_sig_idxs):
            f['sig dim idxs'].create_dataset(str(i), data=sig_idxs)
        if type(total_dims[0]) == str:
            f.attrs['total dims'] = [a.encode('utf8') for a in list(total_dims)] # necessary because h5py cannot handle
                                                            # unicode; will need to decode this back to unicode later
        else:
            f.attrs['total dims'] = total_dims

for t in range(total_dims.size):
    plt.figure()
    num_points = len(L_outs)-1
    plt.errorbar(list(range(1, len(L_outs))), [all_plot_params[0]['mean'][t]]*num_points,
                 yerr=([all_plot_params[0]['err'][0][t]]*num_points, [all_plot_params[0]['err'][1][t]]*num_points),
                 label='base model', fmt='o')
    mod_models_mean = [all_plot_params[i]['mean'][t] for i in range(1, len(L_outs))]
    mod_models_neg_err = [all_plot_params[i]['err'][0][t] for i in range(1, len(L_outs))]
    mod_models_pos_err = [all_plot_params[i]['err'][1][t] for i in range(1, len(L_outs))]
    plt.errorbar(list(range(1, len(L_outs))), mod_models_mean, yerr=(mod_models_neg_err, mod_models_pos_err),
                 label='modified models', fmt='o')
    plt.xticks(list(range(1, len(L_outs))), regions[conn_idxs], rotation=90)
    plt.xlabel('Regions with direct connectivity to seed region')
    plt.ylabel(perf_metric)
    plt.title(perf_metric + ' for base model vs. modified models, t=' + str(total_dims[t]))
    plt.legend()
    plt.show()

