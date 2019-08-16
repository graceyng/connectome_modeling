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
bootstrap = 5  # number of bootstrapped samples to generate
n_threshold = 50  # number of regions for which performance metric data must exist for both the base model and the
                    # modified model
log_shift = "no shift" #'no shift' or 'shift' or 'no log'
group = 'NTG' # group to consider
seed_region = 'R CPu'
c_params = {'c range type': 'lin', 'c range': (0.001, 10.), 'num c': 100}
alpha = 0.05
do_linregress = False
verbose = True
plot = True

####
# Constants (do not change)
####
perf_metric = 'corr'
perf_eval_dim = 'times' # the dimension along which performance is evaluated


def run_models(L_outs, Xo, times, regions, data, total_dims):
    all_c_values, all_predicts, all_perfs, all_dims = [], [], [], []
    for n, L_out in enumerate(L_outs):
        c_values, predicts, perfs = [], [], []
        for i in range(bootstrap):
            all_c, all_perf, _, this_dims, best_c, _, best_predict, _, linregress_params = \
                mdl_fxns.fit(Xo, L_out, times, regions, np.array(data)[:, i, :],
                             c_params['c range type'], c_params['c range'],
                             c_params['num c'], perf_metric, perf_eval_dim, log_shift, do_linregress,
                             plot=False)
            c_idx = np.where(all_c == best_c)[0][0]
            c_values.append(best_c)
            predicts.append(best_predict)
            this_perf = np.full(total_dims.size, np.nan)
            this_perf[[i for i in range(total_dims.size) if total_dims[i] in this_dims]] = all_perf[:, c_idx]
            perfs.append(this_perf)
            if i % 100 == 0 and verbose:
                print('Computed performance for matrix ' + str(n) + ', trial ' + str(i))
        perfs = np.array(perfs)
        all_perfs.append(perfs)
        all_predicts.append(predicts)
        all_c_values.append(c_values)
        all_dims.append(np.array(this_dims))
    all_perfs = np.array(all_perfs)

    #### Save results
    if output_file is not None:
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('c values', data=all_c_values)
            f.create_dataset('predictions', data=all_predicts)
            f.create_dataset('performance scores', data=all_perfs)
            f.create_dataset('dimensions', data=all_dims)
    return all_c_values, all_predicts, all_perfs, all_dims


def stats_analysis(all_perfs, L_outs, total_dims, output_file):
    all_plot_params, all_sig_idxs = [], []
    for n in range(len(L_outs)):
        qual_count_dict = collections.Counter(np.where(np.isfinite(all_perfs[n]))[1])
        # There should be more than 2 valid bootstrapped samples per time point in order to analyze that time point
        qual_dim_idxs = np.array([idx for idx in range(total_dims.size) if qual_count_dict[idx] > 2])
        mean = np.nanmean(all_perfs[n][:, qual_dim_idxs], axis=0)
        pctls = np.percentile(all_perfs[n][:, qual_dim_idxs],
                              [50 - (1. - alpha) / 2. * 100., 50 + (1. - alpha) / 2. * 100.],
                              interpolation='nearest', axis=0)
        full_mean, low_bound, upp_bound = np.full(total_dims.size, np.nan), np.full(total_dims.size, np.nan), \
                                          np.full(total_dims.size, np.nan)
        full_mean[qual_dim_idxs] = mean
        low_bound[qual_dim_idxs] = pctls[0]
        upp_bound[qual_dim_idxs] = pctls[1]
        plot_params = {'mean': full_mean, 'bounds': (low_bound, upp_bound)}
        if n > 0:
            diffs = all_perfs[n][:, qual_dim_idxs] - all_perfs[0][:, qual_dim_idxs]
            greater_tail = np.divide(np.sum(diffs > 0., axis=0), np.sum(np.logical_and(
                np.isfinite(all_perfs[n][:, qual_dim_idxs]), np.isfinite(all_perfs[0][:, qual_dim_idxs])), axis=0))
            lesser_tail = np.divide(np.sum(diffs <= 0., axis=0), np.sum(np.logical_and(
                np.isfinite(all_perfs[n][:, qual_dim_idxs]), np.isfinite(all_perfs[0][:, qual_dim_idxs])), axis=0))
            p_vals = 2 * np.minimum(greater_tail, lesser_tail)
            all_sig_idxs.append([qual_dim_idxs[i] for i in np.where(p_vals < alpha)[0]])
            mean = np.nanmean(diffs, axis=0)
            pctls = np.percentile(diffs, [50 - (1. - alpha) / 2. * 100., 50 + (1. - alpha) / 2. * 100.],
                                  interpolation='nearest', axis=0)
            diff_mean, diff_low_bound, diff_upp_bound, full_p_vals = np.full(total_dims.size, np.nan), \
                                                                     np.full(total_dims.size, np.nan), \
                                                                     np.full(total_dims.size, np.nan), \
                                                                     np.full(total_dims.size, np.nan)
            diff_mean[qual_dim_idxs] = mean
            full_p_vals[qual_dim_idxs] = p_vals
            diff_low_bound[qual_dim_idxs] = pctls[0]
            diff_upp_bound[qual_dim_idxs] = pctls[1]
            plot_params.update({'diff mean': diff_mean, 'diff bounds': (diff_low_bound, diff_upp_bound),
                                'p vals': full_p_vals})
        all_plot_params.append(plot_params)

    if output_file is not None:
        with h5py.File(output_file, 'a') as f:
            plot_params_group = f.create_group('plotting parameters')
            for t in range(total_dims.size):
                plot_params_subgrp = plot_params_group.create_group('time ' + str(total_dims[t]))
                plot_params_subgrp.create_dataset('mean',
                                                  data=[all_plot_params[i]['mean'][t] for i in range(len(L_outs))])
                plot_params_subgrp.create_dataset('low bounds',
                                                  data=[all_plot_params[i]['bounds'][0][t] for i in range(len(L_outs))])
                plot_params_subgrp.create_dataset('upp bounds',
                                                  data=[all_plot_params[i]['bounds'][1][t] for i in range(len(L_outs))])
                plot_params_subgrp.create_dataset('diff mean', data=[all_plot_params[i]['diff mean'][t] for
                                                                     i in range(1, len(L_outs))])
                plot_params_subgrp.create_dataset('diff low bounds', data=[all_plot_params[i]['diff bounds'][0][t] for
                                                                           i in range(1, len(L_outs))])
                plot_params_subgrp.create_dataset('diff upp bounds', data=[all_plot_params[i]['diff bounds'][1][t] for
                                                                           i in range(1, len(L_outs))])
                plot_params_subgrp.create_dataset('p vals', data=[all_plot_params[i]['p vals'][t]
                                                                  for i in range(1, len(L_outs))])
            f.create_group('sig dim idxs')
            for i, sig_idxs in enumerate(all_sig_idxs):
                f['sig dim idxs'].create_dataset(str(i + 1), data=sig_idxs)
    return all_plot_params, all_sig_idxs


def load_from_file(save_filename, total_dims):
    with h5py.File(save_filename, 'r') as f:
        all_c_values = f['c values'][()]
        all_predicts = f['predictions'][()]
        all_perfs = f['performance scores'][()]
        all_dims = f['dimensions'][()]
        if 'plotting parameters' in f.keys():
            all_plot_params = [{'mean': [], 'bounds': [[], []], 'diff mean': [], 'diff bounds': [[], []], 'p vals': []}
                               for i in range(all_predicts.shape[0])]
            del all_plot_params[0]['diff mean']
            del all_plot_params[0]['diff bounds']
            del all_plot_params[0]['p vals']
            for t in range(total_dims.size):
                str_time = 'time ' + str(total_dims[t])
                if str_time in f['plotting parameters'].keys():
                    all_means = f['plotting parameters'][str_time]['mean'][()]
                    for i in range(all_means.size):
                        all_plot_params[i]['mean'].append(all_means[i])
                        all_plot_params[i]['bounds'][0].append(f['plotting parameters'][str_time]['low bounds'][i])
                        all_plot_params[i]['bounds'][1].append(f['plotting parameters'][str_time]['upp bounds'][i])
                        if i > 0:
                            all_plot_params[i]['diff mean'].append(
                                f['plotting parameters'][str_time]['diff mean'][i - 1][()])
                            all_plot_params[i]['diff bounds'][0].append(
                                f['plotting parameters'][str_time]['diff low bounds'][i - 1][()])
                            all_plot_params[i]['diff bounds'][1].append(
                                f['plotting parameters'][str_time]['diff upp bounds'][i - 1][()])
                            all_plot_params[i]['p vals'].append(f['plotting parameters'][str_time]['p vals'][i - 1][()])
                else:
                    raise Exception('Time ' + str(total_dims[t]) + ' is not saved in this file.')
            if 'sig dim idxs' in f.keys():
                all_sig_dim_idxs = []
                for idx in f['sig dim idxs'].keys():
                    all_sig_dim_idxs.append(f['sig dim idxs'][idx][()])
        else:
            all_plot_params = None
            all_sig_dim_idxs = None
    return all_c_values, all_predicts, all_perfs, all_dims, all_plot_params, all_sig_dim_idxs


def plot_robustness_diff(total_dims, all_plot_params, L_outs, conn_idxs):
    for t in range(total_dims.size):
        plt.figure()
        mod_models_mean = np.array([all_plot_params[i]['diff mean'][t] for i in range(1, len(L_outs))])
        mod_models_neg_err = np.array([all_plot_params[i]['diff mean'][t] - all_plot_params[i]['diff bounds'][0][t]
                                       for i in range(1, len(L_outs))])
        mod_models_pos_err = np.array([all_plot_params[i]['diff bounds'][1][t] - all_plot_params[i]['diff mean'][t]
                                       for i in range(1, len(L_outs))])
        plt.errorbar(list(range(1, len(L_outs))), mod_models_mean, yerr=(mod_models_neg_err, mod_models_pos_err),
                     fmt='o')
        p_vals = np.array([all_plot_params[i]['p vals'][t] for i in range(1, len(L_outs))])
        qual_idxs = np.where(np.isfinite(p_vals))[0]
        asterisks = np.array(['' for i in range(p_vals.size)])
        asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.05)[0]]] = '*'
        # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.01)[0]]] = '**'
        # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.001)[0]]] = '***'
        # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.0001)[0]]] = '****'
        y_max = np.max(mod_models_mean + mod_models_pos_err)
        y_min = np.min(mod_models_mean - mod_models_neg_err)
        for i, asterisk in enumerate(asterisks):
            if asterisk:
                plt.text(1 + i, y_max + abs(y_max - y_min) * 0.01, asterisk, horizontalalignment='center',
                         verticalalignment='center')
        plt.xticks(list(range(1, len(L_outs))), regions[conn_idxs], rotation=90)
        plt.xlabel('Regions with direct connectivity to seed region')
        plt.ylabel('Difference between base and modified models (' + perf_metric + ')')
        plt.title(perf_metric + ' for base model vs. modified models, t=' + str(total_dims[t]))
        plt.show()


def plot_robustness(total_dims, all_plot_params, L_outs, conn_idxs):
    for t in range(total_dims.size):
        plt.figure()
        mod_models_mean = np.array([all_plot_params[i]['mean'][t] for i in range(1, len(L_outs))])
        mod_models_neg_err = np.array([all_plot_params[i]['mean'][t] - all_plot_params[i]['bounds'][0][t]
                                       for i in range(1, len(L_outs))])
        mod_models_pos_err = np.array([all_plot_params[i]['bounds'][1][t] - all_plot_params[i]['mean'][t]
                                       for i in range(1, len(L_outs))])
        plt.errorbar(list(range(1, len(L_outs))), mod_models_mean, yerr=(mod_models_neg_err, mod_models_pos_err),
                     fmt='o', label='modified models', alpha=0.5)
        base_mean = np.array([all_plot_params[0]['mean'][t] for i in range(1, len(L_outs))])
        base_neg_err = np.array([all_plot_params[0]['mean'][t] - all_plot_params[0]['bounds'][0][t]
                                 for i in range(1, len(L_outs))])
        base_pos_err = np.array([all_plot_params[0]['bounds'][1][t] - all_plot_params[0]['mean'][t]
                                 for i in range(1, len(L_outs))])
        plt.errorbar(list(range(1, len(L_outs))), base_mean, yerr=(base_neg_err, base_pos_err), fmt='o',
                     label='base model', alpha=0.5)
        p_vals = np.array([all_plot_params[i]['p vals'][t] for i in range(1, len(L_outs))])
        qual_idxs = np.where(np.isfinite(p_vals))[0]
        asterisks = np.array(['' for i in range(p_vals.size)])
        asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.05)[0]]] = '*'
        # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.01)[0]]] = '**'
        # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.001)[0]]] = '***'
        # asterisks[qual_idxs[np.where(p_vals[qual_idxs] < 0.0001)[0]]] = '****'
        y_max = max(np.max(mod_models_mean + mod_models_pos_err), np.max(base_mean + base_pos_err))
        y_min = max(np.min(mod_models_mean - mod_models_neg_err), np.min(base_mean - base_neg_err))
        for i, asterisk in enumerate(asterisks):
            if asterisk:
                plt.text(1 + i, y_max + abs(y_max - y_min) * 0.01, asterisk, horizontalalignment='center',
                         verticalalignment='center')
        plt.xticks(list(range(1, len(L_outs))), regions[conn_idxs], rotation=90)
        plt.xlabel('Regions with direct connectivity to seed region')
        plt.ylabel(perf_metric)
        plt.title(perf_metric + ' for base model vs. modified models, t=' + str(total_dims[t]))
        plt.show()


def plot_perf_vs_path(all_plot_params, conn_idxs, orig_data):
    func = lambda x: x > 0. and not np.isnan(x)
    colors = ['b', 'g', 'm']
    plt.figure()
    plt.title('Robustness Testing: Performance Change vs. Pathology Amount')
    path_to_plot, diff_to_plot = [[] for t in range(total_dims.size)], [[] for t in range(total_dims.size)]
    for i, conn_idx in enumerate(conn_idxs):
        mean_perf_diffs = all_plot_params[i + 1]['diff mean']
        for t, diff in enumerate(mean_perf_diffs):
            if not np.isnan(diff):
                path = orig_data[t][conn_idx]
                if func(path):
                    path_to_plot[t].append(np.log10(path))
                    diff_to_plot[t].append(diff)
    for t in range(total_dims.size):
        plt.scatter(path_to_plot[t], diff_to_plot[t], label='Time ' + str(total_dims[t]), color=colors[t])
    plt.xlabel('Pathology (log)')
    plt.ylabel('Performance Change (' + perf_metric + ') compared to base model')
    plt.legend()
    plt.show()


def calc_node_in_degree(W, region_idx):
    return np.sum(np.where(W[:, region_idx] > 0.)[0])


def calc_weighted_node_in_degree(W, region_idx):
    return np.sum(W[:, region_idx])


def plot_perf_vs_network_metric(all_plot_params, conn_idxs, W, network_metric_func=calc_node_in_degree,
                                metric_name='Node In-Degree'):
    colors = ['b', 'g', 'm']
    plt.figure()
    plt.title('Robustness Testing: Performance Change vs. ' + metric_name)
    metric_to_plot, diff_to_plot = [[] for t in range(total_dims.size)], [[] for t in range(total_dims.size)]
    for i, conn_idx in enumerate(conn_idxs):
        node_metric = network_metric_func(W, conn_idx)
        mean_perf_diffs = all_plot_params[i + 1]['diff mean']
        for t, diff in enumerate(mean_perf_diffs):
            if not np.isnan(diff):
                metric_to_plot[t].append(node_metric)
                diff_to_plot[t].append(diff)
    for t in range(total_dims.size):
        plt.scatter(metric_to_plot[t], diff_to_plot[t], label='Time ' + str(total_dims[t]), color=colors[t])
    plt.xlabel(metric_name)
    plt.ylabel('Performance Change (' + perf_metric + ') compared to base model')
    plt.legend()
    plt.show()


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
data, orig_data = [], []
for time in process_path_data[group]:
    if bootstrap is None:
        data.append(np.nanmean(process_path_data[group][time], axis=0))
    else:
        # Need to bootstrap if computing correlation
        n_mice = process_path_data[group][time].shape[0]
        combos = np.random.randint(0, n_mice, (bootstrap, n_mice))
        data.append(np.nanmean(process_path_data[group][time][combos, :], axis=1))
    orig_data.append(np.nanmean(process_path_data[group][time], axis=0))

#####
# Fit a Model Using Pathology Data with Seeding from the iCPu Region
#####
Xo = mdl_fxns.make_Xo(seed_region, regions)
L_outs = [mdl_fxns.get_L_out(W)]
conn_idxs = np.where(W[:, seed_idx] != 0.)[0]
for conn_idx in conn_idxs:
    this_W = np.copy(W)
    this_W[conn_idx, seed_idx] = 0.
    L_outs.append(mdl_fxns.get_L_out(this_W))

times = np.array(list(process_path_data[group].keys()))
total_dims = times

# all_c_values, all_predicts, all_perfs, all_dims, all_plot_params = run_models(L_outs, Xo, times, regions, data, total_dims)
all_c_values, all_predicts, all_perfs, all_dims, all_plot_params, all_sig_dim_idxs = \
    load_from_file('test_robustness_bootstrap_1000.hdf5', total_dims)
# all_plot_params, all_sig_idxs = stats_analysis(all_perfs, L_outs, total_dims, output_file)
plot_robustness_diff(total_dims, all_plot_params, L_outs, conn_idxs)
plot_robustness(total_dims, all_plot_params, L_outs, conn_idxs)
plot_perf_vs_path(all_plot_params, conn_idxs, orig_data)
plot_perf_vs_network_metric(all_plot_params, conn_idxs, W, calc_node_in_degree, 'Node In-Degree')
plot_perf_vs_network_metric(all_plot_params, conn_idxs, W, calc_weighted_node_in_degree, 'Weighted Node In-Degree')
