import numpy as np
import math
import h5py
from scipy.linalg import expm
import mdl_fxns
import matplotlib.pyplot as plt

__author__ = 'Grace Ng'

#####
# Parameters
#####
connectome_file = 'W.hdf5'
create_matrix_func = mdl_fxns.get_norm_W
process_path_data_file = 'process_path_data.hdf5'
perf_metric = 'corr'
perf_eval_dim = 'times'
group = 'NTG'
max_walk_len = 12
seed_region = 'R CPu'
log_shift = 'no shift'
c_range_type = 'lin' #'lin' or 'log'
# the range of c values that will be tested to find the best-fitting model. Note that if the range type is 'log',
# then these values will be used for  10^x
c_range = (0.01, 10.)
num_c = 100 # the number of c values to test


def single_walk_predict(Xo, A, seed_idx, c, time, walk_len):
    int_mask = np.zeros(A.shape)
    int_mask[:, seed_idx] = 1
    mask = ~np.ma.make_mask(int_mask)
    new_A = np.linalg.matrix_power(A, walk_len)
    np.putmask(new_A, mask, 0.)
    L_out = mdl_fxns.get_L_out(new_A, normalize=False)
    return np.matmul(expm(L_out) * c * time, Xo)


def single_walk_predict_alt(Xo, A, seed_idx, c, time, walk_len):
    int_mask = np.zeros(A.shape)
    int_mask[:, seed_idx] = 1
    mask = ~np.ma.make_mask(int_mask)
    new_A = np.linalg.matrix_power(A, walk_len)
    np.putmask(new_A, mask, 0.)
    return np.matmul(expm(new_A) * c * time, Xo)
    # new_A = np.linalg.matrix_power(A, walk_len)
    # return np.multiply(new_A[:, seed_idx].reshape(Xo.size) * time**walk_len * c, Xo)


def walk_len_predict(Xo, A, time, walk_len):
    return np.matmul((np.linalg.matrix_power(A, walk_len)) * time ** walk_len, Xo)


def walk_len_fit(Xo, A, times, data, c_range_type, c_range, num_c, perf_metric, perf_eval_dim, seed_idx,
                 log_shift="no shift", single_walk_predict_func=single_walk_predict):
    """

    :param Xo:
    :param A:
    :param time:
    :param data (ndarray):
    :param perf_metric:
    :param c_range_type (str)
    :param c_range (tuple of int):
    :param num_c (int):
    :param qual_idxs:
    :param save: tuple, with save[0] being the string of the filename and save[1] being a string with additional
                            information to be stored in the file's attributes, e.g. 'NTG' (the group name)
    :return:
    """
    if c_range_type == 'lin':
        all_c = np.linspace(c_range[0], c_range[1], num=num_c)  # array with all the values of c being tested
    elif c_range_type == 'log':
        all_c = np.logspace(c_range[0], c_range[1], num=num_c)
    raw_predicts, single_walk_predicts, cum_walk_predicts = [], [], []
    for walk_len in range(1, max_walk_len+1):
        single_walk_len_predict = []
        for time in times:
            single_walk_len_predict.append(np.array([single_walk_predict_func(Xo, A, seed_idx, c, time, walk_len)
                                                     for c in all_c]))
        single_walk_predicts.append(single_walk_len_predict)

        raw_predict = np.array([walk_len_predict(Xo, A, time, walk_len) for time in times])
        this_walk_predicts = np.tile(raw_predict, (all_c.size, 1, 1)) * all_c.reshape(all_c.size, 1, 1)
        if walk_len == 1:
            cum_walk_predict = this_walk_predicts
        else:
            cum_walk_predict = cum_walk_predict[-1] + math.factorial(walk_len) * this_walk_predicts
        cum_walk_predicts.append(cum_walk_predict)
    single_walk_predicts = np.array(single_walk_predicts)
    single_walk_predicts = single_walk_predicts.reshape(single_walk_predicts.shape[0], single_walk_predicts.shape[2],
                                                        single_walk_predicts.shape[1], single_walk_predicts.shape[3])
    cum_walk_predicts = np.array(cum_walk_predicts)

    # func used to find the indices of regions where pathology data is not 0 or nan so that we can take the log afterward
    if log_shift == "no shift":
        func = lambda x: x > 0. and not np.isnan(x)
        qual_mat = np.vectorize(func)(
            data)  # matrix has True if element satisfies the above conditions, and False otherwise
    else:
        qual_mat = np.ones((data.shape[0], data.shape[1]), dtype=bool)

    # evaluate model performance according to the performance metric
    single_walk_perfs, cum_walk_perfs = [], []
    if perf_eval_dim == 'times':  # take the mean performance across time points, giving mean performance for each
        for t in range(data.shape[0]):
            qual_idxs = np.where(qual_mat[t])[0]
            if qual_idxs.size > 0:
                single_walk_perf, cum_walk_perf = [], []
                for w in range(max_walk_len):
                    single_walk_perf.append(np.apply_along_axis(mdl_fxns.get_perf, 0,
                                                                single_walk_predicts[w,:,t,qual_idxs], data[t,qual_idxs],
                                                                perf_metric, log_shift))
                    cum_walk_perf.append(np.apply_along_axis(mdl_fxns.get_perf, 0, cum_walk_predicts[w,:,t,qual_idxs],
                                                             data[t,qual_idxs], perf_metric, log_shift))
                single_walk_perfs.append(single_walk_perf)
                cum_walk_perfs.append(cum_walk_perf)
    else:
        raise Exception('perf_mean_type must be "times".')
    single_walk_perfs = np.array(single_walk_perfs)
    cum_walk_perfs = np.array(cum_walk_perfs)

    best_single_walk_perfs, best_cum_walk_perfs = [], []
    best_single_walk_c_idx, best_cum_walk_c_idx = [], []
    for w in range(max_walk_len):
        single_qual_idxs = np.where(np.sum(np.isfinite(single_walk_perfs[:,w,:]), axis=0)
                                    > 0)[
            0]  # single_walk_perfs.shape[0] / 2. #only consider c values for which at least half
                                                                #of the performance evaluations are valid (i.e. not nan)
        if single_qual_idxs.size == 0:
            best_single_walk_perfs.append(np.nan)
            best_single_walk_c_idx.append(np.nan)
        else:
            gen_single_walk_perf = np.nanmean(single_walk_perfs[:,w,single_qual_idxs], axis=0)  # vector with the mean
                                                                                # performance score for each of the c values
            best_single_walk_perfs.append(np.nanmax(gen_single_walk_perf))
            best_single_walk_c_idx.append(single_qual_idxs[np.nanargmax(gen_single_walk_perf)])
        cum_qual_idxs = np.where(np.sum(np.isfinite(cum_walk_perfs[:, w, :]), axis=0)
                                    > cum_walk_perfs.shape[0] / 2.)[0]
        if cum_qual_idxs.size == 0:
            best_cum_walk_perfs.append(np.nan)
            best_cum_walk_c_idx.append(np.nan)
        else:
            gen_cum_walk_perf = np.nanmean(cum_walk_perfs[:, w, cum_qual_idxs], axis=0)
            # if performance metric is correlation, find the c value that gives the highest mean performance score
            best_cum_walk_perfs.append(np.nanmax(gen_cum_walk_perf))
            best_cum_walk_c_idx.append(cum_qual_idxs[np.nanargmax(gen_cum_walk_perf)])
    return single_walk_perfs, best_single_walk_perfs, best_single_walk_c_idx, cum_walk_perfs, best_cum_walk_perfs, \
           best_cum_walk_c_idx

#####
# Load Data from Files
#####
process_path_data = {}
with h5py.File(process_path_data_file, 'r') as f:
    if group in f.keys():
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

# compute the mean of the pathology data across mice for each group and time point
times = []
mean_data = []
for time in process_path_data[group]:
    times.append(time)
    mean_data.append(np.nanmean(process_path_data[group][time], axis=0))
times = np.array(times)
mean_data = np.array(mean_data)

Xo = mdl_fxns.make_Xo(seed_region, regions)
A = create_matrix_func(W)  # generate adjacency matrix
seed_idx = np.where(regions == seed_region)[0]

single_walk_perfs, best_single_walk_perfs, best_single_walk_c_idx, cum_walk_perfs, best_cum_walk_perfs, \
best_cum_walk_c_idx = walk_len_fit(Xo, W, times, mean_data, c_range_type, c_range, num_c, perf_metric,
                                   perf_eval_dim, seed_idx, log_shift, single_walk_predict_func=single_walk_predict)

plt.figure()
plt.plot(list(range(1, max_walk_len+1)), best_single_walk_perfs, marker='.', label='average')
for t in range(times.size):
    if np.any(np.isnan(best_single_walk_c_idx)):
        raise Exception('Performance scores for single walks were invalid.')
    plt.plot(list(range(1, max_walk_len+1)),
             [single_walk_perfs[t,w,best_single_walk_c_idx[w]] for w in range(max_walk_len)], marker='.',
             label='time ' + str(times[t]))
plt.xticks(list(range(1, max_walk_len+1)))
plt.xlabel('Walk Length (Individual)')
plt.ylabel(perf_metric)
plt.legend()
plt.title('Analysis of Individual Walks')
plt.show()

plt.figure()
plt.plot(list(range(1, max_walk_len+1)), best_cum_walk_perfs, marker='.', label='average')
for t in range(times.size):
    if np.any(np.isnan(best_cum_walk_c_idx)):
        raise Exception('Performance scores for cumulative walks were invalid.')
    qual_idxs = np.where(np.isfinite(best_cum_walk_c_idx))
    cum_walk_perfs_to_plot = [cum_walk_perfs[t,w,best_cum_walk_c_idx[w]] for w in list(range(max_walk_len))[qual_idxs]]
    plt.plot(list(range(1, max_walk_len+1))[qual_idxs], cum_walk_perfs_to_plot,
             marker='.', label='time ' + str(times[t]))
plt.xticks(list(range(1, max_walk_len+1)))
plt.xlabel('Walk Length (Cumulative)')
plt.ylabel(perf_metric)
plt.legend()
plt.title('Analysis of Cumulative Walks')
plt.show()