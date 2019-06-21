import numpy as np
from scipy.linalg import expm
from scipy.stats import pearsonr
import plot_fxns

__author__ = 'Grace Ng'

def make_Xo(ROI, ROI_names):
    """
    Initialize the Xo vector, which represents the pathology at t=0. All regions should have 0 pathology, except the
    region of interest being seeded with a pathology value of 1 (arbitrary units).
    :param ROI (str)
    :param ROI_names (ndarray of str)
    :return: Xo (ndarray of floats)
    """
    Xo = np.zeros(len(ROI_names))
    Xo[np.where(ROI_names == ROI)[0]] = 1.
    return Xo

def get_L_out(W):
    """
    Compute the out-degree Laplacian matrix.
    :param W (ndarray): the weighted adjacency matrix
    :return: L_out (ndarray): the weighted out-degree Laplacian matrix
    """
    np.fill_diagonal(W, 0.) # zero out the diagonal since regions should not be connected to themselves
    eigvals = np.linalg.eigvals(W)
    real_eigvals = eigvals[np.isreal(eigvals)]
    if len(real_eigvals) == 0:
        raise Exception('The adjacency matrix does not have any real eigenvalues.')
    W = W / max(real_eigvals).real  # must normalize the adjacency matrix so that the largest eigenvalue is 1
    out_deg = np.sum(W, axis=1)
    return np.diag(out_deg) - W

def get_perf(predicted, actual, perf_metric):
    """

    :param predicted:
    :param actual:
    :param perf_metric:
    :return:
    """
    if perf_metric == 'corr':
        return pearsonr(predicted, actual)[0]
    elif perf_metric == 'dist':
        x = 1 #TODO: fill this in
    else:
        raise Exception('perf_metric should either be "corr" or "dist".')

def predict(Xo, L_out, c, time):
    """

    :param Xo:
    :param L_out:
    :param c:
    :param time:
    :return:
    """
    return np.matmul(expm(-L_out*c*time), Xo)

def fit(Xo, L_out, times, data, c_range_type, c_range, num_c, perf_metric, plot=False):
    """

    :param Xo:
    :param L_out:
    :param time:
    :param data (ndarray):
    :param perf_metric:
    :param c_range_type (str)
    :param c_range (tuple of int):
    :param num_c (int):
    :param qual_idxs:
    :return:
    """
    if c_range_type == 'lin':
        all_c = np.linspace(c_range[0], c_range[1], num=num_c) # array with all the values of c being tested
    elif c_range_type == 'log':
        all_c = np.logspace(c_range[0], c_range[1], num=num_c)
    all_perf = [] # stores the performance scores at each time point for each value of c
    all_predicts = [] # stores the predicted pathology values at each time point for each value of c
    all_log_data = []
    all_data_idxs = []
    for i, time in enumerate(times):
        # find the indices of regions where pathology data is not 0 or nan so that we can take the log afterward
        func = lambda x: x != 0. and not np.isnan(x)
        qual_idxs = np.where(np.vectorize(func)(data[i]))[0]
        all_data_idxs.append(qual_idxs)
        log_data = np.log(data[i][qual_idxs])
        all_log_data.append(log_data)

        # propagate the model according to the tested values of c to generate predicted pathology values
        predictions = np.array([predict(Xo, L_out, c, time) for c in all_c])
        all_predicts.append(predictions)

        # evaluate model performance according to the performance metric
        all_perf.append(np.apply_along_axis(get_perf, 1, np.log(predictions[:,qual_idxs]), log_data, perf_metric))
    gen_perf = np.mean(np.array(all_perf), axis=0) # vector with the mean performance score for each of the c values
    #TODO: try computing mean across times instead of regions
    if plot:
        plot_fxns.plot_perf_vs_c(times, all_c, all_perf, perf_metric)
    # if performance metric is correlation, find the c value that gives the highest mean performance score
    if perf_metric == 'corr':
        best_gen_c = all_c[np.argmax(gen_perf)]
        best_perf = np.max(gen_perf)
        best_predict = np.array(all_predicts)[:,np.argmax(gen_perf)]
        best_c_per_time = all_c[np.argmax(all_perf, axis=1)]
    # if performance metric is distance, find the c value that gives the lowest mean performance score
    elif perf_metric == 'dist':
        best_gen_c = all_c[np.argmin(gen_perf)]
        best_perf = np.min(gen_perf)
        best_predict = np.array(all_predicts)[:,np.argmin(gen_perf)]
        best_c_per_time = all_c[np.argmin(all_perf, axis=1)]
    else:
        raise Exception('perf_metric must either be "corr" or "dist".')
    if plot:
        plot_fxns.plot_logpredict_vs_logdata(times, best_predict, all_data_idxs, all_log_data, best_gen_c)
    return best_gen_c, best_perf, best_predict, best_c_per_time, all_perf