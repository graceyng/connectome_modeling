import numpy as np
import h5py
from scipy.linalg import expm
from scipy.stats import pearsonr, linregress
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
    #real_eigvals = eigvals[np.isreal(eigvals)] #TODO: look into ignoring the imaginary components and taking the max of all
    #if len(real_eigvals) == 0:
    #    raise Exception('The adjacency matrix does not have any real eigenvalues.')
    W = W / max(eigvals).real  # must normalize the adjacency matrix so that the largest eigenvalue is 1
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
        # need a minimum of three points to calculate a meaningful Pearson correlation, and the Pearson correlation is
        # undefined if the std dev of either dataset is 0
        if predicted.size <=2:
            return None
        elif np.std(predicted) == 0. or np.std(actual) == 0.:
            return None
        return pearsonr(predicted, actual)[0]
    elif perf_metric == 'dist':
        #slope, intercept, r_value, p_value, std_err = linregress(predicted, actual)
        #adj_predicted = slope*predicted + intercept

        # calculate the Euclidean distance between the predicted and actual data, and then divide by the dimensionality
        # of the data (the dimensionality of the data may vary between calls to this function)
        return np.linalg.norm(predicted - actual) / predicted.size

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

def fit(Xo, L_out, times, regions, data, c_range_type, c_range, num_c, perf_metric, perf_eval_dim, plot=False, save=None):
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
    :param save: tuple, with save[0] being the string of the filename and save[1] being a string with additional
                            information to be stored in the file's attributes, e.g. 'NTG' (the group name)
    :return:
    """
    if c_range_type == 'lin':
        all_c = np.linspace(c_range[0], c_range[1], num=num_c) # array with all the values of c being tested
    elif c_range_type == 'log':
        all_c = np.logspace(c_range[0], c_range[1], num=num_c)
    all_perf = [] # stores the performance scores at each time point for each value of c
    all_predicts = [] # stores the predicted pathology values at each time point for each value of c
    for i, time in enumerate(times):
        # propagate the model according to the tested values of c to generate predicted pathology values
        all_predicts.append(np.array([predict(Xo, L_out, c, time) for c in all_c]))
    all_predicts = np.array(all_predicts)
    # used to find the indices of regions where pathology data is not 0 or nan so that we can take the log afterward
    func = lambda x: x != 0. and not np.isnan(x)
    qual_mat = np.vectorize(func)(data) # matrix with True if element satisfies the above conditions, and False otherwise
    for j in range(data.shape[1]):
        qual_idxs = np.where(qual_mat[:,j])[0]
        if qual_idxs.size > 0:
            np.log(all_predicts[qual_idxs,:,j])
    if perf_metric == 'dist':
        swapped = np.swapaxes(all_predicts, 1, 0)
        all_slopes, all_intercepts = [], []
        for i in range(swapped.shape[0]):
            slope, intercept, r_value, p_value, std_err = linregress(swapped[i][qual_mat], data[qual_mat])
            all_slopes.append(slope)
            all_intercepts.append(intercept)
        all_predicts = np.add(np.dot(swapped, slope), np.array(all_intercepts).reshape(len(all_intercepts),1,1))
        all_predicts = np.swapaxes(all_predicts, 1, 0)

    # evaluate model performance according to the performance metric
    dims = [] # dimensions along which performance is evaluated
    perf_idxs = []
    if perf_eval_dim == 'times':  # take the mean performance across time points, giving mean performance for each
        for i in range(data.shape[0]):
            qual_idxs = np.where(qual_mat[i])[0]
            perf = np.apply_along_axis(get_perf, 0, np.log(all_predicts[i,:,qual_idxs]), np.log(data[i,qual_idxs]),
                                       perf_metric)
            if not None in perf:
                all_perf.append(perf)
                dims.append(times[i])
                perf_idxs.append(i)
    elif perf_eval_dim == 'regions':  # take the mean performance across regions
        for j in range(data.shape[1]):
            qual_idxs = np.where(qual_mat[:,j])[0]
            perf = np.apply_along_axis(get_perf, 0, np.log(all_predicts[qual_idxs,:,j]), np.log(data[qual_idxs,j]),
                                       perf_metric)
            if not None in perf:
                all_perf.append(perf)
                dims.append(regions[j])
                perf_idxs.append(j)
    else:
        raise Exception('perf_mean_type must be either "times" or "regions".')
    gen_perf = np.mean(np.array(all_perf), axis=0) # vector with the mean performance score for each of the c values
    if plot:
        plot_fxns.plot_perf_vs_c(perf_eval_dim, dims, all_c, all_perf, perf_metric)

    # if performance metric is correlation, find the c value that gives the highest mean performance score
    if perf_metric == 'corr':
        best_c_idx = np.argmax(gen_perf)
        best_c_per_ctgry = all_c[np.argmax(all_perf, axis=1)]
    # if performance metric is distance, find the c value that gives the lowest mean performance score
    elif perf_metric == 'dist':
        best_c_idx = np.argmin(gen_perf)
        best_c_per_ctgry = all_c[np.argmin(all_perf, axis=1)]
    else:
        raise Exception('perf_metric must either be "corr" or "dist".')
    best_gen_c = all_c[best_c_idx]
    best_predict = np.array(all_predicts)[:, best_c_idx]
    best_perf = gen_perf[best_c_idx]
    if plot:
        if perf_eval_dim == 'times':
            all_idxs = np.array(range(len(times)))
        elif perf_eval_dim == 'regions':
            all_idxs = np.array(range(len(regions)))
        else:
            raise Exception('perf_eval_dim must either be "times" or "regions".')
        plot_fxns.plot_predict_vs_data(times, regions, all_idxs, best_predict, data, qual_mat, perf_eval_dim, best_gen_c)
        if len(perf_idxs) > 10:
            best_5_idxs = np.argpartition(np.array(all_perf)[:,best_c_idx], -5)[-5:]
            worst_5_idxs = np.argpartition(np.array(all_perf)[:,best_c_idx], 5)[:5]
            plot_fxns.plot_perf_vs_c(perf_eval_dim, np.array(dims)[best_5_idxs], all_c,
                                     np.array(all_perf)[best_5_idxs],
                                     perf_metric, 'Performance in the 5 Best Dimensions')
            plot_fxns.plot_perf_vs_c(perf_eval_dim, np.array(dims)[worst_5_idxs], all_c,
                                     np.array(all_perf)[worst_5_idxs],
                                     perf_metric, 'Performance in the 5 Worst Dimensions')
            plot_fxns.plot_predict_vs_data(times, regions, np.array(perf_idxs)[best_5_idxs], best_predict,
                                           data, qual_mat, perf_eval_dim, best_gen_c, title='5 Best Dimensions')
            plot_fxns.plot_predict_vs_data(times, regions, worst_5_idxs, best_predict, data, qual_mat, perf_eval_dim,
                                           best_gen_c, title='5 Worst Dimensions')

    if save is not None:
        save_mdl_results(save, times, regions, perf_metric, perf_eval_dim, data, qual_mat, all_c, np.array(all_perf),
                         gen_perf, best_gen_c)
    return best_gen_c, best_perf, best_predict, best_c_per_ctgry

def save_mdl_results(save, times, regions, perf_metric, perf_eval_dim, data, qual_mat, all_c, all_perf, gen_perf,
                     best_gen_c):
    with h5py.File(save[0], 'w') as f:
        f.attrs['info'] = save[1]
        f.attrs['times'] = times
        f.attrs['regions'] = regions
        f.attrs['perf_metric'] = perf_metric
        f.attrs['perf_eval_dim'] = perf_eval_dim
        f.attrs['best_gen_c'] = best_gen_c
        f.create_dataset('data', data=data)
        f.create_dataset('all_c', data=all_c)
        f.create_dataset('all_perf', data=all_perf)
        f.create_dataset('qual_mat', data=qual_mat)
        f.create_dataset('gen_perf', data=gen_perf)

def load_mdl_results(save_file):
    with h5py.File(save_file, 'r') as f:
        info = f.attrs['info']
        times = f.attrs['times']
        regions = f.attrs['regions']
        perf_metric = f.attrs['perf_metric']
        perf_eval_dim = f.attrs['perf_eval_dim']
        best_gen_c = f.attrs['best_gen_c']
        data = f.get('data')[()]
        all_c = f.get('all_c')[()]
        all_perf = f.get('all_perf')[()]
        qual_mat = f.get('qual_mat')[()]
        gen_perf = f.get('gen_perf')[()]
    return info, times, regions, perf_metric, perf_eval_dim, data, qual_mat, all_c, all_perf, gen_perf, best_gen_c