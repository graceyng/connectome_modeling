import numpy as np
import h5py
from scipy.linalg import expm
from scipy.stats import pearsonr, linregress
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import plot_fxns

__author__ = 'Grace Ng'

def make_Xo(seed_region, regions):
    """
    Initialize the Xo vector, which represents the pathology at t=0. All regions should have 0 pathology, except the
    region of interest being seeded with a pathology value of 1 (arbitrary units).
    :param seed_region (str)
    :param regions (ndarray of str)
    :return: Xo (ndarray of floats)
    """
    Xo = np.zeros(len(regions))
    Xo[np.where(regions == seed_region)[0]] = 1.
    return Xo

def get_norm_W(W):
    np.fill_diagonal(W, 0.)
    W = W / max(np.linalg.eigvals(W).real)
    return W - np.identity(W.shape[0])

def get_L_out(W, normalize=True):
    """
    Compute the out-degree Laplacian matrix.
    :param W (ndarray): the weighted adjacency matrix
    :return: L_out (ndarray): the weighted out-degree Laplacian matrix
    """
    np.fill_diagonal(W, 0.) # zero out the diagonal since regions should not be connected to themselves
    if normalize:
        W = W / max(np.linalg.eigvals(W).real)
    out_deg = np.sum(W, axis=1)
    return np.diag(out_deg) - W

def get_perf(predicted, actual, perf_metric, log_shift="shift"):
    """

    :param predicted:
    :param actual:
    :param perf_metric:
    :param log: can take values of None, "shift", or "no shift"
    :return:
    """
    if log_shift == "no shift" or log_shift == "shift":
        if log_shift == "shift":
            actual = actual + 1.
            predicted = predicted + 1.
        func = lambda x: x > 0. and not np.isnan(x)
        qual_idxs = np.where(np.vectorize(func)(predicted))[0]
        if qual_idxs.size < 2:
            return np.nan
        actual = np.log(actual[qual_idxs])
        predicted = np.log(predicted[qual_idxs])
    if perf_metric == 'corr':
        # need a minimum of three points to calculate a meaningful Pearson correlation, and the Pearson correlation is
        # undefined if the std dev of either dataset is 0
        if predicted.size < 3:
            return np.nan
        elif np.std(predicted) == 0. or np.std(actual) == 0.:
            return np.nan
        return pearsonr(predicted, actual)[0]
    elif perf_metric == 'dist':
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

def fit(Xo, L_out, times, regions, data, c_range_type, c_range, num_c, perf_metric, perf_eval_dim,
        log_shift="no shift", do_linregress=True, plot=False, save=None):
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
    # func used to find the indices of regions where pathology data is not 0 or nan so that we can take the log afterward
    if log_shift == "no shift":
        func = lambda x: x > 0. and not np.isnan(x)
        qual_mat = np.vectorize(func)(data) # matrix has True if element satisfies the above conditions, and False otherwise
    else:
        qual_mat = np.ones((data.shape[0], data.shape[1]), dtype=bool)
    if do_linregress:
        swapped_predicts = np.swapaxes(all_predicts, 1, 2)
        flat_predicts = swapped_predicts[qual_mat].flatten('F')
        flat_data = np.tile(data[qual_mat],num_c)
        slope, intercept, r_value, p_value, std_err = linregress(flat_predicts, flat_data)
        all_predicts = slope * all_predicts + intercept
        linregress_params = [slope, intercept]
    else:
        linregress_params = None

    # evaluate model performance according to the performance metric
    dims = [] # dimensions along which performance is evaluated
    perf_idxs = []
    if perf_eval_dim == 'times':  # take the mean performance across time points, giving mean performance for each
        for i in range(data.shape[0]):
            qual_idxs = np.where(qual_mat[i])[0]
            if qual_idxs.size > 0:
                perf = np.apply_along_axis(get_perf, 0, all_predicts[i,:,qual_idxs], data[i,qual_idxs], perf_metric,
                                           log_shift)
                if np.where(np.isnan(perf))[0].size <= perf.size/2.:
                    all_perf.append(perf)
                    dims.append(times[i])
                    perf_idxs.append(i)
    elif perf_eval_dim == 'regions':  # take the mean performance across regions
        for j in range(data.shape[1]):
            qual_idxs = np.where(qual_mat[:,j])[0]
            if qual_idxs.size > 0:
                perf = np.apply_along_axis(get_perf, 0, all_predicts[qual_idxs,:,j], data[qual_idxs,j], perf_metric,
                                           log_shift)
                if np.where(np.isnan(perf))[0].size <= perf.size/2.:
                    all_perf.append(perf)
                    dims.append(regions[j])
                    perf_idxs.append(j)
    else:
        raise Exception('perf_mean_type must be either "times" or "regions".')
    all_perf = np.array(all_perf)
    qual_idxs = np.where(~(~np.isfinite(all_perf)).all(axis=0))[0]
    gen_perf = np.nanmean(all_perf[:,qual_idxs], axis=0) # vector with the mean performance score for each of the c values
    if np.all(np.isnan(gen_perf)):
        raise Exception('All c values resulted in performance values that are not a number.')
    if plot:
        if perf_metric == 'dist':
            use_log = True
        else:
            use_log = False
        plot_fxns.plot_perf_vs_c(perf_eval_dim, dims, all_c, all_perf, perf_metric, use_log=use_log)

    # if performance metric is correlation, find the c value that gives the highest mean performance score
    if perf_metric == 'corr':
        best_perf = np.nanmax(gen_perf)
        best_c_idx = qual_idxs[np.nanargmax(gen_perf)]
        best_c_per_ctgry = all_c[np.nanargmax(all_perf, axis=1)]
    # if performance metric is distance, find the c value that gives the lowest mean performance score
    elif perf_metric == 'dist':
        best_perf = np.nanmin(gen_perf)
        best_c_idx = qual_idxs[np.nanargmin(gen_perf)]
        best_c_per_ctgry = all_c[np.nanargmin(all_perf, axis=1)]
    else:
        raise Exception('perf_metric must either be "corr" or "dist".')
    best_gen_c = all_c[best_c_idx]
    best_predict = np.array(all_predicts)[:, best_c_idx]
    if plot:
        if perf_eval_dim == 'times':
            all_idxs = np.array(range(len(times)))
        elif perf_eval_dim == 'regions':
            all_idxs = np.array(range(len(regions)))
        else:
            raise Exception('perf_eval_dim must either be "times" or "regions".')
        plot_fxns.plot_predict_vs_data(times, regions, all_idxs, best_predict, data, qual_mat, perf_eval_dim,
                                       best_gen_c, log_shift, title='Evaluated by ' + perf_metric + ', averaged over ' +
                                                                    perf_eval_dim + ', log ' + log_shift)
        if len(perf_idxs) > 10:
            if perf_metric == "corr":
                best_5_idxs = np.argpartition(np.array(all_perf)[:,best_c_idx], -5)[-5:]
                worst_5_idxs = np.argpartition(np.array(all_perf)[:,best_c_idx], 5)[:5]
            elif perf_metric == "dist":
                best_5_idxs = np.argpartition(np.array(all_perf)[:,best_c_idx], 5)[:5]
                worst_5_idxs = np.argpartition(np.array(all_perf)[:,best_c_idx], -5)[-5:]
            plot_fxns.plot_perf_vs_c(perf_eval_dim, np.array(dims)[best_5_idxs], all_c,
                                     np.array(all_perf)[best_5_idxs],
                                     perf_metric, 'Performance in the 5 Best Dimensions', use_log)
            plot_fxns.plot_perf_vs_c(perf_eval_dim, np.array(dims)[worst_5_idxs], all_c,
                                     np.array(all_perf)[worst_5_idxs],
                                     perf_metric, 'Performance in the 5 Worst Dimensions', use_log)
            plot_fxns.plot_predict_vs_data(times, regions, np.array(perf_idxs)[best_5_idxs], best_predict,
                                           data, qual_mat, perf_eval_dim, best_gen_c, log_shift,
                                           title='5 Best Dimensions')
            plot_fxns.plot_predict_vs_data(times, regions, worst_5_idxs, best_predict, data, qual_mat, perf_eval_dim,
                                           best_gen_c, log_shift, title='5 Worst Dimensions')
            for idx in np.concatenate((best_5_idxs, worst_5_idxs)):
                plot_fxns.plot_predict_vs_actual_timecourse(times, regions, np.array(dims)[idx], data, predict, Xo,
                                                            L_out, best_gen_c, best_perf, log_shift=log_shift,
                                                            linregress_params=linregress_params)

    if save is not None:
        save_mdl_results(save, times, regions, perf_metric, perf_eval_dim, data, qual_mat, all_c, np.array(all_perf),
                         gen_perf, best_gen_c)
    return all_c, np.array(all_perf), np.array(all_predicts), dims, best_gen_c, best_perf, best_predict, \
           best_c_per_ctgry, linregress_params

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

def get_bestperf_bestc_pairs(all_c, all_perf, perf_metric):
    if perf_metric == "corr":
        best_c_idxs = np.nanargmax(all_perf, axis=1)
    elif perf_metric == "dist":
        best_c_idxs = np.nanargmin(all_perf, axis=1)
    else:
        raise Exception('perf_metric should either be "corr" or "dist".')
    best_c = all_c[best_c_idxs]
    best_perf = np.array([all_perf[i,best_c_idxs[i]] for i in range(all_perf.shape[0])])
    return np.array([[best_c[i], best_perf[i]] for i in range(best_c.size)])

def silhouette_cluster_bestperf_bestc(perf_eval_dim, all_c, all_perf, perf_metric, n_range=list(range(2,8))):
    points = get_bestperf_bestc_pairs(all_c, all_perf, perf_metric)
    for n_clusters in n_range:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        #ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, points.shape[0] + (n_clusters + 1) * 10])
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
        labels = kmeans.fit_predict(points)
        silhouette_avg = silhouette_score(points, labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is:", silhouette_avg)
        sample_silhouette_values = silhouette_samples(points, labels)
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(labels.astype(float) / n_clusters)
        ax2.scatter(points[:, 0], points[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = kmeans.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for c values")
        ax2.set_ylabel("Feature space for " + perf_eval_dim)

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    plt.show()



def cluster_bestperf_bestc(perf_eval_dim, dims, all_c, all_perf, perf_metric, n_clusters=7, plot=False):
    """

    :param perf_eval_dim:
    :param dims:
    :param all_c:
    :param all_perf:
    :param perf_metric:
    :return:
    """
    points = get_bestperf_bestc_pairs(all_c, all_perf, perf_metric)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
    labels = kmeans.fit_predict(points)
    for label in set(labels):
        idxs = np.where(labels == label)[0]
        print('Cluster ' + str(label))
        print([dims[idx] for idx in idxs])
    if plot:
        plot_fxns.plot_bestperf_vs_bestc(points, labels, perf_metric, perf_eval_dim)
    return labels, points
