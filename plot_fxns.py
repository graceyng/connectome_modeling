import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Grace Ng'

def plot_perf_vs_c(dim_type, dims, all_c, all_perf, perf_metric, title=None):
    plt.figure()
    for i, dim_value in enumerate(dims):
        if type(dim_value) != str:
            dim_value = str(dim_value)
        plt.plot(all_c, all_perf[i], label=dim_type + ': ' + dim_value)
    plt.xlabel('c value')
    plt.ylabel(perf_metric)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()

def plot_predict_vs_data(times, regions, idxs, best_predict, data, qual_mat, perf_eval_dim, best_gen_c, log=True,
                         title=None):
    plt.figure()
    if perf_eval_dim == 'times':  # take the mean performance across time points, giving mean performance for each
        for i in idxs:
            qual_idxs = np.where(qual_mat[i])[0]
            if log:
                predict_plot = np.log(best_predict[i,qual_idxs])
                data_plot = np.log(data[i,qual_idxs])
            else:
                predict_plot = best_predict[i,qual_idxs]
                data_plot = data[i, qual_idxs]
            plt.scatter(predict_plot, data_plot, label='time: '+str(times[i]))
    elif perf_eval_dim == 'regions':  # take the mean performance across regions
        for j in idxs:
            qual_idxs = np.where(qual_mat[:,j])[0]
            if log:
                predict_plot = np.log(best_predict[qual_idxs,j])
                data_plot = np.log(data[qual_idxs,j])
            else:
                predict_plot = best_predict[qual_idxs,j]
                data_plot = data[qual_idxs,j]
            plt.scatter(predict_plot, data_plot, label=regions[j])
    else:
        raise Exception('perf_mean_type must be either "times" or "regions".')
    plt.xlabel('Best Predicted Pathology Amount (log)')
    plt.ylabel('Actual Pathology Amount (log)')
    plt.legend()
    if title is not None:
        plt.title(title + ': Predicted vs. Actual Pathology, c={:.2f}'.format(best_gen_c))
    else:
        plt.title('Predicted vs. Actual Pathology, c={:.2f}'.format(best_gen_c))
    plt.show()