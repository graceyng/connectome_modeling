import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Grace Ng'

def plot_perf_vs_c(times, all_c, all_perf, perf_metric):
    plt.figure()
    for i, time in enumerate(times):
        plt.plot(all_c, all_perf[i], label='time: ' + str(time))
    plt.xlabel('c value')
    plt.ylabel(perf_metric)
    plt.legend()
    plt.show()

def plot_logpredict_vs_logdata(times, regions, best_predict, data, qual_mat, perf_eval_dim, best_gen_c):
    plt.figure()
    if perf_eval_dim == 'times':  # take the mean performance across time points, giving mean performance for each
        for i, time in enumerate(times):
            qual_idxs = np.where(qual_mat[i])[0]
            plt.scatter(np.log(best_predict[i,qual_idxs]), np.log(data[i,qual_idxs]), label='time: '+str(time))
    elif perf_eval_dim == 'regions':  # take the mean performance across regions
        for j, region in enumerate(regions):
            qual_idxs = np.where(qual_mat[:,j])[0]
            plt.scatter(np.log(best_predict[qual_idxs,j]), np.log(data[qual_idxs,j]), label=region)
    else:
        raise Exception('perf_mean_type must be either "times" or "regions".')
    plt.xlabel('Best Predicted Pathology Amount (log)')
    plt.ylabel('Actual Pathology Amount (log)')
    plt.legend()
    plt.title('Predicted vs. Actual Pathology, c={:.2f}'.format(best_gen_c))
    plt.show()