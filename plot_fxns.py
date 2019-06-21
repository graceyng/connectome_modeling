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

def plot_logpredict_vs_logdata(times, best_predict, all_data_idxs, all_log_data, best_gen_c):
    plt.figure()
    for i, time in enumerate(times):
        plt.scatter(np.log(best_predict[i][all_data_idxs[i]]), all_log_data[i], label='time: ' + str(time))
    plt.xlabel('Best Predicted Pathology Amount (log)')
    plt.ylabel('Actual Pathology Amount (log)')
    plt.legend()
    plt.title('Predicted vs. Actual Pathology, c={:.2f}'.format(best_gen_c))
    plt.show()