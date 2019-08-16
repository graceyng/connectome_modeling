import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

__author__ = 'Grace Ng'

def plot_perf_vs_c(perf_eval_dim, dims, all_c, all_perf, perf_metric, title=None, use_log=False):
    plt.figure()
    for i, dim_value in enumerate(dims):
        if type(dim_value) != str:
            dim_value = str(dim_value)
        if use_log:
            y = np.log10(all_perf[i])
            ylabel = perf_metric + ' (log)'
        else:
            y = all_perf[i]
            ylabel = perf_metric
        plt.plot(all_c, y, label=perf_eval_dim + ': ' + dim_value)
    plt.xlabel('c value')
    plt.ylabel(ylabel)
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()


def plot_predict_vs_data(times, regions, idxs, best_predict, A, Xo, data, qual_mat, perf_eval_dim, best_gen_c,
                         log_shift="no shift", title=None):
    np.fill_diagonal(A, 0.)
    seed_idx = np.where(Xo != 0.)[0]
    retro_conn_str_values = A[:, seed_idx]
    retro_no_conn_idxs = np.where(retro_conn_str_values == 0.)[0]
    retro_conn_idxs = np.where(retro_conn_str_values != 0.)[0]
    if log_shift == "shift":
        data = data + 1.
        best_predict = best_predict + 1.
    func = lambda x: x > 0. and not np.isnan(x)
    if perf_eval_dim == 'times':  # take the mean performance across time points, giving mean performance for each
        for i in idxs:
            colors = {'no conn': '#2ca02c', 'conn': '#9467bd'}
            plt.figure(figsize=(7, 6))
            qual_idxs = np.where(qual_mat[i])[0]
            if log_shift == "no shift" or "shift":
                log_label = ' (log)'
                qual_idxs = np.intersect1d(qual_idxs, np.where(np.vectorize(func)(best_predict[i]))[0])
                no_conn_qual_idxs = np.intersect1d(qual_idxs, retro_no_conn_idxs)
                no_conn_predict_plot = np.log10(best_predict[i, no_conn_qual_idxs])
                no_conn_data_plot = np.log10(data[i, no_conn_qual_idxs])
                conn_qual_idxs = np.intersect1d(qual_idxs, retro_conn_idxs)
                conn_predict_plot = np.log10(best_predict[i, conn_qual_idxs])
                conn_data_plot = np.log10(data[i, conn_qual_idxs])
            else:
                log_label = ''
                no_conn_qual_idxs = np.intersect1d(qual_idxs, retro_no_conn_idxs)
                no_conn_predict_plot = best_predict[i, no_conn_qual_idxs]
                no_conn_data_plot = data[i, no_conn_qual_idxs]
                conn_qual_idxs = np.intersect1d(qual_idxs, retro_conn_idxs)
                conn_predict_plot = best_predict[i, conn_qual_idxs]
                conn_data_plot = data[i, conn_qual_idxs]
            no_conn_corr = stats.pearsonr(no_conn_predict_plot, no_conn_data_plot)[0]
            conn_corr = stats.pearsonr(conn_predict_plot, conn_data_plot)[0]
            slope_no_conn, intercept_no_conn, _, _, _ = stats.linregress(no_conn_predict_plot, no_conn_data_plot)
            line_no_conn = slope_no_conn * no_conn_predict_plot + intercept_no_conn
            plt.plot(no_conn_predict_plot, line_no_conn, color=colors['no conn'])
            slope_conn, intercept_conn, _, _, _ = stats.linregress(conn_predict_plot, conn_data_plot)
            line_conn = slope_conn * conn_predict_plot + intercept_conn
            plt.plot(conn_predict_plot, line_conn, color=colors['conn'])
            plt.scatter(no_conn_predict_plot, no_conn_data_plot, label='Regions without 1st order\nretrograde '
                                                                       'connectivity, r={:.3f}'.format(no_conn_corr),
                        color=colors['no conn'], alpha=0.5)
            plt.scatter(conn_predict_plot, conn_data_plot, label='Regions with 1st order\nretrograde connectivity, '
                                                                 'r={:.3f}'.format(conn_corr),
                        color=colors['conn'], alpha=0.5)
            plt.xlabel('Best Predicted Pathology Amount' + log_label)
            plt.ylabel('Actual Pathology Amount' + log_label)
            plt.legend()
            if title is not None:
                plt.title('Actual vs. Predicted Pathology: Time Point ' + str(times[i]) + ' ' + title +
                          ', c={:.2f}'.format(best_gen_c))
            else:
                plt.title('Actual vs. Predicted Pathology, Time Point ' + str(times[i]) + ' ' +
                          'c={:.2f}'.format(best_gen_c))
            plt.show()
    else:
        raise Exception(
            'Use the "plot_predict_vs_data_combined" function to plot results if perf_eval_dim is "regions".')


def plot_predict_vs_data_combined(times, regions, idxs, best_predict, data, qual_mat, perf_eval_dim, best_gen_c,
                                  log_shift="no shift", title=None):
    plt.figure()
    if log_shift == "shift":
        data = data + 1.
        best_predict = best_predict + 1.
    func = lambda x: x > 0. and not np.isnan(x)
    if perf_eval_dim == 'times':  # take the mean performance across time points, giving mean performance for each
        for i in idxs:
            qual_idxs = np.where(qual_mat[i])[0]
            if log_shift == "no shift" or "shift":
                qual_idxs = np.intersect1d(qual_idxs, np.where(np.vectorize(func)(best_predict[i]))[0])
                predict_plot = np.log10(best_predict[i, qual_idxs])
                data_plot = np.log10(data[i, qual_idxs])
            else:
                predict_plot = best_predict[i,qual_idxs]
                data_plot = data[i, qual_idxs]
            plt.scatter(predict_plot, data_plot, label='time: '+str(times[i]))
    elif perf_eval_dim == 'regions':  # take the mean performance across regions
        for j in idxs:
            qual_idxs = np.where(qual_mat[:,j])[0]
            if log_shift == "no shift" or "shift":
                qual_idxs = np.intersect1d(qual_idxs, np.where(np.vectorize(func)(best_predict[:,j]))[0])
                predict_plot = np.log10(best_predict[qual_idxs, j])
                data_plot = np.log10(data[qual_idxs, j])
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
        plt.title(title + ': Actual vs. Predicted Pathology, c={:.2f}'.format(best_gen_c))
    else:
        plt.title('Actual vs. Predicted Pathology, c={:.2f}'.format(best_gen_c))
    plt.show()


def plot_predict_vs_actual_timecourse(times, regions, region_to_plot, data, predict_func, Xo, A, c, perf,
                                      log_shift="no shift", linregress_params=None):
    plt.figure()
    region_idx = np.where(regions == region_to_plot)[0]
    predict_x = np.linspace(0, max(times), 100)
    predict_y = np.array([predict_func(Xo, A, c, time)[region_idx] for time in predict_x]).reshape(predict_x.size)
    if linregress_params is not None:
        predict_y = linregress_params[0] * predict_y + linregress_params[1]
    actual_x = np.array(times)
    actual_y = data[:,region_idx].reshape(actual_x.size)

    if log_shift == "shift" or "no shift":
        if log_shift == "shift":
            predict_y = predict_y + 1.
            actual_y = actual_y + 1.
        func = lambda x: x > 0. and not np.isnan(x)
        qual_idxs_predicted = np.vectorize(func)(predict_y)
        qual_idxs_actual = np.vectorize(func)(data[:, region_idx]).reshape(actual_x.size)
        predict_x = predict_x[qual_idxs_predicted]
        predict_y = np.log10(predict_y[qual_idxs_predicted])
        actual_x = actual_x[qual_idxs_actual]
        actual_y = np.log10(actual_y[qual_idxs_actual])
    plt.plot(predict_x, predict_y, color='b', label='Predicted')
    plt.scatter(actual_x, actual_y, color='m', label='Actual')
    plt.legend()
    plt.xlabel('Time (Months)')
    plt.ylabel('Amount of Pathology')
    plt.title('Timecourse of Pathology for ' + region_to_plot + ', c={:.2f}'.format(c) + ', r={:.3f}'.format(perf))
    plt.show()

def plot_bestperf_vs_bestc(points, labels, perf_metric, perf_eval_dim):
    """

    :return:
    """
    plt.figure()
    for label in set(labels):
        idxs = np.where(labels == label)[0]
        plot_c = np.array([points[idx][0] for idx in idxs])
        plot_perf = np.array([points[idx][1] for idx in idxs])
        plt.scatter(plot_c, plot_perf, label='Cluster ' + str(label))
    plt.xlabel('c values')
    plt.ylabel(perf_metric)
    plt.title('Best Performance vs. Best c for all ' + perf_eval_dim)
    plt.legend()
    plt.show()


def compare_model_perfs(orig_perfs, new_perfs, dims, perf_metric, perf_eval_dim,
                        labels=['original model', 'new model']):
    plt.figure()
    plt.scatter(dims, orig_perfs, label=labels[0])
    plt.scatter(dims, new_perfs, label=labels[1])
    plt.xlabel(perf_eval_dim)
    plt.ylabel('Model Performance (' + perf_metric + ')')
    plt.title('Comparing ' + labels[0] + ' vs. ' + labels[1] + ' Performance')
    plt.legend()
    plt.show()
