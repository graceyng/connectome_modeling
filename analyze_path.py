import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy import stats

####################
#### Parameters ####
####################
process_path_data_file = 'process_path_data.hdf5'
connectome_file = 'W.hdf5'
snca_file = 'Data83018/SncaExpression.csv'
group_list = ['NTG']
seed_region = 'R CPu'

def compute_percent_change_path(group_data):
    differences = []
    for i in range(group_data.shape[1]):
        if np.all(group_data[:,i] != 0.):
            this_diff = np.diff(group_data[:, i])
            #this_diff = np.diff(group_data[:,i]) / group_data[:-1,i]
            if np.all(np.isfinite(this_diff)):
                differences.append(this_diff)
    return np.array(differences)

def plot_silhouette_clusters(points, n_range=list(range(2,8))):
    for n_clusters in n_range:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
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
        ax2.set_xlabel("Percentage change between time points 1 and 2")
        ax2.set_ylabel("Percentage change between time points 2 and 3")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    plt.show()


def plot_connectivity_str_by_path(W, path_data, snca_data, times, regions, seed_region, logshift=False):
    np.fill_diagonal(W, 0.)
    seed_idx = np.where(regions == seed_region)[0]
    sort_order = np.argsort(path_data, axis=1)
    sides = ('left', 'right', 'top', 'bottom')
    nolabels = {s: False for s in sides}
    nolabels.update({'label%s' % s: False for s in sides})
    for t, time in enumerate(times):
        labels = regions[sort_order[t]]
        retro_conn_str_values = W[:, seed_idx][sort_order[t]]
        antero_conn_str_values = W.T[:, seed_idx][sort_order[t]]
        snca_values = snca_data[sort_order[t]].reshape(labels.size,1)
        fig, axes = plt.subplots(4, 1, figsize=(30, 120))
        fig.suptitle('Time Point ' + str(time))
        if logshift:
            this_path_data = np.log10(path_data[t] + 1.)
        else:
            this_path_data = path_data[t]
        plot_mats = (this_path_data.reshape(labels.size, 1)[sort_order[t]], retro_conn_str_values,
                     antero_conn_str_values, snca_values)
        names = ('Pathology', 'Retrograde', 'Anterograde', 'a-Synuclein')
        cmap_list = ('Blues', 'Oranges', 'Greens', 'Reds')
        for i in range(len(axes)):
            ax = axes[i]
            arr = np.ma.array(plot_mats[i], mask=(plot_mats[i] == 0.))
            cmap = plt.get_cmap(cmap_list[i])
            cmap.set_bad('0.75', 1.)
            img = ax.matshow(arr.T, cmap=cmap)
            if i == len(axes) - 1:
                ax.tick_params(axis='y', which='both', **nolabels)
                ax.tick_params(axis='x', which='both', top=False, bottom=True, labelbottom=True, labeltop=False)
                ax.set_xticks(np.arange(labels.size))
                ax.set_xticklabels([''] + labels, rotation=90, fontsize=7)
            else:
                ax.tick_params(axis='both', which='both', **nolabels)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="8%", pad=0.15)
            cbar = fig.colorbar(img, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=7)
            if i > 0:
                corr = stats.pearsonr(plot_mats[i], plot_mats[0])[0][0]  # finds the correlation between the vector of
                # interest and the pathology vector
                title = names[i] + ',\nr={:.3f}'.format(corr)
            else:
                title = names[i]
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.05
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, title, va='center', ha='center', fontsize=10)
        plt.show()


def plot_connectivity_path_corr(W, path_data, times, regions, seed_region):
    np.fill_diagonal(W, 0.)
    seed_idx = np.where(regions == seed_region)[0]
    retro_conn_str_values = W[:, seed_idx]
    antero_conn_str_values = W.T[:, seed_idx]
    colors = {'antero': 'orange', 'retro': '#17becf'}
    for t, time in enumerate(times):
        fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 5], 'height_ratios': [5, 1]}, sharex='col')
        fig.suptitle('Time Point ' + str(time))
        retro_no_conn_idxs = np.where(retro_conn_str_values == 0.)[0]
        retro_no_conn_zero_idxs = np.where(path_data[t][retro_no_conn_idxs] == 0.)[0]
        retro_no_conn_log_idxs = np.where(path_data[t][retro_no_conn_idxs] != 0.)[0]
        antero_no_conn_idxs = np.where(antero_conn_str_values == 0.)[0]
        antero_no_conn_zero_idxs = np.where(path_data[t][antero_no_conn_idxs] == 0.)[0]
        antero_no_conn_log_idxs = np.where(path_data[t][antero_no_conn_idxs] != 0.)[0]

        retro_conn_idxs = np.where(retro_conn_str_values != 0.)[0]
        retro_conn_zero_idxs = np.where(path_data[t][retro_conn_idxs] == 0.)[0]
        retro_conn_log_idxs = np.where(path_data[t][retro_conn_idxs] != 0.)[0]
        antero_conn_idxs = np.where(antero_conn_str_values != 0.)[0]
        antero_conn_zero_idxs = np.where(path_data[t][antero_conn_idxs] == 0.)[0]
        antero_conn_log_idxs = np.where(path_data[t][antero_conn_idxs] != 0.)[0]
        ax[0, 0].scatter([0.] * antero_no_conn_log_idxs.size,
                         np.log10(path_data[t][antero_no_conn_idxs][antero_no_conn_log_idxs]),
                         alpha=0.5, label='anterograde', color=colors['antero'])
        ax[0, 0].scatter([0.] * retro_no_conn_log_idxs.size,
                         np.log10(path_data[t][retro_no_conn_idxs][retro_no_conn_log_idxs]),
                         alpha=0.5, label='retrograde', color=colors['retro'])
        ax[1, 0].scatter([0.] * antero_no_conn_zero_idxs.size, [0.] * antero_no_conn_zero_idxs, alpha=0.5,
                         color=colors['antero'])
        ax[1, 0].scatter([0.] * retro_no_conn_zero_idxs.size, [0.] * retro_no_conn_zero_idxs, alpha=0.5,
                         color=colors['retro'])
        ax_antero_log = ax[0, 1].twiny()
        ax_antero_zero = ax[1, 1].twiny()
        ax_antero_log.get_shared_x_axes().join(ax_antero_log, ax_antero_zero)
        ax_antero_log.scatter(antero_conn_str_values[antero_conn_idxs][antero_conn_log_idxs],
                              np.log10(path_data[t][antero_conn_idxs][antero_conn_log_idxs]), alpha=0.5,
                              label='anterograde',
                              color=colors['antero'])
        ax[0, 1].scatter(retro_conn_str_values[retro_conn_idxs][retro_conn_log_idxs],
                         np.log10(path_data[t][retro_conn_idxs][retro_conn_log_idxs]), alpha=0.5, label='retrograde',
                         color=colors['retro'])
        ax[1, 1].scatter(antero_conn_str_values[antero_conn_idxs][antero_conn_zero_idxs],
                         path_data[t][antero_conn_idxs][antero_conn_zero_idxs], alpha=0.5, label='anterograde',
                         color=colors['antero'])
        ax[1, 1].scatter(retro_conn_str_values[retro_conn_idxs][retro_conn_zero_idxs],
                         path_data[t][retro_conn_idxs][retro_conn_zero_idxs], alpha=0.5, label='retrograde',
                         color=colors['retro'])
        antero_corr = stats.pearsonr(antero_conn_str_values[antero_conn_idxs][antero_conn_log_idxs][:, 0],
                                     np.log10(path_data[t][antero_conn_idxs][antero_conn_log_idxs]))[0]
        retro_corr = stats.pearsonr(retro_conn_str_values[retro_conn_idxs][retro_conn_log_idxs][:, 0],
                                    np.log10(path_data[t][retro_conn_idxs][retro_conn_log_idxs]))[0]
        slope_antero, intercept_antero, _, _, _ = stats.linregress(
            antero_conn_str_values[antero_conn_idxs][antero_conn_log_idxs][:, 0],
            np.log10(path_data[t][antero_conn_idxs][antero_conn_log_idxs]))
        line_antero = slope_antero * antero_conn_str_values[antero_conn_idxs][antero_conn_log_idxs][:,
                                     0] + intercept_antero
        antero_plot = ax[0, 1].plot(antero_conn_str_values[antero_conn_idxs][antero_conn_log_idxs][:, 0], line_antero,
                                    color=colors['antero'])
        slope_retro, intercept_retro, _, _, _ = stats.linregress(
            retro_conn_str_values[retro_conn_idxs][retro_conn_log_idxs][:, 0],
            np.log10(path_data[t][retro_conn_idxs][retro_conn_log_idxs]))
        line_retro = slope_retro * retro_conn_str_values[retro_conn_idxs][retro_conn_log_idxs][:, 0] + intercept_retro
        retro_plot = ax[0, 1].plot(retro_conn_str_values[retro_conn_idxs][retro_conn_log_idxs][:, 0], line_retro,
                                   color=colors['retro'])
        ax[0, 1].set_title('retrograde r={:.3f}\nanterograde r={:.3f}'.format(retro_corr, antero_corr), {'fontsize': 8},
                           loc='right')
        ax[0, 0].set_ylabel('Pathology (log)')
        ax[1, 0].set_xlabel('No Connectivity')
        ax[1, 1].set_xlabel('Connection Strength (Retrograde)')
        ax_antero_log.set_xlabel('Connection Strength (Anterograde)')
        ax[1, 0].set_yticks([0.])
        ax[1, 0].set_yticklabels(['-inf'])
        ax[0, 1].legend(antero_plot + retro_plot, ['anterograde', 'retrograde'], loc='lower right')
        ax[1, 0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        for this_ax in [ax[0, 0], ax[1, 0], ax[0, 1], ax[1, 1]]:
            this_ax.spines['top'].set_visible(False)
            this_ax.spines['right'].set_visible(False)
        for this_ax in [ax[0, 0], ax[0, 1]]:
            this_ax.spines['bottom'].set_visible(False)
            this_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        for this_ax in [ax[0, 1], ax[1, 1]]:
            this_ax.spines['left'].set_visible(False)
            this_ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_antero_zero.spines['top'].set_visible(False)
        ax_antero_zero.tick_params(axis='x', which='both', top=False, labeltop=False)
        ax_antero_log.spines['bottom'].set_visible(False)
        for this_ax in [ax_antero_zero, ax_antero_log]:
            this_ax.spines['right'].set_visible(False)
            this_ax.spines['left'].set_visible(False)

        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax[0, 0].transAxes, color='k', clip_on=False)
        ax[0, 0].plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        kwargs.update(transform=ax[1, 0].transAxes)  # switch to the bottom axes
        ax[1, 0].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        plt.show()


def plot_snca_path_corr(W, path_data, snca_data, times, regions, seed_region):
    np.fill_diagonal(W, 0.)
    seed_idx = np.where(regions == seed_region)[0]
    retro_conn_str_values = W[:, seed_idx]
    antero_conn_str_values = W.T[:, seed_idx]
    retro_no_conn_idxs = np.where(retro_conn_str_values == 0.)[0]
    antero_no_conn_idxs = np.where(antero_conn_str_values == 0.)[0]

    colors = {'antero': 'orange', 'retro': '#17becf'}
    for t, time in enumerate(times):
        plt.figure()
        retro_no_conn_log_idxs = np.where(path_data[t][retro_no_conn_idxs] != 0.)[0]
        antero_no_conn_log_idxs = np.where(path_data[t][antero_no_conn_idxs] != 0.)[0]
        retro_corr = stats.pearsonr(snca_data[retro_no_conn_idxs][retro_no_conn_log_idxs],
                                    np.log10(path_data[t][retro_no_conn_idxs][retro_no_conn_log_idxs]))[0]
        antero_corr = stats.pearsonr(snca_data[antero_no_conn_idxs][antero_no_conn_log_idxs],
                                     np.log10(path_data[t][antero_no_conn_idxs][antero_no_conn_log_idxs]))[0]
        plt.scatter(snca_data[retro_no_conn_idxs][retro_no_conn_log_idxs],
                    np.log10(path_data[t][retro_no_conn_idxs][retro_no_conn_log_idxs]), alpha=0.5,
                    label='retrograde, r={:.3f}'.format(retro_corr), color=colors['retro'])
        plt.scatter(snca_data[antero_no_conn_idxs][antero_no_conn_log_idxs],
                    np.log10(path_data[t][antero_no_conn_idxs][antero_no_conn_log_idxs]), alpha=0.5,
                    label='anterograde, r={:.3f}'.format(antero_corr), color=colors['antero'])
        slope_retro, intercept_retro, _, _, _ = stats.linregress(snca_data[retro_no_conn_idxs][retro_no_conn_log_idxs],
                                                                 np.log10(path_data[t][retro_no_conn_idxs][
                                                                              retro_no_conn_log_idxs]))
        line_retro = slope_retro * snca_data[retro_no_conn_idxs][retro_no_conn_log_idxs] + intercept_retro
        plt.plot(snca_data[retro_no_conn_idxs][retro_no_conn_log_idxs], line_retro, color=colors['retro'])
        slope_antero, intercept_antero, _, _, _ = stats.linregress(
            snca_data[antero_no_conn_idxs][antero_no_conn_log_idxs],
            np.log10(path_data[t][antero_no_conn_idxs][antero_no_conn_log_idxs]))
        line_antero = slope_antero * snca_data[antero_no_conn_idxs][antero_no_conn_log_idxs] + intercept_antero
        plt.plot(snca_data[antero_no_conn_idxs][antero_no_conn_log_idxs], line_antero, color=colors['antero'])
        plt.legend()
        plt.xlabel('Snca Gene Expression')
        plt.ylabel('Pathology (log)')
        plt.title('Pathology vs. Snca Expression for\nRegions Without Connectivity: Time Point ' + str(time))
        plt.show()

process_path_data = {}
with h5py.File(process_path_data_file, 'r') as f:
    for group in f.keys():
        if group_list == 'all' or group in group_list:
            process_path_data[group] = {}
            for time in f[group].keys():
                process_path_data[group][int(time)] = f[group].get(time)[()]
with h5py.File(connectome_file, 'r') as f:
    W = f.get('W')[()]
    ipsi_regions = f['W'].attrs['Ipsi_Regions']
    contra_regions = f['W'].attrs['Contra_Regions']

times = np.array(list(process_path_data[group].keys()))

# create a vector of the names of all the regions of interest -- first, with the right-sided regions, and then the
# left-sided regions
regions = np.concatenate((np.add('R ', ipsi_regions), np.add('L ', contra_regions)))

snca_df = pd.read_csv(snca_file, header=None)
snca_regions = snca_df.iloc[:,0].values
snca_data = snca_df.iloc[:,1].values
for i, snca_region in enumerate(snca_regions):
    if snca_region[0] == 'i':
        snca_regions[i] = 'R ' + snca_region[1:]
    elif snca_region[0] == 'c':
        snca_regions[i] = 'L ' + snca_region[1:]

# indices to sort the elements of the Snca expression data (for ipsilateral and contralateral regions) to have the same
# order (by region) as the connectivity matrix
sort_snca_idxs = np.argsort(snca_regions)[np.array([sorted(regions).index(x) for x in regions])]
sort_snca_data = snca_data[sort_snca_idxs]

# compute the mean of the pathology data across mice for each group and time point
mean_data = {}
for group in process_path_data:
    mean_data[group] = []
    for time in process_path_data[group]:
        mean_data[group].append(np.nanmean(process_path_data[group][time], axis=0))
    #points = compute_percent_change_path(np.array(mean_data[group]))
    #plot_silhouette_clusters(points)

path_data = np.array(mean_data[group])
# plot_connectivity_str_by_path(W, path_data, sort_snca_data, times, regions, seed_region, logshift=False)
plot_connectivity_path_corr(W, path_data, times, regions, seed_region)
#plot_snca_path_corr(W, path_data, sort_snca_data, times, regions, seed_region)
