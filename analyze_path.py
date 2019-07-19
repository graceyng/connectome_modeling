import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

####################
#### Parameters ####
####################
process_path_data_file = 'process_path_data.hdf5'
connectome_file = 'W.hdf5'
group_list = ['NTG']

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

# create a vector of the names of all the regions of interest -- first, with the right-sided regions, and then the
# left-sided regions
regions = np.concatenate((np.add('R ', ipsi_regions), np.add('L ', contra_regions)))

# compute the mean of the pathology data across mice for each group and time point
mean_data = {}
for group in process_path_data:
    mean_data[group] = []
    for time in process_path_data[group]:
        mean_data[group].append(np.nanmean(process_path_data[group][time], axis=0))
    points = compute_percent_change_path(np.array(mean_data[group]))
    plot_silhouette_clusters(points)

