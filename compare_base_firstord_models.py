import numpy as np
import h5py
import mdl_fxns
import plot_fxns

__author__ = 'Grace Ng'

#####
# Parameters
#####
connectome_file = 'retro_firstord_W.hdf5' #'W.hdf5'
process_path_data_file = 'process_path_data.hdf5'
perf_metric = 'dist' # 'corr' or 'dist'
perf_eval_dim = 'regions' #'times' or 'regions'; the dimension along which performance is evaluated
log_shift = "shift"
do_linregress = True
group_list = ['NTG'] # list of groups to consider, e.g. ['G20', 'NTG'] or 'all'
seed_region = 'R CPu'
c_range_type = 'log' #'lin' or 'log'
# the range of c values that will be tested to find the best-fitting model. Note that if the range type is 'log',
# then these values will be used for  10^x
#c_range = (0.0001, 10.)
c_range = (-5., 1.)
num_c = 100 # the number of c values to test
verbose = True
plot = True
cluster_analysis = 'check silhouettes' #'check silhouettes', 'plot clusters only', 'plot clusters and timecourses', 'None'
n_clusters = 3 #get this value by finding the cluster number with the maximum silhouette value
cluster_to_analyze = 0

#####
# Load Data from Files
#####
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

#####
# Fit a Model Using Pathology Data with Seeding from the iCPu Region
#####

Xo = mdl_fxns.make_Xo(seed_region, regions)

L_out = mdl_fxns.get_L_out(W) # generate weighted out-degree Laplacian matrix

c = {} # store the fit constant for each time point in this dictionary
perf = {} # data from the performance metric used to evaluate best fit
predicts = {}
c_per_ctgry = {}

for group in mean_data:
    times = np.array(list(process_path_data[group].keys()))
    all_c, all_perf, all_predicts, dims, best_c, best_perf, best_predict, best_c_per_ctgry, linregress_params = \
        mdl_fxns.fit(Xo, L_out, times, regions, np.array(mean_data[group]), c_range_type, c_range, num_c, perf_metric,
                     perf_eval_dim, log_shift, do_linregress, plot=True)