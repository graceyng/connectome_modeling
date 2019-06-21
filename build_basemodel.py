import numpy as np
import h5py
import mdl_fxns

__author__ = 'Grace Ng'

#####
# Parameters
#####
connectome_file = 'W.hdf5'
process_path_data_file = 'process_path_data.hdf5'
perf_metric = 'corr' # or 'dist'
group_list = ['NTG'] # list of groups to consider, e.g. ['G20', 'NTG'] or 'all'
ROI = 'R CPu'
c_range_type = 'lin' #'log'
c_range = (0.01, 10.) # the range of c values that will be tested to find the best-fitting model
#c_range = (-3., 2.)
num_c = 100 # the number of c values to test
verbose = True
plot = True

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
ROI_names = np.concatenate((np.add('R ', ipsi_regions), np.add('L ', ipsi_regions)))

# compute the mean of the pathology data across mice for each group and time point
mean_data = {}
for group in process_path_data:
    mean_data[group] = []
    for time in process_path_data[group]:
        mean_data[group].append(np.nanmean(process_path_data[group][time], axis=0))

#####
# Fit a Model Using Pathology Data with Seeding from the iCPu Region
#####

Xo = mdl_fxns.make_Xo(ROI, ROI_names)
L_out = mdl_fxns.get_L_out(W) # generate weighted out-degree Laplacian matrix
c = {} # store the fit constant for each time point in this dictionary
perf = {} # data from the performance metric used to evaluate best fit
predicts = {}
c_per_time = {}

for group in mean_data:
    times = list(process_path_data[group].keys())
    best_c, best_perf, best_predict, best_c_per_time, all_perf = mdl_fxns.fit(Xo, L_out, times, np.array(mean_data[group]), \
                                                                    c_range_type, c_range, num_c, perf_metric, plot=True)
    c[group] = best_c
    perf[group] = best_perf
    predicts[group] = best_predict
    c_per_time[group] = best_c_per_time
    if verbose:
        print('Best c: ', best_c)
        print('Best performance score (', perf_metric, '): ', best_perf)
        print('Best c values for each time point: ', best_c_per_time)

#TODO: figure out if there is a better way of evaluating c instead of just averaging across time points








