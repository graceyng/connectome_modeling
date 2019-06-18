import numpy as np
import h5py

########################
###### Parameters ######
########################

connectome_file = 'W.hdf5'
process_path_data_file = 'process_path_data.hdf5'

################################
##### Load Data from Files #####
################################
process_path_data = {}
with h5py.File(process_path_data_file, 'r') as f:
    for group in f.keys():
        process_path_data[group] = {}
        for time in f[group].keys():
            process_path_data[group][int(time)] = f[group].get(time)[()]
with h5py.File(connectome_file, 'r') as f:
    W = f.get('W')[()]
    ipsi_regions = f['W'].attrs['Ipsi_Regions']
    contra_regions = f['W'].attrs['Contra_Regions']


# compute the mean of the pathology data across time points
mean_time = {}
for group in process_path_data:
    mean_time[group] = {}
    for time in process_path_data[group]:
        mean_time[group][time] = np.nanmean(process_path_data[group][time], axis=0)

######################################################################
# Fit a Model Using Pathology Data with Seeding from the iCPu Region #
######################################################################








