import pandas as pd
import numpy as np
import h5py

########################
###### Parameters ######
########################

path_file = 'Data83018/data.csv'
connContra_file = 'Data83018/connectivity_contra.csv'
connIpsi_file = 'Data83018/connectivity_ipsi.csv'
process_path_data_file = 'process_path_data.hdf5'
connectome_file = 'W.hdf5'
tColIdx = 0 #the index of the column in path_file containing time point labels
gColIdx = 1 #the index of the column in path_file containing group labels

########################################
# Load Pathology and Connectivity Data #
########################################

path_df = pd.read_csv(path_file)
connContra_df = pd.read_csv(connContra_file)
connIpsi_df = pd.read_csv(connIpsi_file)

connContra_regions = [string.split('(')[0].strip() for string in connContra_df.columns[1:]]
connIpsi_regions = [string.split('(')[0].strip() for string in connIpsi_df.columns[1:]]

path_time_params = set(path_df.iloc[:,tColIdx]) #the set of time points in the pathology data
path_group_params = set(path_df.iloc[:,gColIdx]) #the experimental groups of mice in the pathology data

path_regions = path_df.columns[2:]
ipsi_idxs = np.where(np.array([region[0] for region in path_regions]) == 'i')[0]
pathIpsi_regions = [string[1:] for string in path_regions[ipsi_idxs]]
contra_idxs = np.where(np.array([region[0] for region in path_regions]) == 'c')[0]
pathContra_regions = [string[1:] for string in path_regions[contra_idxs]]

sort_pathIpsi_idxs = np.argsort(pathIpsi_regions)[np.array([sorted(connIpsi_regions).index(x) for x in connIpsi_regions])]
sort_pathContra_idxs = np.argsort(pathContra_regions)[np.array([sorted(connContra_regions).index(x) for x in connContra_regions])]

with h5py.File(process_path_data_file, 'w') as f:
    for group in path_group_params:
        g = f.create_group(group)
        for time in path_time_params:
            time_idxs = np.where(path_df.iloc[:,tColIdx] == time)[0]
            group_idxs = np.where(path_df.iloc[:,gColIdx] == group)[0]

            #idxs: row indices of qualifying datapoints (have the group and time point of interest)
            idxs = list(set(time_idxs).intersection(group_idxs))
            data = path_df.iloc[idxs,2:].to_numpy()

            #sort the columns of the pathology data to have the same order (by region) as the connectivity matrix
            ipsi_data = data[:,ipsi_idxs][:,sort_pathIpsi_idxs]
            contra_data = data[:, contra_idxs][:, sort_pathContra_idxs]

            #concatenate the matrices of the processed ipsilateral and contralateral pathology data, and save it to a
            # HD5 file
            save_data = np.concatenate((ipsi_data, contra_data), axis=1)
            g.create_dataset(str(time), data=save_data)

#Tile the connectivity matrix such that rows are source regions and columns are target regions, and save it to a HDF5 file
W_upper = np.concatenate((connIpsi_df.iloc[:,1:].to_numpy(), connContra_df.iloc[:,1:].to_numpy()), axis=1)
W_lower = np.concatenate((connContra_df.iloc[:,1:].to_numpy(), connIpsi_df.iloc[:,1:].to_numpy()), axis=1)
W = np.concatenate((W_upper, W_lower), axis=0)
with h5py.File(connectome_file, 'w') as f:
    dset = f.create_dataset('W', data=W)
    dset.attrs['Ipsi_Regions'] = connIpsi_regions
    dset.attrs['Contra_Regions'] = connContra_regions