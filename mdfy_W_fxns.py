import numpy as np
import h5py

def get_first_ord_W(connectome_file, seed_region, conn_dir, name_map=None, save_file=None):
    with h5py.File(connectome_file, 'r') as f:
        W = f.get('W')[()]
        ipsi_regions = f['W'].attrs['Ipsi_Regions']
        contra_regions = f['W'].attrs['Contra_Regions']
    if name_map is None:
        name_map = {'ipsi': 'R ', 'contra': 'L '}
    regions = np.concatenate((np.add(name_map['ipsi'], ipsi_regions), np.add(name_map['contra'], contra_regions)))
    seed_idx = np.where(regions == seed_region)[0]
    int_mask = np.zeros(W.shape)
    if conn_dir == 'antero':
        W.transpose()
    if conn_dir == 'retro' or conn_dir == 'antero':
        int_mask[:,seed_idx] = 1
    else:
        raise Exception('"conn dir" must be either "retro" or "antero".')
    mask = ~np.ma.make_mask(int_mask)
    np.putmask(W, mask, 0.)
    if save_file is not None:
        with h5py.File(save_file, 'w') as f:
            dset = f.create_dataset('W', data=W)
            dset.attrs['Ipsi_Regions'] = ipsi_regions
            dset.attrs['Contra_Regions'] = contra_regions
            dset.attrs['conn_dir'] = conn_dir
            dset.attrs['Notes'] = 'First order connections only.'
    return W