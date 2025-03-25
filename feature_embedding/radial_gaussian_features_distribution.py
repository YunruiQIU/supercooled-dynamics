import numpy as np
import MDAnalysis as md
from tqdm import tqdm, trange
import gc
import numba as nb  


@nb.jit
def distance_to_radial_features(distance, a, b, c, sigma_a, sigma_b, sigma_c):
    
    _features = np.zeros(int(len(a)+len(b)+len(c)))
    _features[:len(a)] = np.exp(-(a - distance)**2 / 2 / sigma_a**2)
    _features[len(a):int(len(a)+len(b))] = np.exp(-(b - distance)**2 / 2 / sigma_b**2)
    _features[int(len(a)+len(b)):int(len(a)+len(b)+len(c))] = np.exp(-(c - distance)**2 / 2 / sigma_c**2)
    return _features



traj = md.coordinates.LAMMPS.DCDReader("traj.dcd")
xyz = traj.timeseries()
box_size_x = traj.dimensions[0]

a = np.linspace(start=0.5, stop=2.0, num=60, endpoint=False)
b = np.linspace(start=2.0, stop=3.0, num=20, endpoint=False)
c = np.linspace(start=3.0, stop=5.0, num=20, endpoint=False)
a = a+1.5/50
b = b+1/20
c = c+2/20
sigma_a = 0.025
sigma_b = 0.05
sigma_c = 0.1

for idx_frame in trange(0, 20):
    _grid = md.lib.nsgrid.FastNS(cutoff=7.5, coords=xyz[:, idx_frame, :].astype(np.float32), box=traj.dimensions, pbc=True)
    grid = _grid.self_search()
    _contact = grid.get_pair_distances(); _idx = grid.get_pairs()
    a_radial_features = np.zeros((4096, 100)); b_radial_features = np.zeros((4096, 100))
    for i in range(len(_idx)):
        _radial_feature = distance_to_radial_features(distance=_contact[i], a=a, b=b, c=c, sigma_a=sigma_a, sigma_b=sigma_b, sigma_c=sigma_c)
        if _idx[i, 0] > 3276 and _idx[i, 1] > 3276:
            b_radial_features[_idx[i, 0]] += _radial_feature
            b_radial_features[_idx[i, 1]] += _radial_feature
        elif _idx[i, 0] < 3277 and _idx[i, 1] < 3277:
            a_radial_features[_idx[i, 0]] += _radial_feature
            a_radial_features[_idx[i, 1]] += _radial_feature
        elif _idx[i, 0] > 3276 and _idx[i, 1] < 3277:
            a_radial_features[_idx[i, 0]] += _radial_feature
            b_radial_features[_idx[i, 1]] += _radial_feature
        elif _idx[i, 0] < 3277 and _idx[i, 1] > 3276:
            a_radial_features[_idx[i, 1]] += _radial_feature
            b_radial_features[_idx[i, 0]] += _radial_feature
    radial_features = np.hstack((a_radial_features, b_radial_features))
    np.save("./No%07d_frame_radial_gaussian_features.npy"%idx_frame, radial_features)
