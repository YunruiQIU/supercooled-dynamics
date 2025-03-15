import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm, trange

def traj2transitionpair(traj, lagtime=50):
    """
    Transform the time-sequence trajectory into the transition pairs with given lagtime

    Parameters
    ----------
    traj: list-like, shape: (num_trajs, )(length_traj, num_features)
        The ensemble of trajectories used to create time lagged dataset.
    lagtime: int, default=None
        The lag time to construct the transition pairs, default unit is the frame
    ----------

    """
    try:
        len(traj.shape) == 3
    except:
        ValueError("The input data should have 3 dimensions;")
    
    _pastdata = []
    _futuredata = []
    for i in range(len(traj)):
        for j in range(len(traj[i])-lagtime):
            _pastdata.append(traj[i][j])
            _futuredata.append(traj[i][j+lagtime])
    return torch.from_numpy(np.array(_pastdata)), torch.from_numpy(np.array(_futuredata))


def _calculate_mean(dataloader):
    
    """
    Calculate the mean value of different features of the data; the past and future are calculated seperately

    Parameters
    ----------
    dataloader: TensorDataset
        The input dataloader containing past-future transition pairs to calculate the mean value
    ----------

    """
    pastmean, futuremean = None, None
    for _pastdata, _futuredata in dataloader:
        try:
            pastmean.add_(_pastdata.sum(dim=0))
        except AttributeError:
            pastmean = _pastdata.sum(dim=0)
        try:
            futuremean.add_(_futuredata.sum(dim=0))
        except AttributeError:
            futuremean = _futuredata.sum(dim=0)
    pastmean.div_(float(len(dataloader.dataset)))
    futuremean.div_(float(len(dataloader.dataset)))
    return pastmean, futuremean


def rao_blackwell_ledoit_wolf(covmatrix=None, num_data=None):
    """
    Rao-Blackwellized Ledoit-Wolf shrinkaged estimator of the covariance matrix.
    [1] Chen, Yilun, Ami Wiesel, and Alfred O. Hero III. "Shrinkage
    estimation of high dimensional covariance matrices" ICASSP (2009)

    Parameters
    ----------
    covmatrix: torch.Tensor, default=None
        The covariance matrix used to perform the Rao-Blackwellized Ledoit-Wolf shrinkaged estimation.
    num_data: int, default=None
        The number of the data used to compute the covariance matrix.
    
    ----------

    """
    matrix_dim = covmatrix.shape[0]
    assert covmatrix.shape == (matrix_dim, matrix_dim), "The input covariance matrix does not have the squared shape;"

    alpha = (num_data-2)/(num_data*(num_data+2))
    beta = ((matrix_dim+1)*num_data - 2) / (num_data*(num_data+2))

    trace_covmatrix_squared = torch.sum(covmatrix*covmatrix)  
    U = ((matrix_dim * trace_covmatrix_squared / torch.trace(covmatrix)**2) - 1)
    rho = min(alpha + beta/U, 1)

    F = (torch.trace(covmatrix) / matrix_dim) * torch.eye(matrix_dim).to(device=covmatrix.device)
    return (1-rho)*covmatrix + rho*F, rho


def covariance(dataloader, remove_mean=True, shrinkage=False):
    
    """
    Calculation of self, instantaneous, time-lagged correlation matrix. C_{00}, c_{0t}, c_{tt}

    Parameters
    ----------
    dataloader: TensorDataset
        The input dataloader containing past-future transition pairs to calculate the mean value
    remove_mean: bool, default:True
        The bool value used to decide if to remove the mean values for both the pastdata and futuredata.
    shrinkage: bool, default:True
        The bool value used to decide if the rao_blackwell_ledoit_wolf esimation method is used for covariance matrix calculation.
    Returns
    ----------
    pastcovmat: torch.Tensor, shape: (num_features, num_features) 
        Self-instantaneous correlation matrix generated from pastdata.
    overlapcovmat: torch.Tensor, shape: (num_features, num_features)
        Time lagged correlation matrix generated from pastdata.
    futurecovmat: torch.Tensor, shape: (num_features, num_features)
        Self-instantaneous correlation matrix generated from futuredata. 
    """

    pastmean, futuremean = _calculate_mean(dataloader=dataloader)
    pastcovmat, overlapcovmat, futurecovmat = None, None, None
    for _pastdata, _futuredata in dataloader:
        if remove_mean:
            _pastdata.sub_(pastmean[None, :]); _futuredata.sub_(futuremean[None, :])
        try:
            pastcovmat.add_(torch.matmul(_pastdata.T, _pastdata))
        except AttributeError:
            pastcovmat = torch.matmul(_pastdata.T, _pastdata)
        try:
            overlapcovmat.add_(torch.matmul(_pastdata.T, _futuredata))
        except AttributeError:
            overlapcovmat = torch.matmul(_pastdata.T, _futuredata)
        try:
            futurecovmat.add_(torch.matmul(_futuredata.T, _futuredata))
        except AttributeError:
            futurecovmat = torch.matmul(_futuredata.T, _futuredata)
    pastcovmat.div_(float(len(dataloader.dataset)))
    overlapcovmat.div_(float(len(dataloader.dataset)))
    futurecovmat.div_(float(len(dataloader.dataset)))
    
    if shrinkage:
        pastcovmat, _ = rao_blackwell_ledoit_wolf(covmatrix=pastcovmat, num_data=float(len(dataloader.dataset)));
        overlapcovmat, _ = rao_blackwell_ledoit_wolf(covmatrix=overlapcovmat, num_data=float(len(dataloader.dataset)));
        futurecovmat, _ = rao_blackwell_ledoit_wolf(covmatrix=futurecovmat, num_data=float(len(dataloader.dataset)))
    return pastcovmat, overlapcovmat, futurecovmat


def sqrt_root_matrix(matrix=None, eps=1e-6):
    """
    Calculate the sqrt root of the given matrix

    Parameters
    ----------
    matrix: torch.Tensor, default=None
        The input matrix to be calculated
    eps: float, default:1e-6
        The epsilon coefficient to ensure numerical stabilities
    Returns
    ----------
    _sqrt_matrix: torch.Tensor

    """
    eigenval, eigenvec = torch.linalg.eigh(matrix)
    _diag = torch.diag(1 / torch.sqrt(torch.abs(eigenval) + eps))
    _sqrt_matrix = torch.matmul(torch.matmul(eigenvec, _diag), eigenvec.T)
    
    return _sqrt_matrix


def whiten_data(dataloader, eps=1e-6, shrinkage=False):
    """
    Whiten the input data by the sqrt root of the covariance matrix. cov^{-0.5} * (data-data_mean)

    Parameters
    ----------
    dataloader: TensorDataset
        The input dataloader containing past-future transition pairs to be whiten
    eps: float, default:1e-12
        The epsilon coefficient to ensure numerical stabilities
    shrinkage: bool, default:True
        The bool value used to decide if the rao_blackwell_ledoit_wolf esimation method is used for covariance matrix calculation.
    Returns
    ----------
    whiten_data: torch.Tensor

    """
    
    past_mean, future_mean = _calculate_mean(dataloader=dataloader)
    c00, c0t, ctt = covariance(dataloader=dataloader, remove_mean=True, shrinkage=shrinkage)
    past_data, future_data = next(iter(dataloader))
    past_data -= past_mean; future_data -= future_mean
    sqrt_c00 = sqrt_root_matrix(matrix=c00, eps=eps); sqrt_c11 = sqrt_root_matrix(matrix=ctt, eps=eps)
    
    past_data = past_data.mm(sqrt_c00); future_data = future_data.mm(sqrt_c11)
    
    whiten_data = torch.utils.data.TensorDataset(pastdata, futuredata)
    whiten_dataloader = torch.utils.data.DataLoader(dataset=whiten_data, batch_size=dataloader.batch_size, shuffle=True, num_workers=1)
    return whiten_dataloader
