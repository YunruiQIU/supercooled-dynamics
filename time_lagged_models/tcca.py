import numpy as np
import torch
from utils import traj2transitionpair, _calculate_mean, covariance, sqrt_root_matrix, whiten_data

class linear_cca(object):
    
    """
    A linear canonial correlation analysis (CCA) model which can be fit to identify the best time-lagged correlation order parameters
    between time-lagged time;

    Parameters
    ----------
    symmetrize : bool, optional, default=False
        The bool value to decide if the dynamical propagator is symmetrized, if the process is reversible and detailed-balanced, should set as True;
        For glass dynamics, where only sinlge pair of configurations is utilized, should set as False
    kinetic_mapping : bool, optional, default=True
        If the projection will be weighted by the singular values;
        According to the diffusion map theory, the weighted singular functions provide better kinetic distance description.
        Reference: J. Chem. Theory Comput. 2015, 11, 10, 5002–5011
    shrinkage : bool, optional, default=True
        The bool value used to decide if the rao_blackwell_ledoit_wolf esimation method is used for covariance matrix calculation.
        Set to True enable the numerical stabilities of calculations.

    """

    def __init__(self, symmetrize=False, kinetic_mapping=True, shrinkage=True):
        self.symmetrize = symmetrize
        self.kinetic_mapping = kinetic_mapping
        self.shrinkage = shrinkage
        
        
    @property
    def encoder(self):
        """ encoder: torch.Tensor, the transformation functions to project the data on the order parameters
        :getter: Gets the encoder for order parameters.
        """
        return self._encoder
    
    
    @property
    def decoder(self):
        """ decoder: torch.Tensor, the transformation functions to reconstruct the data from low-dimensional order parameters
        :getter: Gets the encoder for reconstruction.
        """
        return self._decoder
    
    
    @property
    def koopman_matrix(self):
        """ koopman_matrix: torch.Tensor, the linear dynamical propagator fitted from linear cca method
        :getter: Gets the koopman_matrix dynamical propagator.
        """
        return self._koopman_matrix
    
    
    @property
    def c00(self):
        """ c00: torch.Tensor, the covarianc matrix for the past data
        """
        return self._c00
    
    
    @property
    def sqrt_c00(self):
        """ sqrt_c00: torch.Tensor, the sqrt root of covarianc matrix for the past data
        """
        return self._sqrt_c00 
    
    
    @property
    def c11(self):
        """ c11: torch.Tensor, the covarianc matrix for the future data
        """
        return self._c11
    
    
    @property
    def sqrt_c11(self):
        """ sqrt_c11: torch.Tensor, the sqrt root of covarianc matrix for the future data
        """
        return self._sqrt_c11
    
    
    @property
    def c01(self):
        """ c01: torch.Tensor, the time-lagged covarianc matrix for the time-lagged data
        """
        return self._c01
    
    
    @property
    def past_mean(self):
        """ past_mean: torch.Tensor, the mean values for the past data
        """
        return self._pastmean
    
    
    @property
    def future_mean(self):
        """ future_mean: torch.Tensor, the mean values for the future data
        """
        return self._futuremean
    
    
    @property
    def sigma(self):
        """ sigma: torch.Tensor, the singular values for the linear dynamical propogator
        """
        return self._sigma
    
    
    def _fit(self, dataloader, dim=20):
        
        """
        Train the linear cca model with the dataloader for one step.
        
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            The dataloader contains the traning data.
        dim: int, default=20
            The dimensionalities of the order parameters;
        """
        self._pastmean, self._futuremean = _calculate_mean(dataloader)
        self._c00, self._c01, self._c11 = covariance(dataloader=dataloader, shrinkage=self.shrinkage)
        self._c00 = (self._c00 + self._c00.t()) / 2
        self._c11 = (self._c11 + self._c11.t()) / 2
        
        if self.symmetrize:
            self._c00 = (self._c00 + self._c11) * 0.5
            self._c11.copy_(self._c00)
            self._c01 = (self._c01 + self._c01.t()) * 0.5
        
        self._sqrt_c00 = sqrt_root_matrix(self._c00); self._sqrt_c11 = sqrt_root_matrix(self._c11)

        u, _sigma, v = torch.linalg.svd((self._sqrt_c00.mm(self._c01)).mm(self._sqrt_c11))
        if dim == None:
            self.dim = sigma.size()[0] 
        self._decoder = v[:, :dim]
        self._encoder = u.t()[:dim, ]
        self._sigma = _sigma

        if self.kinetic_mapping:
            self._encoder = torch.diag(self._sigma[:dim]).mm(self._encoder)
        else:
            self._decoder = self._decoder.mm(torch.diag(self._sigma[:dim]))
        self._koopman_matrix = self._decoder.mm(self._encoder)
        
    
    def fit(self, dataloader, dim):
        """
        Train the linear cca model with the dataloader for one step.
                
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            The dataloader contains the traning data.
        dim: int, default=20
            The dimensionalities of the order parameters;
            
        Retunrs
        ----------
        encoder: torch.Tensor
            the transformation functions to project the data on the order parameters
        decoder: torch.Tensor
            the transformation functions to reconstruct the data from low-dimensional order parameters
        koopman_matrix: torch.Tensor
            the linear dynamical propagator fitted from linear cca method
        
        """
        self._fit(dataloader=dataloader, dim=dim)
        return self._encoder, self._decoder, self._koopman_matrix
    
    
    def transform(self, dataloader):
        """
        Trasnform the data in dataloader by the trained model
        
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            The dataloader contains the data to be transformed.
        """
        project_pastdata = []; project_futuredata = []
        for pastdata, futuredata in dataloader:
            pastdata.sub_(self._pastmean[None, :]); futuredata.sub_(self._futuremean[None, :])
            pastdata = pastdata.mm(self._sqrt_c00.t()); futuredata = futuredata.mm(self._sqrt_c11.t())
            project_pastdata.append(pastdata.mm(self._encoder.t())); project_futuredata.append(futuredata.mm(self._encoder.t()))
        project_pastdata = torch.cat(project_pastdata); project_futuredata = torch.cat(project_futuredata)
        return project_pastdata.numpy(), project_futuredata.numpy()
    
    
    def score(self, dataloader):
        """
        Score the reconstruction error of the model on the dataloader; 
        The MSE error will be calculated based on the recontruction from the low-dimensional order parameters
        
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            The dataloader contains the data to be evaluated.
        """
        self.loss_function = nn.MSELoss(reduction='sum')

        loss = 0.0
        for pastdata, futuredata in dataloader:
            pastdata.sub_(self._pastmean[None, :]); futuredata.sub_(self._futuremean[None, :])
            pastdata = pastdata.mm(self._sqrt_c00); futuredata = futuredata.mm(self._sqrt_c11)
            loss += self.loss_function(pastdata.mm(self._koopman_matrix), futuredata).item()
        loss /= len(dataloader.dataset)
        return loss
    
