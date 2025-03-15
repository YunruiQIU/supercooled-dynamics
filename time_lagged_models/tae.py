import numpy as np
import random
import os
import torch
from utils import traj2transitionpair, _calculate_mean, covariance, sqrt_root_matrix, whiten_data


class time_lagged_autoencoder(torch.nn.Module):
    
    """
    time-lagged autoencoder (TAE) model which can be fit to identify the best time-lagged correlation order parameters
    between time-lagged time;

    Parameters
    ----------
    input_dim : int, optional, default=None
        The dimensionality (number of features) of the input data
    latent_dim : int, optional, default=2
        The dimensionality (number of order parameters) of the autoencoder latent space
    hidden_dim : int, optional, default=1000
        The number of neural nodes for the hidden layers, by default, only one non-linear layer is utilized for encoder and decoder network
    encoder: torch.nn.Module, default=None
        The encoder neural network for the autoencder, could be input by user
        by default, the encoder is implemented by single non-linear layer with Tanh activation function
    decoder: torch.nn.Module, default=None
        The decoder neural network for the autoencder, could be input by user
        by default, the encoder is implemented by single non-linear layer with Tanh activation function
    device: device : device, default=torch.device("cpu")
        The device for the network and training. Can be None which defaults to CPU.
    learning_rate: float, default=5e-4
        The learning rate used to weight the gradient to update the autoencoder
    optimizer: torch.optim, default: Adam
        The optimization algorithm used to update the network,
        by default, the Adam optimizer is adopted, could be defined by user from outside
    regularization: string, default='None'
        The regularization for the loss function, can be chosen from: 'None', 'L1', 'L2'
    weight_decay: float, default=1e-4
        The weight of the regularization term in the loss function
    schedular: torch.torch.optim.lr_scheduler, default: StepLR
        The schedular used to update the learning rate of optimizer, could be defined by user from outside
        by default, the StepLR with step_size=15 and gamma=0.98 is adopted
    """

    def __init__(self, input_dim=None, latent_dim=2, hidden_dim=1000, encoder=None, decoder=None, device=torch.device("cpu"), 
                 learning_rate=5e-4, optimizer=None, regularization='None', weight_decay=1e-4, schedular=None):

        super(time_lagged_autoencoder, self).__init__()
        assert not(input_dim == None and encoder ==None), "input_dim parameter must be specified without input encoder architecture"
        if encoder != None and input_dim != encoder[0].weight.shape[1]:
            raise ValueError("input_dim should be consistent with the encoder architecture {} != {}".format(input_dim, encoder[0].weight.shape[1]))
        
        if encoder == None:
            self._input_dim = input_dim
            self._latent_dim = latent_dim
            self._encoder = torch.nn.Sequential(
                                    torch.nn.Linear(self._input_dim, self._input_dim*5), torch.nn.Tanh(),
                                    torch.nn.Linear(self._input_dim*5, self._latent_dim))
        else:
            self._input_dim = encoder[0].weight.shape[1]
            self._latent_dim = encoder[-1].weight.shape[0]
            self._encoder = encoder
        
        if decoder == None:
            self._decoder = torch.nn.Sequential(
                        torch.nn.Linear(self._latent_dim, self._input_dim*5), torch.nn.Tanh(),
                        torch.nn.Linear(self._input_dim*5, self._input_dim))
        else:
            self._decoder = decoder
            
        self._encoder.double(); self._decoder.double()
        self._device = device
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._regularization = regularization
        self._weight_decay = weight_decay
        

        if optimizer == None:
            self._optimizer = torch.optim.Adam(list(self._encoder.parameters()) + list(self._decoder.parameters()), lr=self._learning_rate)
        if schedular == None:
            self._scheduler =  torch.optim.lr_scheduler.StepLR(optimizer=self._optimizer, step_size=15, gamma=0.98)
        regularization_list = ['None', 'L1', 'L2']
        assert self._regularization in regularization_list, "Invalid network type "+str(self._regularization)+", should adopt from "+str(regularization_list)
        self._train_accuracy = []; self._validate_accuracy = []
        
        
    @staticmethod
    def set_random_seed(seed=0):

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    
    @property
    def encoder(self):
        
        return self._encoder
    
    
    @encoder.setter
    def encoder(self, encoder_network):
        
        self._encoder = encoder_network
        
        
    @property
    def decoder(self):
        
        return self._decoder
    
    
    @encoder.setter
    def decoder(self, decoder_network):
        
        self._decoder = decoder_network
    
        
    @property
    def optimizer(self):
        
        return self._optimizer
    
    
    @optimizer.setter
    def optimizer(self, opt):
        
        self._optimizer = opt
        
        
    @property
    def schedular(self):
        
        return self._schedular
    
    
    @schedular.setter
    def schedular(self, schedular_input):
        
        self._schedular = schedular_input
    
        
    def forward(self, x=None):
        
        """
        forward function used to transform input data to output of neural network

        Parameters
        ----------
        x: torch.Tensor, default=None
            The input data used to be transformed through the autoencoder neural network

        Returns
        ----------
        latent_x: torch.Tensor, shape: (number of data points, dimension of the order parameters)
            The values of latent space order parameters for embedded data
        reconstruct_x: torch.Tensor, shape: (number of data points, dimension of input data)
            The reconstructed data from decoder neural network
        """

        latent_x = self._encoder(x)
        reconstruct_x = self._decoder(latent_x)
        return latent_x, reconstruct_x
    
    
    def _compute_error(self, dataloader):
        
        """
        compute the reconstructed error of the autoencoder based on the dataloader data

        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
            The input data used to calculate the reconstruction error of autoencoder neural network

        Returns
        ----------
        _accuracy: torch.Tensor, 
            The reconstruction error of the dataloader on the trained autoencoder
        """
                
        self._encoder.eval(); self._decoder.eval()
        _past_data, _future_data = next(iter(dataloader))
        with torch.no_grad():
            _past_data = _past_data.to(self._device); _future_data = _future_data.to(self._device)
            latent_x, reconstruct = self.forward(_past_data)
            _accuracy = torch.nn.MSELoss(reduction='mean')(reconstruct, _future_data).detach().float()
        return _accuracy.cpu()
    
    
    def partial_fit(self, train_loader):
        
        """
        train the TAE model for singe epoch using the train_loader data

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The input data used to train the TAE model for single epoch; different kinds of loss functions are adopted according to 
            different initial setups.

        """
        
        self._encoder.train(); self._decoder.train()
        self._optimizer.zero_grad()
        for past_data, future_data in train_loader:
            past_data = past_data.to(self._device); future_data = future_data.to(self._device)
            latent_x, reconstruct = self.forward(past_data)
            if self._regularization == 'L1':
                _reglurization_loss = torch.tensor(0.,).to(self._device)
                for name, para in self._encoder.named_parameters():
                    if 'weight' in name:
                        _reglurization_loss += torch.norm(para, p=1)
                for name, para in self._decoder.named_parameters():
                    if 'weight' in name:
                        _reglurization_loss += torch.norm(para, p=1)
                _loss = torch.nn.MSELoss(reduction='mean')(reconstruct, future_data) + self._weight_decay * _reglurization_loss
            elif self._regularization == 'L2':
                _reglurization_loss = torch.tensor(0.,).to(self._device)
                for name, para in self._encoder.named_parameters():
                    if 'weight' in name:
                        _reglurization_loss += torch.norm(para, p=2)
                for name, para in self._decoder.named_parameters():
                    if 'weight' in name:
                        _reglurization_loss += torch.norm(para, p=2)
                _loss = torch.nn.MSELoss(reduction='mean')(reconstruct, future_data) + self._weight_decay * _reglurization_loss
            else:
                _loss = torch.nn.MSELoss(reduction='mean')(reconstruct, future_data)

        _loss.backward()
        self._optimizer.step()

        return self
    
    
    def fit(self, train_loader, num_epochs=30, validation_loader=None, print_log=True, shrinkage=True, eps=1e-6):
        
        """
        train the TAE model using the train_loader data

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The input data used to train the TAE model;
        num_epochs: int, default=30
            The number of epochs for the training of the TAE model;
        validation_loader: torch.utils,data.DataLoader, default=None
            The validation dataset used to validate the model, could be None if there is no validation dataset
        print_log: bool, default=True
            The bool value to decide if the training process is printed out;
        shrinkage: bool, default=True
            The bool value to decide if the rao_blackwell_ledoit_wolf method is used to ensure numerical stabilities of the covariance matrix
        eps: float, default=1e-6
            The epsilon value which used for numerical stabilities of the covariance matrix;

        """
        
        self._past_mean, self._future_mean = _calculate_mean(dataloader=train_loader)
        self._c00, self._c0t, self.ctt = covariance(dataloader=train_loader, remove_mean=True, shrinkage=shrinkage)
        self._sqrt_c00 = sqrt_root_matrix(matrix=self._c00, eps=eps); self._sqrt_ctt = sqrt_root_matrix(matrix=self.ctt, eps=eps)
        
        _past_data, _future_data = next(iter(train_loader))
        _past_data -= self._past_mean; _future_data -= self._future_mean
        _past_data = _past_data.mm(self._sqrt_c00); _future_data = _future_data.mm(self._sqrt_ctt)
        
        whiten_traindata = torch.utils.data.TensorDataset(_past_data, _future_data)
        whiten_trainloader = torch.utils.data.DataLoader(dataset=whiten_traindata, batch_size=train_loader.batch_size, shuffle=False, num_workers=1)
        
        if  validation_loader != None:
            _past_validation, _future_validation = next(iter(validation_loader))
            _past_validation -= self._past_mean; _future_validation -= self._future_mean
            _past_validation = _past_validation.mm(self._sqrt_c00); _future_validation = _future_validation.mm(self._sqrt_ctt)
            
            whiten_validatedata = torch.utils.data.TensorDataset(_past_validation, _future_validation)
            whiten_validateloader = torch.utils.data.DataLoader(dataset=whiten_validatedata, batch_size=validation_loader.batch_size, shuffle=False, num_workers=1)
            
        for epoch in range(num_epochs):

            self.partial_fit(whiten_trainloader)
            self._train_accuracy.append(self._compute_error(whiten_trainloader))
            if validation_loader != None:
                self._validate_accuracy.append(self._compute_error(whiten_validateloader))
            if print_log:
                print("==>epoch={}, training process={:.2f}%, the loss on training dataset={:.5f};".format(epoch, 100*(epoch+1)/num_epochs, self._train_accuracy[-1]))
                if validation_loader != None:
                    print("==>epoch={}, training process={:.2f}%, the loss on validation dataset={:.5f};".format(epoch, 100*(epoch+1)/num_epochs, self._validate_accuracy[-1]))
    
            self._scheduler.step()
        
        
    def transform(self, data):
        
        """
        transform the input data to the order parameters

        Parameters
        ----------
        data: torch.Tensor
            The data used to be transformed by the trained TAE model.
        """
        
        data = data.to(self._device)
        data -= self._past_mean; data = data.mm(self._sqrt_c00)
        self._encoder.eval(); self._decoder.eval()
        
        with torch.no_grad():
            latent_x= self._encoder(data)
        return latent_x.detach().cpu()
