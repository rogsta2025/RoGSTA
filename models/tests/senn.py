import torch.nn as nn
import torch

from layers.trend_seasonal import TS_Model

class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.
        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        """
        super(SENNGC, self).__init__()

        # Networks for amortising generalised coefficient matrices.
        #self.coeff_nets = nn.ModuleList()
##
        ## Instantiate coefficient networks
        #for k in range(order):
        #    modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
        #    if num_hidden_layers > 1:
        #        for j in range(num_hidden_layers - 1):
        #            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
        #    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2), nn.Tanh()))
        #    self.coeff_nets.append(nn.Sequential(*modules))

        self.pre_model = TS_Model(seq_len=order, num_nodes=num_vars, d_model=32).to(device)
        self.temporal_weights = nn.Parameter(torch.ones(order)) 
        self.rank = 64  # Number of coefficients per variable
        self.A = nn.Linear(num_vars, self.rank)
        self.B = nn.Linear(self.rank, num_vars*num_vars)

        # Some bookkeeping
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers
        self.device = device


    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forwardaaaaaaa(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        for k in range(self.order):
            coeff_net_k = self.coeff_nets[k]
            coeffs_k = coeff_net_k(inputs[:, k, :])
            coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
            if coeffs is None:
                coeffs = torch.unsqueeze(coeffs_k, 1)
            else:
                coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
            # coeffs[:, k, :, :] = coeffs_k
            preds = preds + torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
        return preds, coeffs
    

    def forward(self, inputs):
        # inputs shape: [batch, order, num_vars]
        B, O, N = inputs.shape
        
        preds = self.pre_model(inputs)  # shape [batch, order, num_vars]
        
        A_proj = self.A(preds)              # [B*O, N, rank]
        B_proj = self.B(A_proj)  # [B*O, rank, N]

        coeffs = B_proj.view(B, O, N, N)
        #preds = preds.mean(dim=1)  # [B, N] (simple temporal pooling)
        preds = preds[:, -1, :]
        #weights = torch.softmax(self.temporal_weights, dim=0)  # [O]
        #preds = (preds * weights.view(1,O,1)).sum(dim=1)
        #preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
        #for k in range(self.order):
        #    preds += torch.matmul(coeffs[:, k, :, :], inputs[:, k, :].unsqueeze(-1)).squeeze(-1)
        return preds, coeffs