import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class DenseLayer(nn.Module):

    def __init__(self, 
                 c_in, 
                 c_out, 
                 zero_init=False, 
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(
                6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, 
    			node_feats, # input node features
    			):

        node_feats = self.linear(node_feats)

        return node_feats

class SineLayer(nn.Module):
    def __init__(self, c_in, c_out, bias=True, zero_init=False,
                 omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.zero_init = zero_init
        self.in_features = c_in
        self.linear = nn.Linear(c_in, c_out, bias=bias)
        
        if self.zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(
                6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
            nn.init.zeros_(self.linear.bias.data)
            
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class DeconvNet(nn.Module):

    def __init__(self, 
                 hidden_dims,
                 n_celltypes, 
                 n_slices,
                 n_heads, 
                 slice_emb_dim, 
                 adj_dim,
                 coef_fe,
                 ):
        import torch
        import random
        seed=1
        torch.manual_seed(seed) 
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)  
        np.random.seed(seed)  
        random.seed(seed)  
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        super().__init__()
        self.training_steps = 14001
        mid_channel = 200
        
        self.encoder_layer0 = nn.Sequential(SineLayer(3, mid_channel),
                                            SineLayer(mid_channel, mid_channel),
                                            SineLayer(mid_channel, 30),
                                            DenseLayer(30, hidden_dims[0]))
                    
        self.encoder_layer1 = DenseLayer(hidden_dims[0],hidden_dims[2])
        
        self.decoder = nn.Sequential(SineLayer(hidden_dims[2], mid_channel),
                                            DenseLayer(mid_channel, hidden_dims[0]))
        
        self.deconv_alpha_layer = DenseLayer(hidden_dims[2] + slice_emb_dim, 
                                             1, zero_init=True)
        
        self.deconv_beta_layer = nn.Sequential(DenseLayer(hidden_dims[2], n_celltypes, 
                                                          zero_init=True))

        self.gamma = nn.Parameter(torch.Tensor(n_slices, 
                                               hidden_dims[0]).zero_())

        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim) # n_slice, 16

        self.coef_fe = coef_fe

    def forward(self, 
                coord, # n1*3 coordinate matrix
                adj_matrix,
                node_feats, # input node features n1*n2
                count_matrix, # gene expression counts
                library_size, # library size (based on Y)
                slice_label, # slice label
                basis, # basis matrix
                step
                ):
        # encoder
        
        self.node_feats = node_feats
        
        self.coord = coord/100 # Scale the coordinates
        
        Z, mid_fea = self.encoder(node_feats)

        # deconvolutioner
        slice_label_emb = self.slice_emb(slice_label)
        
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)
        
        self.node_feats_recon = self.decoder(Z)

        # deconvolution loss
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[slice_label]
        lam = torch.exp(log_lam)
        self.decon_loss = - 5*torch.mean(torch.sum(count_matrix * 
                                        (torch.log(library_size + 1e-6) + log_lam
                                         ) - library_size * lam, axis=1))
        
        self.fea_loss = 1*torch.norm(node_feats-mid_fea, 2)+2*torch.norm(node_feats-self.node_feats_recon, 2)
        
        # Total loss
        loss = 1*(self.decon_loss + self.fea_loss)  
        
        denoise = torch.matmul(beta, basis)
        return loss, mid_fea, denoise, Z, 0, 0

    def evaluate(self, adj_matrix, coord, node_feats, slice_label):
        slice_label_emb = self.slice_emb(slice_label)
        Z, _ = self.encoder(node_feats)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)
        
        return Z, beta, alpha, self.gamma
            
    def encoder(self, H):
        self.mid_fea = self.encoder_layer0(self.coord)
        Z = self.encoder_layer1(self.mid_fea)
        return Z, self.mid_fea
    
    def deconvolutioner(self, Z, slice_label_emb):
        beta = self.deconv_beta_layer(torch.sin(Z))
        beta = F.softmax(beta, dim=1)
        H = torch.sin(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha
