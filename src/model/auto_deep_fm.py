from __future__ import print_function
from abc import abstractmethod
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import os

from typing import Optional
# from src.config.tf_config import tf_config
# from src.utils import split_data_mask, layer_normalization, activate
from config.tf_config import tf_config
from utils import split_data_mask, layer_normalization, activate

dtype = tf_config['dtype']

if dtype.lower() == 'float32' or dtype.lower() == 'float':
    dtype = torch.float32
elif dtype.lower() == 'float64':
    dtype = torch.float64


class Model:
    inputs: Optional[torch.Tensor] = None
    outputs: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    learning_rate: Optional[float] = None
    loss: Optional[torch.Tensor] = None
    l2_loss: Optional[torch.Tensor] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    grad: Optional[torch.Tensor] = None

    @abstractmethod
    def compile(self, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__


def generate_pairs(ranges=range(1, 100), mask=None, order=2):
    res = []
    for i in range(order):
        res.append([])
    for i, pair in enumerate(list(combinations(ranges, order))):
        if mask is None or len(mask) == 0 or mask[i] == 1:
            for j in range(order):
                res[j].append(pair[j])
    return res


def save_mask(comb_mask, third_comb_mask=None):
    base_dir = os.path.abspath(os.path.dirname('__file__'))
    file_name = os.path.join(base_dir, 'comb_mask.npz')
    if third_comb_mask:
        np.savez(file_name, comb_mask=comb_mask, third_comb_mask=third_comb_mask)
    else:
        np.savez(file_name, comb_mask=comb_mask)

class AutoDeepFM(nn.Module):
    def __init__(self, init='xavier', num_inputs=None, input_dim=None, embed_size=None, l2_w=None, l2_v=None,
                 layer_sizes=None, layer_acts=None, layer_keeps=None, layer_l2=None, norm=False, real_inputs=None,
                 batch_norm=False, layer_norm=False, comb_mask=None, weight_base=0.6, third_prune=True,
                 comb_mask_third=None, weight_base_third=0.6, retrain_stage=0, batch_size:int= None, device:str= None):
        super(AutoDeepFM, self).__init__()
        self.l2_w = l2_w
        self.l2_v = l2_v
        self.l2_ps = l2_v
        self.layer_l2 = layer_l2
        self.retrain_stage = retrain_stage
        self.training = True
        self.layer_keeps = layer_keeps
        self.init = init
        self.norm = norm
        self.num_inputs = num_inputs
        self.real_inputs = real_inputs
        self.input_dim = input_dim
        self.embed_size = embed_size
        self.third_prune = third_prune
        self.device = device
        self.xw = None
        self.xv = None
        if third_prune:
            self.xps = None
        
        self.xw_embed = FeaturesEmbedding(init=init, input_dims=num_inputs, embed_dim=embed_size)
        self.xv_embed = FeaturesEmbedding(init=init, input_dims=input_dim, embed_dim=embed_size, expand=True)
        if third_prune:
            self.xps_embed = FeaturesEmbedding(init=init, input_dims=input_dim, embed_dim=embed_size, expand=True)
        
        self.bin_mlp = MLP(init, layer_sizes, layer_acts, layer_keeps, num_inputs * embed_size,
                                            batch_norm=batch_norm, layer_norm=layer_norm)
        self.linear = nn.Linear(embed_size, 1)
        # # Combining 2-way interactions
        self.level_2_matrix = level_2_matrix(num_inputs, comb_mask, weight_base)

        if third_prune:
            self.level_3_matrix = level_3_matrix(num_inputs, comb_mask_third, weight_base_third, self.device)

    def forward(self, input):
        inputs, mask, flag, num_inputs = split_data_mask(input, self.num_inputs, norm=self.norm, real_inputs=self.real_inputs)
        xw = self.xw_embed(inputs, mask, apply_mask=flag)
        xv = self.xv_embed(inputs, mask, apply_mask=flag)
        self.xw, self.xv = xw, xv

        if self.third_prune:
            xps = self.xps_embed(inputs, mask, apply_mask=flag)
            self.xps = xps
        h = self.bin_mlp(xv)
        l = self.linear(xw)
        fm_out = self.level_2_matrix(xv)
        
        if self.third_prune:
            fm_out2 = self.level_3_matrix(xps)
        if self.third_prune:
            outputs = torch.sum(torch.stack([l, fm_out, fm_out2, h]), dim=0)
        else:
            outputs = torch.sum(torch.stack([l, fm_out, h]), dim=0)
        return outputs.squeeze()

    def analyse_structure(self, print_full_weight=False, epoch=None):
        wts = self.level_2_matrix.edge_weights.cpu().detach().numpy()
        mask = self.level_2_matrix.edge_weights.cpu().detach().numpy()
        if print_full_weight:
            outline = ""
            for j in range(wts.shape[0]):
                outline += str(wts[j]) + ","
            outline += "\n"
            print("log avg auc all weights for(epoch:%s)" % (epoch), outline)
        print("wts", wts[:10])
        print("mask", mask[:10])
        zeros_ = np.zeros_like(mask, dtype=np.float32)
        zeros_[mask == 0] = 1
        print("masked edge_num", sum(zeros_))
        comb_mask = np.zeros_like(mask, dtype=np.float32)
        comb_mask[mask > 0] = 1
        comb_mask_third = None
        if self.third_prune:
            wts_third = self.level_3_matrix.third_edge_weights.cpu().detach().numpy()
            mask_third = self.level_3_matrix.third_edge_weights.cpu().detach().numpy()
            if print_full_weight:
                outline = ""
                for j in range(wts.shape[0]):
                    outline += str(wts[j]) + ","
                outline += "\n"
                print("third log avg auc all third weights for(epoch:%s)" % (epoch), outline)
            print("third wts", wts_third[:10])
            print("third mask", mask_third[:10])
            zeros_ = np.zeros_like(mask_third, dtype=np.float32)
            zeros_[mask_third == 0] = 1
            print("third masked edge_num", sum(zeros_))
            comb_mask_third = np.zeros_like(wts_third, dtype=np.float32)
            comb_mask_third[mask_third > 0] = 1
        return comb_mask, comb_mask_third
    
    def save_mask(comb_mask, third_comb_mask = None):
        import os
        base_dir = os.path.abspath(os.path.dirname('__file__'))
        file_name = os.path.join(base_dir, 'comb_mask.npz')
        if third_comb_mask:
            np.savez(file_name, comb_mask=comb_mask, third_comb_mask=third_comb_mask)
        else:
            np.savez(file_name, comb_mask=comb_mask)

class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, init, input_dims:int = 0, embed_dim:int=0, expand:bool = False):
        super().__init__()
        self.expand = expand
        if expand == True:
            self.embedding = nn.Embedding(input_dims, embed_dim)
        else:
            self.embedding = nn.Linear(input_dims, embed_dim)
        if init == 'xavier':
            self.init_embedding()

    def forward(self, x, mask, apply_mask:bool = False):
        if self.expand:
            embed = self.embedding(x.long())
        else:
            embed = self.embedding(x)
        if apply_mask == True:
            embed = embed * mask
        return embed
    
    def init_embedding(self):
        shape = self.embedding.weight.data.shape
        maxval = np.sqrt(6. / np.sum(shape))
        torch.nn.init.xavier_uniform_(self.embedding.weight.data, gain=maxval)

class MLP(torch.nn.Module):
    def __init__(self, init, layer_sizes, layer_acts, layer_keeps, node_in, batch_norm=False, layer_norm=False,
            res_conn=False):
        super().__init__()
        self.node_in = node_in
        layers = list()
        for i in range(len(layer_sizes)):
            layers.append(nn.Linear(node_in, layer_sizes[i]))
            if i < len(layer_sizes) - 1:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i]))
                elif layer_norm:
                    # layers.append(layer_normalization(h, out_dim=layer_sizes[i], bias=False))
                    pass
                layers.append(activate(layer_acts[i]))
                layers.append(nn.Dropout())
            node_in = layer_sizes[i]
        self.mlp = torch.nn.Sequential(*layers)
        if init == 'xavier':
            self.init_mlp(self.mlp)
    
    def forward(self, x):
        h = x.view(-1, self.node_in)
        output = self.mlp(h)
        return output
            
    def init_mlp(self, mlp):
        for layer in mlp:
            if type(layer) == torch.nn.Linear:
                shape = [layer.in_features, layer.out_features]
                maxval = np.sqrt(6. / np.sum(shape))
                torch.nn.init.xavier_uniform_(layer.weight.data, gain=maxval)
                
class level_2_matrix(nn.Module):
    def __init__(self, num_inputs:int=None, comb_mask=None, weight_base:float=0.6):
        super().__init__()
        self.comb_mask = comb_mask
        length, _ = generate_pairs(range(num_inputs), mask=self.comb_mask)
        self.length = len(length)
        self.batchnorm = nn.BatchNorm1d(self.length)
        self.edge_weights = nn.Parameter(torch.Tensor(self.length).uniform_(weight_base - 0.001, weight_base + 0.001))
        
    def forward(self, x):
        cols, rows = generate_pairs(range(x.shape[1]), mask=self.comb_mask)
        self.cols, self.rows = torch.tensor(cols), torch.tensor(rows)
        t_embedding_matrix = x.permute(1, 0, 2)
        left = t_embedding_matrix[self.rows,:,:].permute(1, 0, 2)
        right = t_embedding_matrix[self.cols,:,:].permute(1, 0, 2)
        level_2_matrix = torch.sum(left * right, dim=-1)
        level_2_matrix = self.batchnorm(level_2_matrix)

        # Weight pruning for 2-way interactions
        mask = self.edge_weights.unsqueeze(0)
        level_2_matrix *= mask
        fm_out = level_2_matrix.sum(-1, keepdim=True)
        return fm_out
    
class level_3_matrix(nn.Module):
    def __init__(self, num_inputs:int=None, comb_mask_third=None, weight_base_third:float=0.6, device=None): #self.xv, self.xv.shape[1], comb_mask, weight_base
        super().__init__()
        self.comb_mask_third = comb_mask_third
        length, _ = generate_pairs(range(num_inputs), mask=self.comb_mask_third )
        self.length = len(length)
        self.third_edge_weights = torch.Tensor(self.length).uniform_(weight_base_third - 0.001, weight_base_third + 0.001)
        self.third_edge_weights = nn.Parameter(nn.functional.normalize(self.third_edge_weights, p=2, dim=0),requires_grad=True)
        self.third_edge_weights.unpruned_mask = self.third_edge_weights.clone().detach().to(device)
        self.batchnorm = nn.BatchNorm1d(self.length, momentum=0.1, eps=1e-05, track_running_stats=True)
        
        
    def forward(self, x):
        first, second, third = generate_pairs(range(x.shape[1]), mask=self.comb_mask_third, order=3)
        t_embedding_matrix = x.permute(1, 0, 2)
        first_embed = t_embedding_matrix[first].permute(1, 0, 2)
        second_embed = t_embedding_matrix[second].permute(1, 0, 2)
        third_embed = t_embedding_matrix[third].permute(1, 0, 2)
        level_3_matrix = (first_embed * second_embed * third_embed).sum(-1)
        level_3_matrix *= self.third_edge_weights.unpruned_mask.squeeze(0)
        fm_out2 = level_3_matrix.sum(-1, keepdim=True)
        return fm_out2