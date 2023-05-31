
from random import weibullvariate
import torch
import torch.nn as nn
from layers import GraphConvolution
import math
from utils import get_sequent_mask

class GCN_block(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, nb_layer,
                act=torch.relu, dropout=0., bias=False, **kwargs):
        super(GCN_block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.nb_layer = nb_layer
        self.bias = bias
        self.dropout = dropout
        self.act = act

        self.GCN_layers = nn.ModuleList()
        self.GCN_layers.append(
                GraphConvolution(
                    self.input_dim,
                    self.hidden_dim,
                    dropout=self.dropout,
                    act=self.act))
        for ii in range(self.nb_layer - 2):
            self.GCN_layers.append(
                GraphConvolution(self.hidden_dim, self.hidden_dim, dropout=self.dropout, act=self.act))
        self.GCN_layers.append(
            GraphConvolution(
                self.hidden_dim,
                self.output_dim,
                dropout=self.dropout,
                act=self.act))

    def forward(self, input, adj, device):
        x = input.to(device)
        adj = adj.to(device)
        for layer in self.GCN_layers:
            x = layer(x, adj)
        return x



class multi_head_attention(nn.Module):
    def __init__(self, h, nb_node, input_dim, SFhidden_dim ,output_dim, block_size ,act = torch.softmax, dropout = 0.1):
        super(multi_head_attention,self).__init__()
        self.nb_node = nb_node
        self.input_dim = input_dim
        self.hidden_dim = SFhidden_dim
        self.block_size = block_size
        self.output_dim = output_dim
        self.h_num = h
        self.act = act
        self.dropout = dropout
        self.w_o = nn.Linear(h*self.hidden_dim, self.output_dim, bias=False)
        self.pe = PositionalEncodingAuto(self.nb_node, self.input_dim, dropout, block_size)
        self.head_attentions = nn.ModuleList()
        for i in range(self.h_num):
            self.head_attentions.append(self_attention(self.nb_node, self.input_dim, self.hidden_dim, self.output_dim, self.block_size, self.act, self.dropout))


    def forward(self, inputs, device,isMask = True):


        if isMask:
            mask = get_sequent_mask(inputs).to(device)
        else:
            mask = None


        inputs = self.pe(inputs,device)
        heads = self.head_attentions[0](inputs,device,mask)
        for i in range(1,self.h_num):
            current_heads = self.head_attentions[i](inputs,device,mask)
            heads = torch.cat([heads,current_heads], dim = 2)
        if self.h_num == 1:
            return heads
        else:
            return self.w_o(heads)



class self_attention(nn.Module):
    def __init__(self, nb_node, input_dim, SFhidden_dim ,output_dim, batch_size ,act = torch.softmax, dropout = 0.1):
        super(self_attention,self).__init__()
        self.nb_node = nb_node
        self.input_dim = input_dim
        self.hidden_dim = SFhidden_dim
        self.output_dim = output_dim
        self.act = act

        self.w_q = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(self.input_dim,self.hidden_dim, bias=False)
        self.w_v = nn.Linear(self.input_dim,self.output_dim, bias=False)
        self.layerNorm = nn.LayerNorm([self.nb_node,self.output_dim])
        

    def forward(self, inputs,device,mask):
        '''
        input : GCN输出的矩阵的集合  T x N x input_dim
        '''
        T = inputs.size(0)
        Q = self.w_q(inputs) #T x N x hidden_dim
        K = self.w_k(inputs) #T x N x hidden_dim
        V = self.w_v(inputs) #T x N x output_dim
        V = V.permute(2,1,0) #output_dim x N x T
        qk = torch.empty([0,self.nb_node,T]).to(device)
        for t in range(T):
            qk_t = torch.sum(Q[t]*K,dim = -1).permute(1,0).unsqueeze(0)
            qk = torch.cat([qk,qk_t],dim = 0)
        scores = qk / math.sqrt(self.hidden_dim) # T x N x T
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = self.act(scores,dim = -1)       # T x N x T
        out = torch.empty([0,self.nb_node,self.output_dim]).to(device)
        for t in range(T):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            out_t = torch.sum(p_attn[t]*V,dim = -1).permute(1,0).unsqueeze(0)
            out = torch.cat([out,out_t],dim = 0)
        #return self.layerNorm(out) # T x N x output_dim
        return out



class PositionalEncodingAuto(nn.Module):
    def __init__(self, nb_node, input_dim, dropout, batch_size):
        super(PositionalEncodingAuto, self).__init__()
        self.n = nb_node
        self.dim = input_dim
        self.dropout = nn.Dropout(p=dropout)
        self.PositionEmbedding=torch.nn.Embedding(batch_size,nb_node*input_dim)
    
    def forward(self,x,device):
        batch_len = x.size(0)
        idx = torch.LongTensor(range(0,batch_len)).to(device)
        pe = self.PositionEmbedding(idx).view(batch_len,self.n,self.dim)
        x = x+pe
        return self.dropout(x)