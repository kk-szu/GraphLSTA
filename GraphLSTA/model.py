import torch
from torch._C import device
import torch.nn as nn
from block import *

class GraphLSTA(nn.Module):

    def __init__(self,args,nb_node):
        super(GraphLSTA,self).__init__()

        self.dropout = args.dropout
        self.input_dim = args.input_dim  # k
        self.GCNhidden_dim = args.GCNhidden_dim # d
        self.SFhidden_dim = args.SFhidden_dim
        self.nb_layer = args.nb_layer
        self.weight_decay = args.weight_decay
        self.learning_rate =args.learning_rate
        self.batch_size = args.batch_size
        self.h = args.h_num
        self.nb_node = nb_node
        self.margin = 0.1

        self.block = nn.ModuleList()
        self.block.append(GCN_block(self.input_dim,self.GCNhidden_dim
                                    ,self.GCNhidden_dim,self.nb_layer,droupout = self.dropout))
        self.block.append(multi_head_attention(self.h,self.nb_node,self.GCNhidden_dim,self.SFhidden_dim,self.GCNhidden_dim,
                                         self.batch_size+1,dropout=self.dropout))
        self.linear1 = nn.Linear(self.GCNhidden_dim, 1, bias=False)
        self.linear2 = nn.Linear(self.GCNhidden_dim, 1, bias=False)
        self.linear3 = nn.Linear(self.GCNhidden_dim*2, 1, bias=True)
        self.func = self.linear
        self.act = nn.Sigmoid()
        self.lossFun = nn.BCELoss()


    def linear(self,a,b):
        ab = torch.cat([a,b],dim = 1)
        #ab = a*b
        o = self.linear3(ab)
        #o = self.linear3(ab) / (self.linear1(a) + self.linear2(b))
        return o.squeeze()

    def sorce(self,h,t):
        #a = (self.func(h,t) - self.miu)*self.beta
        a = self.func(h,t)
        return self.act(a)

    def semiForward(self,memroy_block, device, start, end, dataPkg):
        in_features = dataPkg['features']
        seq_len = end - start
        adjs = dataPkg['adjs_train'][start:end]

        GCN_hiddens, offset = memroy_block.unsqueeze(0).to(device), 1
        for t in range(seq_len):
            current = self.block[0](in_features,adjs[t],device).unsqueeze(0)
            GCN_hiddens = torch.cat([GCN_hiddens,current],dim = 0)
        hidden_ts = self.block[1](GCN_hiddens,device)
        loss = torch.tensor(0,dtype = torch.float32,requires_grad = True).to(device)
        for t in range(seq_len):
            i = t + offset
            loss = loss + self.cal_loss_semi(hidden_ts[i], start, end, t, dataPkg, device)
        mem = hidden_ts[0]

        return mem,loss/seq_len

    def test(self,memroy_block, device, start, end, dataPkg):
        in_features = dataPkg['features']
        seq_len = end - start
        adjs = dataPkg['adjs_test'][start:end]
        rows = dataPkg['rows_test'][start:end]
        cols = dataPkg['cols_test'][start:end]

        GCN_hiddens, offset = memroy_block.unsqueeze(0).to(device), 1
        for t in range(seq_len):
            current = self.block[0](in_features,adjs[t],device).unsqueeze(0)
            GCN_hiddens = torch.cat([GCN_hiddens,current],dim = 0)
        hidden_ts = self.block[1](GCN_hiddens,device)
        posis = []
        for t in range(seq_len):
            i = t + offset
            h_vec, t_vec = hidden_ts[i][rows[t],:], hidden_ts[i][cols[t],:]
            posi = self.sorce(h_vec,t_vec).cpu().detach().numpy()
            posis.append(posi)

        mem = hidden_ts[0]

        return mem,posis

    def cal_loss_semi(self,hidden_t, start, end, t, dataPkg, device):
        rowt = dataPkg['rows_train'][start:end][t]
        colt = dataPkg['cols_train'][start:end][t]
        row_negt = dataPkg['rows_neg'][start:end][t]
        col_negt = dataPkg['cols_neg'][start:end][t]

        margin = torch.tensor(self.margin).to(device)
        h_vec, t_vec = hidden_t[rowt,:], hidden_t[colt,:]
        h_vec_neg, t_vec_neg = hidden_t[row_negt,:], hidden_t[col_negt,:]
        posi = self.sorce(h_vec,t_vec)
        nega = self.sorce(h_vec_neg,t_vec_neg)
        loss_pair = torch.mean((torch.maximum((posi + margin - nega),torch.tensor(0.0).to(device))).type(torch.float32))
        return loss_pair


