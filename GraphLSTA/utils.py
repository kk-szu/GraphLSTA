import numpy as np
import scipy.sparse as sp
import torch
import random
import math
from torch.nn.parameter import Parameter
import time
import json



def get_adjs(rows,cols,weis,nb_nodes):
    l = len(rows)
    adjs = [preprocess_adj(sp.csr_matrix((weis[i],(rows[i],cols[i])),shape=(nb_nodes,nb_nodes),dtype=np.float32)) for i in range(l)]
    return adjs

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def hidden_mat_squeeze(input):
    temp = input.permute(1,2,0)
    c_t = input.permute(1,2,0)
    tup = []
    for i in range(temp.shape[0]):
        tup.append(temp[i])
    tup = tuple(tup)
    temp = torch.cat(tup,1)
    return temp,c_t

def negative_sampling(heads,tails,adjacency_table,degrees,n,not_neighbor = False):
    h_len = len(heads)
    head_degrees = degrees[heads]
    tail_degrees = degrees[tails]

    bernoulli = head_degrees/(head_degrees+tail_degrees)
    prob_replace_h = np.random.random(h_len) < bernoulli
    sampled_nodes = (np.random.random(h_len) * n).astype(np.int32)

    if not_neighbor:
        # the sampled node should not be the neighbor of the head/tail
        # Need to accelerate
        # The complexity here is O(|E|)
        for ii in range(h_len):
            neighbors = adjacency_table[tails[ii]] if prob_replace_h[ii] else adjacency_table[heads[ii]]
            while (sampled_nodes[ii] in neighbors):
                sampled_nodes[ii] = random.randint(0, n - 1)
    else:
        pass
        # Considering the sparsity of the graph
        # the sampled node is mostly not the neighbor of the head/tail
        # when the graph size is big
    prob_replace_h =prob_replace_h.astype(np.int32)
    prob_replace_t = 1 - prob_replace_h
    heads_neg = sampled_nodes* prob_replace_h + heads * prob_replace_t
    tails_neg = sampled_nodes* prob_replace_t + tails * prob_replace_h

    return heads_neg.astype(np.int32),tails_neg.astype(np.int32)

def weight_variable_glorot(dim):
    
    if len(dim) > 1:
        input_dim = dim[0]
        output_dim = dim[1]
        stdv = math.sqrt(6.0/ (input_dim + output_dim))
        weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        weight.data.uniform_(-stdv, stdv)
        return weight
    else:
        input_dim = dim[0]
        stdv = math.sqrt(6.0/ input_dim)
        weight = Parameter(torch.FloatTensor(input_dim))
        weight.data.uniform_(-stdv, stdv)
        return weight


def getNegSample(size, rows, cols, headtail, degrees,n,not_neighbor=True):
    row_negs, col_negs = [], []
    neg_len = int(len(rows[0])*1)
    for t in range(size):
        row_neg_t,col_neg_t = negative_sampling(rows[t],cols[t],headtail,degrees,n,not_neighbor)
        row_negs.append(row_neg_t[0:neg_len])
        col_negs.append(col_neg_t[0:neg_len])
    return row_negs, col_negs

def AnormalReCall(lab_test,lab_ture):#true for 0 false for 1
    lab_test_t = np.where(lab_test>=0.5,1,0)
    false_idx = np.where(lab_ture==1)
    false_num = len(false_idx[0])
    ReCallRate = sum(lab_test_t[false_idx]) / false_num
    return ReCallRate

def percision(lab_test,lab_ture):#true for 0 false for 1
    lab_test_t = np.where(lab_test>=0.5,1,0)
    false_idx = np.where(lab_test_t==1)
    real_false_num = sum(lab_ture[false_idx])
    per = real_false_num/sum(lab_test_t)
    return per

def getTrainData(train_size,rows,cols,headtail,degrees,n):
    rows_train, cols_train = rows[0:train_size], cols[0:train_size]
    rows_neg, cols_neg = getNegSample(train_size,rows,cols,headtail,degrees,n)#获取负采样的样本
    return rows_train,cols_train,rows_neg,cols_neg

def model_train_semi(model,args,optimizer,device,dataPkg):
    losses = []
    nb_nodes = dataPkg['nb_nodes']
    batch_num_train = dataPkg['batch_num_train']
    train_size = dataPkg['train_size']

    model.train()
    for epoch in range(args.max_epoch):
        memroy_block = torch.zeros(nb_nodes,args.GCNhidden_dim)
        for i in range(batch_num_train):
            start, end = i*args.batch_size, (i+1)*args.batch_size
            if end >= train_size:
                end = train_size
            t0 = time.time()
            losses_train = []
            optimizer.zero_grad()
            memroy_block_t,loss = model.semiForward(memroy_block, device, start, end, dataPkg)
            memroy_block = memroy_block_t.detach()
            loss.backward()
            optimizer.step()
            losses_train.append(loss.item())

        print("Epoch : %03d  |  Loss: %9.8f  | Time: %6.3f " % (epoch, np.average(losses_train), time.time() - t0))
        losses.append(np.average(losses_train))

def model_test(model,args,device,dataPkg):
    model.eval()
    nb_nodes = dataPkg['nb_nodes']
    batch_num_test = dataPkg['batch_num_test']
    test_size = dataPkg['test_size']
    
    memroy_block = torch.zeros(nb_nodes,args.GCNhidden_dim)
    values = []
    for i in range(batch_num_test):
        start, end = i*args.batch_size, (i+1)*args.batch_size
        if end >= test_size:
            end = test_size
        memroy_block_t,value = model.test(memroy_block, device, start, end, dataPkg)
        memroy_block = memroy_block_t.detach()
        values.extend(value)
    return values

def get_sequent_mask(inputs):

    t = inputs.size(0)
    n = inputs.size(1)
    attn_shape = (t,n,t)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask[:,0,:] = 0


    return torch.from_numpy(subsequent_mask) == 0

def getConfig(parser, dataName, anomaly_per):

    with open("./dataConfig.json", "r", encoding="utf-8") as f:
        content = json.load(f)

    key = dataName + '-' + str(anomaly_per)

    parser.add_argument('--dataset', action = 'store', default = content[key]['dataName'],
                    help ='data to use')
    parser.add_argument('--anomaly_per', action = 'store', type = float,
                        default = anomaly_per, help = 'anomaly percent in test data')
    parser.add_argument('--snap_size', action = 'store', type = int, default = content[key]['snap_size'],
                        help = 'edge num per graph')
    parser.add_argument('--train_per', action = 'store',type = float, default = 0.5,
                        help='train percent in data')
    parser.add_argument('--input_dim', action = 'store', type = int, default = content[key]['dim'],
                        help = 'Dimension of feature')

    # ===== training parameters =====
    parser.add_argument('--random_seed', action = 'store', type = int, default = content[key]['seed'],
                        help = 'Random seed for tf np random')
    parser.add_argument('--learning_rate', action = 'store', type = float, default = 0.001,
                        help = 'Learning rate for training')
    parser.add_argument('--max_epoch', action = 'store', type = int, default = content[key]['epoch'],
                        help = 'Number of epochs to train.')
    parser.add_argument('--dropout', action = 'store', type = float, default = content[key]['droupout'],
                        help = 'Dropout rate (1 - keep probability).')
    parser.add_argument('--early_stopping', action = 'store', type = int, default = 70,
                        help = 'Tolerance for early stopping (# of epochs).')
    parser.add_argument('--batch_size', action = 'store', type = int, default = content[key]['batch_size'],
                        help = 'batch_size')
    parser.add_argument('--device', action = 'store', default = 'gpu',
                        help ='run device')


    # ===== model parameters =====
    parser.add_argument('--nb_layer', action = 'store', type = int, default = 3,
                        help = 'layer numbers of GCN')
    parser.add_argument('--GCNhidden_dim', action = 'store', type = int, default = content[key]['dim'],
                        help = 'Dimension of hidden states')
    parser.add_argument('--SFhidden_dim', action = 'store', type = int, default = content[key]['dim'],
                        help = 'Dimension of hidden states')
    parser.add_argument('--h_num', action = 'store', type = int, default = 4,
                        help = 'Number of heads in muti-head attention.')

    # ===== hyper-parameters =====
    parser.add_argument('--weight_decay', action = 'store', type = float, default = content[key]['weight_decay'],
                        help = 'Weight for L2 loss on basic models.')