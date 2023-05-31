import numpy as np
from sklearn import preprocessing,metrics
from utils import *
import pickle
import torch
import os


# Generate a DataDict for the tasks
# =======  Dataset Information  =======
# dataset = 'digg'            # uci   digg
# snaps =6000                 # 1000  6000
# train_percent=0.5           # train/test split
# anomaly_percent =0.01       # anomaly in test 0.10/0.05/0.01
# =====================================


def load_dataset_raw(args):
    dataset = args.dataset
    train_per = args.train_per
    anomaly_per = args.anomaly_per
    snap_size = args.snap_size

    path = './data/' + dataset + '_' + str(train_per) + '_' + str(anomaly_per) + '_' + str(snap_size) + '.pkl'

    if not os.path.exists(path):
        print("no such dataset")
        return
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_dataset(args):
    dataPkg = dict()
    rows,cols,labs,weis,headtail,train_size,test_size,nb_nodes,nb_edges,_ = load_dataset_raw(args)
    degrees = np.array([len(x) for x in headtail])
    features = np.random.rand(nb_nodes,args.input_dim)
    features = preprocessing.normalize(features,axis=1)
    features = torch.from_numpy(features).to(torch.float32)
    adjs = get_adjs(rows,cols,weis,nb_nodes) #adjs 中每个adj都是torch.sp_tensor格式
    adjs_train = adjs[0:train_size]
    adjs_test = adjs[train_size:]
    rows_train,cols_train,rows_neg,cols_neg = getTrainData(train_size,rows,cols,headtail,degrees,nb_nodes)
    rows_test, cols_test = rows[train_size:], cols[train_size:]
    labs_test = labs[train_size:]
    labs_test_all = np.hstack(labs[train_size:])
    batch_num_train = (train_size+args.batch_size-1) // args.batch_size
    batch_num_test = (test_size+args.batch_size-1) // args.batch_size

    dataPkg['train_size'] = train_size
    dataPkg['test_size'] = test_size
    dataPkg['nb_nodes'] = nb_nodes
    dataPkg['features'] = features
    dataPkg['adjs_train'] = adjs_train
    dataPkg['adjs_test'] = adjs_test
    dataPkg['rows_train'] = rows_train
    dataPkg['cols_train'] = cols_train
    dataPkg['rows_neg'] = rows_neg
    dataPkg['cols_neg'] = cols_neg
    dataPkg['rows_test'] = rows_test
    dataPkg['cols_test'] = cols_test
    dataPkg['labs_test'] = labs_test
    dataPkg['labs_test_all'] = labs_test_all
    dataPkg['batch_num_train'] = batch_num_train
    dataPkg['batch_num_test'] = batch_num_test

    return dataPkg
