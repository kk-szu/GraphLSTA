from prepareData import load_dataset
from model import GraphLSTA
from utils import *
from block import *
from sklearn import metrics
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

# ===== dataset Setting =====
dataName = 'alp' # 选择实验数据集
anomaly_per = 0.1 # 选择注入异常比例

# ===== hyperparameter Setting =====
getConfig(parser, dataName, anomaly_per) #根据配置文件注入参数
args = parser.parse_args()


# ===== set device =====
if args.device != 'cpu' and torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# ===== set random seeds =====
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)

# ===== Load data =====
dataPkg= load_dataset(args)

# ===== Model Initialization =====

model = GraphLSTA(args,dataPkg['nb_nodes']).to(device)
optimizer = optim.Adam(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)

# ===== Train =====
model_train_semi(model,args,optimizer,device,dataPkg)

# ===== Test =====
with torch.no_grad():
    values = model_test(model,args,device,dataPkg)

# ===== Print =====
labs_test = dataPkg['labs_test']
labs_test_all = dataPkg['labs_test_all']

#打印每个快照的结果
for t in range(dataPkg['test_size']):
    print("Snap: %02d | AUC: %.6f  recall: %.6f percision: %.6f"% (t,metrics.roc_auc_score(labs_test[t],values[t]),AnormalReCall(values[t],labs_test[t]),percision(values[t],labs_test[t])))

#打印整个数据集的总体结果
values_test = np.hstack(values)
auc_test = metrics.roc_auc_score(labs_test_all, values_test)

print("Test AUC: %.6f"%(auc_test))


