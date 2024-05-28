import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers import GraphConvolution
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import sparse_mx_to_torch_sparse_tensor

class AHDSLE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(AHDSLE, self).__init__()
        self.w1 = nn.Linear(nhid, nhid)
        self.w2 = nn.Linear(nhid, nhid)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.INT = nn.Linear(nhid, 1)
        
        self.dropout = dropout

    
    def forward(self, x,adj, adj_v,adj_e, PeT,wv):

        adj_v = sparse_mx_to_torch_sparse_tensor(adj_v)
        adj_e = sparse_mx_to_torch_sparse_tensor(adj_e)
        if torch.cuda.is_available():
            adj_v = adj_v.cuda(x.device.index)
            adj_e = adj_e.cuda(x.device.index)


        x = F.relu(self.gc1(x, adj_v*adj*wv)+self.gc1(x, adj_e*adj*(2-wv)))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc2(x, adj_v*adj*wv)+self.gc2(x, adj_e*adj*(2-wv)))
        x = F.dropout(x, self.dropout, training=self.training)

        x = torch.spmm(PeT, x) # 矩阵乘法
        x = torch.sigmoid(self.INT(x))

        return x
    
    def metrics(self, scores, test={'flag': False}):
        
        S, S_ = scores['S'], scores['S_']

        if test['flag']: M = test['m']
        else: M = torch.sum(S_)/(len(S_))

        loss = 0
        for i in range(S.size()[0]): loss += torch.log(1+torch.exp(M - S[i]))

        if S.is_cuda: S, S_ = S.cpu().data.numpy(), S_.cpu().data.numpy()

        Y = [1] * S.shape[0] + [0] * S_.shape[0]
        Z = list(S) + list(S_)

        roc, ap = roc_auc_score(Y, Z), average_precision_score(Y, Z)

        R = {'loss': loss, 'auc': round(roc, 3), 'acc': round(ap, 3)}

        if test['flag']:
            A = np.vstack((S, S_))
            A = A.squeeze(1)
            idx = np.argsort(A)[::-1][:test['k']]
            
            d = S.shape[0]
            n = 0
            for i in [int(j) for j in idx]: 
                if Y[i] == 1: n += 1 
            
            R['r@k'] = round(float(n/d), 3)
        
        return R
