import os,pickle,torch,math
import numpy as np
import scipy.sparse as sp

from torch.nn.init import xavier_normal_
from SLE import transform


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def catchSplitByRatio(dataLen, ratio):
    trainLen = math.ceil(dataLen*ratio)
    testLen = dataLen - trainLen
    id_train = np.random.choice(np.arange(dataLen), trainLen, replace=False)
    id_val = np.random.choice(np.array(list(set(np.arange(dataLen)) - set(id_train))), testLen, replace=False)

    return id_train, id_val, id_val


def load_data(args):
    print('Loading {} dataset...'.format(args.dataset))

    pkldatapath = os.path.join(os.getcwd(), 'data', args.dataset,args.dataset+'.pkl')
    if args.re_calc == 0:
        if os.path.exists(pkldatapath):
            with open(pkldatapath, 'rb') as h:
                return pickle.load(h)

    featurespath = os.path.join(os.getcwd(), 'data', args.dataset,args.dataset+'.npz')
    
    if os.path.isfile(featurespath): 
        X = sp.load_npz(featurespath)
        features = X.todense()
    else:
        X = xavier_normal_(torch.zeros(args.n, args.d))
    
    # build graph

    pairs_pos_path = os.path.join(os.getcwd(), 'data', args.dataset,args.dataset+'.edges.pos')
    pairs_neg_path = os.path.join(os.getcwd(), 'data', args.dataset,args.dataset+'.edges.neg')
    pairs_pos = np.genfromtxt(pairs_pos_path, dtype=np.int32)
    pairs_neg = np.genfromtxt(pairs_neg_path, dtype=np.int32)
    print("len(pairs_pos) = {}, len(pairs_nav) = {}".format(len(pairs_pos),len(pairs_neg)))    
    print("pos edges num = {}, nav edge num = {}".format(len(np.unique(pairs_pos[:, 1])),len(np.unique(pairs_neg[:, 1]))))
    

    labels = np.concatenate((np.ones(len(np.unique(pairs_pos[:, 1]))),
                             np.zeros(len(np.unique(pairs_neg[:, 1])))))
    labels = encode_onehot(labels)
    pairs = np.concatenate((pairs_pos,pairs_neg))
    pairs = np.unique(pairs,axis=0)

    print ('Loaded edge pairs...')

    features = features[np.unique(pairs[:, 0]).tolist(),:]    
    # transform into LE 
    adj_v, adj_e, Pv, PvT, Pe, PeT = transform(pairs)
    adj = adj_v+adj_e
    print ('get LE adjacency and projections')

    # get dataset split
    idx_train, idx_val, idx_test = catchSplitByRatio(len(labels),args.train_ratio)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    # build symmetric adjacency matrix
    adj_v = adj_v + adj_v.T.multiply(adj_v.T > adj_v) - adj.multiply(adj_v.T > adj_v)
    adj_e = adj_e + adj_e.T.multiply(adj_e.T > adj_e) - adj.multiply(adj_e.T > adj_e)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + 2.0 * sp.eye(adj.shape[0]))
    
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    labels = torch.LongTensor(np.where(labels)[1])
    
    # project features to LE
    features = np.array(Pv @ features)
    features = normalize(features)
    features = torch.FloatTensor(features)
    
    # sparse back projection matrix
    PvT = sparse_mx_to_torch_sparse_tensor(PvT)
    PeT = sparse_mx_to_torch_sparse_tensor(PeT)

    D = {
            'adj_v': adj_v, 
            'adj_e': adj_e, 
            'adj': adj, 
            'PeT': PeT, 
            'features': features, 
            'labels': labels, 
            'idx_train': idx_train, 
            'idx_val': idx_val, 
            'idx_test': idx_test
        }
    with open(pkldatapath, 'wb') as h: 
        pickle.dump(D, h, protocol=pickle.HIGHEST_PROTOCOL)
        
    return D

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = (r_mat_inv).dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def symnormalise(M):
    """
    symmetrically normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    
    d = np.array(M.sum(1))
    
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 

def normalise(M):
    """
    row-normalise sparse matrix
    arguments:
    M: scipy sparse matrix
    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)
