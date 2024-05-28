import time, argparse, torch
import numpy as np
import torch.optim as optim

from utils import load_data
from models import AHDSLE

import logging
logger = logging.getLogger(__name__)
import logutil

def train(model, epoch):
    tic = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj, adj_v, adj_e, PeT,args.wv)

    loss_train = 0
    Mean = 0

    output_train = output[idx_train]
    labels_train = labels[idx_train]
    labels_train_pos = torch.where(labels_train==1)
    labels_train_nav = torch.where(labels_train==0)
    S = output_train[labels_train_pos]
    S_ = output_train[labels_train_nav]
    scores = {"S": S, "S_": S_}
    Mean = torch.sum(scores['S_'])/(len(scores['S_']))

    M_train = model.metrics(scores)
    loss_train = M_train['loss']
    auc_train = M_train['auc']

    M_train['loss'].backward()
    optimizer.step()

    logger.info('Epoch: {:04d}'.format(epoch+1),
          '[Train] loss: {:.4f}, '.format(loss_train.item()),
          'auc: {:.4f}, '.format(auc_train.item()),
          'time: {:.4f}s'.format(time.time() - tic))
    M_val = None
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features,adj, adj_v, adj_e, PeT)
        # val
        output_val = output[idx_val]
        labels_val = labels[idx_val]
        labels_val_pos = torch.where(labels_val==1)
        labels_val_nav = torch.where(labels_val==0)
        S = output_val[labels_val_pos]
        S_ = output_val[labels_val_nav]
        scores = {"S": S, "S_": S_}
        
        M_val = model.metrics(scores)
        loss_val = M_val['loss']
        auc_val = M_val['auc']
        
        logger.info('Epoch: {:04d}'.format(epoch+1),
            '[Val] loss: {:.4f}, '.format(loss_val.item()),
            'auc: {:.4f}, '.format(auc_val.item()),
            'time: {:.4f}s'.format(time.time() - tic))

    return Mean,M_train,M_val

def test(model, Mean):
    model.eval()
    with torch.no_grad():
        output = model(features,adj, adj_v, adj_e, PeT,args.wv)

    output_test = output[idx_test]
    labels_test = labels[idx_test]
    labels_test_pos = torch.where(labels_test==1)
    labels_test_nav = torch.where(labels_test==0)
    S = output_test[labels_test_pos]
    S_ = output_test[labels_test_nav]
    k = int(len(S)/2)
    scores = {"S": S, "S_": S_}

    test = {'flag': True, 'm': Mean,'k': k}
    M = model.metrics(scores,test)
    return M

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=200, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--fastmode', type=int, default=1, help='Validate during training pass.')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--train_ratio', type=float, default=0.2, help='Train ratio.')
    parser.add_argument('--dataset', type=str, default="iAF1260b", help='Name of dataset')
    parser.add_argument('--wv', type=float, default=0.5, help='Hyperparameters of w_v')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--re_calc', type=int, default=1, help='recalculate the dataset')

    return parser.parse_args()

if __name__ == '__main__':
    
    logger=logutil.logs()
    # Training settings
    args = parse()

    logger.info('task start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    logger.info('===================args============================')
    for k,v in sorted(vars(args).items()):
        logger.info(k,' = ',v)
    logger.info('===================================================')

    args.cuda = torch.cuda.is_available()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    D = load_data(args)
    adj_v       = D['adj_v']
    adj_e       = D['adj_e']
    adj         = D['adj']
    PeT         = D['PeT']
    features    = D['features']
    labels      = D['labels']
    idx_train   = D['idx_train']
    idx_val     = D['idx_val']
    idx_test    = D['idx_test']


    model = AHDSLE(nfeat=features.shape[1],nhid=args.hidden,nclass=adj.shape[0],dropout=args.dropout)

    if args.cuda:
        features    = features.cuda(args.gpu)
        adj         = adj.cuda(args.gpu)
        PeT         = PeT.cuda(args.gpu)
        labels      = labels.cuda(args.gpu)
        idx_train   = idx_train.cuda(args.gpu)
        idx_val     = idx_val.cuda(args.gpu)
        idx_test    = idx_test.cuda(args.gpu)
        model.cuda(args.gpu)

    optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)

    # train model
    tic = time.time()
    auc_train = []
    auc_test = []
    rk_test = []
    for epoch in range(args.epochs):
        Mean,M_train,M_val = train(model, epoch)

        auc_train.append(M_train['auc'])

        M_test = test(model, Mean)
        auc_test.append(M_test['auc'])
        rk_test.append(M_test['r@k'])

        logger.info("            [Test ] loss= {:.4f}, ".format(M_test['loss'].item()),
                    '\033[0;31;40mauc: {:.4f}, '.format(M_test['auc'].item()),
                    "r@k: {:.4f}\033[0m".format(M_test['r@k']))

    logger.info("Test best_auc = [%d] %f, bset_r@k = [%d] %f " % (auc_test.index(max(auc_test)),max(auc_test),rk_test.index(max(rk_test)),max(rk_test)))

    logger.info('======================= All Done ======================')