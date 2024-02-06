import numpy as np
import torch
import scipy.io
from sklearn.model_selection import KFold
from utils.utils import read_yaml
import os.path as osp

def create_bags_mat(data):
    ids=data['bag_ids'][0]
    f=scipy.sparse.csr_matrix.todense(data['features'])
    l=np.array(scipy.sparse.csr_matrix.todense(data['labels']))[0]
    bags=[]
    labels=[]
    for i in set(ids):
        bags.append(np.array(f[ids==i]))
        labels.append(0 if l[ids==i][0] == -1 else 1)
    bags=np.array(bags, dtype=object)
    labels=np.array(labels)
    return bags, labels

def load_elephant_fox_tiger(args, data, seed):
    bags, labels = create_bags_mat(data)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
    datasets = []
    for train_idx, test_idx in kf.split(bags, labels):
        dataset = {}
        dataset['train'] = [(bags[ibag], [labels[ibag] for i in range(len(bags[ibag]))]) for ibag in train_idx]
        dataset['test'] = [(bags[ibag], [labels[ibag] for i in range(len(bags[ibag]))]) for ibag in test_idx]
        datasets.append(dataset)
    return datasets

def load_musk_messidor(args, data, seed):
    ins_fea = data['x']['data'][0,0]
    if args.dataset.startswith('musk'):
        bags_nm = data['x']['ident'][0,0]['milbag'][0,0]
    else:
        bags_nm = data['x']['ident'][0,0]['milbag'][0,0][:,0]
    bags_label = data['x']['nlab'][0,0][:,0] - 1
    # L2 norm for musk1 and musk2
    if args.dataset.startswith('newsgroups') is False:
        mean_fea = np.mean(ins_fea, axis=0, keepdims=True)+1e-6
        std_fea = np.std(ins_fea, axis=0, keepdims=True)+1e-6
        ins_fea = np.divide(ins_fea-mean_fea, std_fea)
    # store data in bag level
    ins_idx_of_input = {}            # store instance index of input
    for id, bag_nm in enumerate(bags_nm):
        if bag_nm in ins_idx_of_input: ins_idx_of_input[bag_nm].append(id)
        else:                                ins_idx_of_input[bag_nm] = [id]
    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        bag_fea = ([], [])
        for ins_idx in ins_idxs:
            bag_fea[0].append(ins_fea[ins_idx])
            bag_fea[1].append(bags_label[ins_idx])
        bags_fea.append(bag_fea)
    # random select 90% bags as train, others as test
    num_bag = len(bags_fea)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=seed)
    datasets = []
    for train_idx, test_idx in kf.split(bags_fea):
        dataset = {}
        dataset['train'] = [bags_fea[ibag] for ibag in train_idx]
        dataset['test'] = [bags_fea[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets


def load_dataset(args, seed):
    cfg = read_yaml(args.config)
    assert args.dataset in cfg.Data.keys()

    # path = cfg.Data[args.dataset].path
    path = osp.join(cfg.General.data_dir, cfg.Data[args.dataset].path)
    data = scipy.io.loadmat(path)
    if args.dataset in ('musk1', 'musk2', 'messidor'):
        return load_musk_messidor(args, data, seed)
    return load_elephant_fox_tiger(args, data, seed)

def convertToBatch(bags):
    data_set = []
    for ibag, bag in enumerate(bags):
        batch_data = torch.tensor(np.array(bag[0]), dtype=torch.float32)
        batch_label = torch.tensor(np.array(bag[1]), dtype=torch.float32)
        data_set.append((batch_data, batch_label))
    return data_set
