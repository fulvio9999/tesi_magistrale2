import argparse
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch

import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from template.model import Model, Model_with_embs
from utils.dataset import convertToBatch, create_bags_mat, load_dataset
from utils.utils import read_yaml


# Training settings
parser = argparse.ArgumentParser(description='FLV')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--folds', type=int, default=10, metavar='N',
                    help='number of folds for cross validation (default: 10)')
parser.add_argument('--run', type=int, default=5, metavar='N',
                    help='number of run (default: 5)')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
#                     help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--config', default='settings.yaml',type=str)
parser.add_argument('--model', type=str, default='MInet', help='Choose b/w minet, MInet, attnet')
parser.add_argument('--dataset', type=str, default='elephant', help='Choose b/w elephant, fox, tiger, musk1, musk2, messidor')
parser.add_argument('--eval-per-epoch', type=int, default=0, 
                    help='Choose 0 if you do not want to save the best model, otherwise choose the number of times per epoch you want to save the best model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

assert args.config
cfg = read_yaml(args.config)
seed = cfg.General.seed

np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)
    print('\nGPU is ON!')

if __name__ == "__main__":   
    accs_base = np.zeros((args.run, args.folds), dtype=float)
    accs = np.zeros((args.run, args.folds), dtype=float)
    seeds = [seed+i*5 for i in range(args.run)]
    for irun in range(args.run):
        dataset = load_dataset(args, seeds[irun])
        for ifold in range(args.folds):
            print(f"\nRUN {irun+1}/{args.run}\t FOLD {ifold+1}/{args.folds}: ----------------------------------------")
            
            args.train_loader = convertToBatch(dataset[ifold]['train'])
            args.test_loader = convertToBatch(dataset[ifold]['test'])

            print("\nTRAIN BASE MODEL:")
            model = Model(args)
            for e in range(args.epochs):
                model.train()
            best_model = model.get_best_model()
            accs_base[irun][ifold] = 1 - best_model.eval_error()[0]

            args2 = copy.deepcopy(args)
            args2.embeddings = best_model.get_training_features()
            args2.model = 'flvnet'
            args2.base_model = best_model

            print("\nTRAIN FLV MODEL:")
            flv_model = Model_with_embs(args2)
            for e in range(args2.epochs):
                flv_model.train()
            best_model_flv = flv_model.get_best_model()
            accs[irun][ifold] = 1 - best_model_flv.eval_error()[0]

    print('\n\nDone!!!')
    print('FINAL (base): mean accuracy = ', np.mean(accs_base))
    print('FINAL (base): std = ', np.std(accs_base))
    print('\nFINAL: mean accuracy = ', np.mean(accs))
    print('FINAL: std = ', np.std(accs))
            


    

