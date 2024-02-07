import argparse
import csv
import torch
import numpy as np
import os.path as osp
import os

from template.model import Model
from utils.dataset import convertToBatch, load_dataset
from utils.utils import load_checkpoint, read_yaml, save_checkpoint, save_results

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
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--config', default='settings.yaml',type=str)
parser.add_argument('--model', type=str, default='minet', help='Choose b/w minet, MI_net, attnet, sa_abmilp')
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

ckpt_dir = cfg.Data[args.dataset].Models[args.model].ckpt_dir
ckpt_file = osp.join(ckpt_dir, 'accs.tar')

if __name__ == "__main__":   
    # accs = np.zeros((args.run, args.folds), dtype=float)
    accs, curr_run, curr_fold = load_checkpoint(args.run, args.folds, ckpt_file)
    seeds = [seed+i*5 for i in range(args.run)]
    for irun in range(curr_run, args.run):
        dataset = load_dataset(args, seeds[irun])
        for ifold in range(curr_fold, args.folds):
            print(f"\nRUN {irun+1}/{args.run}\t FOLD {ifold+1}/{args.folds}: ----------------------------------------")
            args.train_loader = convertToBatch(dataset[ifold]['train'])
            args.test_loader = convertToBatch(dataset[ifold]['test'])

            model = Model(args)
            for e in range(args.epochs):
                model.train()
            best_model = model.get_best_model()
            accs[irun][ifold] = 1 - best_model.eval_error()[0]
            save_checkpoint(accs, irun, ifold, args.folds, ckpt_file)
        curr_fold = 0

    print('\n\nDone!!!')
    print('FINAL: mean accuracy = ', np.mean(accs))
    print('FINAL: std = ', np.std(accs))

    os.remove(ckpt_file)
    save_results(accs, cfg.General.results_dir, args.dataset, args.model)
