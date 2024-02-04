import argparse
import torch
import numpy as np

from template.model import Model
from utils.dataset import convertToBatch, load_dataset
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
parser.add_argument('--model', type=str, default='minet', help='Choose b/w minet, MInet, attnet')
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
    accs = np.zeros((args.run, args.folds), dtype=float)
    seeds = [seed+i*5 for i in range(args.run)]
    for irun in range(args.run):
        dataset = load_dataset(args, seeds[irun])
        for ifold in range(args.folds):
            print(f"\nRUN {irun+1}/{args.run}\t FOLD {ifold+1}/{args.folds}: ----------------------------------------")
            args.train_loader = convertToBatch(dataset[ifold]['train'])
            args.test_loader = convertToBatch(dataset[ifold]['test'])

            model = Model(args)
            for e in range(args.epochs):
                model.train()
            accs[irun][ifold] = model.test_best_model()

    print('\n\nDone!!!')
    print('FINAL: mean accuracy = ', np.mean(accs))
    print('FINAL: std = ', np.std(accs))