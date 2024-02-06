#---->read yaml
import csv
import yaml
from addict import Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp

def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
    
def write_results(accs):
    import numpy as np
import csv

def save_results(accuracy_matrix, results_dir, dataset, model):
    accuracy_matrix = np.round(accuracy_matrix, 3)
    run = accuracy_matrix.shape[0]
    run_means = np.round(np.mean(accuracy_matrix, axis=1), 3)
    accuracy_matrix_with_run_means = np.column_stack((accuracy_matrix, run_means))
    fold_means = np.round(np.mean(accuracy_matrix_with_run_means, axis=0),3)
    fold_std = np.round(np.std(accuracy_matrix_with_run_means, axis=0), 3)
    accuracy_matrix_with_fold_means = np.row_stack((accuracy_matrix_with_run_means, fold_means))
    accuracy_matrix_with_fold_std = np.row_stack((accuracy_matrix_with_fold_means, fold_std))
    run_std = np.round(np.std(accuracy_matrix_with_fold_std.T[:][:-1].T, axis=1), 3)
    final_matrix = np.column_stack((accuracy_matrix_with_fold_std, run_std))
    column_labels = [f"Fold_{i+1}" for i in range(accuracy_matrix.shape[1])] + ["Acc_run_mean", "Acc_run_std"]

    path = osp.join(results_dir, dataset)
    csv_file_path = osp.join(path, model+".csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_labels)
        for id, row in enumerate(final_matrix):
            if id == run:
                writer.writerow([])
                writer.writerow(row)
            else:
                writer.writerow(row)
    
    acc_mean = np.round(np.mean(accuracy_matrix), 3)
    acc_std = np.round(np.std(accuracy_matrix), 3)

    txt_file_path = osp.join(path, model+".txt")
    with open(txt_file_path, "w") as file:
        file.write(f"ACCURACY MEAN: {acc_mean}\nACCURACY STD: {acc_std}")

    print(f"\nI dati sono stati scritti con successo nei file: \nCSV = {csv_file_path}\nTXT = {txt_file_path}")

def load_checkpoint(run, folds, ckpt_file):
    if not osp.exists(ckpt_file):
        return np.zeros((run, folds), dtype=float), 0, 0
    state = torch.load(ckpt_file)
    accs = state['accs']
    curr_run = state['curr_run']
    curr_fold = state['curr_fold']
    print('load accuracy checkpoint from {}'.format(ckpt_file))
    return accs, curr_run, curr_fold

def save_checkpoint(accs, curr_run, curr_fold, folds, ckpt_file):
    state = {}
    if curr_fold == folds-1:
        state['curr_run'] = curr_run+1
        state['curr_fold'] = 0
    else:
        state['curr_run'] = curr_run
        state['curr_fold'] = curr_fold+1
    state['accs'] = accs
    torch.save(state, ckpt_file)
    print('save accuracy checkpoint at {}'.format(ckpt_file))
    
class Score_pooling(nn.Module):
    def __init__(self, input_dim=64, output_dim=1, pooling_mode='max', net='minet'):
        super(Score_pooling, self).__init__()
        self.output_dim = output_dim
        self.pooling_mode = pooling_mode
        self.fc = nn.Linear(input_dim, output_dim)
        self.net = net
    
    def choice_pooling(self, x):
        if self.pooling_mode == 'max':
            return torch.max(x, dim=0, keepdim=True)[0]
        if self.pooling_mode == 'lse':
            return torch.log(torch.mean(torch.exp(x), dim=0, keepdim=True))
        if self.pooling_mode == 'ave':
            return torch.mean(x, dim=0, keepdim=True)

    def forward(self, x):
        if self.net == 'MInet':
            x = self.choice_pooling(x)
            emb = x

        x = self.fc(x)
        output = torch.sigmoid(x)

        if self.net == 'minet':
            output = self.choice_pooling(output)
            emb = None

        # if self.net == 'minet':
        #     x = self.fc(x)
        #     x = torch.sigmoid(x)
        #     output = self.choice_pooling(x)
        # else: #MI-net
        #     x = self.choice_pooling(x)
        #     x = self.fc(x)
        #     output = torch.sigmoid(x)
        return output, emb