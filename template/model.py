import torch
import torch.optim as optim
import torchvision.transforms as tfs
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models.Attnet import Attnet
from models.MI_net import MINet
from models.minet import MiNet
from template.template import TemplateModel
import tensorboardX as tX
from utils.utils import read_yaml
import os.path as osp

class Model(TemplateModel):

    def __init__(self, args=None):
        super().__init__()
        self.args = args
        cfg = read_yaml(args.config)

        self.writer = tX.SummaryWriter(log_dir=cfg.General.log_dir, comment=args.model)
        self.train_logger = None
        self.eval_logger = None

        self.step = 0
        self.epoch = 0
        self.best_error = float('Inf')

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.model, self.optimizer = self.get_model(cfg)
        self.model = self.model.to(self.device)
        self.criterion = self.get_criterion(cfg)
        self.metric = metric

        self.train_loader = args.train_loader
        self.test_loader = args.test_loader

        # self.ckpt_dir = cfg.Models[args.model].ckpt_dir
        self.ckpt_dir = cfg.Data[args.dataset].Models[args.model].ckpt_dir
        self.log_per_step = cfg.General.log_per_step
        # self.eval_per_epoch = cfg.General.eval_per_epoch
        self.eval_per_epoch = args.eval_per_epoch

        # self.best_model_path = cfg.Models[args.model].ckpt_dir + '/best.pth.tar'
        self.best_model_path = cfg.Data[args.dataset].Models[args.model].ckpt_dir + '/best.pth.tar'

        self.check_init()

    def get_model(self, cfg):
        assert self.args.model in ('minet', 'MInet', 'attnet')

        input_dim = cfg.Data[self.args.dataset].input_dim
        model_params = cfg.Data[self.args.dataset].Models[self.args.model]
        if self.args.model == 'attnet':
            model = Attnet(input_dim)
            # if model_params.weight_std:
            #     torch.nn.init.normal_(model.weight, mean=0, std=model_params.weight_std)
            optimizer = optim.SGD(model.parameters(), lr=5e-4, weight_decay=0.005, momentum=0.9, nesterov=True)
        elif self.args.model == 'minet':
            model = MiNet(input_dim, pooling_mode=model_params.pooling_mode)
            # optim.Adam(self.model.parameters(), lr=cfg[args.model].lr)
            optimizer = optim.SGD(model.parameters(), 
                                  lr=model_params.lr, 
                                  weight_decay=model_params.weight_decay, 
                                  momentum=model_params.momentum, 
                                  nesterov=model_params.nesterov)
        elif self.args.model == 'MInet':
            model = MINet(input_dim, pooling_mode=model_params.pooling_mode)
            # optim.Adam(self.model.parameters(), lr=cfg[args.model].lr)
            optimizer = optim.SGD(model.parameters(), 
                                  lr=model_params.lr, 
                                  weight_decay=model_params.weight_decay, 
                                  momentum=model_params.momentum, 
                                  nesterov=model_params.nesterov)
        # elif args.model == 'SA-ABMILP':
        #     model = SA_ABMILP(True, self.args.input_dim)
        #     # optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.9, 0.999), weight_decay=10e-5,)
        else:
            print("ERRORE: nome modello errato!")
            exit(1)
        return model, optimizer
    
    def get_criterion(self, cfg):
        model_params = cfg.Data[self.args.dataset].Models[self.args.model]
        if model_params.criterion == 'cross-entropy':
            criterion = torch.nn.CrossEntropyLoss()
        elif model_params.criterion == 'binary-cross-entropy':
            criterion = torch.nn.BCELoss()
        elif model_params.criterion == 'neg-log-likelihood':
            criterion = lambda Y_prob, Y: -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob)) 
        else:
            print("ERRORE: nome criterio errato!")
            exit(1)
        return criterion
    
    def test_best_model(self):
        assert osp.exists(self.best_model_path)

        best_model = Model(self.args)
        best_model.load_state(self.best_model_path)        
        error, _ = best_model.eval_error()
        return 1-error


def metric(pred, target):
    # pred = torch.argmax(pred, dim=0)
    pred = torch.ge(pred, 0.5).float()
    correct_num = torch.sum(pred == target).item()
    total_num = target.size(0)
    accuracy = correct_num / total_num
    return 1. - accuracy