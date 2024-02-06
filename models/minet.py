import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import Score_pooling

class MiNet(nn.Module):
    def __init__(self, cfg, input_dim, output_dim=1, pooling_mode='max'):
        super(MiNet, self).__init__()

        seed = cfg.General.seed
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.dim_emb = 64
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.dim_emb)
        self.dropout = nn.Dropout(p=0.5)
        self.score_pooling = Score_pooling(input_dim=self.dim_emb, output_dim=output_dim, pooling_mode=pooling_mode, net='minet')

    def forward(self, x):
        if x.dim() > 2:
            x = x.squeeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        Y_prob, emb = self.score_pooling(x)
        Y_prob = Y_prob.squeeze()
        # Y_prob, emb = Y_prob.squeeze(), emb.squeeze() 
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, emb
    
