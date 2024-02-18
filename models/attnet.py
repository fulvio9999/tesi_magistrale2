import torch
import torch.nn as nn
import torch.nn.functional as F

class Attnet(nn.Module):
    def __init__(self, cfg, input_dim):
        super(Attnet, self).__init__()

        seed = cfg.General.seed
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.FC1 = 256
        self.FC2 = 128
        self.FC3 = 64
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Linear(input_dim, self.FC1),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.FC1, self.FC2),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.feature_extractor_part3 = nn.Sequential(
            nn.Linear(self.FC2, self.FC3),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.FC3, self.FC3), # matrix V
            nn.Tanh(),
            nn.Linear(self.FC3, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.FC3*self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.squeeze(0)
        # print(x.dim())

        H = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(H) 
        H = self.feature_extractor_part3(H) 

        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = torch.transpose(A, 1, 0)  # ATTENTION_BRANCHESxK
        A = F.softmax(A, dim=1)  # softmax over K

        Z = torch.mm(A, H)  # ATTENTION_BRANCHESxM
        emb = Z.squeeze()

        Y_prob = self.classifier(Z).squeeze()
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, emb#, A