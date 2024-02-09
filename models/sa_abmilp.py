import torch
import torch.nn as nn
import torch.nn.functional as F

class SA_ABMILP(nn.Module):
    def __init__(self, cfg, input_dim, device, self_att=True):
        super(SA_ABMILP, self).__init__()

        seed = cfg.General.seed
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.FC1 = 256
        self.FC2 = 128
        self.FC3 = 64
        self.ATTENTION_BRANCHES = 1
        self.self_att = self_att
        self.dim_emb = self.FC3

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

        # self.feature_extractor_part1 = nn.Sequential(
        #     nn.Conv2d(1, 20, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2),
        #     nn.Conv2d(20, 50, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2)
        # )

        # self.feature_extractor_part2 = nn.Sequential(
        #     nn.Linear(50 * 4 * 4, self.L),
        #     nn.ReLU(),
        # )

        if self.self_att:
            self.self_att = SelfAttention(self.dim_emb, device)

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

        # H = self.feature_extractor_part1(x)
        # H = H.view(-1, 50 * 4 * 4)
        # H = self.feature_extractor_part2(H)  # NxL
        H = self.feature_extractor_part1(x)
        H = self.feature_extractor_part2(H) 
        H = self.feature_extractor_part3(H) 

        if self.self_att:
            H, self_attention, gamma, gamma_kernel = self.self_att(H)

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        emb = M.squeeze()

        Y_prob = self.classifier(M).squeeze()
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        # Y_hat = torch.ge(Y_prob, 0.5).float()
        return Y_prob, emb


class SelfAttention(nn.Module):
    def __init__(self, in_dim, device):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter((torch.zeros(1)).to(device))
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter((torch.ones(1)).to(device))

    def forward(self, x):
        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x
        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att
