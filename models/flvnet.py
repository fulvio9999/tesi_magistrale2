import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MI_net import MINet
from utils.utils import Score_pooling

class FLV(nn.Module):
    def __init__(self, cfg, input_dim, base_model, device, output_dim=1, num_references=100, pooling_mode='max'):
        super(FLV, self).__init__()

        seed = cfg.General.seed
        # np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.dim_emb = 64
        self.num_references=num_references
        self.base_model = base_model.model
        self.self_att = SelfAttention(self.dim_emb, self.dim_emb//8, self.dim_emb, device)
        self.fc = nn.Linear(self.dim_emb, output_dim)

    def forward(self, x, tr_bags, tr_mask):
        # training from zero
        self.base_model.eval()
        with torch.no_grad():
            emb2 = self.base_model(x)[1].unsqueeze(0)

        # #training from basemodel
        # emb2 = self.base_model(x)[1].unsqueeze(0)

        tr_bags_tensor = torch.stack(tr_bags, dim=0) #num_refs x 64

        emb2 = self.self_att(tr_bags_tensor, q = emb2) #1x64
        # emb2 = self.self_att(tr_bags_tensor, q = emb2)
        # emb2 = self.self_att(tr_bags_tensor, q = emb2)

        Y_prob = torch.sigmoid(self.fc(emb2)).squeeze()
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        return Y_prob, emb2
    
class SelfAttention(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int, output_size: int, device):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=feature_size, out_channels=hidden_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=feature_size, out_channels=hidden_dim, kernel_size=1)
        # self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=feature_size, out_channels=output_size, kernel_size=1)
        self.gamma = nn.Parameter((torch.zeros(1)).to(device))
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter((torch.ones(1)).to(device))

    def forward(self, x, q=None):
        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        if q is not None:
            q = q.view(1, q.shape[0], q.shape[1]).permute((0, 2, 1))
            bs_q, C_q, length_q = q.shape
            proj_query  = self.query_conv(q).view(bs_q, -1, length_q).permute(0, 2, 1)  # B X CX(N)
        else:
            proj_query  = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        # dissimilarity_matrix = 1 / (attention + 1e-8)
        # attention = torch.softmax(dissimilarity_matrix, dim=1)

        # attention2 = 1-attention
        # row_sums = torch.sum(attention2, dim=1, keepdim=True)
        # attention = attention2 / row_sums

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(bs, C, length)
        out = out.view(bs_q, C_q, length_q)

        out = self.gamma_att * out + q
        # out = out + q
        return out[0].permute(1, 0)#, attention, self.gamma, self.gamma_att