import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import Score_pooling

class FLV(nn.Module):
    def __init__(self, input_dim, base_model, output_dim=1, num_references=100, pooling_mode='max'):
        super(FLV, self).__init__()
        self.dim_emb = 64
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.dim_emb)
        self.dropout = nn.Dropout(p=0.5)
        self.score_pooling = Score_pooling(input_dim=self.dim_emb, output_dim=output_dim, pooling_mode=pooling_mode, net='MInet')
        # self.fcq = nn.Linear(input_dim, 32)
        # self.fck = nn.Linear(input_dim, 32)
        # self.fcv = nn.Linear(input_dim, 32)
        self.num_references=num_references
        self.base_model = base_model
        self.self_att = SelfAttention(self.dim_emb)
        self.fc = nn.Linear(self.dim_emb, output_dim)

    def forward(self, x, tr_bags, tr_mask):
        # emb = self.base_model.calculate_embedding2(x).unsqueeze(0) #1x64
        emb2 = self.calculate_embedding3(x).unsqueeze(0) #1x64
        tr_bags_tensor = torch.stack(tr_bags, dim=0) #num_refs x 64

        # emb2 = self.self_att(emb, tr_bags_tensor)[0] #1x64
        # emb2 = self.self_att(emb2, tr_bags_tensor)[0]
        # emb2 = self.self_att(emb2, tr_bags_tensor)[0]

        Y_prob = torch.sigmoid(self.fc(emb2)).squeeze()
        return Y_prob, emb2
    
    # def forward(self, x, tr_bags, tr_mask):
    #     if x.dim() > 2:
    #         x = x.squeeze(0)

    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = F.relu(self.fc3(x))
    #     x = self.dropout(x)

    #     Y_prob, emb = self.score_pooling(x)
    #     Y_prob, emb = Y_prob.squeeze(), emb.squeeze() 
    #     Y_hat = torch.ge(Y_prob, 0.5).float()

    #     return Y_prob, Y_hat
    
    def calculate_embedding3(self, x):
        if x.dim() > 2:
            x = x.squeeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        Y_prob, emb = self.score_pooling(x)
        Y_prob, emb = Y_prob.squeeze(), emb.squeeze() 
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return emb
    
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        # self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter((torch.zeros(1)).cuda())
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        # self.gamma_att = nn.Parameter((torch.ones(1)).cuda())
        self.gamma_att = nn.Parameter(torch.ones(1))

    def forward(self, q, x):
        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        q = q.view(1, q.shape[0], q.shape[1]).permute((0, 2, 1))
        bs_q, C_q, length_q = q.shape
        proj_query = self.query_conv(q).view(bs_q, -1, length_q).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1, length)  # B X C x (*W*H)

        energy = torch.bmm(proj_query, proj_key)  # transpose check

        attention = self.softmax(energy)  # BX (N) X (N)

        proj_value = self.value_conv(x).view(bs, -1, length)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(bs, C, length)
        out = out.view(bs_q, C_q, length_q)

        out = self.gamma * out + q
        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att