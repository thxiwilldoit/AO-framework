import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

def orthogonal_projection(personality, commonality):
    B_norm_squared = torch.sum(commonality ** 2, dim=-1, keepdim=True)
    dot_product = torch.einsum('ijk,ilk->ijl', personality, commonality)
    projection_coeff = dot_product / B_norm_squared.permute(0, 2, 1)
    orthogonal_projection = torch.einsum('ijk,ikl->ijl', projection_coeff, commonality)
    op_mean = orthogonal_projection.mean(dim=1, keepdim=True)
    op_std = orthogonal_projection.std(dim=1, keepdim=True)
    op_std = op_std + 1e-8
    orthogonal_projection = (orthogonal_projection - op_mean) / op_std

    return orthogonal_projection


def rotate_vectors(torch_theta, torch_com):
    theta = torch_theta * 2 * torch.pi
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    N, M, D = torch_com.shape
    R = torch.zeros(N, D // 2, 2, 2, device=torch_com.device)
    R[..., 0, 0] = cos_theta[:, ::2]
    R[..., 0, 1] = -sin_theta[:, ::2]
    R[..., 1, 0] = sin_theta[:, ::2]
    R[..., 1, 1] = cos_theta[:, ::2]
    com_reshaped = torch_com.view(N * M, D)
    for d in range(0, D - 1, 2):
        v = com_reshaped[:, d:d + 2].view(-1, 2, 1).transpose(1, 2)
        v_rotated = torch.bmm(v, R[:, d // 2])
        com_reshaped[:, d:d + 2] = v_rotated.squeeze(2)
    com_rotated = com_reshaped.view(N, M, D)
    return com_rotated


class AngleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AngleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim*input_dim)

    def forward(self, A):
        N, M, D = A.size()
        x = torch.relu(self.fc1(A))
        x = torch.relu(self.fc2(x))
        B_flat = self.fc3(x)
        B = B_flat.view(N, M, D, D)
        C = torch.bmm(A.view(N * M, 1, D), B.view(N * M, D, D)).view(N, M, D)
        return C


class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.fc_query = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.fc_key = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.fc_value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, fig_tensor, fea_tensor):
        q = self.fc_query(fig_tensor)
        K = self.fc_key(fea_tensor)
        K_expend = K.unsqueeze(-1).repeat(1, 1, 1, int(q.shape[1]/40))
        K_re_expend = K_expend.flatten(start_dim=2)
        V = self.fc_value(fea_tensor)
        scores = torch.bmm(K_re_expend, q)
        attention_weights = F.softmax(scores, dim=1)
        final_output = attention_weights + V
        return final_output


class Pnet(nn.Module):
    def __init__(self, input_size, hidden_size, pic_size):
        super(Pnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(int(pic_size/input_size*3*pic_size)),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(int(pic_size/input_size*3*pic_size)),
            nn.ELU(),
            nn.Linear(hidden_size, input_size),
            nn.BatchNorm1d(int(pic_size/input_size*3*pic_size)),
            nn.ELU(),
        )
    def forward(self, x):
        personality = self.fc(x)
        return personality


class Cnet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Cnet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, input_size),
            nn.ELU(),
        )
    def forward(self, x):
        commonality = self.fc(x)
        return commonality


if __name__ == "__main__":
    model = AngleNN(40, 128)
    input_tensor = torch.rand(20, 80, 40)  # instance: batch_size = 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)
