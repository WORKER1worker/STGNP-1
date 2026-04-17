import torch
import torch.nn as nn
import torch.nn.functional as F


class FCStack(nn.Module):
    def __init__(self, in_dim, units, activations, use_bias=True):
        super().__init__()
        assert len(units) == len(activations)
        layers = []
        prev = in_dim
        for out_dim, act in zip(units, activations):
            layers.append(nn.Linear(prev, out_dim, bias=use_bias))
            if act == "relu":
                layers.append(nn.ReLU())
            elif act == "tanh":
                layers.append(nn.Tanh())
            elif act is None:
                pass
            else:
                raise ValueError(f"Unsupported activation: {act}")
            prev = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RelationAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = FCStack(d, [d], ["tanh"])
        self.fc2 = FCStack(d, [1], [None], use_bias=False)

    def forward(self, x):
        # x: [N, h, R, d]
        a = self.fc1(x)
        a = self.fc2(a)  # [N, h, R, 1]
        a = torch.softmax(a.transpose(2, 3), dim=-1)  # [N, h, 1, R]
        x = torch.matmul(a, x)  # [N, h, 1, d]
        return x.squeeze(2)


class SPFSModel(nn.Module):
    def __init__(self, T=24, d=64, mean=0.0, std=1.0):
        super().__init__()
        self.T = T
        self.d = d
        self.mean = float(mean)
        self.std = float(std) if float(std) > 0 else 1.0

        self.input_gp = FCStack(1, [d, d], ["relu", None])
        self.input_fs = FCStack(1, [d, d], ["relu", None])

        self.x_post_gp = FCStack(d, [d, d], ["relu", None])
        self.x_post_fs = FCStack(d, [d, d], ["relu", None])
        self.y_post_gp = FCStack(d, [d, d], ["relu", None])
        self.y_post_fs = FCStack(d, [d, d], ["relu", None])

        self.delta_gp = FCStack(d, [d, d], ["relu", "tanh"])
        self.delta_fs = FCStack(d, [d, d], ["relu", "tanh"])
        self.res_gp = FCStack(d, [d], [None])
        self.res_fs = FCStack(d, [d], [None])

        self.x2_gp = FCStack(d, [d, d], ["relu", None])
        self.x2_fs = FCStack(d, [d, d], ["relu", None])
        self.y2_gp = FCStack(d, [d, d], ["relu", None])
        self.y2_fs = FCStack(d, [d, d], ["relu", None])

        self.te_embed = FCStack(T, [d, d], ["relu", None])

        self.g1_gp = FCStack(d, [d, d], ["relu", "relu"])
        self.g1_fs = FCStack(d, [d, d], ["relu", "relu"])
        self.g2_gp = nn.Linear(d, d)
        self.g2_fs = nn.Linear(d, d)

        self.cell_gp = nn.GRUCell(input_size=2 * d, hidden_size=d)
        self.cell_fs = nn.GRUCell(input_size=2 * d, hidden_size=d)

        self.attn = RelationAttention(d)
        self.out_head = FCStack(d, [d, d, 1], ["relu", "relu", None])

    def forward(self, x_gp, x_fs, gp, fs, TE):
        # x_*: [N, h, K, 1], gp/fs: [N, 1, 1, K], TE: [1, h]
        N, h = x_gp.shape[0], x_gp.shape[1]

        x_gp = (x_gp - self.mean) / self.std
        x_fs = (x_fs - self.mean) / self.std

        x_gp = self.input_gp(x_gp)
        x_fs = self.input_fs(x_fs)

        gp_t = gp.repeat(1, h, 1, 1)
        fs_t = fs.repeat(1, h, 1, 1)

        y_gp = torch.matmul(gp_t, x_gp)
        y_fs = torch.matmul(fs_t, x_fs)

        x_gp = self.x_post_gp(x_gp)
        x_fs = self.x_post_fs(x_fs)
        y_gp = self.y_post_gp(y_gp)
        y_fs = self.y_post_fs(y_fs)

        x_gp = torch.abs(y_gp - x_gp)
        x_fs = torch.abs(y_fs - x_fs)

        x_gp = torch.matmul(gp_t, x_gp)
        x_fs = torch.matmul(fs_t, x_fs)

        x_gp = self.delta_gp(x_gp)
        x_fs = self.delta_fs(x_fs)
        y_gp = x_gp + self.res_gp(y_gp)
        y_fs = x_fs + self.res_fs(y_fs)

        x_gp = self.x2_gp(x_gp)
        x_fs = self.x2_fs(x_fs)
        y_gp = self.y2_gp(y_gp)
        y_fs = self.y2_fs(y_fs)

        y_gp = y_gp.squeeze(2)
        y_fs = y_fs.squeeze(2)
        x_gp = x_gp.squeeze(2)
        x_fs = x_fs.squeeze(2)

        TE_onehot = F.one_hot(TE, num_classes=self.T).float()  # [1, h, T]
        TE_emb = self.te_embed(TE_onehot).repeat(N, 1, 1)      # [N, h, d]

        g1_gp = torch.exp(-self.g1_gp(x_gp))
        g1_fs = torch.exp(-self.g1_fs(x_fs))

        y_gp = torch.cat([g1_gp * y_gp, TE_emb], dim=-1)
        y_fs = torch.cat([g1_fs * y_fs, TE_emb], dim=-1)

        state_gp = torch.zeros((N, self.d), device=x_gp.device)
        state_fs = torch.zeros((N, self.d), device=x_gp.device)

        pred_gp = []
        pred_fs = []
        zeros_state = torch.zeros_like(state_gp)

        for i in range(h):
            if i == 0:
                gate_gp = torch.exp(-F.relu(self.g2_gp(zeros_state)))
                gate_fs = torch.exp(-F.relu(self.g2_fs(zeros_state)))
            else:
                gate_gp = torch.exp(-F.relu(self.g2_gp(x_gp[:, i - 1, :])))
                gate_fs = torch.exp(-F.relu(self.g2_fs(x_fs[:, i - 1, :])))

            state_gp = self.cell_gp(y_gp[:, i, :], gate_gp * state_gp)
            state_fs = self.cell_fs(y_fs[:, i, :], gate_fs * state_fs)

            pred_gp.append(state_gp.unsqueeze(1))
            pred_fs.append(state_fs.unsqueeze(1))

        pred_gp = torch.cat(pred_gp, dim=1).unsqueeze(2)
        pred_fs = torch.cat(pred_fs, dim=1).unsqueeze(2)

        pred = torch.cat([pred_gp, pred_fs], dim=2)
        pred = self.attn(pred)
        pred = self.out_head(pred)

        return pred * self.std + self.mean


def masked_mse_loss(pred, label):
    mask = (label != 0).float()
    mask = mask / (mask.mean() + 1e-10)
    mask = torch.nan_to_num(mask, nan=0.0)

    loss = (pred - label) ** 2
    loss = loss * mask
    loss = torch.nan_to_num(loss, nan=0.0)
    return loss.mean()
