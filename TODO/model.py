import torch
import torch.nn as nn
import torch.nn.functional as F


def NLLloss(y, mean, var):
    return torch.mean(torch.log(var) + (y - mean)**2/(2*var))


class CNPs(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    # Output : mean, variance
    def forward(self, x_ctx, y_ctx, x_obs):
        r_agg = self.encoder(x_ctx, y_ctx)
        mean, variance = self.decoder(x_obs, r_agg)
        return mean, variance


class Encoder(nn.Module):
    def __init__(self, x_dim=1, y_dim=1, r_dim=128, h_dim=128):
        super().__init__()
        self.mlp1 = torch.nn.Linear(x_dim + y_dim, h_dim)
        self.mlp2 = torch.nn.Linear(h_dim, h_dim)
        self.mlp3 = torch.nn.Linear(h_dim, r_dim)

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        h1 = F.relu(self.mlp1(input))
        h2 = F.relu(self.mlp2(h1))
        r = F.relu(self.mlp3(h2))
        return torch.mean(r, dim=0)


class Decoder(nn.Module):
    def __init__(self, x_dim, r_dim, y_dim, h_dim=128):
        super().__init__()
        self.mlp1 = torch.nn.Linear(x_dim + r_dim, h_dim)
        self.mlp2 = torch.nn.Linear(h_dim, h_dim)
        self.mlp3 = torch.nn.Linear(h_dim, h_dim)
        self.mlp4 = torch.nn.Linear(h_dim, h_dim)
        self.mlp_mu = torch.nn.Linear(h_dim, y_dim)
        self.mlp_var = torch.nn.Linear(h_dim, y_dim)
        pass

    def forward(self, x, r):
        r_ = r.repeat(x.size(0)).view(x.size(0), -1)
        input = torch.cat([x, r_], dim=-1)
        h1 = F.relu(self.mlp1(input))
        h2 = F.relu(self.mlp2(h1))
        h3 = F.relu(self.mlp3(h2))
        h4 = F.relu(self.mlp3(h3))

        mu = self.mlp_mu(h4)
        # std = F.relu(self.mlp_var(h4)) + 1e-6
        var = F.softplus(self.mlp_var(h4)) + 1e-6
        # return mu, std
        return mu, var
