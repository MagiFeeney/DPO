import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

# Beta
class FixedBeta(torch.distributions.Beta):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean
    
class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        self.mean    = nn.Linear(num_inputs, num_outputs)
        self.log_std = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        mean = self.mean(x)
        std  = F.softplus(self.log_std(x))

        return FixedNormal(mean, std)

class Beta(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Beta, self).__init__()

        self.log_alpha = nn.Linear(num_inputs, num_outputs)
        self.log_beta  = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        alpha = F.softplus(self.log_alpha(x)) + 1.
        beta  = F.softplus(self.log_beta(x)) + 1.

        return FixedBeta(alpha, beta)

