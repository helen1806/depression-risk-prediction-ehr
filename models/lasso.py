import torch
import torch.nn as nn

class LassoLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LassoLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def lasso_penalty(self, l1_lambda):
        return l1_lambda * torch.sum(torch.abs(self.linear.weight))
