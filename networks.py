import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class JointPolVal(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden=64):
        super(JointPolVal, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden)
        self.affine2 = nn.Linear(hidden, hidden)
        self.affine3 = nn.Linear(hidden, hidden)

        self.action_mean = nn.Linear(hidden, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        #self.action_log_std = nn.Linear(hidden, num_outputs)
        self.action_log_std = nn.Parameter(-1 * torch.ones(1, num_outputs))

        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))
        x = F.tanh(self.affine3(x))

        action_mean = self.action_mean(x)
        #action_log_std = self.action_log_std(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        value = self.value_head(x)

        return action_mean, action_log_std, action_std, value
