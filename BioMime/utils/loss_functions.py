import torch
import math


def mse(x, y):
    assert(x.shape == y.shape)
    return torch.mean((x - y) ** 2)


def nrmse_matrix_torch(x, y):
    '''
    y is the target signal
    '''
    assert(x.shape == y.shape)

    bs = x.shape[0]
    x = torch.reshape(x, (bs, -1))
    y = torch.reshape(y, (bs, -1))
    rmse = torch.linalg.norm(y - x, dim=1) / math.sqrt(y.shape[-1]) / (torch.max(y, dim=1)[0] - torch.min(y, dim=1)[0])
    return rmse
