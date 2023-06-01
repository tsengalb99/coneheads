import torch
import torch.nn.functional as F


def map_xi(x):
    x_x = x[..., :-1]
    x_y = torch.exp(x[..., -1] / x.shape[-1])
    return x_x * x_y.unsqueeze(-1), x_y


def umbral(q, k, r=1, gamma=1):
    q_x, q_y = map_xi(q)
    k_x, k_y = map_xi(k)
    q_y = q_y.unsqueeze(2)
    k_y = k_y.unsqueeze(1)
    out = torch.maximum(torch.maximum(q_y, k_y),
                        (torch.cdist(q_x, k_x) / torch.sinh(torch.tensor(r)) +
                         torch.add(q_y, k_y)) / 2)
    return -gamma * out


# [b, n, d]
q = torch.randn(100, 10, 2)
k = torch.randn(100, 10, 2)

# [100, 10, 10]
print(umbral(q, k).shape)
