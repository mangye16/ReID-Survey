import math 
import torch
import torch.nn.functional as F

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

def adaptive_gem2d(x, output_size=(1, 1), p=3, eps=1e-6):
    return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), output_size).pow(1. / p)