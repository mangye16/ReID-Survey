""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from functools import partial
from itertools import repeat
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch._six import container_abcs
from torch import einsum
import collections.abc as container_abcs

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool1d(x, self.output_size).pow(1. / self.p)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         #############
#         self.bn1 = nn.BatchNorm1d(1)
#         # self.fc2 = nn.Linear(self.seq_len, dim_feedforward)
#         # self.bn2 = nn.BatchNorm1d(dim_feedforward)
#         # self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(193, 1)
#         self.bn2 = nn.BatchNorm1d(1)

#         self.bn1_g = nn.BatchNorm1d(1)
#         # self.fc2 = nn.Linear(self.seq_len, dim_feedforward)
#         # self.bn2 = nn.BatchNorm1d(dim_feedforward)
#         # self.relu = nn.ReLU()
#         self.fc1_g = nn.Linear(193, 1)
#         self.bn2_g = nn.BatchNorm1d(1)
#         self.bn3 = nn.BatchNorm1d(1)
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def forward(self, x,gallery_1=None,gallery_2=None):
#         if (gallery_1 is None) and (gallery_1 is None):
#             B, N, C = x.shape
#             qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#             q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)

#             x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#             x = self.proj(x)
#             x = self.proj_drop(x)
#             return x

#         elif (gallery_1 is not None) and (gallery_2 is not None):
#             x_1= x[:gallery_1.size(0)]
#             x_2 = x[gallery_1.size(0):]
#             B_x, N_x, C_x = x_1.shape
#             qkv_x = self.qkv(x_1).reshape(B_x, N_x, 3, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)
#             q_x, k_x, v_x = qkv_x[0], qkv_x[1], qkv_x[2]   # make torchscript happy (cannot use tensor as tuple)  # B H N C//H 

#             attn_x = (q_x @ k_x.transpose(-2, -1)) * self.scale # B H N N 
#             attn_x = attn_x.softmax(dim=-1)
#             attn_x = self.attn_drop(attn_x)

#             x_1 = (attn_x @ v_x).transpose(1, 2).reshape(B_x, N_x, C_x) # B H N C//H -> B N H C//H  -> B N C 
#             x_1 = self.proj(x_1)
#             x_1 = self.proj_drop(x_1)

#             B_x_2, N_x_2, C_x_2 = x_2.shape
#             qkv_x_2 = self.qkv(x_2).reshape(B_x_2, N_x_2, 3, self.num_heads, C_x_2 // self.num_heads).permute(2, 0, 3, 1, 4)
#             q_x_2, k_x_2, v_x_2 = qkv_x_2[0], qkv_x_2[1], qkv_x_2[2]   # make torchscript happy (cannot use tensor as tuple)  # B H N C//H 

#             attn_x_2 = (q_x_2 @ k_x_2.transpose(-2, -1)) * self.scale # B H N N 
#             attn_x_2 = attn_x_2.softmax(dim=-1)
#             attn_x_2 = self.attn_drop(attn_x_2)

#             x_2 = (attn_x_2 @ v_x_2).transpose(1, 2).reshape(B_x_2, N_x_2, C_x_2) # B H N C//H -> B N H C//H  -> B N C 
#             x_2 = self.proj(x_2)
#             x_2 = self.proj_drop(x_2)

#             x = torch.cat((x_1,x_2),dim=0)
# #################
#             B_g, N_g, C_g = gallery_1.shape
#             qkv_g = self.qkv(gallery_1).reshape(B_g, N_g, 3, self.num_heads, C_g // self.num_heads).permute(2, 0, 3, 1, 4)
#             q_g, k_g, v_g = qkv_g[0], qkv_g[1], qkv_g[2]   # make torchscript happy (cannot use tensor as tuple)  # B H N C//H 

#             attn_g = (q_g @ k_g.transpose(-2, -1)) * self.scale # B H N N 
#             attn_g = attn_g.softmax(dim=-1)
#             attn_g = self.attn_drop(attn_g)

#             gallery_1 = (attn_g @ v_g).transpose(1, 2).reshape(B_g, N_g, C_g) # B H N C//H -> B N H C//H  -> B N C 
#             gallery_1 = self.proj(gallery_1)
#             gallery_1 = self.proj_drop(gallery_1)
# ####################
#             B_g_2, N_g_2, C_g_2 = gallery_2.shape
#             qkv_g_2 = self.qkv(gallery_2).reshape(B_g_2, N_g_2, 3, self.num_heads, C_g_2 // self.num_heads).permute(2, 0, 3, 1, 4)
#             q_g_2, k_g_2, v_g_2 = qkv_g_2[0], qkv_g_2[1], qkv_g_2[2]   # make torchscript happy (cannot use tensor as tuple)  # B H N C//H 

#             attn_g_2 = (q_g_2 @ k_g_2.transpose(-2, -1)) * self.scale # B H N N 
#             attn_g_2 = attn_g_2.softmax(dim=-1)
#             attn_g_2 = self.attn_drop(attn_g_2)

#             gallery_2 = (attn_g_2 @ v_g_2).transpose(1, 2).reshape(B_g_2, N_g_2, C_g_2) # B H N C//H -> B N H C//H  -> B N C 
#             gallery_2 = self.proj(gallery_2)
#             gallery_2 = self.proj_drop(gallery_2)
# ################
#             q_match = q_x.transpose(1, 2).reshape(B_x, N_x, C_x)
#             g_match = k_g.transpose(1, 2).reshape(B_g, N_g, C_g)
#             score_1 = einsum('q t d, k s d -> q k s t', q_match, g_match)
#             score_1 = score_1.reshape(B_x * B_g, N_x, N_g)
#             # print('score_1 1',score_1.size())
#             score_1 = torch.cat((score_1.max(dim=1)[0], score_1.max(dim=2)[0]), dim=-1)
#             # print('score_1 2',score_1.size())
#             score_1 = score_1.view(-1, 1, N_x)
#             score_1 = self.bn1(score_1).view(-1, N_x)
#             # score = self.fc2(score)
#             # score = self.bn2(score)
#             # score = self.relu(score)
#             score_1 = self.fc1(score_1)
#             score_1 = score_1.view(-1, 2).sum(dim=-1, keepdim=True)
#             score_1 = self.bn2(score_1)
#             score_1 = score_1.view(B_x, B_g)

#             # score = einsum('q t d, k s d -> q k s t', q_x, k_g)
#             q_match = q_g.transpose(1, 2).reshape(B_g, N_g, C_g)
#             g_match = k_x.transpose(1, 2).reshape(B_x, N_x, C_x)
#             score_2 = einsum('q t d, k s d -> q k s t', q_match, g_match)
#             score_2 = score_2.reshape(B_g * B_x, N_x, N_x)
#             score_2 = torch.cat((score_2.max(dim=1)[0], score_2.max(dim=2)[0]), dim=-1)
#             score_2 = score_2.view(-1, 1, N_x)
#             score_2 = self.bn1_g(score_2).view(-1, N_x)
#             # score = self.fc2(score)
#             # score = self.bn2(score)
#             # score = self.relu(score)
#             score_2 = self.fc1_g(score_2)
#             score_2 = score_2.view(-1, 2).sum(dim=-1, keepdim=True)
#             score_2 = self.bn2_g(score_2)
#             score_2 = score_2.view(B_g, B_x).transpose(0, 1)
 
#             # print('score_1,score_2',score_1.size(),score_2.size())
#             score_g1 = 0.5*score_1 + 0.5*score_2 

#             q, k = score_g1.size()
#             score_g1 = score_g1.view(-1, 1)
#             score_g1 = self.bn3(score_g1)
#             score_g1 = score_g1.view(q, k)
# ########################
#             q_match = q_x_2.transpose(1, 2).reshape(B_x_2, N_x_2, C_x)
#             g_match = k_g_2.transpose(1, 2).reshape(B_g_2, N_g_2, C_g)
#             score_1 = einsum('q t d, k s d -> q k s t', q_match, g_match)
#             score_1 = score_1.reshape(B_x_2 * B_g_2, N_x_2, N_x_2)
#             score_1 = torch.cat((score_1.max(dim=1)[0], score_1.max(dim=2)[0]), dim=-1)
#             score_1 = score_1.view(-1, 1, N_x_2)
#             score_1 = self.bn1(score_1).view(-1, N_x_2)
#             # score = self.fc2(score)
#             # score = self.bn2(score)
#             # score = self.relu(score)
#             score_1 = self.fc1(score_1)
#             score_1 = score_1.view(-1, 2).sum(dim=-1, keepdim=True)
#             score_1 = self.bn2(score_1)
#             score_1 = score_1.view(B_x_2, B_g_2)

#             # score = einsum('q t d, k s d -> q k s t', q_x, k_g)
#             q_match = q_g_2.transpose(1, 2).reshape(B_g_2 , N_x, C_x)
#             g_match = k_x_2.transpose(1, 2).reshape(B_x_2, N_g, C_g)
#             score_2 = einsum('q t d, k s d -> q k s t', q_match, g_match)
#             score_2 = score_2.reshape(B_g_2 * B_x_2, N_x, N_x)
#             score_2 = torch.cat((score_2.max(dim=1)[0], score_2.max(dim=2)[0]), dim=-1)
#             score_2 = score_2.view(-1, 1, N_x)
#             score_2 = self.bn1_g(score_2).view(-1, N_x)
#             # score = self.fc2(score)
#             # score = self.bn2(score)
#             # score = self.relu(score)
#             score_2 = self.fc1_g(score_2)
#             score_2 = score_2.view(-1, 2).sum(dim=-1, keepdim=True)
#             score_2 = self.bn2_g(score_2)
#             score_2 = score_2.view(B_g_2, B_x_2).transpose(0, 1)
 
#             # print('score_1,score_2',score_1.size(),score_2.size())
#             score_g2 = 0.5*score_1 + 0.5*score_2 

#             q, k = score_g2.size()
#             score_g2 = score_g2.view(-1, 1)
#             score_g2 = self.bn3(score_g2)
#             score_g2 = score_g2.view(q, k)

#             return x, gallery_1,gallery_2,score_g1,score_g2

#         elif (gallery_1 is not None):
#             gallery = gallery_1
#             B_x, N_x, C_x = x.shape
#             qkv_x = self.qkv(x).reshape(B_x, N_x, 3, self.num_heads, C_x // self.num_heads).permute(2, 0, 3, 1, 4)
#             q_x, k_x, v_x = qkv_x[0], qkv_x[1], qkv_x[2]   # make torchscript happy (cannot use tensor as tuple)  # B H N C//H 

#             attn_x = (q_x @ k_x.transpose(-2, -1)) * self.scale # B H N N 
#             attn_x = attn_x.softmax(dim=-1)
#             attn_x = self.attn_drop(attn_x)

#             x = (attn_x @ v_x).transpose(1, 2).reshape(B_x, N_x, C_x) # B H N C//H -> B N H C//H  -> B N C 
#             x = self.proj(x)
#             x = self.proj_drop(x)

#             B_g, N_g, C_g = gallery.shape
#             qkv_g = self.qkv(gallery).reshape(B_g, N_g, 3, self.num_heads, C_g // self.num_heads).permute(2, 0, 3, 1, 4)
#             q_g, k_g, v_g = qkv_g[0], qkv_g[1], qkv_g[2]   # make torchscript happy (cannot use tensor as tuple)  # B H N C//H 

#             attn_g = (q_g @ k_g.transpose(-2, -1)) * self.scale # B H N N 
#             attn_g = attn_g.softmax(dim=-1)
#             attn_g = self.attn_drop(attn_g)

#             gallery = (attn_g @ v_g).transpose(1, 2).reshape(B_g, N_g, C_g) # B H N C//H -> B N H C//H  -> B N C 
#             gallery = self.proj(gallery)
#             gallery = self.proj_drop(gallery)


#             q_match = q_x.transpose(1, 2).reshape(B_x, N_x, C_x)
#             g_match = k_g.transpose(1, 2).reshape(B_g, N_g, C_g)
#             score_1 = einsum('q t d, k s d -> q k s t', q_match, g_match)
#             score_1 = score_1.reshape(B_x * B_g, N_x, N_x)
#             score_1 = torch.cat((score_1.max(dim=1)[0], score_1.max(dim=2)[0]), dim=-1)
#             score_1 = score_1.view(-1, 1, N_x)
#             score_1 = self.bn1(score_1).view(-1, N_x)
#             # score = self.fc2(score)
#             # score = self.bn2(score)
#             # score = self.relu(score)
#             score_1 = self.fc1(score_1)
#             score_1 = score_1.view(-1, 2).sum(dim=-1, keepdim=True)
#             score_1 = self.bn2(score_1)
#             score_1 = score_1.view(B_x, B_g)

#             # score = einsum('q t d, k s d -> q k s t', q_x, k_g)
#             q_match = q_g.transpose(1, 2).reshape(B_g, N_g, C_g)
#             g_match = k_x.transpose(1, 2).reshape(B_x , N_x, C_x)
#             score_2 = einsum('q t d, k s d -> q k s t', q_match, g_match)
#             score_2 = score_2.reshape(B_x * B_g, N_g, N_g)
#             score_2 = torch.cat((score_2.max(dim=1)[0], score_2.max(dim=2)[0]), dim=-1)
#             score_2 = score_2.view(-1, 1, N_x)
#             score_2 = self.bn1_g(score_2).view(-1, N_x)
#             # score = self.fc2(score)
#             # score = self.bn2(score)
#             # score = self.relu(score)
#             score_2 = self.fc1_g(score_2)
#             score_2 = score_2.view(-1, 2).sum(dim=-1, keepdim=True)
#             score_2 = self.bn2_g(score_2)
#             score_2 = score_2.view(B_g, B_x).transpose(0, 1)
 
#             # print('score_1,score_2',score_1.size(),score_2.size())
#             score = 0.5*score_1 + 0.5*score_2 

#             q, k = score.size()
#             score = score.view(-1, 1)
#             score = self.bn3(score)
#             score = score.view(q, k)


#             return x, gallery,score


#####ori
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


    # def forward(self, x,gallery_1=None,gallery_2=None):
    #     if (gallery_1 is None) and (gallery_2 is None):
    #         x = x + self.drop_path(self.attn(self.norm1(x)))
    #         x = x + self.drop_path(self.mlp(self.norm2(x)))
    #         return x
    #     elif (gallery_1 is not None) and (gallery_2 is not None):
    #         x_att,gallery_1_att,gallery_2_att,score_g1,score_g2 = self.attn(self.norm1(x),gallery_1,gallery_2)
    #         x = x + self.drop_path(x_att)
    #         x = x + self.drop_path(self.mlp(self.norm2(x)))

    #         gallery_1 = gallery_1 + self.drop_path(gallery_1_att)
    #         gallery_1 = gallery_1 + self.drop_path(self.mlp(self.norm2(gallery_1)))

    #         gallery_2 = gallery_2 + self.drop_path(gallery_2_att)
    #         gallery_2 = gallery_2 + self.drop_path(self.mlp(self.norm2(gallery_2)))
    #         return x,gallery_1,gallery_2,score_g1,score_g2

    #     elif (gallery_1 is not None):
    #         x_att,gallery_att,score = self.attn(self.norm1(x),gallery_1)
    #         x = x + self.drop_path(x_att)
    #         x = x + self.drop_path(self.mlp(self.norm2(x)))

    #         gallery_1 = gallery_1 + self.drop_path(gallery_att)
    #         gallery_1 = gallery_1 + self.drop_path(self.mlp(self.norm2(gallery_1)))

    #     return x,gallery_1,score


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, embed_dim=768, stem_conv=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        self.num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size

        self.stem_conv = stem_conv
        if self.stem_conv:
            hidden_dim = 64
            stem_stride = 2
            stride_size = patch_size = patch_size[0] // stem_stride
            self.conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=7, stride=stem_stride, padding=3,bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1,bias=False),
                IBN(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,padding=1,bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            in_chans = hidden_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)

    def forward(self, x):
        if self.stem_conv:
            x = self.conv(x)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2) # [64, 8, 768]
        return x





class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), local_feature=False, sie_xishu =1.0, hw_ratio=1, gem_pool = False, stem_conv=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.local_feature = True
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
            embed_dim=embed_dim, stem_conv =stem_conv)#stem_conv

        self.patch_embed2 = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
            embed_dim=embed_dim, stem_conv = stem_conv) 

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        self.in_planes = 768
        self.gem_pool = gem_pool
        if self.gem_pool:
            print('using gem pooling')
        # Initialize SIE Embedding
        # self.sie_embed_rgb = nn.Parameter(torch.zeros(4, 1, embed_dim))
        # self.sie_embed_ir = nn.Parameter(torch.zeros(2, 1, embed_dim))
        # if camera > 1 and view > 1:
        #     self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
        #     trunc_normal_(self.sie_embed, std=.02)
        # elif camera > 1:
        #     self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
        #     trunc_normal_(self.sie_embed, std=.02)
        # elif view > 1:
        #     self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
        #     trunc_normal_(self.sie_embed, std=.02)

        #  print('using drop_out rate is : {}'.format(drop_rate))
        #  print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        #  print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)
        self.decoder_norm = nn.BatchNorm1d(1)



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x1_input, x2_input,modal=0,label_1=None,label_2=None):#,gallery_1=None,gallery_2=None):#(self, x, camera_id, view_id):
        # B = x.shape[0]
        # x = self.patch_embed(x)
        # gallery =None
        single_size_1 = x1_input.size(0)
        single_size_2 = x2_input.size(0)
        if modal == 0:
            x1 = self.patch_embed(x1_input)
            x2 = self.patch_embed2(x2_input)
            # x2_reverse = self.patch_embed(x2_input)
            # x1_reverse = self.patch_embed2(x1_input)
            x = torch.cat((x1, x2), 0)#x1_reverse,x2_reverse
            label = torch.cat((label_1, label_2), -1)
            # sie = torch.cat((self.sie_embed_rgb[camera_id_1],self.sie_embed_ir[camera_id_2]),0 )
        elif modal == 1:
            x = self.patch_embed(x1_input)
            # sie = self.sie_embed_rgb[camera_id_1]
        elif modal == 2:
            x = self.patch_embed2(x2_input)
            # sie = self.sie_embed_ir[camera_id_2]
        # if (gallery_1 is not None) and (gallery_2 is not None):
        #     gallery_1 = self.patch_embed(gallery_1)
        #     gallery_2 = self.patch_embed2(gallery_2)
        #     gallery = torch.cat((gallery_1, gallery_2), 0)    
        # elif (gallery_1 is not None):
        #     gallery = self.patch_embed(gallery_1)
        # elif (gallery_2 is not None):
        #     gallery = self.patch_embed2(gallery_2)

        B = x.shape[0]
        B_gallery = x.shape[0]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # if gallery is not None:
        #     cls_tokens_g = self.cls_token.expand(B_gallery, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        #     gallery = torch.cat((cls_tokens_g, gallery), dim=1)
        #     gallery = gallery + self.pos_embed
        #     gallery = self.pos_drop(gallery)
        #     gallery_1 = gallery[:single_size]
        #     gallery_2 = gallery[single_size:]
        #     target = label.unsqueeze(1)
        #     mask = (target == target.t())
        #     pair_labels= mask.float()

        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            if self.training:
                return x,x[:single_size_1],x[single_size_1:],label[:single_size_1],label[single_size_1:]
            else:
                return x,x[:single_size_1],x[single_size_1:],label_1,label_2
        else:
            for blk in self.blocks:
                x = blk(x)    



            x = self.norm(x)
        # if self.gem_pool:
        #     gf = self.gem(x[:,1:].permute(0,2,1)).squeeze()
        #     feat = x[:, 0] + gf
        #     feat =torch.cat((x[:, 0], gf), dim=1)
        #     return feat,feat[:single_size],feat[single_size:],label_1,label_2

        feat=x#[:, 0]
        # if gallery is not None:
        #     # if modal == 0:
        #     return feat,feat[:single_size],feat[single_size:],label_1,label_2#,score_1,score_2,pair_labels[:single_size,:single_size],pair_labels[single_size:,single_size:]
        #     # else:
        #     #     return feat,feat[:single_size],feat[single_size:],label_1,label_2,score,pair_labels
        # else:
        if self.training:
            return x,x[:single_size_1],x[single_size_1:],label[:single_size_1],label[single_size_1:]
        else:
            return x,x[:single_size_1],x[single_size_1:],label_1,label_2
        # return feat,feat[:single_size_1],feat[single_size_1:],label[:single_size_1],label[single_size_1:]
    # def forward(self, x, cam_label=None, view_label=None):
    #     x = self.forward_features(x, cam_label, view_label)
    #     return x

    def load_param(self, model_path,hw_ratio):
        param_dict = torch.load(model_path, map_location='cpu')
        count=0
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        if 'teacher' in param_dict: ### for dino
            obj = param_dict["teacher"]
            print('Convert dino model......')
            newmodel = {}
            for k, v in obj.items():
                if k.startswith("module."):
                    k = k.replace("module.", "")
                if not k.startswith("backbone."):
                    continue
                old_k = k
                k = k.replace("backbone.", "")
                newmodel[k] = v
                param_dict = newmodel
        # print(param_dict.keys())
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k or 'pre_logits' in k:
            # if 'head' in k or 'dist' in k or 'pre_logits' in k or 'IN' in k or 'BN' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x,hw_ratio)
            try:
                self.state_dict()[k].copy_(v)
                count +=1
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
        print('Load %d / %d layers.'%(count,len(self.state_dict().keys())))


def resize_pos_embed(posemb, posemb_new, hight, width, hw_ratio):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old_h = int(math.sqrt(len(posemb_grid)*hw_ratio))
    gs_old_w = gs_old_h // hw_ratio
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = TransReID(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, camera=camera, view=view, drop_path_rate=drop_path_rate, sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)
    return model

def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,drop_path_rate=drop_path_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,  **kwargs)
    model.in_planes = 384
    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
