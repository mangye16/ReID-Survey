import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID
from .backbones.swin_transformer import swin_base_patch4_window7_224, swin_small_patch4_window7_224
# from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from torch.nn.modules import Module
from torch import einsum
from torch.nn.init import xavier_uniform_
from torch import Tensor
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1) # [5:] [1]
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))


        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        if self.reduce_feat_dim:
            self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            self.fcneck.apply(weights_init_xavier)
            self.in_planes = cfg.MODEL.FEAT_DIM

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)

        if pretrain_choice == 'self':
            self.load_param(model_path)


    def forward(self, x, label=None, **kwargs):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        if self.reduce_feat_dim:
            global_feat = self.fcneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)
        if self.dropout_rate > 0:
            feat = self.dropout(feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i:
                continue
            elif 'module' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            else:
                self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    #  def load_param(self, trained_path):
        #  param_dict = torch.load(trained_path, map_location = 'cpu')
        #  for i in param_dict:
            #  try:
                #  self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            #  except:
                #  continue
        #  print('Loading pretrained model from {}'.format(trained_path))

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



class Matcher(Module):
    r"""TransformerDecoderLayer is made up of feature matching and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, dim_feedforward=2048)
        >>> memory = torch.rand(10, 24, 8, 512)
        >>> tgt = torch.rand(20, 24, 8, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, seq_len):
        super(Matcher, self).__init__()
        self.seq_len = seq_len
        self.d_model = 768
        self.num_heads=12
        # score_embed = torch.randn(seq_len, seq_len)
        # score_embed = score_embed + score_embed.t()
        # self.score_embed = nn.Parameter(score_embed.view(1, 1, seq_len, seq_len))
        # self.qkv = nn.Linear(d_model, d_model * 3, bias=False)#nn.Linear(d_model, d_model)
        # self.bn1 = nn.BatchNorm1d(1)
        # self.fc2 = nn.Linear(self.seq_len, dim_feedforward)
        # self.bn2 = nn.BatchNorm1d(dim_feedforward)
        # self.relu = nn.ReLU()
        # self.fc3 = nn.Linear(dim_feedforward, 1)
        # self.bn3 = nn.BatchNorm1d(768*self.seq_len)
        # self.bn3 = nn.LayerNorm(768)
        # self.l2norm = Normalize(2)
    def forward(self, tgt, memory, label=None):
        r"""Pass the inputs through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).

        Shape:
            tgt: [q, h, w, d], where q is the query length, d is d_model, and (h, w) is feature map size
            memory: [k, h, w, d], where k is the memory length
        """

        q, d_5 = tgt.size() # b d*5,  
        k, d_5 = memory.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)
        # query_t = F.normalize(tgt.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(memory.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        query_t = tgt.view(q, -1, z)#@self.bn3(tgt.view(q, -1, z))  #B N C tgt.view(q, -1, z)#
        key_m = memory.view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #Q N C memory.view(k, -1, d)#
        query_t = tgt.reshape(q, self.seq_len, self.num_heads, z // self.num_heads).permute(0, 2, 1, 3) #B N H,C//H -> B H N C//H 
        key_m = memory.reshape(k, self.seq_len, self.num_heads, z // self.num_heads).permute(0, 2, 1, 3) #Q N C memory.view(k, -1, d)#

        # score = einsum('q t d, k s d -> q k s t', query_t, key_m) # B Q N N

        score = einsum('q h t d, k h s d  -> q k h s t', query_t, key_m) # k q H N N

        score_h = torch.cat((score.max(dim=3)[0], score.max(dim=4)[0]), dim=-1).mean(-1).view(q, k,self.num_heads)
        Score_TOPK=self.num_heads//2
        score_h_topk, ins_indices_rgb_ir_2 = torch.topk(score_h, int(Score_TOPK))#20
        score_p = score_h_topk.mean(-1).view(q, k)
        score_p = torch.sigmoid(score_p)
        # score_h = score_h.view(q, k)

        if self.training: 
            return score_p,score_p,label
        else:
            return score_p, score_p
#ori
class Matcher(Module):
    r"""TransformerDecoderLayer is made up of feature matching and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, dim_feedforward=2048)
        >>> memory = torch.rand(10, 24, 8, 512)
        >>> tgt = torch.rand(20, 24, 8, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, seq_len):
        super(Matcher, self).__init__()
        self.seq_len = seq_len
        self.d_model = 768
        # score_embed = torch.randn(seq_len, seq_len)
        # score_embed = score_embed + score_embed.t()
        # self.score_embed = nn.Parameter(score_embed.view(1, 1, seq_len, seq_len))
        # self.qkv = nn.Linear(d_model, d_model * 3, bias=False)#nn.Linear(d_model, d_model)
        # self.bn1 = nn.BatchNorm1d(1)
        # self.fc2 = nn.Linear(self.seq_len, dim_feedforward)
        # self.bn2 = nn.BatchNorm1d(dim_feedforward)
        # self.relu = nn.ReLU()
        # self.fc3 = nn.Linear(dim_feedforward, 1)
        # self.bn3 = nn.BatchNorm1d(1)
        # self.l2norm = Normalize(2)
        # self.bn3 = nn.LayerNorm(768)#nn.BatchNorm1d(768*self.seq_len)#
    def forward(self, tgt, memory, label=None):
        r"""Pass the inputs through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).

        Shape:
            tgt: [q, h, w, d], where q is the query length, d is d_model, and (h, w) is feature map size
            memory: [k, h, w, d], where k is the memory length
        """

        q, d_5 = tgt.size() # b d*5,  
        # print('tgt.size()',tgt.size())
        # print('tgt.view(q, -1, z)',tgt.view(q, -1, int(d_5/5)).size())
        # assert(h * w == self.seq_len and d == self.d_model)
        k, d_5 = memory.size()
        # assert(h * w == self.seq_len and d == self.d_model)

        # q, h, w, d = tgt.size() # b d*5,  
        # assert(h * w == self.seq_len and d == self.d_model)
        # k, h, w, d = memory.size()
        # assert(h * w == self.seq_len and d == self.d_model)

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)
        query_t =  tgt.view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        key_m = memory.view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(tgt.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(memory.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        score = einsum('q t d, k s d -> q k s t', query_t, key_m) # B Q N N
        # print(score.size())
        # score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1).mean(-1).view(q, k)
        # score_n = torch.cat((score.min(dim=2)[0], score.min(dim=3)[0]), dim=-1).mean(-1).view(q, k)
        score_p = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1).mean(-1).view(q, k)
        score_p = torch.sigmoid(score_p)
        # print(score.max(dim=2)[0].size())
        # score_p = score.max(dim=3)[0].mean(-1).view(q, k)#torch.cat((score.min(dim=2)[0], score.min(dim=3)[0]), dim=-1).mean(-1).view(q, k)
        # score_n = score.min(dim=3)[0].mean(-1).view(q, k)#torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1).mean(-1).view(q, k)

        # score_test = score.reshape(q, k,-1).mean(-1).view(q, k)
        if self.training: 
            return score_p,score_p,label
        else:
            return score_p, score_p


        # if self.training: 
            
        #     return score,label
        # else:
        #     return score
# ori
# class TransformerDecoderLayer(Module):
#     r"""TransformerDecoderLayer is made up of feature matching and feedforward network.

#     Args:
#         d_model: the number of expected features in the input (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).

#     Examples::
#         >>> decoder_layer = TransformerDecoderLayer(d_model=512, dim_feedforward=2048)
#         >>> memory = torch.rand(10, 24, 8, 512)
#         >>> tgt = torch.rand(20, 24, 8, 512)
#         >>> out = decoder_layer(tgt, memory)
#     """

#     def __init__(self, seq_len, d_model=768, dim_feedforward=2048):
#         super(TransformerDecoderLayer, self).__init__()
#         self.seq_len = seq_len
#         self.d_model = d_model
#         score_embed = torch.randn(seq_len, seq_len)
#         score_embed = score_embed + score_embed.t()
#         self.score_embed = nn.Parameter(score_embed.view(1, 1, seq_len, seq_len))
#         self.fc1 = nn.Linear(d_model, d_model)
#         self.bn1 = nn.BatchNorm1d(1)
#         self.fc2 = nn.Linear(self.seq_len, dim_feedforward)
#         self.bn2 = nn.BatchNorm1d(dim_feedforward)
#         self.relu = nn.ReLU()
#         self.fc3 = nn.Linear(dim_feedforward, 1)
#         self.bn3 = nn.BatchNorm1d(1)

#     def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
#         r"""Pass the inputs through the decoder layer.

#         Args:
#             tgt: the sequence to the decoder layer (required).
#             memory: the sequence from the last layer of the encoder (required).

#         Shape:
#             tgt: [q, h, w, d], where q is the query length, d is d_model, and (h, w) is feature map size
#             memory: [k, h, w, d], where k is the memory length
#         """

#         q, d_5 = tgt.size() # b d*5,  
#         # print('tgt.size()',tgt.size())
#         # print('tgt.view(q, -1, z)',tgt.view(q, -1, int(d_5/5)).size())
#         # assert(h * w == self.seq_len and d == self.d_model)
#         k, d_5 = memory.size()
#         # assert(h * w == self.seq_len and d == self.d_model)

#         # q, h, w, d = tgt.size() # b d*5,  
#         # assert(h * w == self.seq_len and d == self.d_model)
#         # k, h, w, d = memory.size()
#         # assert(h * w == self.seq_len and d == self.d_model)

#         z= int(d_5/self.seq_len)
#         d = int(d_5/self.seq_len)
#         tgt = tgt.view(q, -1, z)
#         memory = memory.view(k, -1, d)
#         query = self.fc1(tgt)
#         key = self.fc1(memory)
#         score = einsum('q t d, k s d -> q k s t', query, key) * self.score_embed.sigmoid()
#         score = score.reshape(q * k, self.seq_len, self.seq_len)
#         score = torch.cat((score.max(dim=1)[0], score.max(dim=2)[0]), dim=-1)
#         score = score.view(-1, 1, self.seq_len)
#         score = self.bn1(score).view(-1, self.seq_len)

#         score = self.fc2(score)
#         score = self.bn2(score)
#         score = self.relu(score)
#         score = self.fc3(score)
#         score = score.view(-1, 2).sum(dim=-1, keepdim=True)
#         score = self.bn3(score)
#         score = score.view(q, k)
#         return score



class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, dim_feedforward=2048)
        >>> transformer_decoder = TransformerDecoder(decoder_layer, num_layers=3)
        >>> memory = torch.rand(10, 24, 8, 512)
        >>> tgt = torch.rand(20, 24, 8, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        # self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.layers = ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        r"""Pass the inputs through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).

        Shape:
            tgt: [q, h, w, d*n], where q is the query length, d is d_model, n is num_layers, and (h, w) is feature map size
            memory: [k, h, w, d*n], where k is the memory length
        """

        # tgt = tgt.chunk(self.num_layers, dim=-1)
        # memory = memory.chunk(self.num_layers, dim=-1)
        # for i, mod in enumerate(self.layers):
        #     if i == 0:
        #         score = mod(tgt[i], memory[i])
        #     else:
        #         score = score + mod(tgt[i], memory[i])

        # if self.norm is not None:
        #     q, k = score.size()
        #     score = score.view(-1, 1)
        #     score = self.norm(score)
        #     score = score.view(q, k)

        # return score
        # tgt = tgt.chunk(self.num_layers, dim=-1)
        # memory = memory.chunk(self.num_layers, dim=-1)
        for i, mod in enumerate(self.layers):
            if i == 0:
                score = mod(tgt, memory)
            else:
                score = score + mod(tgt, memory)

        # if self.norm is not None:
        #     q, k = score.size()
        #     score = score.view(-1, 1)
        #     score = self.norm(score)
        #     score = score.view(q, k)

        return score

class TransMatcher(nn.Module):

    def __init__(self, seq_len, d_model=512, num_decoder_layers=3, dim_feedforward=2048):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        self.decoder_layer = TransformerDecoderLayer(seq_len, d_model, dim_feedforward)
        decoder_norm = nn.BatchNorm1d(1)
        self.decoder = TransformerDecoder(self.decoder_layer, num_decoder_layers, decoder_norm)
        self.memory = None
        # self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def make_kernel(self, features):
        self.memory = features

    def forward(self, features,label = None):
        score = self.decoder(self.memory, features)
        if self.training: 
            return score,label
        else:
            return score


# class build_transformer_matchv2(nn.Module):
#     def __init__(self, num_classes, camera_num, view_num, cfg, factory):
#         super(build_transformer, self).__init__()
#         last_stride = cfg.MODEL.LAST_STRIDE
#         model_path = cfg.MODEL.PRETRAIN_PATH
#         model_name = cfg.MODEL.NAME
#         pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
#         self.cos_layer = cfg.MODEL.COS_LAYER
#         self.neck = cfg.MODEL.NECK
#         self.neck_feat = cfg.TEST.NECK_FEAT
#         # self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
#         self.feat_dim = cfg.MODEL.FEAT_DIM
#         self.dropout_rate = cfg.MODEL.DROPOUT_RATE

#         print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

#         if cfg.MODEL.SIE_CAMERA:
#             camera_num = camera_num
#         else:
#             camera_num = 0
#         if cfg.MODEL.SIE_VIEW:
#             view_num = view_num
#         else:
#             view_num = 0

#         self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV)
#         self.in_planes = self.base.in_planes
#         # if pretrain_choice == 'imagenet':
#         #     self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
#         #     print('Loading pretrained ImageNet model......from {}'.format(model_path))

#         self.num_classes = num_classes
#         # self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
#         # if self.ID_LOSS_TYPE == 'arcface':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = Arcface(self.in_planes, self.num_classes,
#         #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'cosface':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = Cosface(self.in_planes, self.num_classes,
#         #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'amsoftmax':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = AMSoftmax(self.in_planes, self.num_classes,
#         #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # elif self.ID_LOSS_TYPE == 'circle':
#         #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
#         #     self.classifier = CircleLoss(self.in_planes, self.num_classes,
#         #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
#         # else:
#             # if self.reduce_feat_dim:
#             #     self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
#             #     self.fcneck.apply(weights_init_xavier)
#             #     self.in_planes = cfg.MODEL.FEAT_DIM
#             # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
#             # self.classifier.apply(weights_init_classifier)

#         self.bottleneck = nn.BatchNorm1d(self.in_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.bottleneck.apply(weights_init_kaiming)

#         self.dropout = nn.Dropout(self.dropout_rate)

#         # if pretrain_choice == 'self':
#         self.base.load_param(model_path,hw_ratio=2)
#         self.base.patch_embed2 = copy.deepcopy(self.base.patch_embed)

#         self.l2norm = Normalize(2)

#         self.gem = GeneralizedMeanPooling(norm=1)
#         # self.part = 5
        
#         # self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck_1.bias.requires_grad_(False)
#         # self.bottleneck_1.apply(weights_init_kaiming)
#         # self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck_2.bias.requires_grad_(False)
#         # self.bottleneck_2.apply(weights_init_kaiming)
#         # self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck_3.bias.requires_grad_(False)
#         # self.bottleneck_3.apply(weights_init_kaiming)
#         # self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
#         # self.bottleneck_4.bias.requires_grad_(False)
#         # self.bottleneck_4.apply(weights_init_kaiming)
#         # self.matcher_ir = TransMatcher(self.part, 768, 3, 768)
#         # self.matcher_rgb = TransMatcher(self.part, 768, 3, 768)
#         # block = self.base.blocks[-1]
#         # layer_norm = self.base.norm
#         # self.b1 = nn.Sequential(
#         #     copy.deepcopy(block),
#         #     copy.deepcopy(layer_norm)
#         # )
#         # self.b2 = nn.Sequential(
#         #     copy.deepcopy(block),
#         #     copy.deepcopy(layer_norm)
#         # )


#     def forward(self, x1, x2,modal=0,label_1=None,label_2=None, gallery_1=None,gallery_2=None):#(self, x, label=None, cam_label= None, view_label=None):
#         single_size = x1.size(0)
#         if (gallery_1 is None) and (gallery_2 is None):
#             features,feat1,feat2,label_1,label_2 = self.base(x1,x2,gallery_1=None,gallery_2=None,modal=modal,label_1=label_1,label_2=label_2)
#         else:
#             if modal==0:
#                 features,feat1,feat2,label_1,label_2,score_rgb, score_ir,pair_labels_rgb,pair_labels_ir= self.base(x1,x2,gallery_1=gallery_1,gallery_2=gallery_2,modal=modal,label_1=label_1,label_2=label_2)
#             else:
#                 features,feat1,feat2,label_1,label_2,score,pair_labels= self.base(x1,x2,gallery_1=gallery_1,gallery_2=gallery_2,modal=modal,label_1=label_1,label_2=label_2)
#         feat = self.bottleneck(features)

#         if self.training:
#             if (gallery_1 is None) and (gallery_2 is None):
#                 return feat,feat[:single_size],feat[single_size:],label_1,label_2#,local_feat_bn[:single_size],local_feat_bn[single_size:]  # global feature for triplet loss
#             else:
#                 if modal==0:
#                     return feat,feat[:single_size],feat[single_size:],label_1,label_2,score_rgb, score_ir,pair_labels_rgb,pair_labels_ir
#                 else:
#                     return feat,feat[:single_size],feat[single_size:],label_1,label_2,score,pair_labels
#         else:
#             if (gallery_1 is None) and (gallery_2 is None):
#                 return self.l2norm(feat)#,self.l2norm(local_feat_bn)
#             else:
#                 return self.l2norm(feat),score
 

#     def load_param(self, trained_path):
#         param_dict = torch.load(trained_path, map_location = 'cpu')
#         for i in param_dict:
#             try:
#                 self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
#             except:
#                 continue
#         print('Loading pretrained model from {}'.format(trained_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        # self.reduce_feat_dim = cfg.MODEL.REDUCE_FEAT_DIM
        self.feat_dim = cfg.MODEL.FEAT_DIM
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH, drop_rate= cfg.MODEL.DROP_OUT,attn_drop_rate=cfg.MODEL.ATT_DROP_RATE, gem_pool=cfg.MODEL.GEM_POOLING, stem_conv=cfg.MODEL.STEM_CONV)
        self.in_planes = self.base.in_planes
        # if pretrain_choice == 'imagenet':
        #     self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
        #     print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.num_classes = num_classes
        # self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        # if self.ID_LOSS_TYPE == 'arcface':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = Arcface(self.in_planes, self.num_classes,
        #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'cosface':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = Cosface(self.in_planes, self.num_classes,
        #                               s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'amsoftmax':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = AMSoftmax(self.in_planes, self.num_classes,
        #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # elif self.ID_LOSS_TYPE == 'circle':
        #     print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
        #     self.classifier = CircleLoss(self.in_planes, self.num_classes,
        #                                 s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        # else:
            # if self.reduce_feat_dim:
            #     self.fcneck = nn.Linear(self.in_planes, self.feat_dim, bias=False)
            #     self.fcneck.apply(weights_init_xavier)
            #     self.in_planes = cfg.MODEL.FEAT_DIM
        # self.classifier_rgb = nn.Linear(self.in_planes, 1000, bias=False)
        # self.classifier_rgb.apply(weights_init_classifier)

        # self.classifier_ir = nn.Linear(self.in_planes, 1000, bias=False)
        # self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.dropout = nn.Dropout(self.dropout_rate)

        # if pretrain_choice == 'self':
        self.base.load_param(model_path,hw_ratio=2)
        self.base.patch_embed2 = copy.deepcopy(self.base.patch_embed)
        # del self.base.patch_embed.conv[1]
        # del self.base.patch_embed.conv[3]
        # del self.base.patch_embed2.conv[1]
        # del self.base.patch_embed2.conv[3]
        # print('self.base.patch_embed',self.base.patch_embed.conv)
        # print('self.base.patch_embed2',self.base.patch_embed2.conv)

        self.l2norm = Normalize(2)

        self.gem = GeneralizedMeanPooling(norm=1)
        # self.part = 5
        
        # self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck_1.bias.requires_grad_(False)
        # self.bottleneck_1.apply(weights_init_kaiming)
        # self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck_2.bias.requires_grad_(False)
        # self.bottleneck_2.apply(weights_init_kaiming)
        # self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck_3.bias.requires_grad_(False)
        # self.bottleneck_3.apply(weights_init_kaiming)
        # self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck_4.bias.requires_grad_(False)
        # self.bottleneck_4.apply(weights_init_kaiming)
        # self.ir_softmax_dim=[]
        # self.rgb_softmax_dim=[]

        # self.matcher_ir = TransMatcher(self.part, 768, 3, 768)
        # self.matcher_rgb = TransMatcher(self.part, 768, 3, 768)
        # self.matcher =Matcher(self.part)
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        # for l_num in range(len(self.matcher_ir.decoder.layers)):
        #     self.matcher_ir.decoder.layers[l_num].qkv = copy.deepcopy(self.base.blocks[-1].attn.qkv)

        # for l_num in range(len(self.matcher_rgb.decoder.layers)):
        #     self.matcher_rgb.decoder.layers[l_num].qkv = copy.deepcopy(self.base.blocks[-1].attn.qkv)

    def forward(self, x1, x2, modal=0,label_1=None,label_2=None,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
    # def forward(self, x1, x2, modal=0,label_1=None,label_2=None):#(self, x, label=None, cam_label= None, view_label=None):
        # single_size = x1.size(0)
        single_size_1 = x1.size(0)
        single_size_2 = x2.size(0)
        features,feat1,feat2,label_1,label_2 = self.base(x1,x2,modal=modal,label_1=label_1,label_2=label_2)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # # JPM branch
        # feature_length = features.size(1) - 1
        # patch_length = feature_length // 4#4
        # token = features[:, 0:1]

        # # if self.rearrange:
        # x = shuffle_unit(features, 5, 2)#
        # # else:
        # # x = features[:, 1:]
        # # lf_1
        # b1_local_feat = x[:, :patch_length]
        # b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        # local_feat_1 = b1_local_feat[:, 0]

        # # lf_2
        # b2_local_feat = x[:, patch_length:patch_length*2]
        # b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        # local_feat_2 = b2_local_feat[:, 0]

        # # lf_3
        # b3_local_feat = x[:, patch_length*2:patch_length*3]#
        # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        # local_feat_3 = b3_local_feat[:, 0]

        # # # lf_4
        # b4_local_feat = x[:, patch_length*3:]#patch_length*4
        # b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        # local_feat_4 = b4_local_feat[:, 0]

        clsfeat = self.bottleneck(global_feat)

        # # if self.reduce_feat_dim:
        # #     global_feat = self.fcneck(global_feat)
        # feature_length = feat.size(1) - 1
        # clsfeat = self.bottleneck(feat[:, 0])
        # # feat_cls = self.dropout(feat)

        ############part token
        # feature_length = features.size(1) - 1
        # patch_length = feature_length // 4
        # x = features[:, 1:]
        # b1_local_feat = x[:, :patch_length]
        # local_feat_1 = self.gem(b1_local_feat.permute(0,2,1)).squeeze() 

        # # lf_2
        # b2_local_feat = x[:, patch_length:patch_length*2]
        # local_feat_2 = self.gem(b2_local_feat.permute(0,2,1)).squeeze() 

        # # lf_3
        # b3_local_feat = x[:, patch_length*2:patch_length*3]
        # # b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        # local_feat_3 = self.gem(b3_local_feat.permute(0,2,1)).squeeze() 

        # # lf_4
        # b4_local_feat = x[:, patch_length*3:]#patch_length*4
        # # b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        # local_feat_4 = self.gem(b4_local_feat.permute(0,2,1)).squeeze() 

        # feat = self.bottleneck(global_feat)



        # local_feat_1_bn = self.bottleneck_1(local_feat_1)
        # local_feat_2_bn = self.bottleneck_2(local_feat_2)
        # local_feat_3_bn = self.bottleneck_3(local_feat_3)
        # local_feat_4_bn = self.bottleneck_4(local_feat_4)

        # local_feat_bn = torch.cat((local_feat_1_bn,local_feat_2,local_feat_3,local_feat_4),dim=1)
        # feat = torch.cat((global_feat,local_feat_bn),dim=1).view(-1,768)
        # feat = self.bottleneck_g(feat).view(-1,768*5)
        # feat = torch.cat((local_feat_1_bn,local_feat_2_bn,local_feat_3_bn),dim=1)
        feat = clsfeat
        # feat = torch.cat((clsfeat,local_feat_1_bn,local_feat_2_bn,local_feat_3_bn,local_feat_4_bn),dim=1)#,local_feat_4_bn clsfeat,,local_feat_4_bn ,local_feat_3_bn
        # feat = torch.cat((clsfeat,local_feat_bn),dim=1)#.view(-1,768)
        # feat = self.bottleneck_g(feat).view(-1,768*5)

        # part_feat=torch.cat((clsfeat,local_feat_bn),dim=1)
        if self.training:
            # self.matcher_rgb.make_kernel(feat[:single_size]) #matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            # score_query_rgb,labels_rgb_match = self.matcher_ir(feat[single_size:],label = label_1)#.detach()
            # self.matcher_ir.make_kernel(feat[single_size:])            
            # score_query_ir,labels_ir_match = self.matcher_ir(feat[single_size:],label = label_2)
            # score_query_rgb,labels_rgb_match = self.matcher(feat[:single_size],feat[:single_size])
            # score_query_rgb,labels_rgb_match = self.matcher(feat[:single_size],feat[:single_size])
            # target_ir = label_2.unsqueeze(1)
            # mask_query_ir = (target_ir == target_ir.t())
            # pair_labels_query_ir = mask_query_ir.float()
            # target_rgb = label_1.unsqueeze(1)
            # mask_query_rgb = (target_rgb == target_rgb.t())
            # pair_labels_query_rgb = mask_query_rgb.float()
            return feat,feat[:single_size_1],feat[single_size_1:],label_1,label_2,cid_rgb,cid_ir,index_rgb,index_ir#,feat[single_size_1+single_size_2:single_size_1*2+single_size_2],feat[single_size_1*2+single_size_2:]
            # return feat,feat[:single_size],feat[single_size:],label_1,label_2,feat,feat,pair_labels_query_rgb,pair_labels_query_ir#,local_feat_bn[:single_size],local_feat_bn[single_size:]  # global feature for triplet loss
        else:
            return feat#self.l2norm(feat)#torch.cat([self.l2norm(feat[:,i*768:(i+1)*768]) for i in range(5)],dim=1)#self.l2norm(feat)# #self.l2norm(feat)#,  #self.l2norm(feat)#,self.l2norm(local_feat_bn)
 


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location = 'cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            except:
                continue
        print('Loading pretrained model from {}'.format(trained_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)
        self.in_planes = self.base.in_planes
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path,hw_ratio=cfg.MODEL.PRETRAIN_HW_RATIO)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))



__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
    'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
}

def make_model(cfg, num_class, camera_num, view_num):
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
