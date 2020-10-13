import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalized features
        x = F.normalize(input)
        # normalized weights 
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits 
        # add margin 
        theta = torch.acoss(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale 
        output *= self.s
        
        return output

# class ArcCos(nn.Module):
#     def __init__(self, in_features, out_features, s=30.0, m=0.50, bias=False)
#         super(ArcCos, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)

#         self.th = math.cos(math.pi - m)
#         self.mm = math.sin(math.pi - m) * m
        
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.register_parameters()

#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             boud = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
        
#     def forward(self, input, label):
#         # fix for

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos,self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)
    def forward(self, input, label=None):
        # normalize features 
        x = F.normalize(input)
        # normalize weights 
        W = F.normalize(self.W)
        # dot product
        print('debug logging')
        print(x.shape)
        print(W.shape)
        logits = F.linear(x, W)        
        if label is None:
            return logits
        # feature re-scale 
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits 
        return output

class CosFace(nn.Module):
    def __init__(self, num_features, num_classes, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.num_features = num_features 
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize feature 
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits 

        # # * add margin version 
        # target_logits = logits - self.m
        # one_hot = torch.zeros_like(logits)
        # one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        # output = logits - (1 - one_hot) + target_logits * one_hot

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, abel.view(-1, 1).long(), 1)
        output = logits - one_hot * self.m
        # feature re-scale
        output *= self.s

        return output
    
    def __repr__(self):
        return self.__class__.__name__ +\
               '(' + 'num_features='+'{}'.format(self.num_features) + \
               ','+'num_classes=' + '{}'.format(self.num_classes) +\
               ', ' + 's=' + str(self.s) + \
               ', ' + 'm=' + str(self.m) +\
               ')'