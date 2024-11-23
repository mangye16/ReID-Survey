import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from collections import defaultdict
import random
class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory_all(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory_all, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        # if self.use_hard:
        outputs1 = cm_hard(inputs, targets, self.features, self.momentum)
        # else:
        outputs2 = cm(inputs, targets, self.features, self.momentum)

        outputs1 /= self.temp
        outputs2 /= self.temp
        loss = F.cross_entropy(outputs2, targets) + 0.1*F.cross_entropy(outputs1, targets)
        return loss





#ori
class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets,training_momentum=None):

        inputs = F.normalize(inputs, dim=1).cuda()
        if training_momentum == None:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features, self.momentum)
            else:
                outputs = cm(inputs, targets, self.features, self.momentum)
        else:
            if self.use_hard:
                outputs = cm_hard(inputs, targets, self.features, training_momentum)
            else:
                outputs = cm(inputs, targets, self.features, training_momentum)

        outputs /= self.temp
        # if i_score== None:
        loss = F.cross_entropy(outputs, targets)
        # else:
        #     loss = (F.cross_entropy(outputs, targets,reduction='none')*i_score).mean()

        return loss

class EM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update, not applied for meta learning
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def em(inputs, indexes, features, momentum=0.5):
    return EM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class Memory(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638
        self.sce = CrossEntropyLabelSmooth(num_cluster)
        self.global_std, self.global_mean = torch.zeros(num_features).to(self.devices), \
                                            torch.zeros(num_features).to(self.devices)
    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            # self.features[y] /= self.features[y].norm()

    def __update_params(self):
        camSet = set(self.cam.cpu().numpy().tolist())
        temp_std, temp_mean = [], []
        for cam in camSet:
            cam_feat = self.features[self.cam == cam]
            if len(cam_feat) <= 1: continue
            temp_std.append(cam_feat.std(0))
            temp_mean.append(cam_feat.mean(0))
        self.global_std = self.momentum * torch.stack(temp_std).mean(0) + \
                          (1 - self.momentum) * self.global_std
        self.global_mean = self.momentum * torch.stack(temp_mean).mean(0) + \
                           (1 - self.momentum) * self.global_mean

    def forward(self, inputs, indexes,cameras, symmetric=False):
        # self.__update_params()
        # inputs: B*2048, features: L*2048
        # get scores for all samples, inputs--(64*13638)
        inputs = F.normalize(inputs, dim=1).cuda()
        ###################cam
        # num_cams, cam_set, loss_cam = len(set(self.cam)), set(self.cam.cpu().numpy().tolist()), []
        # # print(cameras)
        # for cur_cam in range(len(cam_set)):
        #     cam_feat = inputs[cur_cam == cameras]
        #     # print(cam_feat.size())
        #     if len(cam_feat) <= 1:
        #         continue
        #     temp_mean, temp_std = cam_feat.mean(0), cam_feat.std(0)
        #     # print(temp_mean.size(),self.global_mean.size())
        #     loss_mean = (temp_mean - self.global_mean).pow(2).sum()
        #     loss_std = (temp_std - self.global_std).pow(2).sum()
        #     loss_cam.append(loss_mean)
        #     loss_cam.append(loss_std)
        # loss_cam = torch.stack(loss_cam).mean()

        #####################


        inputs = em(inputs, indexes, self.features, self.momentum)#B N 
        input_forinc=inputs
        inputs /= self.temp  # 64*13638
        B = inputs.size(0)

        targets = self.labels[indexes].clone()
        # print('targets',targets)
        labels = self.labels.clone()  # 16522, whole labels

        # get centroids for each id
        sim = torch.zeros(labels.max() + 1, B).float().cuda() # C B
        # re-arange simi matrix according to labels to find centroids
        sim.index_add_(0, labels[labels != -1], inputs[:, labels != -1].t().contiguous())

        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        # get counter
        nums.index_add_(0, labels[labels != -1], torch.ones(labels[labels != -1].shape[0], 1).float().cuda())
        sim /= nums.clone().expand_as(sim)  # compute centroids # C B
        sim = sim.t() #B,C
        # loss = self.sce(sim,targets)#torch.tensor([0.]).cuda()#
        # loss = F.cross_entropy(sim, targets)

        # softMask = torch.zeros(sim.t().shape).cuda()
        # softMask.scatter_(1, targets.view(-1, 1), 1)
        # loss = F.cross_entropy(sim, targets)
        ########soft instance
        loss = -(F.softmax(input_forinc/10, 1) * F.log_softmax(inputs, dim=1)).sum(1).mean()


        return loss#,loss_cam 




class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

# class Memory(nn.Module):
#     def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
#         super(Memory, self).__init__()
#         self.num_features = num_features
#         self.num_samples = num_samples
#         self.momentum = momentum
#         self.temp = temp

#         self.register_buffer('features', torch.zeros(num_samples, num_features))
#         # features--(source centers+tgt features)
#         self.register_buffer('labels', torch.zeros(num_samples).long())
#         # labels--(each src and predicted tgt id and outliers), 13638
#         self.ccloss = CrossEntropyLabelSmooth(num_cluster)
#     def updateEM(self, inputs, indexes):
#         # momentum update
#         for x, y in zip(inputs, indexes):
#             self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
#             self.features[y] /= self.features[y].norm()

#     def forward(self, inputs, indexes, symmetric=False):
#         # inputs: B*2048, features: L*2048
#         # get scores for all samples, inputs--(64*13638)
#         targets = self.labels[indexes].clone()
#         labels = self.labels.clone()  # 16522, whole labels #N N
#         inputs = F.normalize(inputs, dim=1).cuda()
#         inputs = em(inputs, indexes, self.features, self.momentum) #B N
        
#         rce_temp = 0.001
#         hce_temp = 1
#         # inputs_rce = inputs/self.temp
#         inputs_hce = inputs/self.temp
#         inputs /= self.temp  # 64*13638
#         B = inputs.size(0)
#         # get centroids for each id
#         sim = torch.zeros(labels.max() + 1, B).float().cuda() #N,B
#         # sim_rce = torch.zeros(labels.max() + 1, B).float().cuda()
#         # re-arange simi matrix according to labels to find centroids
        
#         sim.index_add_(0, labels[labels != -1], inputs[:, labels != -1].t().contiguous())
#         # sim_rce.index_add_(0, labels[labels != -1], inputs_rce[:, labels != -1].t().contiguous())

#         nums = torch.zeros(labels.max() + 1, 1).float().cuda()
#         # get counter
#         nums.index_add_(0, labels[labels != -1], torch.ones(labels[labels != -1].shape[0], 1).float().cuda())
#         # sim_rce /= nums.clone().expand_as(sim_rce)  # compute centroids

#         sim /= nums.clone().expand_as(sim)

#         softMask = torch.zeros(sim.t().shape).cuda()#B,N
#         softMask.scatter_(1, targets.view(-1, 1), 1)#B,N
#         # loss = -(softMask * F.log_softmax(sim.t(), dim=1)).sum(1).mean()
#         loss_sym = 0
#         # loss_ic = F.cross_entropy(inputs_hce, targets)
#         # if symmetric:
#         # loss_sym = -(F.softmax(sim.t(), 1) * F.log_softmax(softMask/self.temp, dim=1)).sum(1).mean()
#         loss_soft = -(F.softmax(softMask, 1) * F.log_softmax(sim.t(), dim=1)).sum(1).mean()
#         return loss_soft#loss_sym,loss_ic#loss,

# class Memory(nn.Module):
#     def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
#         super(Memory, self).__init__()
#         self.num_features = num_features
#         self.num_samples = num_samples
#         self.momentum = momentum
#         self.temp = temp

#         self.register_buffer('features', torch.zeros(num_samples, num_features))
#         # features--(source centers+tgt features)
#         self.register_buffer('labels', torch.zeros(num_samples).long())
#         # labels--(each src and predicted tgt id and outliers), 13638

#     def updateEM(self, inputs, indexes):
#         # momentum update
#         for x, y in zip(inputs, indexes):
#             self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
#             self.features[y] /= self.features[y].norm()

#     def forward(self, inputs, indexes, symmetric=False):
#         # inputs: B*2048, features: L*2048
#         # get scores for all samples, inputs--(64*13638)
#         inputs = em(inputs, indexes, self.features, self.momentum)
#         inputs /= self.temp  # 64*13638
#         B = inputs.size(0)

#         targets = self.labels[indexes].clone()
#         labels = self.labels.clone()  # 16522, whole labels

#         # get centroids for each id
#         sim = torch.zeros(labels.max() + 1, B).float().cuda()
#         # re-arange simi matrix according to labels to find centroids
#         sim.index_add_(0, labels[labels != -1], inputs[:, labels != -1].t().contiguous())

#         nums = torch.zeros(labels.max() + 1, 1).float().cuda()
#         # get counter
#         nums.index_add_(0, labels[labels != -1], torch.ones(labels[labels != -1].shape[0], 1).float().cuda())
#         sim /= nums.clone().expand_as(sim)  # compute centroids
#         softMask = torch.zeros(sim.t().shape).cuda()
#         softMask.scatter_(1, targets.view(-1, 1), 1)
#         loss = -(softMask * F.log_softmax(sim.t(), dim=1)).sum(1).mean()
#         loss_sym = 0
#         if symmetric:
#             loss_sym = -(F.softmax(sim.t(), 1) * F.log_softmax(softMask, dim=1)).sum(1).mean()
#         return loss + loss_sym




class CamMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(CamMemory, self).__init__()
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, num_features).to(self.devices))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long().to(self.devices))

        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638

        self.global_std, self.global_mean = torch.zeros(num_features).to(self.devices), \
                                            torch.zeros(num_features).to(self.devices)

    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def __update_params(self):
        camSet = set(self.cam.cpu().numpy().tolist())
        temp_std, temp_mean = [], []
        for cam in camSet:
            cam_feat = self.features[self.cam == cam]
            if len(cam_feat) <= 1: continue
            temp_std.append(cam_feat.std(0))
            temp_mean.append(cam_feat.mean(0))
        self.global_std = self.momentum * torch.stack(temp_std).mean(0) + \
                          (1 - self.momentum) * self.global_std
        self.global_mean = self.momentum * torch.stack(temp_mean).mean(0) + \
                           (1 - self.momentum) * self.global_mean

    def forward(self, features, indexes, cameras, symmetric=False):
        # inputs: B*2048, features: L*2048
        # get scores for all samples, inputs--(64*13638)
        self.__update_params()  # update camera-level params
        inputs = em(features, indexes, self.features, self.momentum)
        inputs /= self.temp  # 64*13638
        B = inputs.size(0)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()  # 13638, whole labels

        # get centroids for each id
        sim = torch.zeros(labels.max() + 1, B).float().cuda()  # 12123(maxID)*64
        # re-arange simi matrix according to labels
        sim.index_add_(0, labels, inputs.t().contiguous())  # labels--13638(centroids+tgt IDs), inputs--13638*64

        nums = torch.zeros(labels.max() + 1, 1).float().cuda()  # 12123(maxID)
        # get counter
        nums.index_add_(0, labels, torch.ones(self.num_samples, 1).float().cuda())

        sim /= nums.clone().expand_as(sim)

        # get camera loss
        num_cams, cam_set, loss_cam = len(set(self.cam)), set(self.cam.cpu().numpy().tolist()), []
        for cur_cam in range(len(cam_set)):
            cam_feat = features[cur_cam == cameras]
            if len(cam_feat) <= 1:
                continue
            temp_mean, temp_std = cam_feat.mean(0), cam_feat.std(0)

            loss_mean = (temp_mean - self.global_mean).pow(2).sum()
            loss_std = (temp_std - self.global_std).pow(2).sum()
            loss_cam.append(loss_mean)
            loss_cam.append(loss_std)
        loss_cam = 0 if len(loss_cam) == 0 else torch.stack(loss_cam).mean()
        softMask = torch.zeros(sim.t().shape).cuda()
        softMask.scatter_(1, targets.view(-1, 1), 1)
        loss = -(softMask * F.log_softmax(sim.t(), dim=1)).sum(1).mean()
        loss_sym = 0
        if symmetric:
            loss_sym = -(F.softmax(sim.t(), 1) * F.log_softmax(softMask, dim=1)).sum(1).mean()



        return loss,loss_sym,loss_cam

#######ori
class Memory_wise(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)

    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9):
        self.thresh=0.5
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1).cuda()
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)
        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), indexes, cameras, sim.device)
        # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
        # Compute masks for intra-camera neighbors
        # print('mask_instance.sum(1)',mask_instance.sum(1))
        # print('mask_intra.sum(1)',mask_intra.sum(1))
        # print('mask_inter.sum(1)',mask_inter.sum(1))
        # print('sim',sim)
        sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        # print('sim_intra.sum(1)',sim_intra.sum(1))
        nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
        # print('nearest_intra',nearest_intra)
        mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
        # print('num_neighbor_intra',num_neighbor_intra)
        # Activate intra-camera candidates
        sim_exp_intra = sim_exp * mask_intra
        # print('sim_exp_intra',sim_exp_intra)
        score_intra =  F.softmax(sim_exp_intra,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        # print('score_intra',score_intra)
        intra_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
        intra_loss = intra_loss.div(num_neighbor_intra)
        # print('score_intra,intra_loss',score_intra,intra_loss)
        ##########################

        # weight_intra = sim.data * mask_neighbor_intra
        # weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        # weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        # intra_loss = intra_loss.mul(weight_intra)
        # if self.thresh >0:
        #     # Weighting intra-camera neighborhood consistency
        #     weight_intra = sim.data * mask_neighbor_intra
        #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        #     intra_loss = intra_loss.mul(weight_intra)
        # Instance consistency
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log()
        # -------------------------- Inter-Camera Neighborhood Loss --------------------------#
        # Compute masks for inter-camera neighbors
        sim_inter = (sim.data + 1) * mask_inter - 1
        nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        mask_neighbor_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        num_neighbor_inter = mask_neighbor_inter.sum(dim=1)+1
        # print('num_neighbor_inter',num_neighbor_inter)
        # Activate inter-camera candidates
        sim_exp_inter = sim_exp * mask_inter
        score_inter = F.softmax(sim_exp_inter,dim=1) #sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True) #
        score_inter = score_inter.clamp_min(1e-8)
        inter_loss = -score_inter.log().mul(mask_neighbor_inter).sum(dim=1)
        inter_loss = inter_loss.div(num_neighbor_inter)
        # print('score_inter,inter_loss',score_inter,inter_loss)
        # if self.thresh >0:
        #     # Weighting inter-camera neighborhood consistency
        #     weight_inter = sim.data * mask_neighbor_inter
        #     weight_inter = weight_inter.sum(dim=1) / num_neighbor_inter
        #     weight_inter = torch.where(weight_inter > self.thresh, 1, 0)
        #     inter_loss = inter_loss.mul(weight_inter)

########################
        # weight_inter = sim.data * mask_neighbor_inter
        # weight_inter = weight_inter.sum(dim=1) / num_neighbor_inter
        # weight_inter = torch.where(weight_inter > self.thresh, 1, 0)
        # inter_loss = inter_loss.mul(weight_inter)


        # loss = ins_loss + intra_loss * 1.0 + inter_loss * 0.6
        # loss = loss.mean()

        return ins_loss.mean(),intra_loss.mean(),inter_loss.mean()* 0.6#loss#,loss_cam inter_loss.mean()
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter

    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter



class Memory_wise_v1(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v1, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).long())
        self.register_buffer('cam', torch.zeros(num_samples).long())
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9):
        self.thresh=0.6
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1).cuda()
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)

        intrawise_loss_total=torch.tensor([0.]).cuda()

        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), indexes, cameras, sim.device)
        # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
        # Compute masks for intra-camera neighbors
        # print('mask_instance.sum(1)',mask_instance.sum(1))
        # print('mask_intra.sum(1)',mask_intra.sum(1))
        # print('mask_inter.sum(1)',mask_inter.sum(1))
        # print('sim',sim)
        sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        # print('sim_intra.sum(1)',sim_intra.sum(1))
        nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
        # print('nearest_intra',nearest_intra)
        mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
        # print('num_neighbor_intra',num_neighbor_intra)
        # Activate intra-camera candidates
        sim_exp_intra = sim_exp * mask_intra
        # print('sim_exp_intra',sim_exp_intra)
        score_intra =  F.softmax(sim_exp_intra,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        # print('score_intra',score_intra)
        intra_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
        intra_loss = intra_loss.div(num_neighbor_intra)
        # print('score_intra,intra_loss',score_intra,intra_loss)
        ##########################

        # weight_intra = sim.data * mask_neighbor_intra
        # weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        # weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        # intra_loss = intra_loss.mul(weight_intra)
        # if self.thresh >0:
        #     # Weighting intra-camera neighborhood consistency
        #     weight_intra = sim.data * mask_neighbor_intra
        #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        #     intra_loss = intra_loss.mul(weight_intra)
        # Instance consistency
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log()
        # -------------------------- Inter-Camera Neighborhood Loss --------------------------#
        # Compute masks for inter-camera neighbors
        sim_inter = (sim.data + 1) * mask_inter - 1
        nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        mask_neighbor_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        num_neighbor_inter = mask_neighbor_inter.sum(dim=1)+1
        # print('num_neighbor_inter',num_neighbor_inter)
        # Activate inter-camera candidates
        sim_exp_inter = sim_exp * mask_inter
        score_inter = F.softmax(sim_exp_inter,dim=1) #sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True) #
        score_inter = score_inter.clamp_min(1e-8)
        inter_loss = -score_inter.log().mul(mask_neighbor_inter).sum(dim=1)
        inter_loss = inter_loss.div(num_neighbor_inter)

        for c in self.allcam:
            cam_wise = [int(c) for i in range(inputs.size(0))]
            mask_instance, mask_intra, mask_inter = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
            # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
            # Compute masks for intra-camera neighbors
            # print('mask_instance.sum(1)',mask_instance.sum(1))
            # print('mask_intra.sum(1)',mask_intra.sum(1))
            # print('mask_inter.sum(1)',mask_inter.sum(1))
            # print('sim',sim)
            sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
            # print('sim_intra.sum(1)',sim_intra.sum(1))
            nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
            # print('nearest_intra',nearest_intra)
            mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
            num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
            # print('num_neighbor_intra',num_neighbor_intra)
            # Activate intra-camera candidates
            sim_exp_intra = sim_exp * mask_intra
            # print('sim_exp_intra',sim_exp_intra)
            score_intra =  F.softmax(sim_exp_intra,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # print('score_intra',score_intra)
            score_intra = score_intra.clamp_min(1e-8)
            # print('score_intra',score_intra)
            intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
            intrawise_loss = intrawise_loss.div(num_neighbor_intra)
            # print('score_intra,intra_loss',score_intra,intra_loss)
            if self.thresh >0:
                # Weighting intra-camera neighborhood consistency
                weight_intra = sim.data * mask_neighbor_intra
                weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
                weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
                intrawise_loss = intrawise_loss.mul(weight_intra)
            intrawise_loss_total = intrawise_loss_total+intrawise_loss.mean()
    ########################
            # weight_inter = sim.data * mask_neighbor_inter
            # weight_inter = weight_inter.sum(dim=1) / num_neighbor_inter
            # weight_inter = torch.where(weight_inter > self.thresh, 1, 0)
            # inter_loss = inter_loss.mul(weight_inter)
            # print('c:ins_loss.mean(),intra_loss.mean(),inter_loss.mean()',c,ins_loss.mean().item(),intra_loss.mean().item(),inter_loss.mean().item())
            # ins_loss_total=ins_loss_total+ins_loss.mean()
            # intra_loss_total=intra_loss_total+intra_loss.mean()
            # inter_loss_total=inter_loss_total+inter_loss.mean()
        # ins_loss_total=ins_loss_total/len(self.allcam)
        # intra_loss_total=intra_loss_total/len(self.allcam)
        # inter_loss_total=inter_loss_total/len(self.allcam)
        return ins_loss.mean(),intra_loss.mean(),inter_loss.mean()* 0.6,intrawise_loss_total* 0.6#ins_loss_total,intra_loss_total,inter_loss_total* 0.6#loss#,loss_cam inter_loss.mean()* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter

    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter

def pairwise_distance(features_q, features_g):
    x = features_q#torch.from_numpy(features_q)
    y = features_g#torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m#.numpy()


class Memory_wise_v2(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v2, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)
        # self.cam_mem = torch.cat([self.cam_mem[i] for i in self.allcam], 0).detach().data
        # self.cluster = self.generate_cluster_features_all(self.labels,self.features)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)
        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), indexes, cameras, sim.device)
        sim_exp_intra = sim_exp #* mask_intra
        # print('sim_exp_intra',sim_exp_intra)
        score_intra =   F.softmax(sim_exp_intra,dim=1)#sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log().mean()
        return ins_loss#* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter



    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.zeros(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers


class Memory_wise_v2_ori(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v2_ori, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)
        # self.cam_mem = torch.cat([self.cam_mem[i] for i in self.allcam], 0).detach().data
        # self.cluster = self.generate_cluster_features_all(self.labels,self.features)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)


        # # sim_all = self.features.mm(F.normalize(self.cluster, dim=1).t()) #N C
        # # # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
        # # sim_all_B = sim_all[indexes]#B C
        # # sim_all = sim_all_B.mm(sim_all.t())#B N
        # # sim_all = F.softmax(sim_all.detach().data/self.temp,dim=1)


        intrawise_loss_total=torch.tensor([0.]).cuda()
        inswise_loss_total =torch.tensor([0.]).cuda()
        # mask_instance, mask_intra,mask_inter  = self.compute_mask(sim.size(), indexes, cameras, sim.device)#
        # # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
        # # Compute masks for intra-camera neighbors
        # # print('mask_instance.sum(1)',mask_instance.sum(1))
        # # print('mask_intra.sum(1)',mask_intra.sum(1))
        # # print('mask_inter.sum(1)',mask_inter.sum(1))
        # # print('sim',sim)
        # sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        # # print('sim_intra.sum(1)',sim_intra.sum(1))
        # nearest_intra =  sim_intra.max(dim=1, keepdim=True)[0]
        # # print('nearest_intra',nearest_intra)
        # mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        # num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
        # # print('num_neighbor_intra',num_neighbor_intra)
        # # Activate intra-camera candidates
        # sim_exp_intra = sim_exp * mask_intra
        # # print('sim_exp_intra',sim_exp_intra)
        # score_intra =  F.softmax(sim_exp_intra,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # # print('score_intra',score_intra)
        # score_intra = score_intra.clamp_min(1e-8)
        # # print('score_intra',score_intra)
        # intra_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
        # intra_loss = intra_loss.div(num_neighbor_intra)
        # # print('score_intra,intra_loss',score_intra,intra_loss)
        # ##########################

        # # weight_intra = sim.data * mask_neighbor_intra
        # # weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        # # weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        # # intra_loss = intra_loss.mul(weight_intra)
        # # if self.thresh >0:
        # #     # Weighting intra-camera neighborhood consistency
        # #     weight_intra = sim.data * mask_neighbor_intra
        # #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        # #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        # #     intra_loss = intra_loss.mul(weight_intra)
        # # Instance consistency
        # ins_loss = -score_intra.masked_select(mask_instance.bool()).log()
        # # # -------------------------- Inter-Camera Neighborhood Loss --------------------------#
        # # Compute masks for inter-camera neighbors
        # sim_inter = (sim.data + 1) * mask_inter - 1
        # nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        # mask_neighbor_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        # num_neighbor_inter = mask_neighbor_inter.sum(dim=1)+1
        # # print('num_neighbor_inter',num_neighbor_inter)
        # # Activate inter-camera candidates
        # sim_exp_inter = sim_exp * mask_inter
        # score_inter = F.softmax(sim_exp_inter,dim=1) #sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True) #
        # score_inter = score_inter.clamp_min(1e-8)
        # inter_loss = -score_inter.log().mul(mask_neighbor_inter).sum(dim=1)
        # inter_loss = inter_loss.div(num_neighbor_inter)
        # # intra_loss = torch.tensor([0.]).cuda()
        # # inter_loss = torch.tensor([0.]).cuda()
        # # del mask_instance, mask_intra, mask_inter
        for c in self.allcam:
            if stage3==True:
                cam_wise = 1-cameras
            else:
                cam_wise = [int(c) for i in range(inputs.size(0))]
            mask_intra,mask_instance = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
            # sim_wise = self.features.mm(F.normalize(self.cam_mem.detach().data, dim=1).t()) #N C
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t()),dim=1) for i in self.allcam],dim=1).detach().data  #N C/0.05
            # sim_wise = F.softmax(self.features.mm(F.normalize(self.cam_mem[c].detach().data, dim=1).t())/0.05,dim=1).detach().data  #N C
            sim_wise_B = sim_wise[indexes]#B C
            sim_wise = F.normalize(sim_wise_B, dim=1).mm(F.normalize(sim_wise.t(),dim=1))#B N
            # sim_wise = pairwise_distance(sim_wise_B,sim_wise)
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            # sim_wise = sim_wise[indexes]
            # print('sim_wise',sim_wise.size())
            # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
            # Compute masks for intra-camera neighbors
            # print('mask_instance.sum(1)',mask_instance.sum(1))
            # print('mask_intra.sum(1)',mask_intra.sum(1))
            # print('mask_inter.sum(1)',mask_inter.sum(1))
            # print('sim',sim)
            sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
            sim_wise = (sim_wise.data + 1) * mask_intra * (1 - mask_instance) - 1
#########################
            # topk, indices_nearest_intra = torch.topk(sim_intra, 20)#20
            # mask_neighbor_intra = torch.zeros_like(sim_intra)
            # mask_neighbor_intra = mask_neighbor_intra.scatter(1, indices_nearest_intra, 1)

            # topk, indices_sim_wise = torch.topk(sim_wise, 20)#20
            # mask_sim_wise = torch.zeros_like(sim_intra)
            # sim_wise = mask_sim_wise.scatter(1, indices_sim_wise, 1)
#########################
            # print('sim_intra.sum(1)',sim_intra.sum(1))
            nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
            sim_wise_max = sim_wise.max(dim=1, keepdim=True)[0]
            # print('nearest_intra',nearest_intra)
            # print('sim_wise_max',sim_wise_max)
            mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)#nearest_intra * self.neighbor_eps)self.neighbor_eps
            sim_wise = torch.gt(sim_wise, sim_wise_max * self.neighbor_eps)#sim_wise_max * self.neighbor_eps)
            ####################
            num_neighbor_intra = mask_neighbor_intra.sum(dim=1)#.mul(sim_wise).
            num_neighbor_sim_wise = sim_wise.sum(dim=1)#+1
            # print('ori num_neighbor_intra',num_neighbor_intra)
            # print('num_neighbor_sim_wise',num_neighbor_sim_wise)
            # Activate intra-camera candidates
            # num_neighbor_intra = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1#sim_wise.sum(dim=1)+1#
            sim_exp_intra = sim_exp# * mask_intra
            # print('sim_exp_intra',sim_exp_intra)
            score_intra =   F.softmax(sim_exp_intra,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # print('score_intra',score_intra)
            score_intra = score_intra.clamp_min(1e-8)
            # print('score_intra',score_intra)
            # print('sim_wise',sim_wise.size())
            # print('mask_neighbor_intra',mask_neighbor_intra.size())
            # print('mask_neighbor_intra',mask_neighbor_intra.sum(dim=1).view(-1))
            # print('sim_wise',sim_wise.sum(dim=1).view(-1))
            cam_id_count = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1e-8
            # print('cameras',cameras)
            # print('c',c,cam_id_count)
            # print()
            # intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise)
            # print('cam_id_count',cam_id_count)
            mask_neighbor_intra_soft = F.softmax(cam_id_count.float(),dim=-1)
            # print('mask_neighbor_intra_soft',mask_neighbor_intra_soft)

            intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            intrawise_loss = intrawise_loss.div(cam_id_count).mul(mask_neighbor_intra_soft) ##

##################
            # intrcam_mask = cameras.eq(cam_wise).float()
            # intercam_mask = 1-intrcam_mask

            # intrawise_loss_inter = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            # intrawise_loss_inter = intrawise_loss.div(num_neighbor_intra)#.mul(mask_neighbor_intra_soft) 

            # intrawise_loss_intra = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            # intrawise_loss_intra = intrawise_loss.div(cam_id_count)#.mul(mask_neighbor_intra_soft) 

            # intrawise_loss = intrawise_loss_intra*intrcam_mask+intrawise_loss_inter*intrcam_mask
#######################
            # intrawise_loss = intrawise_loss.mul(mask_neighbor_intra_soft) 
            # intrawise_loss = -score_intra.log().mul(sim_wise).sum(dim=1)
            # intrawise_loss = intrawise_loss.div(num_neighbor_sim_wise)
            # print('score_intra,intra_loss',score_intra,intra_loss)
            # if self.thresh >0:
            #     # Weighting intra-camera neighborhood consistency
            #     weight_intra = sim.data * mask_neighbor_intra
            #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
            #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
            #     intrawise_loss = intrawise_loss.mul(weight_intra)
            # intrawise_loss = intrawise_loss
            intrawise_loss_total = intrawise_loss_total+intrawise_loss.sum()#.mean()#
            # inswise_loss_total = inswise_loss_total+inswise_loss.mean()
    ########################
            # weight_inter = sim.data * mask_neighbor_inter
            # weight_inter = weight_inter.sum(dim=1) / num_neighbor_inter
            # weight_inter = torch.where(weight_inter > self.thresh, 1, 0)
            # inter_loss = inter_loss.mul(weight_inter)
            # print('c:ins_loss.mean(),intra_loss.mean(),inter_loss.mean()',c,ins_loss.mean().item(),intra_loss.mean().item(),inter_loss.mean().item())
            # ins_loss_total=ins_loss_total+ins_loss.mean()
            # intra_loss_total=intra_loss_total+intra_loss.mean()
            # inter_loss_total=inter_loss_total+inter_loss.mean()
        inswise_loss = -score_intra.masked_select(mask_instance.bool()).log()
        inswise_loss_total=inswise_loss.mean()#inswise_loss_total/len(self.allcam)
        intrawise_loss_total=intrawise_loss_total/len(self.allcam)
        # inter_loss_total=inter_loss_total/len(self.allcam)
        # if refine==True:
        #     return inswise_loss_total,intrawise_loss_total* 0.6,pseudo_labels_rgb_cm
        # else:
        return inswise_loss_total,intrawise_loss_total#* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter

    # def compute_mask(self, size, img_ids, cam_ids, device):
    #     # print('self.cam2uid',self.cam2uid)
    #     # print('cam_ids',cam_ids)
    #     mask_inter = torch.ones(size, device=device)
    #     for i, cam in enumerate(cam_ids.tolist()):
    #         intra_cam_ids = self.cam2uid[cam]
    #         # print(cam_ids)
            
    #         # print('intra_cam_ids',intra_cam_ids)
    #         mask_inter[i, intra_cam_ids] = 0

    #     mask_intra = 1 - mask_inter
    #     # print(mask_intra)
    #     mask_instance = torch.zeros(size, device=device)
    #     mask_instance[torch.arange(size[0]), img_ids] = 1
    #     return mask_instance, mask_intra#, mask_inter



    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.zeros(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers



class Memory_wise_v2_ori(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v2_ori, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)
        # self.cam_mem = torch.cat([self.cam_mem[i] for i in self.allcam], 0).detach().data
        # self.cluster = self.generate_cluster_features_all(self.labels,self.features)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)


        # # sim_all = self.features.mm(F.normalize(self.cluster, dim=1).t()) #N C
        # # # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
        # # sim_all_B = sim_all[indexes]#B C
        # # sim_all = sim_all_B.mm(sim_all.t())#B N
        # # sim_all = F.softmax(sim_all.detach().data/self.temp,dim=1)


        intrawise_loss_total=torch.tensor([0.]).cuda()
        inswise_loss_total =torch.tensor([0.]).cuda()
        # mask_instance, mask_intra,mask_inter  = self.compute_mask(sim.size(), indexes, cameras, sim.device)#
        # # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
        # # Compute masks for intra-camera neighbors
        # # print('mask_instance.sum(1)',mask_instance.sum(1))
        # # print('mask_intra.sum(1)',mask_intra.sum(1))
        # # print('mask_inter.sum(1)',mask_inter.sum(1))
        # # print('sim',sim)
        # sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        # # print('sim_intra.sum(1)',sim_intra.sum(1))
        # nearest_intra =  sim_intra.max(dim=1, keepdim=True)[0]
        # # print('nearest_intra',nearest_intra)
        # mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        # num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
        # # print('num_neighbor_intra',num_neighbor_intra)
        # # Activate intra-camera candidates
        # sim_exp_intra = sim_exp * mask_intra
        # # print('sim_exp_intra',sim_exp_intra)
        # score_intra =  F.softmax(sim_exp_intra,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # # print('score_intra',score_intra)
        # score_intra = score_intra.clamp_min(1e-8)
        # # print('score_intra',score_intra)
        # intra_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
        # intra_loss = intra_loss.div(num_neighbor_intra)
        # # print('score_intra,intra_loss',score_intra,intra_loss)
        # ##########################

        # # weight_intra = sim.data * mask_neighbor_intra
        # # weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        # # weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        # # intra_loss = intra_loss.mul(weight_intra)
        # # if self.thresh >0:
        # #     # Weighting intra-camera neighborhood consistency
        # #     weight_intra = sim.data * mask_neighbor_intra
        # #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        # #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        # #     intra_loss = intra_loss.mul(weight_intra)
        # # Instance consistency
        # ins_loss = -score_intra.masked_select(mask_instance.bool()).log()
        # # # -------------------------- Inter-Camera Neighborhood Loss --------------------------#
        # # Compute masks for inter-camera neighbors
        # sim_inter = (sim.data + 1) * mask_inter - 1
        # nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        # mask_neighbor_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        # num_neighbor_inter = mask_neighbor_inter.sum(dim=1)+1
        # # print('num_neighbor_inter',num_neighbor_inter)
        # # Activate inter-camera candidates
        # sim_exp_inter = sim_exp * mask_inter
        # score_inter = F.softmax(sim_exp_inter,dim=1) #sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True) #
        # score_inter = score_inter.clamp_min(1e-8)
        # inter_loss = -score_inter.log().mul(mask_neighbor_inter).sum(dim=1)
        # inter_loss = inter_loss.div(num_neighbor_inter)
        # # intra_loss = torch.tensor([0.]).cuda()
        # # inter_loss = torch.tensor([0.]).cuda()
        # # del mask_instance, mask_intra, mask_inter
        for c in self.allcam:
            if stage3==True:
                cam_wise = 1-cameras
            else:
                cam_wise = [int(c) for i in range(inputs.size(0))]
            mask_intra,mask_instance = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
            # sim_wise = self.features.mm(F.normalize(self.cam_mem.detach().data, dim=1).t()) #N C
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t()),dim=1) for i in self.allcam],dim=1).detach().data  #N C/0.05
            # sim_wise = F.softmax(self.features.mm(F.normalize(self.cam_mem[c].detach().data, dim=1).t())/0.05,dim=1).detach().data  #N C
            sim_wise_B = sim_wise[indexes]#B C
            sim_wise = F.normalize(sim_wise_B, dim=1).mm(F.normalize(sim_wise.t(),dim=1))#B N
            # sim_wise = pairwise_distance(sim_wise_B,sim_wise)
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            # sim_wise = sim_wise[indexes]
            # print('sim_wise',sim_wise.size())
            # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
            # Compute masks for intra-camera neighbors
            # print('mask_instance.sum(1)',mask_instance.sum(1))
            # print('mask_intra.sum(1)',mask_intra.sum(1))
            # print('mask_inter.sum(1)',mask_inter.sum(1))
            # print('sim',sim)
            sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
            sim_wise = (sim_wise.data + 1) * mask_intra * (1 - mask_instance) - 1
#########################
            # topk, indices_nearest_intra = torch.topk(sim_intra, 20)#20
            # mask_neighbor_intra = torch.zeros_like(sim_intra)
            # mask_neighbor_intra = mask_neighbor_intra.scatter(1, indices_nearest_intra, 1)

            # topk, indices_sim_wise = torch.topk(sim_wise, 20)#20
            # mask_sim_wise = torch.zeros_like(sim_intra)
            # sim_wise = mask_sim_wise.scatter(1, indices_sim_wise, 1)
#########################
            # print('sim_intra.sum(1)',sim_intra.sum(1))
            nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
            sim_wise_max = sim_wise.max(dim=1, keepdim=True)[0]
            # print('nearest_intra',nearest_intra)
            # print('sim_wise_max',sim_wise_max)
            mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)#nearest_intra * self.neighbor_eps)self.neighbor_eps
            sim_wise = torch.gt(sim_wise, sim_wise_max * self.neighbor_eps)#sim_wise_max * self.neighbor_eps)
            ####################
            num_neighbor_intra = mask_neighbor_intra.sum(dim=1)#.mul(sim_wise).
            num_neighbor_sim_wise = sim_wise.sum(dim=1)#+1
            # print('ori num_neighbor_intra',num_neighbor_intra)
            # print('num_neighbor_sim_wise',num_neighbor_sim_wise)
            # Activate intra-camera candidates
            # num_neighbor_intra = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1#sim_wise.sum(dim=1)+1#
            sim_exp_intra = sim_exp# * mask_intra
            # print('sim_exp_intra',sim_exp_intra)
            score_intra =   F.softmax(sim_exp_intra,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # print('score_intra',score_intra)
            score_intra = score_intra.clamp_min(1e-8)
            # print('score_intra',score_intra)
            # print('sim_wise',sim_wise.size())
            # print('mask_neighbor_intra',mask_neighbor_intra.size())
            # print('mask_neighbor_intra',mask_neighbor_intra.sum(dim=1).view(-1))
            # print('sim_wise',sim_wise.sum(dim=1).view(-1))
            cam_id_count = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1e-8
            # print('cameras',cameras)
            # print('c',c,cam_id_count)
            # print()
            # intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise)
            # print('cam_id_count',cam_id_count)
            mask_neighbor_intra_soft = F.softmax(cam_id_count.float(),dim=-1)
            # print('mask_neighbor_intra_soft',mask_neighbor_intra_soft)

            intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            intrawise_loss = intrawise_loss.div(cam_id_count).mul(mask_neighbor_intra_soft) ##

##################
            # intrcam_mask = cameras.eq(cam_wise).float()
            # intercam_mask = 1-intrcam_mask

            # intrawise_loss_inter = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            # intrawise_loss_inter = intrawise_loss.div(num_neighbor_intra)#.mul(mask_neighbor_intra_soft) 

            # intrawise_loss_intra = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
            # intrawise_loss_intra = intrawise_loss.div(cam_id_count)#.mul(mask_neighbor_intra_soft) 

            # intrawise_loss = intrawise_loss_intra*intrcam_mask+intrawise_loss_inter*intrcam_mask
#######################
            # intrawise_loss = intrawise_loss.mul(mask_neighbor_intra_soft) 
            # intrawise_loss = -score_intra.log().mul(sim_wise).sum(dim=1)
            # intrawise_loss = intrawise_loss.div(num_neighbor_sim_wise)
            # print('score_intra,intra_loss',score_intra,intra_loss)
            # if self.thresh >0:
            #     # Weighting intra-camera neighborhood consistency
            #     weight_intra = sim.data * mask_neighbor_intra
            #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
            #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
            #     intrawise_loss = intrawise_loss.mul(weight_intra)
            # intrawise_loss = intrawise_loss
            intrawise_loss_total = intrawise_loss_total+intrawise_loss.sum()#.mean()#

        inswise_loss = -score_intra.masked_select(mask_instance.bool()).log()
        inswise_loss_total=inswise_loss.mean()#inswise_loss_total/len(self.allcam)
        intrawise_loss_total=intrawise_loss_total/len(self.allcam)
        # inter_loss_total=inter_loss_total/len(self.allcam)
        # if refine==True:
        #     return inswise_loss_total,intrawise_loss_total* 0.6,pseudo_labels_rgb_cm
        # else:
        return inswise_loss_total,intrawise_loss_total#* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter


    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.zeros(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers


class Memory_wise_v3(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_v3, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid_(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)
        # self.cam_mem = torch.cat([self.cam_mem[i] for i in self.allcam], 0).detach().data
        # self.cluster = self.generate_cluster_features_all(self.labels,self.features)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

#     def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):#v5
#         self.thresh=-1
#         self.neighbor_eps  = neighbor_eps
#         inputs = F.normalize(inputs, dim=1)#.cuda()

#         # print(indexes)
#         sim = em(inputs, indexes, self.features, self.momentum)#B N 
#         sim_exp =sim /self.temp  # 64*13638
#         B = inputs.size(0)


#         intrawise_loss_total=torch.tensor([0.]).cuda()
#         inswise_loss_total =torch.tensor([0.]).cuda()
#         # mask_instance, mask_intra,mask_inter  = self.compute_mask(sim.size(), indexes, cameras, sim.device)#
#         # # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
#         intrawise_loss_list=[]
#         # combine_list=[]
#         sim_wise_total = torch.zeros((B,self.features.size(0))).cuda()
#         for c in self.allcam:
#             cam_wise = [int(c) for i in range(inputs.size(0))]
#             mask_intra,mask_instance = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
#             # sim_wise = self.features.mm(F.normalize(self.cam_mem.detach().data, dim=1).t()) #N C
#             # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
#             # sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t()),dim=1) for i in self.allcam],dim=1).detach().data  #N C/0.05
#             sim_wise = F.softmax(self.features.mm(F.normalize(self.cam_mem[c].detach().data, dim=1).t())/0.01,dim=1).detach().data  #N C/0.05
#             # sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.allcam],dim=1).detach().data  #N C/0.05
#             sim_wise_B = sim_wise[indexes]#B C
#             sim_wise = F.normalize(sim_wise_B, dim=1).mm(F.normalize(sim_wise.t(),dim=1))#B N
#             # sim_wise = pairwise_distance(sim_wise_B,sim_wise)
#             # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
#             # sim_wise = sim_wise[indexes]
#             # print('sim_wise',sim_wise.size())
#             # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
#             # Compute masks for intra-camera neighbors
#             # print('mask_instance.sum(1)',mask_instance.sum(1))
#             # print('mask_intra.sum(1)',mask_intra.sum(1))
#             # print('mask_inter.sum(1)',mask_inter.sum(1))
#             # print('sim',sim)
#             sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
#             sim_wise = (sim_wise.data + 1) * mask_intra * (1 - mask_instance) - 1
#     #########################
#             # topk, indices_nearest_intra = torch.topk(sim_intra, 20)#20
#             # mask_neighbor_intra = torch.zeros_like(sim_intra)
#             # mask_neighbor_intra = mask_neighbor_intra.scatter(1, indices_nearest_intra, 1)

#             # topk, indices_sim_wise = torch.topk(sim_wise, 20)#20
#             # mask_sim_wise = torch.zeros_like(sim_intra)
#             # sim_wise = mask_sim_wise.scatter(1, indices_sim_wise, 1)
#     #########################
#             # print('sim_intra.sum(1)',sim_intra.sum(1))
#             nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
#             sim_wise_max = sim_wise.max(dim=1, keepdim=True)[0]
#             # print('nearest_intra',nearest_intra)
#             # print('sim_wise_max',sim_wise_max)
#             mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)#nearest_intra * self.neighbor_eps)self.neighbor_eps
#             sim_wise = torch.gt(sim_wise, sim_wise_max * self.neighbor_eps)#sim_wise_max * self.neighbor_eps)
#             sim_wise_total = sim_wise_total+sim_wise.int()
#         ####################
#         sim_wise = torch.gt(sim_wise, 1)
#         num_neighbor_intra = mask_neighbor_intra.sum(dim=1)#.mul(sim_wise).
#         num_neighbor_sim_wise = sim_wise.sum(dim=1)#+1
#         # print('ori num_neighbor_intra',num_neighbor_intra)
#         # print('num_neighbor_sim_wise',num_neighbor_sim_wise)
#         # Activate intra-camera candidates
#         # num_neighbor_intra = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1#sim_wise.sum(dim=1)+1#
#         sim_exp_intra = sim_exp# * mask_intra
#         # print('sim_exp_intra',sim_exp_intra)
#         score_intra =   F.softmax(sim_exp_intra,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#         # print('score_intra',score_intra)
#         score_intra = score_intra.clamp_min(1e-8)
#         # print('score_intra',score_intra)
#         # print('sim_wise',sim_wise.size())
#         # print('mask_neighbor_intra',mask_neighbor_intra.size())
#         # print('mask_neighbor_intra',mask_neighbor_intra.sum(dim=1).view(-1))
#         # print('sim_wise',sim_wise.sum(dim=1).view(-1))
#         cam_id_count = (sim_wise).sum(dim=1)+1#mask_neighbor_intra.mul
#         # print('len(self.allcam)',len(self.allcam),cam_id_count)
#         # print('cameras',cameras)
#         # print('c',c,cam_id_count)
#         # print()
#         # intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise)
#         # print('cam_id_count',cam_id_count)
#         mask_neighbor_intra_soft = F.softmax(cam_id_count.float(),dim=-1)
#         # print('mask_neighbor_intra_soft',mask_neighbor_intra_soft)
#         intrawise_loss = -score_intra.log().mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#         intrawise_loss = intrawise_loss.div(cam_id_count)#.mul(mask_neighbor_intra_soft) ##
#         intrawise_loss_total = intrawise_loss.mean()
#         # intrawise_loss_list.append(intrawise_loss.view(-1,1))

# ##################
#         # intrcam_mask = cameras.eq(cam_wise).float()
#         # intercam_mask = 1-intrcam_mask

#         # intrawise_loss_inter = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
#         # intrawise_loss_inter = intrawise_loss.div(num_neighbor_intra)#.mul(mask_neighbor_intra_soft) 

#         # intrawise_loss_intra = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
#         # intrawise_loss_intra = intrawise_loss.div(cam_id_count)#.mul(mask_neighbor_intra_soft) 

#         # intrawise_loss = intrawise_loss_intra*intrcam_mask+intrawise_loss_inter*intrcam_mask
# #######################
#         # intrawise_loss = intrawise_loss.mul(mask_neighbor_intra_soft) 
#         # intrawise_loss = -score_intra.log().mul(sim_wise).sum(dim=1)
#         # intrawise_loss = intrawise_loss.div(num_neighbor_sim_wise)
#         # print('score_intra,intra_loss',score_intra,intra_loss)
#         # if self.thresh >0:
#         #     # Weighting intra-camera neighborhood consistency
#         #     weight_intra = sim.data * mask_neighbor_intra
#         #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
#         #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
#         #     intrawise_loss = intrawise_loss.mul(weight_intra)
#         # intrawise_loss = intrawise_loss
#         # intrawise_loss_total = intrawise_loss_total+intrawise_loss.mean()#.mean()#
#         # intrawise_loss_list = torch.cat(intrawise_loss_list,dim=1)
#         # combine_list = torch.cat(combine_list,dim=1)
#         # print('combine_list',combine_list)
#         # combine_list = F.softmax(combine_list.float(),dim=1)
#         # print('combine_list softmax',combine_list)
#         # intrawise_loss_total = intrawise_loss_list.mul(combine_list).sum(1).mean()

#         inswise_loss = -score_intra.masked_select(mask_instance.bool()).log()
#         inswise_loss_total=inswise_loss.mean()#inswise_loss_total/len(self.allcam)
#         # intrawise_loss_total=intrawise_loss_total/len(self.allcam)
#         # inter_loss_total=inter_loss_total/len(self.allcam)
#         # if refine==True:
#         #     return inswise_loss_total,intrawise_loss_total* 0.6,pseudo_labels_rgb_cm
#         # else:
#         return inswise_loss_total,intrawise_loss_total* 0.6#*len(self.allcam)


#     def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):#v4
#         self.thresh=-1
#         self.neighbor_eps  = neighbor_eps
#         inputs = F.normalize(inputs, dim=1)#.cuda()

#         # print(indexes)
#         sim = em(inputs, indexes, self.features, self.momentum)#B N 
#         sim_exp =sim /self.temp  # 64*13638
#         B = inputs.size(0)


#         intrawise_loss_total=torch.tensor([0.]).cuda()
#         inswise_loss_total =torch.tensor([0.]).cuda()
#         # mask_instance, mask_intra,mask_inter  = self.compute_mask(sim.size(), indexes, cameras, sim.device)#
#         # # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
#         intrawise_loss_list=[]
#         # combine_list=[]
#         for c in self.allcam:
#             cam_wise = [int(c) for i in range(inputs.size(0))]
#             mask_intra,mask_instance = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
#             # sim_wise = self.features.mm(F.normalize(self.cam_mem.detach().data, dim=1).t()) #N C
#             # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
#             # sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t()),dim=1) for i in self.allcam],dim=1).detach().data  #N C/0.05
#             sim_wise = F.softmax(self.features.mm(F.normalize(self.cam_mem[c].detach().data, dim=1).t())/0.01,dim=1).detach().data  #N C/0.05
#             # sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.allcam],dim=1).detach().data  #N C/0.05
#             sim_wise_B = sim_wise[indexes]#B C
#             sim_wise = F.normalize(sim_wise_B, dim=1).mm(F.normalize(sim_wise.t(),dim=1))#B N
#             # sim_wise = pairwise_distance(sim_wise_B,sim_wise)
#             # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
#             # sim_wise = sim_wise[indexes]
#             # print('sim_wise',sim_wise.size())
#             # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
#             # Compute masks for intra-camera neighbors
#             # print('mask_instance.sum(1)',mask_instance.sum(1))
#             # print('mask_intra.sum(1)',mask_intra.sum(1))
#             # print('mask_inter.sum(1)',mask_inter.sum(1))
#             # print('sim',sim)
#             sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
#             sim_wise = (sim_wise.data + 1) * mask_intra * (1 - mask_instance) - 1
#     #########################
#             # topk, indices_nearest_intra = torch.topk(sim_intra, 20)#20
#             # mask_neighbor_intra = torch.zeros_like(sim_intra)
#             # mask_neighbor_intra = mask_neighbor_intra.scatter(1, indices_nearest_intra, 1)

#             # topk, indices_sim_wise = torch.topk(sim_wise, 20)#20
#             # mask_sim_wise = torch.zeros_like(sim_intra)
#             # sim_wise = mask_sim_wise.scatter(1, indices_sim_wise, 1)
#     #########################
#             # print('sim_intra.sum(1)',sim_intra.sum(1))
#             nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
#             sim_wise_max = sim_wise.max(dim=1, keepdim=True)[0]
#             # print('nearest_intra',nearest_intra)
#             # print('sim_wise_max',sim_wise_max)
#             mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)#nearest_intra * self.neighbor_eps)self.neighbor_eps
#             sim_wise = torch.gt(sim_wise, sim_wise_max * self.neighbor_eps)#sim_wise_max * self.neighbor_eps)
#             ####################
#             num_neighbor_intra = mask_neighbor_intra.sum(dim=1)#.mul(sim_wise).
#             num_neighbor_sim_wise = sim_wise.sum(dim=1)#+1
#             # print('ori num_neighbor_intra',num_neighbor_intra)
#             # print('num_neighbor_sim_wise',num_neighbor_sim_wise)
#             # Activate intra-camera candidates
#             # num_neighbor_intra = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1#sim_wise.sum(dim=1)+1#
#             sim_exp_intra = sim_exp# * mask_intra
#             # print('sim_exp_intra',sim_exp_intra)
#             score_intra =   F.softmax(sim_exp_intra,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#             # print('score_intra',score_intra)
#             score_intra = score_intra.clamp_min(1e-8)
#             # print('score_intra',score_intra)
#             # print('sim_wise',sim_wise.size())
#             # print('mask_neighbor_intra',mask_neighbor_intra.size())
#             # print('mask_neighbor_intra',mask_neighbor_intra.sum(dim=1).view(-1))
#             # print('sim_wise',sim_wise.sum(dim=1).view(-1))
#             cam_id_count = (sim_wise).sum(dim=1)+1#mask_neighbor_intra.mul
#             # print('len(self.allcam)',len(self.allcam),cam_id_count)
#             # print('cameras',cameras)
#             # print('c',c,cam_id_count)
#             # print()
#             # intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise)
#             # print('cam_id_count',cam_id_count)
#             mask_neighbor_intra_soft = F.softmax(cam_id_count.float(),dim=-1)
#             # print('mask_neighbor_intra_soft',mask_neighbor_intra_soft)
#             intrawise_loss = -score_intra.log().mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#             intrawise_loss = intrawise_loss.div(cam_id_count)#.mul(mask_neighbor_intra_soft) ##
#             intrawise_loss_total = intrawise_loss_total+intrawise_loss.mean()
#         # intrawise_loss_list.append(intrawise_loss.view(-1,1))

# ##################
#         # intrcam_mask = cameras.eq(cam_wise).float()
#         # intercam_mask = 1-intrcam_mask

#         # intrawise_loss_inter = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
#         # intrawise_loss_inter = intrawise_loss.div(num_neighbor_intra)#.mul(mask_neighbor_intra_soft) 

#         # intrawise_loss_intra = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra)
#         # intrawise_loss_intra = intrawise_loss.div(cam_id_count)#.mul(mask_neighbor_intra_soft) 

#         # intrawise_loss = intrawise_loss_intra*intrcam_mask+intrawise_loss_inter*intrcam_mask
# #######################
#         # intrawise_loss = intrawise_loss.mul(mask_neighbor_intra_soft) 
#         # intrawise_loss = -score_intra.log().mul(sim_wise).sum(dim=1)
#         # intrawise_loss = intrawise_loss.div(num_neighbor_sim_wise)
#         # print('score_intra,intra_loss',score_intra,intra_loss)
#         # if self.thresh >0:
#         #     # Weighting intra-camera neighborhood consistency
#         #     weight_intra = sim.data * mask_neighbor_intra
#         #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
#         #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
#         #     intrawise_loss = intrawise_loss.mul(weight_intra)
#         # intrawise_loss = intrawise_loss
#         # intrawise_loss_total = intrawise_loss_total+intrawise_loss.mean()#.mean()#
#         # intrawise_loss_list = torch.cat(intrawise_loss_list,dim=1)
#         # combine_list = torch.cat(combine_list,dim=1)
#         # print('combine_list',combine_list)
#         # combine_list = F.softmax(combine_list.float(),dim=1)
#         # print('combine_list softmax',combine_list)
#         # intrawise_loss_total = intrawise_loss_list.mul(combine_list).sum(1).mean()

#         inswise_loss = -score_intra.masked_select(mask_instance.bool()).log()
#         inswise_loss_total=inswise_loss.mean()#inswise_loss_total/len(self.allcam)
#         intrawise_loss_total=intrawise_loss_total/len(self.allcam)
#         # inter_loss_total=inter_loss_total/len(self.allcam)
#         # if refine==True:
#         #     return inswise_loss_total,intrawise_loss_total* 0.6,pseudo_labels_rgb_cm
#         # else:
#         return inswise_loss_total,intrawise_loss_total* 0.6#*len(self.allcam)


    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9):
        self.thresh=0.5
        self.neighbor_eps  = 0.8
        inputs = F.normalize(inputs, dim=1).cuda()
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)
        mask_instance, mask_intra, mask_inter = self.compute_mask(sim.size(), indexes, cameras, sim.device)
        # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
        # Compute masks for intra-camera neighbors
        # print('mask_instance.sum(1)',mask_instance.sum(1))
        # print('mask_intra.sum(1)',mask_intra.sum(1))
        # print('mask_inter.sum(1)',mask_inter.sum(1))
        # print('sim',sim)
        sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
        # print('sim_intra.sum(1)',sim_intra.sum(1))
        nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
        # print('nearest_intra',nearest_intra)
        mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
        num_neighbor_intra = mask_neighbor_intra.sum(dim=1)+1
        # print('num_neighbor_intra',num_neighbor_intra)
        # Activate intra-camera candidates
        # sim_exp_intra = sim_exp * mask_intra
        # # print('sim_exp_intra',sim_exp_intra)
        score_intra =  F.softmax(sim_exp,dim=1)# sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # print('score_intra',score_intra)
        score_intra = score_intra.clamp_min(1e-8)
        # # print('score_intra',score_intra)
        # intra_loss = -score_intra.log().mul(mask_neighbor_intra).sum(dim=1)
        # intra_loss = intra_loss.div(num_neighbor_intra)
        # print('score_intra,intra_loss',score_intra,intra_loss)
        ##########################

        # weight_intra = sim.data * mask_neighbor_intra
        # weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        # weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        # intra_loss = intra_loss.mul(weight_intra)
        # if self.thresh >0:
        #     # Weighting intra-camera neighborhood consistency
        #     weight_intra = sim.data * mask_neighbor_intra
        #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
        #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
        #     intra_loss = intra_loss.mul(weight_intra)
        # Instance consistency
        ins_loss = -score_intra.masked_select(mask_instance.bool()).log()
        # # -------------------------- Inter-Camera Neighborhood Loss --------------------------#
        # # Compute masks for inter-camera neighbors
        # sim_inter = (sim.data + 1) * mask_inter - 1
        # nearest_inter = sim_inter.max(dim=1, keepdim=True)[0]
        # mask_neighbor_inter = torch.gt(sim_inter, nearest_inter * self.neighbor_eps)
        # num_neighbor_inter = mask_neighbor_inter.sum(dim=1)+1
        # # print('num_neighbor_inter',num_neighbor_inter)
        # # Activate inter-camera candidates
        # sim_exp_inter = sim_exp * mask_inter
        # score_inter = F.softmax(sim_exp_inter,dim=1) #sim_exp_inter / sim_exp_inter.sum(dim=1, keepdim=True) #
        # score_inter = score_inter.clamp_min(1e-8)
        # inter_loss = -score_inter.log().mul(mask_neighbor_inter).sum(dim=1)
        # inter_loss = inter_loss.div(num_neighbor_inter)
        # print('score_inter,inter_loss',score_inter,inter_loss)
        # if self.thresh >0:
        #     # Weighting inter-camera neighborhood consistency
        #     weight_inter = sim.data * mask_neighbor_inter
        #     weight_inter = weight_inter.sum(dim=1) / num_neighbor_inter
        #     weight_inter = torch.where(weight_inter > self.thresh, 1, 0)
        #     inter_loss = inter_loss.mul(weight_inter)

########################
        # weight_inter = sim.data * mask_neighbor_inter
        # weight_inter = weight_inter.sum(dim=1) / num_neighbor_inter
        # weight_inter = torch.where(weight_inter > self.thresh, 1, 0)
        # inter_loss = inter_loss.mul(weight_inter)


        # loss = ins_loss + intra_loss * 1.0 + inter_loss * 0.6
        # loss = loss.mean()

        return ins_loss.mean()#,intra_loss.mean(),inter_loss.mean()* 0.6#loss#,loss_cam inter_loss.mean()
        # return 0
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        # mask_inter = torch.ones(size, device=device)
        # for i, cam in enumerate(cam_ids.tolist()):
        #     intra_cam_ids = self.cam2uid[cam]
        #     # print(cam_ids)
            
        #     # print('intra_cam_ids',intra_cam_ids)
        #     mask_inter[i, intra_cam_ids] = 0

        # mask_intra = 1 - mask_inter
        # # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance,mask_instance,mask_instance# mask_intra, mask_inter

    # def compute_mask(self, size, img_ids, cam_ids, device):
    #     # print('self.cam2uid',self.cam2uid)
    #     # print('cam_ids',cam_ids)
    #     mask_inter = torch.ones(size, device=device)
    #     for i, cam in enumerate(cam_ids.tolist()):
    #         intra_cam_ids = self.cam2uid[cam]
    #         # print(cam_ids)
    #         # print('intra_cam_ids',intra_cam_ids)
    #         mask_inter[i, intra_cam_ids] = 0

    #     mask_intra = 1 - mask_inter
    #     # print(mask_intra)
    #     mask_instance = torch.zeros(size, device=device)
    #     mask_instance[torch.arange(size[0]), img_ids] = 1
    #     return mask_instance, mask_intra, mask_inter


    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.ones(size, device=device)#zeros
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers

class Memory_wise_vbatch(nn.Module):
    def __init__(self, num_features, num_samples,num_cluster, temp=0.05, momentum=0.2):
        super(Memory_wise_vbatch, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.devices = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        # features--(source centers+tgt features)
        self.register_buffer('labels', torch.zeros(num_samples).int())
        self.register_buffer('cam', torch.zeros(num_samples).int())
        self.cam_mem = defaultdict(list)
        # num_encoder_layers=4
        # trans_layer = nn.TransformerEncoderLayer(d_model=2048,nhead=8) 
        # self.encoder=nn.TransformerEncoder(trans_layer,num_layers=num_encoder_layers)
        # labels--(each src and predicted tgt id and outliers), 13638
    def cam2uid(self):
        uid2cam = zip(range(self.num_samples), self.cam)
        self.cam2uid = defaultdict(list)
        for uid, cam in uid2cam:
            self.cam2uid[int(cam.cpu().data)].append(uid)
        # print(self.cam2uid)
        self.allcam = torch.unique(self.cam).cpu().numpy().tolist()
        print(self.allcam)
        # self.cam_mem = defaultdict(list)
    def cam_mem_gen(self):
        num_c_total=0
        for c in self.allcam:
            self.cam_mem[c],num_c = self.generate_cluster_features(self.labels,self.features,c)
            num_c_total= num_c_total+num_c
        print(num_c_total)
        # self.cam_mem = torch.cat([self.cam_mem[i] for i in self.allcam], 0).detach().data
        # self.cluster = self.generate_cluster_features_all(self.labels,self.features)


    def updateEM(self, inputs, indexes):
        # momentum update
        for x, y in zip(inputs, indexes):
            self.features[y] = self.momentum * self.features[y] + (1. - self.momentum) * x
            self.features[y] /= self.features[y].norm()

    def forward(self, inputs, indexes,cameras,neighbor_eps=0.9,refine=False,stage3=False):
        self.thresh=-1
        self.neighbor_eps  = neighbor_eps
        inputs = F.normalize(inputs, dim=1)#.cuda()

        # print(indexes)
        sim = em(inputs, indexes, self.features, self.momentum)#B N 
        sim_exp =sim /self.temp  # 64*13638
        B = inputs.size(0)


        intrawise_loss_total=torch.tensor([0.]).cuda()
        inswise_loss_total =torch.tensor([0.]).cuda()
        topk_list=[]
        for c in self.allcam:
            if stage3==True:
                cam_wise = 1-cameras
            else:
                cam_wise = [int(c) for i in range(inputs.size(0))]
            mask_intra,mask_instance = self.compute_mask_camwise(sim.size(), indexes, cam_wise, sim.device)
            # sim_wise = self.features.mm(F.normalize(self.cam_mem.detach().data, dim=1).t()) #N C
            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            sim_wise = torch.cat([F.softmax(self.features.mm(F.normalize(self.cam_mem[i].detach().data, dim=1).t())/self.temp,dim=1) for i in self.allcam],dim=1).detach().data  #N C
            sim_wise_B = sim_wise[indexes]#B C
            sim_wise = sim_wise_B.mm(sim_wise.t())#B N


            # sim_wise = F.softmax(sim_wise.detach().data/self.temp,dim=1)
            # sim_wise = sim_wise[indexes]
            # print('sim_wise',sim_wise.size())
            # -------------------------- Intra-camera Neighborhood Loss -------------------------- #
            # Compute masks for intra-camera neighbors
            # print('mask_instance.sum(1)',mask_instance.sum(1))
            # print('mask_intra.sum(1)',mask_intra.sum(1))
            # print('mask_inter.sum(1)',mask_inter.sum(1))
            # print('sim',sim)
            sim_intra = (sim.data + 1) * mask_intra * (1 - mask_instance) - 1
#########################
            # topk, indices_nearest_intra = torch.topk(sim_intra, 20)#20
            # mask_neighbor_intra = torch.zeros_like(sim_intra)
            # mask_neighbor_intra = mask_neighbor_intra.scatter(1, indices_nearest_intra, 1)

            topk, indices_sim_wise = torch.topk(sim_wise, 1)#20
            topk_list.append(indices_sim_wise.view(-1))
            # mask_sim_wise = torch.zeros_like(sim_intra)
            # sim_wise = mask_sim_wise.scatter(1, indices_sim_wise, 1)
#########################
            # print('sim_intra.sum(1)',sim_intra.sum(1))
            nearest_intra = sim_intra.max(dim=1, keepdim=True)[0]
            sim_wise_max = sim_wise.max(dim=1, keepdim=True)[0]
            # print('nearest_intra',nearest_intra)
            mask_neighbor_intra = torch.gt(sim_intra, nearest_intra * self.neighbor_eps)
            sim_wise = torch.gt(sim_wise, sim_wise_max * self.neighbor_eps)
            ####################
            num_neighbor_intra = mask_neighbor_intra.mul(sim_wise).sum(dim=1)+1
            # num_neighbor_sim_wise = sim_wise.sum(dim=1)+1
            # print('num_neighbor_intra',num_neighbor_intra)
            # Activate intra-camera candidates
            sim_exp_intra = sim_exp * mask_intra
            # print('sim_exp_intra',sim_exp_intra)
            score_intra =   sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# F.softmax(sim_exp_intra,dim=1)#
            # print('score_intra',score_intra)
            score_intra = score_intra.clamp_min(1e-8)
            # print('score_intra',score_intra)
            # print('sim_wise',sim_wise.size())
            # print('mask_neighbor_intra',mask_neighbor_intra.size())
            
            intrawise_loss = -score_intra.log().mul(mask_neighbor_intra).mul(sim_wise).sum(dim=1)
            intrawise_loss = intrawise_loss.div(num_neighbor_intra)

            # intrawise_loss = -score_intra.log().mul(sim_wise).sum(dim=1)
            # intrawise_loss = intrawise_loss.div(num_neighbor_sim_wise)

            # print('score_intra,intra_loss',score_intra,intra_loss)
            # if self.thresh >0:
            #     # Weighting intra-camera neighborhood consistency
            #     weight_intra = sim.data * mask_neighbor_intra
            #     weight_intra = weight_intra.sum(dim=1).div(num_neighbor_intra)
            #     weight_intra = torch.where(weight_intra > self.thresh, 1, 0)
            #     intrawise_loss = intrawise_loss.mul(weight_intra)
            # intrawise_loss = intrawise_loss
            intrawise_loss_total = intrawise_loss_total+intrawise_loss.mean()
            # inswise_loss_total = inswise_loss_total+inswise_loss.mean()
    ########################
            # weight_inter = sim.data * mask_neighbor_inter
            # weight_inter = weight_inter.sum(dim=1) / num_neighbor_inter
            # weight_inter = torch.where(weight_inter > self.thresh, 1, 0)
            # inter_loss = inter_loss.mul(weight_inter)
            # print('c:ins_loss.mean(),intra_loss.mean(),inter_loss.mean()',c,ins_loss.mean().item(),intra_loss.mean().item(),inter_loss.mean().item())
            # ins_loss_total=ins_loss_total+ins_loss.mean()
            # intra_loss_total=intra_loss_total+intra_loss.mean()
            # inter_loss_total=inter_loss_total+inter_loss.mean()
        inswise_loss = -score_intra.masked_select(mask_instance.bool()).log()
        inswise_loss_total=inswise_loss.mean()#inswise_loss_total/len(self.allcam)
        intrawise_loss_total=intrawise_loss_total/len(self.allcam)
        # inter_loss_total=inter_loss_total/len(self.allcam)
        # if refine==True:
        #     return inswise_loss_total,intrawise_loss_total* 0.6,pseudo_labels_rgb_cm
        # else:
        # trans_feat_list=[]
        # for tok_i in topk_list:
        #     # topk_list=torch.cat(topk_list,dim=0).view(-1)
        # tok_i = topk_list[random.randint(0, len(topk_list)-1)]
        # # print(self.labels[indexes].view(-1,16))
        # all_topk_feat = self.features[self.labels[indexes]].view(-1,2048)#self.features[tok_i].view(-1,2048)#
        # # print('all_topk_feat',all_topk_feat.size())
        # # print('inputs',inputs.size())
        # # all_topk_feat = all_topk_feat.view(80,5,2048)
        # mix_feat = inputs.view(-1,16,2048)#torch.cat((inputs.view(-1,16,2048),all_topk_feat.view(-1,16,2048)),dim=1)#seqlenth  batch  dim #########
        # trans_feat = self.encoder(mix_feat).contiguous().view(-1,2048)#[:,:16,:]
        # # print('trans_feat',trans_feat.size())
        # # trans_feat_list.append(trans_feat)
        # # trans_feat = torch.cat(topk_list,dim=0)
        # # print(trans_feat.size())
        # # trans_feat =trans_feat_all[:inputs.size(0)].view(inputs.size(0),-1)
        # # del trans_feat_all
        return inswise_loss_total,intrawise_loss_total#,trans_feat#,self.labels[indexes]#* 0.6
    def compute_mask(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_inter = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids.tolist()):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            # print('intra_cam_ids',intra_cam_ids)
            mask_inter[i, intra_cam_ids] = 0

        mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_instance, mask_intra, mask_inter





    def compute_mask_camwise(self, size, img_ids, cam_ids, device):
        # print('self.cam2uid',self.cam2uid)
        # print('cam_ids',cam_ids)
        mask_intra = torch.ones(size, device=device)
        for i, cam in enumerate(cam_ids):
            intra_cam_ids = self.cam2uid[cam]
            # print(cam_ids)
            
            # print('intra_cam_ids',intra_cam_ids)
            mask_intra[i, intra_cam_ids] = 1

        # mask_intra = 1 - mask_inter
        # print(mask_intra)
        mask_instance = torch.zeros(size, device=device)
        mask_instance[torch.arange(size[0]), img_ids] = 1
        return mask_intra,mask_instance

    def generate_cluster_features(self,labels, features,cam_id):
        centers = collections.defaultdict(list)
        for i, label in enumerate(self.labels):
            # print(int(self.cam[i]),int(cam_id))
            if (label == -1) or (int(self.cam[i]) != int(cam_id)):
                continue
            centers[int(label)].append(self.features[i])
            # print('cam label',self.cam[i],label)
        # print(centers)
        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        print('cam cluster',cam_id,centers.size(0))
        return centers, centers.size(0)

    def generate_cluster_features_all(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if (label == -1):
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0).cuda()
        return centers