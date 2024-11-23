from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Module
import collections
from torch import einsum
from torch.autograd import Variable
from clustercontrast.models.cm import ClusterMemory
part=5
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=True):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        # correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss#, correct

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3

        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss



class ClusterContrastTrainer_pretrain_camera_confusionrefine(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_confusionrefine, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)#1
        self.tri = TripletLoss_WRT()
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.criterion_ce_soft = SoftEntropy().cuda()
        # self.match_loss = PairwiseMatchingLoss(self.encoder.matcher)
    # def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
    #     all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
    #              print_freq=10, train_iters=400):
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()
        




        lamda_c = 0.1#0.1
        if epoch>=self.cmlabel:
            
            concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        else:
            
            concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)

        part=4

        
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_rgb]
        # ir_softmax_dim=[i.size(0) for i in percam_memory_ir]
        # # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # # self.encoder.module.ir_softmax_dim=ir_softmax_dim

        # distribute_map_rgb = torch.cat(percam_memory_rgb, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb).detach().t()
        # # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())


        # distribute_map_ir = torch.cat(percam_memory_ir, dim=0)
        # distribute_map_ir = F.normalize(distribute_map_ir).detach().t()
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_ir.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())


        # percam_memory_all=percam_memory_rgb+percam_memory_ir
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_all]
        # ir_softmax_dim=[i.size(0) for i in percam_memory_all]
        # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # self.encoder.module.ir_softmax_dim=ir_softmax_dim

        # distribute_map_rgb = torch.cat(percam_memory_all, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb)
        # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())


        # distribute_map_ir = torch.cat(percam_memory_all, dim=0)
        # distribute_map_ir = F.normalize(distribute_map_ir)
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_ir.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()

        # del percam_memory_ir,percam_memory_rgb
        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            # inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1

            # # inputs_ir,inputs_ir1,labels_ir, indexes_ir,cids_ir = self._parse_data_rgb(inputs_ir) #inputs_ir1
            # inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)
            # # forward
            # inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            # labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            # cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)
            # if epoch%2 == 0:
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
            #     cids_ir =  torch.cat((cids_ir,cids_ir),-1)

            #     indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            #     indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()



            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            

            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + self.memory_ir(f_out_ir_r, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)# +self.memory_rgb(f_out_rgb_r, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,percam_tempV_ir)
            loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb)
            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_confusion_ir = torch.tensor([0.]).cuda()
            loss_confusion_rgb = torch.tensor([0.]).cuda()
            thresh=0.8
            lamda_i = 0
################cpsrefine
            
            # if epoch>=0:#self.cmlabel:
#                 #########rgb
#                 percam_memory_all  = percam_memory_rgb+percam_memory_ir
#                 # if epoch %2 == 0:
#                 concate_mem_ir =  torch.cat(percam_memory_ir,dim=0) #C_RGB+C_IR dim
#                 # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
#                 sim_concate_rgb = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(percam_memory_ir[i].detach().data.t()),dim=1) for i in range(len(percam_memory_ir))],dim=1)
#                 # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
#                 # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
#                 sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1).detach()
#                 confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem_ir)# B dim
#                 # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
#                 # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
#                 # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
#                 rgb_ir_loss = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
#                 # loss_confusion_rgb = self.tri(torch.cat((F.normalize(f_out_rgb, dim=1),F.normalize(confusion_feat_rgb, dim=1)),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
#             #############ir
#                 # else:
#                 # concate_mem =  torch.cat((self.memory_rgb.features,self.memory_ir.features),dim=0) #C_RGB+C_IR dim
#                 concate_mem_rgb =  torch.cat(percam_memory_rgb,dim=0)
#                 # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
#                 sim_concate_ir = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(percam_memory_rgb[i].detach().data.t()),dim=1) for i in range(len(percam_memory_rgb))],dim=1)
#                 # sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1) ##B C_RGB+C_IR
#                 # sim_concate_weight_ir = torch.cat((F.softmax(sim_concate_ir[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_ir[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
#                 sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1).detach()
#                 confusion_feat_ir = sim_concate_weight_ir.mm(concate_mem_rgb)# B dim
#                 # loss_confusion_ir = 0.1*self.memory_ir(confusion_feat, labels_ir)
#                 # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_ir.features.t())
#                 # loss_confusion_ir= F.cross_entropy(confusion_out, labels_ir)
#                 ir_rgb_loss = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
#                 # loss_confusion_ir = self.tri(torch.cat((F.normalize(f_out_ir, dim=1),F.normalize(confusion_feat_ir, dim=1)),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
# #################
#                 concate_mem_rgb =  torch.cat(percam_memory_rgb,dim=0) #C_RGB+C_IR dim
#                 # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
#                 sim_concate_rgb = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(percam_memory_rgb[i].detach().data.t()),dim=1) for i in range(len(percam_memory_rgb))],dim=1)
#                 # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
#                 # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
#                 sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1).detach()
#                 confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem_rgb)# B dim
#                 # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
#                 # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
#                 # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
#                 rgb_rgb_loss = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
# #####################
#                 concate_mem_ir =  torch.cat(percam_memory_ir,dim=0) #C_RGB+C_IR dim
#                 # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
#                 sim_concate_ir = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(percam_memory_ir[i].detach().data.t()),dim=1) for i in range(len(percam_memory_ir))],dim=1)
#                 # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
#                 # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
#                 sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1).detach()
#                 confusion_feat_ir = sim_concate_weight_ir.mm(concate_mem_ir)# B dim
#                 # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
#                 # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
#                 # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
#                 ir_ir_loss = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))


#################v1
            # percam_memory_all  = percam_memory_rgb+percam_memory_ir #  f*M.t()*M   B dim   *   dim C   *   C dim    
            # # if epoch %2 == 0:
            # concate_mem =  torch.cat(percam_memory_all,dim=0) #C_RGB+C_IR dim
            # # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
            # sim_concate_rgb = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(percam_memory_all[i].detach().data.t()),dim=1) for i in range(len(percam_memory_all))],dim=1)
            # # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
            # # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
            # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1).detach()
            # confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem)# B dim
            # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
            # # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
            # # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
            # rgb_ir_loss = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
            # # loss_confusion_rgb = self.tri(torch.cat((F.normalize(f_out_rgb, dim=1),F.normalize(confusion_feat_rgb, dim=1)),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
            # # else:
            # # concate_mem =  torch.cat((self.memory_rgb.features,self.memory_ir.features),dim=0) #C_RGB+C_IR dim
            # concate_mem =  torch.cat(percam_memory_all,dim=0)
            # # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
            # sim_concate_ir = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(percam_memory_all[i].detach().data.t()),dim=1) for i in range(len(percam_memory_all))],dim=1)
            # # sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1) ##B C_RGB+C_IR
            # # sim_concate_weight_ir = torch.cat((F.softmax(sim_concate_ir[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_ir[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
            # sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1).detach()
            # confusion_feat_ir = sim_concate_weight_ir.mm(concate_mem)# B dim
            # # loss_confusion_ir = 0.1*self.memory_ir(confusion_feat, labels_ir)
            # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_ir.features.t())
            # # loss_confusion_ir= F.cross_entropy(confusion_out, labels_ir)
            # ir_rgb_loss = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
            # # loss_confusion_ir = self.tri(torch.cat((F.normalize(f_out_ir, dim=1),F.normalize(confusion_feat_ir, dim=1)),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))


###########homo
            # if epoch>=10:
            #     features_rgb = F.normalize(f_out_rgb,dim=1)#features_rgb_.cuda()#f_out_rgb#
            #     features_rgb_input = features_rgb.mm(distribute_map_rgb)#self.encoder.module.classifier_rgb(features_rgb)#*20
            #     features_rgb= features_rgb_input#*20
            #     # rgb_softmax_dim= self.encoder.module.rgb_softmax_dim
            #     # print('rgb_softmax_dim',rgb_softmax_dim)
            #     features_rgb_1 = F.softmax(features_rgb[:,:rgb_softmax_dim[0]], dim=1)
            #     features_rgb_2 = F.softmax(features_rgb[:,rgb_softmax_dim[0]:rgb_softmax_dim[0]+rgb_softmax_dim[1]], dim=1)
            #     features_rgb_3 = F.softmax(features_rgb[:,rgb_softmax_dim[0]+rgb_softmax_dim[1]:rgb_softmax_dim[0]+rgb_softmax_dim[1]+rgb_softmax_dim[2]], dim=1)
            #     features_rgb_4 = F.softmax(features_rgb[:,rgb_softmax_dim[0]+rgb_softmax_dim[1]+rgb_softmax_dim[2]:], dim=1)
            #     # print(features_rgb_1.size(),features_rgb_2.size(),features_rgb_3.size(),features_rgb_4.size())
            #     features_rgb_sim = torch.cat((features_rgb_1,features_rgb_2,features_rgb_3,features_rgb_4), dim=1)

            #     # print('features_rgb_input,features_rgb_sim',features_rgb_input.size(),features_rgb_sim.size())
                
            #     rgb_rgb_loss = self.criterion_ce_soft(features_rgb_input,features_rgb_sim)
            #     # confusion_feat_rgb = features_rgb_sim.mm(self.encoder.module.classifier_rgb.weight.data)
            #     # # rgb_rgb_loss = self.criterion_kl(f_out_rgb, Variable(confusion_feat_rgb))
            #     # rgb_rgb_loss = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))

            #     features_ir =  F.normalize(f_out_ir,dim=1)#features_ir_.cuda()#f_out_ir#
            #     features_ir_input = features_ir.mm(distribute_map_ir)#self.encoder.module.classifier_ir(features_ir) 
            #     features_ir= features_ir_input#*20
            #     # ir_softmax_dim= self.encoder.module.ir_softmax_dim
            #     features_ir_1 = F.softmax(features_ir[:,:ir_softmax_dim[0]], dim=1)
            #     features_ir_2 = F.softmax(features_ir[:,ir_softmax_dim[0]:], dim=1)
            #     features_ir_sim = torch.cat((features_ir_1,features_ir_2), dim=1)
            #     ir_ir_loss = self.criterion_ce_soft(features_ir_input,features_ir_sim)
    # ###########heter
                # if epoch%2 == 0:
                #     # features_ir =  F.normalize(f_out_ir)#features_ir_.cuda()#f_out_ir#
                #     features_rgb_ir_input = self.encoder.module.classifier_rgb(F.normalize(f_out_ir))#*20
                #     features_rgb_ir= features_rgb_ir_input#*20
                #     rgb_softmax_dim= self.encoder.module.rgb_softmax_dim
                #     # print('rgb_softmax_dim',rgb_softmax_dim)
                #     features_rgb_ir_1 = F.softmax(features_rgb_ir[:,:rgb_softmax_dim[0]], dim=1)
                #     features_rgb_ir_2 = F.softmax(features_rgb_ir[:,rgb_softmax_dim[0]:rgb_softmax_dim[0]+rgb_softmax_dim[1]], dim=1)
                #     features_rgb_ir_3 = F.softmax(features_rgb_ir[:,rgb_softmax_dim[0]+rgb_softmax_dim[1]:rgb_softmax_dim[0]+rgb_softmax_dim[1]+rgb_softmax_dim[2]], dim=1)
                #     features_rgb_ir_4 = F.softmax(features_rgb_ir[:,rgb_softmax_dim[0]+rgb_softmax_dim[1]+rgb_softmax_dim[2]:], dim=1)
                #     # print(features_rgb_1.size(),features_rgb_2.size(),features_rgb_3.size(),features_rgb_4.size())
                #     features_rgb_ir_sim = torch.cat((features_rgb_ir_1,features_rgb_ir_2,features_rgb_ir_3,features_rgb_ir_4), dim=1)
                #     # print('features_rgb_input,features_rgb_sim',features_rgb_input.size(),features_rgb_sim.size())
                #     rgb_ir_loss = self.criterion_ce_soft(features_rgb_ir_input,features_rgb_ir_sim)
                # # confusion_feat_rgb = features_rgb_sim.mm(self.encoder.module.classifier_rgb.weight.data)
                # # # rgb_rgb_loss = self.criterion_kl(f_out_rgb, Variable(confusion_feat_rgb))
                # # rgb_rgb_loss = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
                # else:
                #     # features_ir =  f_out_ir#F.normalize(f_out_ir)#features_ir_.cuda()#
                #     features_ir_rgb_input = self.encoder.module.classifier_ir(F.normalize(f_out_rgb)) 
                #     features_ir_rgb= features_ir_rgb_input#*20
                #     ir_softmax_dim= self.encoder.module.ir_softmax_dim
                #     features_ir_rgb_1 = F.softmax(features_ir_rgb[:,:ir_softmax_dim[0]], dim=1)
                #     features_ir_rgb_2 = F.softmax(features_ir_rgb[:,ir_softmax_dim[0]:], dim=1)
                #     features_ir_rgb_sim = torch.cat((features_ir_rgb_1,features_ir_rgb_2), dim=1)
                #     ir_rgb_loss = self.criterion_ce_soft(features_ir_rgb_input,features_ir_rgb_sim)



            # confusion_feat_ir= features_ir_sim.mm(self.encoder.module.classifier_ir.weight.data)
            # ir_ir_loss = self.criterion_kl(f_out_ir, Variable(confusion_feat_ir))
            # ir_ir_loss = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))

            lamda_i = 1
####################
            # lamda_i = 1
            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(ir_ir_loss+rgb_rgb_loss)+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(loss_confusion_rgb+loss_confusion_ir)#+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)

            loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)#+(ir_ir_loss+rgb_rgb_loss)#+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())
            loss_camera_rgb_log.update(loss_camera_rgb.item())
            loss_camera_ir_log.update(loss_camera_ir.item())
            ir_rgb_loss_log.update(ir_rgb_loss.item())
            rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            loss_ins_ir_log.update(loss_ins_ir.item())
            loss_ins_rgb_log.update(loss_ins_rgb.item())

            # ir_rgb_loss_log.update(loss_confusion_ir.item())
            # rgb_ir_loss_log.update(loss_confusion_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'camera ir {:.3f} ({:.3f})\t'
                      'camera rgb {:.3f} ({:.3f})\t'
                      'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                      'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                      'ir_ir_loss_log {:.3f} ({:.3f})\t'
                      'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
                              loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                              ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                              ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            # print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            # print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item(),flush=True)
            # if (i + 1) % print_freq == 0:
            #     print('Epoch: [{}][{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           'Loss {:.3f} ({:.3f})\t'
            #           'Loss ir {:.3f}\t'
            #           'Loss rgb {:.3f}\t'
            #           'camera ir {:.3f}\t'
            #           'camera rgb {:.3f}\t'
            #           #  'adp ir {:.3f}\t'
            #           # 'adp rgb {:.3f}\t'
            #           .format(epoch, i + 1, len(data_loader_rgb),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg,
            #                   losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))
            #     print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            #     print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            #     print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
                # print('loss_query_ir_p,',loss_query_ir_p.item())
                # print('loss_query_ir_n,',loss_query_ir_n.item())
                # print('loss_query_rgb_p,',loss_query_rgb_p.item())
                # print('loss_query_rgb_n,',loss_query_rgb_n.item())
                # print('score_log_ir',score_log_ir)
                # print('score_log_rgb',score_log_rgb)
    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    # def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
    #     return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)



    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            # intra-camera loss
            # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
            # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
            # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
            # percam_inputs /= self.beta  # similarity score before softmax
            # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

            # cross-camera loss
            # if epoch >= self.crosscam_epoch:
            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])

        for idx in sorted(centers.keys()):
            centers[idx] = torch.stack(centers[idx], dim=0).mean(0)

        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        # self.seq_len=5
        # q, d_5 = query_t.size() # b d*5,  
        # k, d_5 = key_m.size()

        # z= int(d_5/self.seq_len)
        # d = int(d_5/self.seq_len)        
        # # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        # score = F.softmax(score.permute(0,2,3,1)/0.01,dim=-1).reshape(q,-1)
        # # score = F.softmax(score,dim=1)
        # return score

        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score



class ClusterContrastTrainer_pretrain_camera_confusionrefine_noice(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_confusionrefine_noice, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)#1
        self.tri = TripletLoss_WRT()
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir
        self.criterion_kl = KLDivLoss()
        self.cmlabel=0
        self.criterion_ce_soft = SoftEntropy().cuda()
        # self.match_loss = PairwiseMatchingLoss(self.encoder.matcher)
    # def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
    #     all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
    #              print_freq=10, train_iters=400):
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()
        




        lamda_c = 0.0#0.1
        # if epoch>=self.cmlabel:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        # else:
            
        #     concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        #     concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)

        part=4

        
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_rgb]
        # ir_softmax_dim=[i.size(0) for i in percam_memory_ir]
        # # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # # self.encoder.module.ir_softmax_dim=ir_softmax_dim

        # distribute_map_rgb = torch.cat(percam_memory_rgb, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb).detach().t()
        # # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())


        # distribute_map_ir = torch.cat(percam_memory_ir, dim=0)
        # distribute_map_ir = F.normalize(distribute_map_ir).detach().t()
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_ir.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())


        # percam_memory_all=percam_memory_rgb+percam_memory_ir
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_all]
        # ir_softmax_dim=[i.size(0) for i in percam_memory_all]
        # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # self.encoder.module.ir_softmax_dim=ir_softmax_dim

        # distribute_map_rgb = torch.cat(percam_memory_all, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb)
        # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())


        # distribute_map_ir = torch.cat(percam_memory_all, dim=0)
        # distribute_map_ir = F.normalize(distribute_map_ir)
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_ir.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_ir.cuda())


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()

        # del percam_memory_ir,percam_memory_rgb
        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            # inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1

            # # inputs_ir,inputs_ir1,labels_ir, indexes_ir,cids_ir = self._parse_data_rgb(inputs_ir) #inputs_ir1
            # inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)
            # # forward
            # inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            # labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            # cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)
            # if epoch%2 == 0:
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
            #     cids_ir =  torch.cat((cids_ir,cids_ir),-1)

            #     indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            #     indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()



            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            

            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + self.memory_ir(f_out_ir_r, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)# +self.memory_rgb(f_out_rgb_r, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            # loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,percam_tempV_ir)
            # loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb)
            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_confusion_ir = torch.tensor([0.]).cuda()
            loss_confusion_rgb = torch.tensor([0.]).cuda()
            thresh=0.8
            lamda_i = 0


            lamda_i = 1
####################
            # lamda_i = 1
            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(ir_ir_loss+rgb_rgb_loss)+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(loss_confusion_rgb+loss_confusion_ir)#+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)

            loss = loss_ir+loss_rgb#+lamda_c*(loss_camera_ir+loss_camera_rgb)#+(ir_ir_loss+rgb_rgb_loss)#+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            loss_ir_log.update(loss_ir.item())
            loss_rgb_log.update(loss_rgb.item())
            loss_camera_rgb_log.update(loss_camera_rgb.item())
            loss_camera_ir_log.update(loss_camera_ir.item())
            ir_rgb_loss_log.update(ir_rgb_loss.item())
            rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            loss_ins_ir_log.update(loss_ins_ir.item())
            loss_ins_rgb_log.update(loss_ins_rgb.item())

            # ir_rgb_loss_log.update(loss_confusion_ir.item())
            # rgb_ir_loss_log.update(loss_confusion_rgb.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'camera ir {:.3f} ({:.3f})\t'
                      'camera rgb {:.3f} ({:.3f})\t'
                      'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                      'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                      'ir_ir_loss_log {:.3f} ({:.3f})\t'
                      'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
                              loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                              ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                              ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            # print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            # print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item(),flush=True)
            # if (i + 1) % print_freq == 0:
            #     print('Epoch: [{}][{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           'Loss {:.3f} ({:.3f})\t'
            #           'Loss ir {:.3f}\t'
            #           'Loss rgb {:.3f}\t'
            #           'camera ir {:.3f}\t'
            #           'camera rgb {:.3f}\t'
            #           #  'adp ir {:.3f}\t'
            #           # 'adp rgb {:.3f}\t'
            #           .format(epoch, i + 1, len(data_loader_rgb),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg,
            #                   losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))
            #     print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
            #     print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
            #     print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
                # print('loss_query_ir_p,',loss_query_ir_p.item())
                # print('loss_query_ir_n,',loss_query_ir_n.item())
                # print('loss_query_rgb_p,',loss_query_rgb_p.item())
                # print('loss_query_rgb_n,',loss_query_rgb_n.item())
                # print('score_log_ir',score_log_ir)
                # print('score_log_rgb',score_log_rgb)
    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    # def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
    #     return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)


    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)



    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
            print(cc,proto_memory.size())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV_ = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV_,percam_memory#memory_class_mapper,
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper):
        beta = 0.07
        bg_knn = 50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):
            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]

            # intra-camera loss
            # mapped_targets = [self.memory_class_mapper[cc][int(k)] for k in percam_targets]
            # mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
            # # percam_inputs = ExemplarMemory.apply(percam_feat, mapped_targets, self.percam_memory[cc], self.alpha)
            # percam_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(self.percam_memory[cc].t()))
            # percam_inputs /= self.beta  # similarity score before softmax
            # loss_cam += F.cross_entropy(percam_inputs, mapped_targets)

            # cross-camera loss
            # if epoch >= self.crosscam_epoch:
            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam

    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i].item()].append(features[i])

        for idx in sorted(centers.keys()):
            centers[idx] = torch.stack(centers[idx], dim=0).mean(0)

        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        # self.seq_len=5
        # q, d_5 = query_t.size() # b d*5,  
        # k, d_5 = key_m.size()

        # z= int(d_5/self.seq_len)
        # d = int(d_5/self.seq_len)        
        # # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        # score = F.softmax(score.permute(0,2,3,1)/0.01,dim=-1).reshape(q,-1)
        # # score = F.softmax(score,dim=1)
        # return score

        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
        score = einsum('q t d, k s d -> q k t s', query_t, key_m)

        score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
        score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

        # score = F.softmax(score,dim=1)
        return score


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class SoftEntropy(nn.Module):
    def __init__(self):
        super(SoftEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, inputs, targets):

        # nearest_rgb_ir = targets.max(dim=1, keepdim=True)[0]
        # mask_neighbor_rgb_ir = torch.gt(targets, nearest_rgb_ir * 0.8)
        # num_neighbor_rgb_ir = mask_neighbor_rgb_ir.sum(dim=1)+1
        # print(num_neighbor_rgb_ir)
        log_probs = self.logsoftmax(inputs)
        loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
        # loss = (- F.softmax(targets*20, dim=1).detach() * log_probs).mean(0).sum()
        # # loss = (- (targets).detach() * log_probs).mul(mask_neighbor_rgb_ir).sum(dim=1)#.mean(0).sum()#.mean()
        # # loss = (- F.softmax(targets*20, dim=1).detach() * log_probs).mul(mask_neighbor_rgb_ir).sum(dim=1)#.mean(0).sum()#.mean()
        # loss = loss.div(num_neighbor_rgb_ir).mean()

        # sim_rgb_ir=inputs
        # sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
        # nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
        # nearest_rgb_ir_2 = targets.max(dim=1, keepdim=True)[0]
        # mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps
        # mask_neighbor_rgb_ir_2 = torch.gt(targets, nearest_rgb_ir_2 * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps

        # num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_rgb_ir_2).sum(dim=1)+1
        # score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
        # score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
        # loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_rgb_ir_2).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
        # loss = loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##



        # loss = (- F.softmax(targets*20, dim=1).detach() * log_probs).sum(1)
        # loss = loss.div(num_neighbor_rgb_ir).mean()
        # print(loss.item())
        return loss



class ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_all =  memory
        self.wise_memory_rgb =  memory
        self.wise_memory_ir =  memory

        self.nameMap_rgb = []
        self.nameMap_ir = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)
        # self.criterion_pa = PredictionAlignmentLoss(lambda_vr=0.5, lambda_rv=0.5)
        self.camstart=0
        self.tri = TripletLoss_WRT()
        self.criterion_kl = KLDivLoss()
        self.criterion_ce_soft = SoftEntropy().cuda()
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,all_label=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        loss_ir_log = AverageMeter()
        loss_rgb_log = AverageMeter()
        loss_camera_rgb_log = AverageMeter()
        loss_camera_ir_log = AverageMeter()
        ir_rgb_loss_log = AverageMeter()
        rgb_ir_loss_log = AverageMeter()
        rgb_rgb_loss_log = AverageMeter()
        ir_ir_loss_log = AverageMeter()
        loss_ins_ir_log = AverageMeter()
        loss_ins_rgb_log = AverageMeter()

        ##########init camera proxy
        # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        
        
        # # percam_memory_rgb = [self.wise_memory_rgb.features, self.wise_memory_ir.features]
        # rgb_softmax_dim=[i.size(0) for i in percam_memory_rgb]
        distribute_map_ir=percam_memory_rgb[1]
        distribute_map_rgb=percam_memory_rgb[0]
        # # ir_softmax_dim=[i.size(0) for i in percam_memory_ir]
        # self.encoder.module.rgb_softmax_dim=rgb_softmax_dim
        # self.encoder.module.ir_softmax_dim=rgb_softmax_dim


        memory_dy_ir = ClusterMemory(self.encoder.module.in_planes*part, distribute_map_ir.size(0), temp=0.05,
                               momentum=0.5, use_hard=False).cuda()
        # memory_dy_rgb = ClusterMemory(self.encoder.module.in_planes*part, distribute_map_rgb.size(0), temp=0.05,
        #                        momentum=0.5, use_hard=False).cuda()

        memory_dy_ir.features = distribute_map_ir#.cuda()
        # memory_dy_rgb.features = distribute_map_ir#.cuda()

        # distribute_map_rgb = torch.cat(percam_memory_rgb, dim=0)
        # distribute_map_rgb = F.normalize(distribute_map_rgb).detach()#.t()


        # self.encoder.module.classifier_rgb = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # self.encoder.module.classifier_rgb.weight.data.copy_(distribute_map_rgb.cuda())
        # self.encoder.module.classifier_ir = nn.Linear(768*part, distribute_map_rgb.size(0), bias=False).cuda()
        # self.encoder.module.classifier_ir.weight.data.copy_(distribute_map_rgb.cuda())

        # Note=open('x.txt',mode='a+')

        start_cam=0
        ir_num = len(all_label_ir)
        rgb_num = len(all_label)-ir_num
        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            # inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            # inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # # forward
            # inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            # labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            # indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            # indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = []#torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # # labels_ir = torch.cat((labels_ir,labels_ir),-1)
            # # for path,cameraid in  zip(name_ir,cids_ir):
            # #     print(path,cameraid)

            # if epoch%2 == 0:
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            # indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
            #     cids_ir =  torch.cat((cids_ir,cids_ir),-1)

            #     indexes_ir = []#torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            #     indexes_rgb = []#torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()




            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            # indexes_all = torch.cat((index_rgb,index_ir),-1)
            cid_all=torch.cat((cid_rgb,cid_ir),-1)

#####################################
            labels_all = torch.cat((labels_rgb,labels_ir),-1)
            # loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            f_out_all=torch.cat((f_out_rgb,f_out_ir),0)
            lamda_c=0.1
            # start=30
            loss_camera_ir=torch.tensor([0.]).cuda()
            loss_camera_rgb=torch.tensor([0.]).cuda()
            loss_camera_all = torch.tensor([0.]).cuda()
            rgb_rgb_loss = torch.tensor([0.]).cuda()
            ir_ir_loss = torch.tensor([0.]).cuda()
            loss_confusion_all  = torch.tensor([0.]).cuda()
            # if epoch >= self.camstart:
            loss_camera_all = self.camera_loss(f_out_all,cid_all,labels_all,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb,cross_m=True)#self.camera_loss(f_out_ir,cid_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
                # loss_camera_rgb = self.camera_loss(f_out_rgb,cid_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)

            # concate_mem = torch.cat(percam_memory_rgb,dim=0)  #C_RGB+C_IR dim
            # # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
            # sim_concate_rgb = torch.cat([F.softmax(F.normalize(f_out_all, dim=1).mm(percam_memory_rgb[i].detach().data.t()),dim=1) for i in range(len(percam_memory_rgb))],dim=1)
            # # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
            # # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
            # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1)
            # confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem)# B dim
            # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
            # # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
            # # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
            # loss_confusion_all = self.tri(torch.cat((f_out_all,confusion_feat_rgb),dim=0),torch.cat((labels_all,labels_all),dim=-1))
            # # loss_confusion_rgb = self.tri(torch.cat((F.normalize(f_out_rgb, dim=1),F.normalize(confusion_feat_rgb, dim=1)),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
            if epoch>=700:
                # features_rgb = F.normalize(f_out_rgb)#features_rgb_.cuda()#
                # features_rgb_input = F.normalize(f_out_all,dim=1).mm(distribute_map_rgb)# self.encoder.module.classifier_rgb()#*20
                # features_rgb= features_rgb_input*20
                # # rgb_softmax_dim= self.encoder.module.rgb_softmax_dim
                # features_rgb_1 = F.softmax(features_rgb[:,:rgb_softmax_dim[0]], dim=1)
                # features_rgb_2 = F.softmax(features_rgb[:,rgb_softmax_dim[0]:], dim=1)
                # features_rgb_sim = torch.cat((features_rgb_1,features_rgb_2), dim=1)
                # rgb_rgb_loss = self.criterion_ce_soft(features_rgb_input,features_rgb_sim)

                # if epoch%2==0:
                # for l in range(part):
                #     if l ==0:
                #         ins_sim_rgb_all = F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(distribute_map_rgb[:,l*768:(l+1)*768].detach().t(), dim=-1))
                #     else:
                #         ins_sim_rgb_all += F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(distribute_map_rgb[:,l*768:(l+1)*768].detach().t(), dim=-1))
                # ins_sim_rgb_sim = ins_sim_rgb_all*20#F.normalize(f_out_all,dim=1).mm(distribute_map_rgb)# self.encoder.module.classifier_rgb()#*20
                # # features_rgb= features_rgb_input*20
                # features_rgb_1 = F.softmax(ins_sim_rgb_sim[:,:rgb_softmax_dim[0]], dim=1)
                # features_rgb_2 = F.softmax(ins_sim_rgb_sim[:,rgb_softmax_dim[0]:], dim=1)
                # features_rgb_sim = torch.cat((features_rgb_1,features_rgb_2), dim=1)
                # features_rgb_input = F.normalize(f_out_all,dim=1).mm(distribute_map_rgb.t()) #ins_sim_rgb_all#best: 26.9% *
                # rgb_rgb_loss = self.criterion_ce_soft(features_rgb_input,features_rgb_sim)

                # for l in range(part):

                #     if l ==0:
                #         ins_sim_rgb_all = F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(distribute_map_rgb[:,l*768:(l+1)*768].detach().t(), dim=-1))
                #     else:
                #         ins_sim_rgb_all += F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(distribute_map_rgb[:,l*768:(l+1)*768].detach().t(), dim=-1))
                # if epoch%2==0:
                #     cluster_label_rgb_rgb=[]
                #     intersect_count_list=[]

                #     for l in range(part):
                #         ins_sim_rgb_rgb= F.normalize(f_out_all[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(memory_dy_rgb.features[:,l*768:(l+1)*768].detach().t(), dim=-1))
                #         Score_TOPK = 3#20#10
                #         topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
                #         # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                #         # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
                #         cluster_label_rgb_rgb.append(cluster_indices_rgb_rgb.detach())#.cpu()
                #         if l == 0:
                #             ins_sim_rgb_rgb_all=ins_sim_rgb_rgb
                #         else:
                #             ins_sim_rgb_rgb_all*=ins_sim_rgb_rgb
                #     cluster_label_rgb_rgb=torch.cat(cluster_label_rgb_rgb,1)
                #     for n in range(Score_TOPK*part):
                #         intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,n].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                #         intersect_count_list.append(intersect_count)
                #     intersect_count_list = torch.cat(intersect_count_list,1)
                #     intersect_count, _ = intersect_count_list.max(1)
                #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
                #     cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1) 
                #     # print(cluster_label_rgb_rgb)

                #     rgb_rgb_loss = memory_dy_rgb(f_out_all, cluster_label_rgb_rgb)

                # else:
                    cluster_label_rgb_rgb=[]
                    intersect_count_list=[]
                    for l in range(part):
                        ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,l*768:(l+1)*768], dim=-1).mm(F.normalize(memory_dy_ir.features[:,l*768:(l+1)*768].detach().t(), dim=-1))
                        Score_TOPK = 3#20#10
                        topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
                        # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                        # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
                        cluster_label_rgb_rgb.append(cluster_indices_rgb_rgb.detach())#.cpu()
                        if l == 0:
                            ins_sim_rgb_rgb_all=ins_sim_rgb_rgb
                        else:
                            ins_sim_rgb_rgb_all*=ins_sim_rgb_rgb

                    cluster_label_rgb_rgb=torch.cat(cluster_label_rgb_rgb,1)
                    for n in range(Score_TOPK*part):
                        intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,n].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                        intersect_count_list.append(intersect_count)
                    intersect_count_list = torch.cat(intersect_count_list,1)
                    intersect_count, _ = intersect_count_list.max(1)
                    topk,cluster_label_index = torch.topk(intersect_count_list,1)
                    cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1) 
                    # rgb_rgb_loss = memory_dy_ir(f_out_all, cluster_label_rgb_rgb)

                    lamda_cm=0.1
                    update_memory = memory_dy_ir.features[cluster_label_rgb_rgb]

                    self.memory_rgb.features[labels_rgb] = l2norm( lamda_cm*self.memory_rgb.features[labels_rgb] + (1-lamda_cm)*(update_memory) )
                    # self.memory_rgb.features[key[1]] = l2norm( lamda_cm*trainer_interm.memory_rgb.features[key[1]] + (1-lamda_cm)*(update_memory) )





            # confusion_feat_rgb = features_rgb_sim.mm(self.encoder.module.classifier_rgb.weight.data)
            # rgb_rgb_loss = self.criterion_kl(f_out_rgb, Variable(confusion_feat_rgb))
            # rgb_rgb_loss = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))

            # # features_ir =  F.normalize(f_out_ir)#features_ir_.cuda()#
            # features_ir_input = self.encoder.module.classifier_ir(f_out_ir) 
            # features_ir= features_ir_input*20
            # ir_softmax_dim= self.encoder.module.ir_softmax_dim
            # features_ir_1 = F.softmax(features_ir[:,:ir_softmax_dim[0]], dim=1)
            # features_ir_2 = F.softmax(features_ir[:,ir_softmax_dim[0]:], dim=1)
            # features_ir_sim = torch.cat((features_ir_1,features_ir_2), dim=1)
            # ir_ir_loss = self.criterion_ce_soft(features_ir_input,features_ir_sim)
            # # confusion_feat_ir= features_ir_sim.mm(self.encoder.module.classifier_ir.weight.data)
            # # ir_ir_loss = self.criterion_kl(f_out_ir, Variable(confusion_feat_ir))
            # # ir_ir_loss = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))

            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#

            loss_all = self.memory_rgb(f_out_all, labels_all) 
            lamda_i = 0
            loss = loss_all+lamda_c*loss_camera_all#+0.1*rgb_rgb_loss#+ir_ir_loss#+loss_confusion_all#all_all_loss+lamda_i*(loss_ins_ir+loss_ins_rgb)#+rgb_rgb_loss+ir_ir_loss#+lamda_i*(loss_ins_ir+loss_ins_rgb)#+rgb_rgb_loss+ir_ir_loss#+#+ir_ir_loss #+ loss_tri+loss_rgb_ir_trans+loss_ir_rgb_trans +(loss_rgb_trans+loss_ir_trans)

            # loss = lamda_cc*(loss_ir+loss_rgb)+loss_camera_rgb+loss_camera_ir #+ loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())


            loss_ir_log.update(loss_all.item())
            loss_rgb_log.update(loss_all.item())
            loss_camera_rgb_log.update(loss_camera_all.item())
            loss_camera_ir_log.update(loss_camera_all.item())
            # ir_rgb_loss_log.update(ir_rgb_loss.item())
            # rgb_ir_loss_log.update(rgb_ir_loss.item())
            rgb_rgb_loss_log.update(rgb_rgb_loss.item())
            ir_ir_loss_log.update(ir_ir_loss.item())
            # loss_ins_ir_log.update(loss_ins_ir.item())
            # loss_ins_rgb_log.update(loss_ins_rgb.item())


            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            # if (i + 1) % print_freq == 0:
            #     print('Epoch: [{}][{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           'Loss {:.3f} ({:.3f})\t'
            #           'Loss all {:.3f}\t'
            #           'Loss all {:.3f}\t'
            #           'camera all {:.3f}\t'
            #           'camera rgb {:.3f}\t'
            #           #  'adp ir {:.3f}\t'
            #           # 'adp rgb {:.3f}\t'
            #           .format(epoch, i + 1, len(data_loader_rgb),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg,
            #                   losses.val, losses.avg,loss_all,loss_all,loss_camera_all.item(),loss_camera_rgb.item()))


            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f} ({:.3f})\t'
                      'camera ir {:.3f} ({:.3f})\t'
                      'camera rgb {:.3f} ({:.3f})\t'
                      'ir_rgb_loss_log {:.3f} ({:.3f})\t'
                      'rgb_ir_loss_log {:.3f} ({:.3f})\t'
                      'ir_ir_loss_log {:.3f} ({:.3f})\t'
                      'rgb_rgb_loss_log {:.3f} ({:.3f})\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
                              loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
                              ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
                              ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            # Note.write('Epoch: [{}][{}/{}]\t'
            #           'Time {:.3f} ({:.3f})\t'
            #           'Data {:.3f} ({:.3f})\t'
            #           'Loss {:.3f} ({:.3f})\t'
            #           'Loss ir {:.3f} ({:.3f})\t'
            #           'Loss rgb {:.3f} ({:.3f})\t'
            #           'camera ir {:.3f} ({:.3f})\t'
            #           'camera rgb {:.3f} ({:.3f})\t'
            #           'ir_rgb_loss_log {:.3f} ({:.3f})\t'
            #           'rgb_ir_loss_log {:.3f} ({:.3f})\t'
            #           'ir_ir_loss_log {:.3f} ({:.3f})\t'
            #           'rgb_rgb_loss_log {:.3f} ({:.3f})\t\n'
            #           # 'ir_ir_loss_log {:.3f}\t'
            #           # 'rgb_rgb_loss_log {:.3f}\t'
            #           # 'loss_ins_ir_log {:.3f}\t'
            #           # 'loss_ins_rgb_log {:.3f}\t'
            #           #  'adp ir {:.3f}\t'
            #           # 'adp rgb {:.3f}\t'
            #           .format(epoch, i + 1, len(data_loader_rgb),
            #                   batch_time.val, batch_time.avg,
            #                   data_time.val, data_time.avg,
            #                   losses.val, losses.avg,loss_ir_log.val,loss_ir_log.avg,loss_rgb_log.val,loss_rgb_log.avg,\
            #                   loss_camera_ir_log.val,loss_camera_ir_log.avg,loss_camera_rgb_log.val,loss_camera_rgb_log.avg,\
            #                   ir_rgb_loss_log.val,ir_rgb_loss_log.avg,rgb_ir_loss_log.val,rgb_ir_loss_log.avg,\
            #                   ir_ir_loss_log.val,ir_ir_loss_log.avg,rgb_rgb_loss_log.val,rgb_rgb_loss_log.avg))
            

                # if epoch >= start_cam:
                # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
                # print('loss_tri',loss_tri.item())
                # print('loss_confusion_all',loss_confusion_all.item())
                    # print('loss_intra_ir,loss_inter_ir,intrawise_loss_ir',loss_ins_ir.item(),loss_intra_ir.item(),loss_inter_ir.item(),intrawise_loss_ir.item())
                    # print('loss_intra_rgb,loss_inter_rgb,intrawise_loss_rgb',loss_ins_rgb.item(),loss_intra_rgb.item(),loss_inter_rgb.item(),intrawise_loss_rgb.item())
            # pseudo_labels_all=self.wise_memory_all.labels.numpy()
            # cluster_features_all = self.generate_cluster_features(pseudo_labels_all, self.wise_memory_all.features)

            # num_cluster_all = len(set(pseudo_labels_all)) - (1 if -1 in pseudo_labels_all else 0)
            # cam_moment-0.1
            # for cc in torch.unique(cid_all):
            #     # print(cc)
            #     inds = torch.nonzero(cid_all == cc).squeeze(-1)
            #     percam_targets = labels_all[inds]
            #     percam_feat = f_out_all[inds].detach().clone()
 
            #     for k in range(len(percam_feat)):
            #         ori_asso_ind = torch.nonzero(concate_intra_class_rgb == percam_targets[k]).squeeze(-1)
            #         percam_tempV_rgb[ori_asso_ind] = (1-cam_moment)*percam_feat[k]+cam_moment*percam_tempV_rgb[ori_asso_ind]

            # # self.memory_rgb.features = F.normalize(cluster_features_all, dim=1).cuda()
        # Note.close()  
    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)

    def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
        all_img_cams = torch.tensor(all_img_cams).cuda()
        unique_cams = torch.unique(all_img_cams)
        # print(self.unique_cams)

        all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
        init_intra_id_feat = intra_id_features
        # print(len(self.init_intra_id_feat))

        # initialize proxy memory
        percam_memory = []
        memory_class_mapper = []
        concate_intra_class = []
        for cc in unique_cams:
            percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
            uniq_class = torch.unique(all_pseudo_label[percam_ind])
            uniq_class = uniq_class[uniq_class >= 0]
            concate_intra_class.append(uniq_class)
            cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
            memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

            if len(init_intra_id_feat) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = init_intra_id_feat[cc]
                proto_memory = proto_memory.cuda()
                percam_memory.append(proto_memory.detach())
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV,percam_memory#memory_class_mapper
    def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper,cross_m=False):
        beta = 0.07#0.07
        bg_knn = 50#100#50
        loss_cam = torch.tensor([0.]).cuda()
        for cc in torch.unique(cids):

            # print(cc)
            inds = torch.nonzero(cids == cc).squeeze(-1)
            percam_targets = targets[inds]
            # print(percam_targets)
            percam_feat = f_out_t1[inds]
            associate_loss = 0
            # target_inputs = percam_feat.mm(percam_tempV.t().clone())
            target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
            temp_sims = target_inputs.detach().clone()
            target_inputs /= beta

            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue  
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
                # sel_ind_2 = torch.sort(temp_sims[k])[1][1:bg_knn*10]
                # sel_ind = torch.cat((sel_ind,sel_ind_2), dim=-1)
                # nearest_intra = temp_sims[k].max(dim=-1, keepdim=True)[0]
                # mask_neighbor_intra = torch.gt(temp_sims[k], nearest_intra * 0.8)
                # sel_ind = torch.nonzero(mask_neighbor_intra).squeeze(-1)
                # if cross_m == True:
                #     concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)#
                #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                #     torch.device('cuda'))
                #     concated_target[0:len(ori_asso_ind)+len(sel_ind)] = 1.0 / (len(ori_asso_ind)+len(sel_ind)+1e-8)
                # else:
                concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)#target_inputs[k, ori_asso_ind]#
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                torch.device('cuda'))
                # concated_target[0:len(ori_asso_ind)] = 1.0 / (len(ori_asso_ind)+1e-8)
                # print('len(concated_input)',len(concated_input))
                # print('len(ori_asso_ind)',len(ori_asso_ind))
                # concated_target[0:len(concated_input)] = 1.0 / (len(concated_input)+1e-8)
                # concated_target[0:len(ori_asso_ind)+len(sel_ind)] = 1.0 / (len(ori_asso_ind)+len(sel_ind)+1e-8)
                concated_target[0:len(ori_asso_ind)] = 1.0 / (len(ori_asso_ind)+1e-8)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_cam +=  associate_loss / len(percam_feat)
        return loss_cam
    @torch.no_grad()
    def generate_cluster_features(self,labels, features):
        centers = collections.defaultdict(list)
        for i, label in enumerate(labels):
            if label == -1:
                continue
            centers[labels[i]].append(features[i])

        centers = [
            torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
        ]

        centers = torch.stack(centers, dim=0)
        return centers

    def mask(self,ones, labels,ins_label):
        for i, label in enumerate(labels):
            ones[i,ins_label==label] = 1
        return ones

    def part_sim(self,query_t, key_m):
        self.seq_len=5
        q, d_5 = query_t.size() # b d*5,  
        k, d_5 = key_m.size()

        z= int(d_5/self.seq_len)
        d = int(d_5/self.seq_len)        
        query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
        key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
        # query_t = F.normalize(tgt.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
        # key_m = F.normalize(memory.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
        score = F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m)/0.01,dim=-1).view(q,-1) # B Q N N
        # score = F.softmax(score,dim=1)
        return score



class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha =1, gamma = 1, square = 0):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap*self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an*self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        
        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)
        
        # squared difference
        if self.square ==0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow =torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            
            loss = self.ranking_loss(diff_pow, y)
        
        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss#, correct




class ClusterContrastTrainer_pretrain_joint(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_joint, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()




        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            # inputs_ir,labels_ir, indexes_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            # inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # # forward
            # inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            # labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)


            # if epoch%2 == 0:
            inputs_ir,labels_ir, indexes_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)


            # else:

            #     inputs_rgb,labels_rgb, indexes_rgb = self._parse_data_ir(inputs_rgb) #inputs_ir1


            #     inputs_ir,inputs_ir1, labels_ir, indexes_ir = self._parse_data_rgb(inputs_ir)
            #     # forward
            #     inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            #     labels_ir = torch.cat((labels_ir,labels_ir),-1)
  




            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)


            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir+loss_rgb# + loss_tri
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

