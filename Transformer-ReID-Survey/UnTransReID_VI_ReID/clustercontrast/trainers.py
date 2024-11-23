from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Module
import collections
from torch import einsum
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

class ClusterContrastTrainer_pretrain_camera_cpsrefine(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_cpsrefine, self).__init__()
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

        self.cmlabel=0
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






        if epoch>=self.cmlabel:
            concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        else:
            concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()
        Note=open('x.txt',mode='a+')
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


            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            lamda_c = 0.1

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
            # part=5
            # if epoch>=0:#self.cmlabel:
                # percam_memory_all  = percam_memory_rgb+percam_memory_ir
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
                # loss_confusion_rgb = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
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
                # loss_confusion_ir = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
                # # loss_confusion_ir = self.tri(torch.cat((F.normalize(f_out_ir, dim=1),F.normalize(confusion_feat_ir, dim=1)),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))



                # # if epoch %2 == 0:
                # concate_mem =  percam_memory_all#torch.cat((self.memory_rgb.features,self.memory_ir.features),dim=0) #C_RGB+C_IR dim
                # # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
                # sim_concate_rgb = F.normalize(f_out_rgb, dim=1).mm(concate_mem.detach().data.t()) #
                # # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
                # # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
                # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1)
                # confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem)# B dim
                # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
                # # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
                # # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
                # # loss_confusion_rgb = self.tri(torch.cat((F.normalize(f_out_rgb, dim=1),F.normalize(confusion_feat_rgb, dim=1)),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
                # loss_confusion_rgb = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
                # # else:
                # # concate_mem =  torch.cat((self.memory_rgb.features,self.memory_ir.features),dim=0) #C_RGB+C_IR dim
                # # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
                # sim_concate_ir = F.normalize(f_out_ir, dim=1).mm(concate_mem.detach().data.t()) #
                # sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1) ##B C_RGB+C_IR
                # # sim_concate_weight_ir = torch.cat((F.softmax(sim_concate_ir[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_ir[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
                # sim_concate_weight_ir = F.softmax(sim_concate_weight_ir/0.05,dim=1)
                # confusion_feat_ir = sim_concate_weight_ir.mm(concate_mem)# B dim
                # # loss_confusion_ir = 0.1*self.memory_ir(confusion_feat, labels_ir)
                # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_ir.features.t())
                # # loss_confusion_ir= F.cross_entropy(confusion_out, labels_ir)
                # # loss_confusion_ir = self.tri(torch.cat((F.normalize(f_out_ir, dim=1),F.normalize(confusion_feat_ir, dim=1)),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
                # loss_confusion_ir = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
             

            # for i in range(part):
            #     intersect_count_list=[]
            #     ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #     Score_TOPK = 20#20#10
            #     topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
            #     # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #     cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
            #     for l in range(Score_TOPK):
            #         intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         intersect_count_list.append(intersect_count)
            #     intersect_count_list = torch.cat(intersect_count_list,1)
            #     intersect_count, _ = intersect_count_list.max(1)
            #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #     cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #     if i==0: 
            #         mask_rgb_rgb = (cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
            #     else:
            #         mask_rgb_rgb = mask_rgb_rgb.mul(cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 

            # num_neighbor_rgb_rgb = mask_rgb_rgb.sum(dim=1)+1
            # sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
            # sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
            # score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
            # rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##


            # for i in range(part):
            #     intersect_count_list=[]
            #     ins_sim_ir_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #     Score_TOPK = 20#20#10
            #     topk, cluster_indices_ir_ir = torch.topk(ins_sim_ir_ir, int(Score_TOPK))#20
            #     # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #     cluster_label_ir_ir = self.wise_memory_ir.labels[cluster_indices_ir_ir].detach()#.cpu()
            #     for l in range(Score_TOPK):
            #         intersect_count=(cluster_label_ir_ir == cluster_label_ir_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         intersect_count_list.append(intersect_count)
            #     intersect_count_list = torch.cat(intersect_count_list,1)
            #     intersect_count, _ = intersect_count_list.max(1)
            #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #     cluster_label_ir_ir = torch.gather(cluster_label_ir_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #     if i==0: 
            #         mask_ir_ir =(cluster_label_ir_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda()  
            #     else:
            #         mask_ir_ir = mask_ir_ir.mul(cluster_label_ir_ir.cuda()==self.wise_memory_ir.cuda().labels).float().cuda()  

            # num_neighbor_ir_ir = mask_ir_ir.sum(dim=1)+1
            # sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
            # sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
            # score_intra_ir_ir=   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
            # ir_ir_loss = -score_intra_ir_ir.log().mul(mask_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##




################cps refine
            part=5
            if epoch>=0:#self.cmlabel:
                if epoch %2 ==0:
                    cluster_label_rgb_ir = []
                    for i in range(part):
                        intersect_count_list=[]
                        ins_sim_rgb_ir = F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
                        Score_TOPK = 20#20#10
                        topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
                        # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                        cluster_label_rgb_ir.append(self.wise_memory_ir.labels[cluster_indices_rgb_ir].detach())#.cpu()
                    cluster_label_rgb_ir=torch.cat(cluster_label_rgb_ir,1)
                    for l in range(Score_TOPK*part):
                        intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                        intersect_count_list.append(intersect_count)
                    intersect_count_list = torch.cat(intersect_count_list,1)
                    intersect_count, _ = intersect_count_list.max(1)
                    topk,cluster_label_index = torch.topk(intersect_count_list,1)
                    cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1)
                    
                    no_neg=torch.where(cluster_label_rgb_ir>=0)
                    # print(cluster_label_rgb_ir[no_neg])
                    rgb_ir_loss = self.memory_ir(f_out_rgb[no_neg], cluster_label_rgb_ir[no_neg],training_momentum=0.9)
                        # if i==0: 
                        #     mask_rgb_ir = (cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda() 
                        # else:
                        #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).cuda().float()
                #     num_neighbor_rgb_ir = mask_rgb_ir.sum(dim=1)+1
                #     sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
                #     sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
                #     score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                #     score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                #     rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                #     rgb_ir_loss = rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##
                else:
                    cluster_label_rgb_ir=[]
                    # mask_rgb_ir = torch.ones((f_out_ir.size(0),self.wise_memory_rgb.features.size(0))).cuda()
                    for i in range(part):
                        intersect_count_list=[]
                        ins_sim_rgb_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
                        Score_TOPK = 20#20#10
                        topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
                        # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                        # cluster_label_rgb_ir = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
                        cluster_label_rgb_ir.append(self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach())#.cpu()
                    cluster_label_rgb_ir=torch.cat(cluster_label_rgb_ir,1)
                    for l in range(Score_TOPK*part):
                        intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                        intersect_count_list.append(intersect_count)
                    intersect_count_list = torch.cat(intersect_count_list,1)
                    intersect_count, _ = intersect_count_list.max(1)
                    topk,cluster_label_index = torch.topk(intersect_count_list,1)
                    cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1) 
                    no_neg=torch.where(cluster_label_rgb_ir>=0)#torch.ge(cluster_label_rgb_ir,0)
                    ir_rgb_loss = self.memory_rgb(f_out_ir[no_neg], cluster_label_rgb_ir[no_neg],training_momentum=0.9)
                    #     if i==0: 
                    #         mask_rgb_ir =(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
                    #     else:
                    #         mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
                    # num_neighbor_rgb_ir = mask_rgb_ir.sum(dim=1)+1
                    # sim_rgb_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
                    # sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
                    # score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                    # score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                    # ir_rgb_loss = -score_intra_rgb_ir.log().mul(mask_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                    # ir_rgb_loss = ir_rgb_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##

            cluster_label_rgb_rgb=[]
            for i in range(part):
                intersect_count_list=[]
                ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
                Score_TOPK = 20#20#10
                topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
                # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
                cluster_label_rgb_rgb.append(self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach())#.cpu()
            cluster_label_rgb_rgb=torch.cat(cluster_label_rgb_rgb,1)
            for l in range(Score_TOPK*part):
                intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                intersect_count_list.append(intersect_count)
            intersect_count_list = torch.cat(intersect_count_list,1)
            intersect_count, _ = intersect_count_list.max(1)
            topk,cluster_label_index = torch.topk(intersect_count_list,1)
            cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1) 
            no_neg_rgb=torch.where(cluster_label_rgb_rgb>=0)#torch.(cluster_label_rgb_rgb,0)
            # print(cluster_label_rgb_rgb[no_neg_rgb])
            rgb_rgb_loss = self.memory_rgb(f_out_rgb[no_neg_rgb], cluster_label_rgb_rgb[no_neg_rgb],training_momentum=0.9)
            #     if i==0: 
            #         mask_rgb_rgb = (cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
            #     else:
            #         mask_rgb_rgb = mask_rgb_rgb.mul(cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 

            # num_neighbor_rgb_rgb = mask_rgb_rgb.sum(dim=1)+1
            # sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
            # sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
            # score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
            # rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##

            cluster_label_ir_ir=[]
            for i in range(part):
                intersect_count_list=[]
                ins_sim_ir_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
                Score_TOPK = 20#20#10
                topk, cluster_indices_ir_ir = torch.topk(ins_sim_ir_ir, int(Score_TOPK))#20
                # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                # cluster_label_ir_ir = self.wise_memory_ir.labels[cluster_indices_ir_ir].detach()#.cpu()
                cluster_label_ir_ir.append(self.wise_memory_ir.labels[cluster_indices_ir_ir].detach())#.cpu()
            cluster_label_ir_ir=torch.cat(cluster_label_ir_ir,1)
            for l in range(Score_TOPK):
                intersect_count=(cluster_label_ir_ir == cluster_label_ir_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                intersect_count_list.append(intersect_count)
            intersect_count_list = torch.cat(intersect_count_list,1)
            intersect_count, _ = intersect_count_list.max(1)
            topk,cluster_label_index = torch.topk(intersect_count_list,1)
            cluster_label_ir_ir = torch.gather(cluster_label_ir_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1) 
            no_neg_ir=torch.where(cluster_label_ir_ir>=0)#torch.ge(cluster_label_rgb_rgb,0)
            ir_ir_loss = self.memory_ir(f_out_ir[no_neg_ir], cluster_label_ir_ir[no_neg_ir],training_momentum=0.9)
            #     if i==0: 
            #         mask_ir_ir =(cluster_label_ir_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda()  
            #     else:
            #         mask_ir_ir = mask_ir_ir.mul(cluster_label_ir_ir.cuda()==self.wise_memory_ir.cuda().labels).float().cuda()  
            

            # num_neighbor_ir_ir = mask_ir_ir.sum(dim=1)+1
            # sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
            # sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
            # score_intra_ir_ir=   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
            # ir_ir_loss = -score_intra_ir_ir.log().mul(mask_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##



            lamda_i = 1
####################
            # lamda_i = 1
            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(ir_ir_loss+rgb_rgb_loss)+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)
            loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(rgb_ir_loss+ir_rgb_loss)+(ir_ir_loss+rgb_rgb_loss)#+(loss_confusion_rgb+loss_confusion_ir)#+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



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
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      'ir_rgb_loss_log {:.3f}\t'
                      'rgb_ir_loss_log {:.3f}\t'
                      'ir_ir_loss_log {:.3f}\t'
                      'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_rgb_log.val,loss_camera_ir_log.val,loss_camera_rgb_log.val,ir_rgb_loss_log.val,rgb_ir_loss_log.val,ir_ir_loss_log.val,rgb_rgb_loss_log.val))
            Note.write('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      'ir_rgb_loss_log {:.3f}\t'
                      'rgb_ir_loss_log {:.3f}\t'
                      'ir_ir_loss_log {:.3f}\t'
                      'rgb_rgb_loss_log {:.3f}\t\n'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'

                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_rgb_log.val,loss_camera_ir_log.val,loss_camera_rgb_log.val,ir_rgb_loss_log.val,rgb_ir_loss_log.val,ir_ir_loss_log.val,rgb_rgb_loss_log.val))
            
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
        Note.close()        
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

        self.cmlabel=0
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
        





        if epoch>=self.cmlabel:
            concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        else:
            concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


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


            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            lamda_c = 0.1

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
            part=5
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
            percam_memory_all  = percam_memory_rgb+percam_memory_ir
            # if epoch %2 == 0:
            concate_mem =  torch.cat(percam_memory_all,dim=0) #C_RGB+C_IR dim
            # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
            sim_concate_rgb = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(percam_memory_all[i].detach().data.t()),dim=1) for i in range(len(percam_memory_all))],dim=1)
            # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
            # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
            sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1).detach()
            confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem)# B dim
            # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
            # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
            # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
            rgb_ir_loss = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
            # loss_confusion_rgb = self.tri(torch.cat((F.normalize(f_out_rgb, dim=1),F.normalize(confusion_feat_rgb, dim=1)),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
            # else:
            # concate_mem =  torch.cat((self.memory_rgb.features,self.memory_ir.features),dim=0) #C_RGB+C_IR dim
            concate_mem =  torch.cat(percam_memory_all,dim=0)
            # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
            sim_concate_ir = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(percam_memory_all[i].detach().data.t()),dim=1) for i in range(len(percam_memory_all))],dim=1)
            # sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1) ##B C_RGB+C_IR
            # sim_concate_weight_ir = torch.cat((F.softmax(sim_concate_ir[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_ir[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
            sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1).detach()
            confusion_feat_ir = sim_concate_weight_ir.mm(concate_mem)# B dim
            # loss_confusion_ir = 0.1*self.memory_ir(confusion_feat, labels_ir)
            # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_ir.features.t())
            # loss_confusion_ir= F.cross_entropy(confusion_out, labels_ir)
            ir_rgb_loss = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
            # loss_confusion_ir = self.tri(torch.cat((F.normalize(f_out_ir, dim=1),F.normalize(confusion_feat_ir, dim=1)),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))



                # # if epoch %2 == 0:
                # concate_mem =  percam_memory_all#torch.cat((self.memory_rgb.features,self.memory_ir.features),dim=0) #C_RGB+C_IR dim
                # # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
                # sim_concate_rgb = F.normalize(f_out_rgb, dim=1).mm(concate_mem.detach().data.t()) #
                # # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
                # # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
                # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1)
                # confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem)# B dim
                # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
                # # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
                # # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
                # # loss_confusion_rgb = self.tri(torch.cat((F.normalize(f_out_rgb, dim=1),F.normalize(confusion_feat_rgb, dim=1)),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
                # loss_confusion_rgb = self.tri(torch.cat((f_out_rgb,confusion_feat_rgb),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
                # # else:
                # # concate_mem =  torch.cat((self.memory_rgb.features,self.memory_ir.features),dim=0) #C_RGB+C_IR dim
                # # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
                # sim_concate_ir = F.normalize(f_out_ir, dim=1).mm(concate_mem.detach().data.t()) #
                # sim_concate_weight_ir = F.softmax(sim_concate_ir/0.05,dim=1) ##B C_RGB+C_IR
                # # sim_concate_weight_ir = torch.cat((F.softmax(sim_concate_ir[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_ir[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
                # sim_concate_weight_ir = F.softmax(sim_concate_weight_ir/0.05,dim=1)
                # confusion_feat_ir = sim_concate_weight_ir.mm(concate_mem)# B dim
                # # loss_confusion_ir = 0.1*self.memory_ir(confusion_feat, labels_ir)
                # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_ir.features.t())
                # # loss_confusion_ir= F.cross_entropy(confusion_out, labels_ir)
                # # loss_confusion_ir = self.tri(torch.cat((F.normalize(f_out_ir, dim=1),F.normalize(confusion_feat_ir, dim=1)),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
                # loss_confusion_ir = self.tri(torch.cat((f_out_ir,confusion_feat_ir),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))
             

            # for i in range(part):
            #     intersect_count_list=[]
            #     ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #     Score_TOPK = 20#20#10
            #     topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
            #     # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #     cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
            #     for l in range(Score_TOPK):
            #         intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         intersect_count_list.append(intersect_count)
            #     intersect_count_list = torch.cat(intersect_count_list,1)
            #     intersect_count, _ = intersect_count_list.max(1)
            #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #     cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #     if i==0: 
            #         mask_rgb_rgb = (cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
            #     else:
            #         mask_rgb_rgb = mask_rgb_rgb.mul(cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 

            # num_neighbor_rgb_rgb = mask_rgb_rgb.sum(dim=1)+1
            # sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
            # sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
            # score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
            # rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##


            # for i in range(part):
            #     intersect_count_list=[]
            #     ins_sim_ir_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #     Score_TOPK = 20#20#10
            #     topk, cluster_indices_ir_ir = torch.topk(ins_sim_ir_ir, int(Score_TOPK))#20
            #     # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #     cluster_label_ir_ir = self.wise_memory_ir.labels[cluster_indices_ir_ir].detach()#.cpu()
            #     for l in range(Score_TOPK):
            #         intersect_count=(cluster_label_ir_ir == cluster_label_ir_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         intersect_count_list.append(intersect_count)
            #     intersect_count_list = torch.cat(intersect_count_list,1)
            #     intersect_count, _ = intersect_count_list.max(1)
            #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #     cluster_label_ir_ir = torch.gather(cluster_label_ir_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #     if i==0: 
            #         mask_ir_ir =(cluster_label_ir_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda()  
            #     else:
            #         mask_ir_ir = mask_ir_ir.mul(cluster_label_ir_ir.cuda()==self.wise_memory_ir.cuda().labels).float().cuda()  

            # num_neighbor_ir_ir = mask_ir_ir.sum(dim=1)+1
            # sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
            # sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
            # score_intra_ir_ir=   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
            # ir_ir_loss = -score_intra_ir_ir.log().mul(mask_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##




#################cps refine
            # if epoch>=1000:#self.cmlabel:
            #     if epoch %2 ==0:
            #         for i in range(part):
            #             intersect_count_list=[]
            #             ins_sim_rgb_ir = F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #             Score_TOPK = 20#20#10
            #             topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #             # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #             cluster_label_rgb_ir = self.wise_memory_ir.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #             for l in range(Score_TOPK):
            #                 intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #                 intersect_count_list.append(intersect_count)
            #             intersect_count_list = torch.cat(intersect_count_list,1)
            #             intersect_count, _ = intersect_count_list.max(1)
            #             topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #             cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1)
            #             if i==0: 
            #                 mask_rgb_ir = (cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda() 
            #             else:
            #                 mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).cuda().float()
            #         # i=0
            #         # intersect_count_list=[]
            #         # ins_sim_rgb_ir = F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #         # Score_TOPK = 20#20#10
            #         # topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #         # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #         # cluster_label_rgb_ir = self.wise_memory_ir.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #         # for l in range(Score_TOPK):
            #         #     intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         #     intersect_count_list.append(intersect_count)
            #         # intersect_count_list = torch.cat(intersect_count_list,1)
            #         # intersect_count, _ = intersect_count_list.max(1)
            #         # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #         # cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1)
            #         # if i==0: 
            #         #     mask_rgb_ir = (cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda() 
            #         # else:
            #         #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).cuda().float()


            #         # i=1
            #         # intersect_count_list=[]
            #         # ins_sim_rgb_ir = F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #         # Score_TOPK = 20#20#10
            #         # topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #         # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #         # cluster_label_rgb_ir = self.wise_memory_ir.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #         # for l in range(Score_TOPK):
            #         #     intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         #     intersect_count_list.append(intersect_count)
            #         # intersect_count_list = torch.cat(intersect_count_list,1)
            #         # intersect_count, _ = intersect_count_list.max(1)
            #         # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #         # cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1)
            #         # if i==0: 
            #         #     mask_rgb_ir = (cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda() 
            #         # else:
            #         #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).cuda().float()

            #         # i=2
            #         # intersect_count_list=[]
            #         # ins_sim_rgb_ir = F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #         # Score_TOPK = 20#20#10
            #         # topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #         # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #         # cluster_label_rgb_ir = self.wise_memory_ir.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #         # for l in range(Score_TOPK):
            #         #     intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         #     intersect_count_list.append(intersect_count)
            #         # intersect_count_list = torch.cat(intersect_count_list,1)
            #         # intersect_count, _ = intersect_count_list.max(1)
            #         # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #         # cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1)
            #         # if i==0: 
            #         #     mask_rgb_ir = (cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda() 
            #         # else:
            #         #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).cuda().float()
            #         # i=3
            #         # intersect_count_list=[]
            #         # ins_sim_rgb_ir = F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #         # Score_TOPK = 20#20#10
            #         # topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #         # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #         # cluster_label_rgb_ir = self.wise_memory_ir.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #         # for l in range(Score_TOPK):
            #         #     intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         #     intersect_count_list.append(intersect_count)
            #         # intersect_count_list = torch.cat(intersect_count_list,1)
            #         # intersect_count, _ = intersect_count_list.max(1)
            #         # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #         # cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1)
            #         # if i==0: 
            #         #     mask_rgb_ir = (cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda() 
            #         # else:
            #         #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_ir.labels.cuda()).cuda().float()

            #         num_neighbor_rgb_ir = mask_rgb_ir.sum(dim=1)+1
            #         sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
            #         sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
            #         score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            #         score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
            #         rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            #         rgb_ir_loss = rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##
            #     else:
            #         # mask_rgb_ir = torch.ones((f_out_ir.size(0),self.wise_memory_rgb.features.size(0))).cuda()
            #         for i in range(part):
            #             intersect_count_list=[]
            #             ins_sim_rgb_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #             Score_TOPK = 20#20#10
            #             topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #             # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #             cluster_label_rgb_ir = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #             for l in range(Score_TOPK):
            #                 intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #                 intersect_count_list.append(intersect_count)
            #             intersect_count_list = torch.cat(intersect_count_list,1)
            #             intersect_count, _ = intersect_count_list.max(1)
            #             topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #             cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #             if i==0: 
            #                 mask_rgb_ir =(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
            #             else:
            #                 mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
            #         # i = 0
            #         # intersect_count_list=[]
            #         # ins_sim_rgb_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #         # Score_TOPK = 20#20#10
            #         # topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #         # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #         # cluster_label_rgb_ir = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #         # for l in range(Score_TOPK):
            #         #     intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         #     intersect_count_list.append(intersect_count)
            #         # intersect_count_list = torch.cat(intersect_count_list,1)
            #         # intersect_count, _ = intersect_count_list.max(1)
            #         # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #         # cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #         # if i==0: 
            #         #     mask_rgb_ir =(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
            #         # else:
            #         #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  


            #         # i = 1
            #         # intersect_count_list=[]
            #         # ins_sim_rgb_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #         # Score_TOPK = 20#20#10
            #         # topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #         # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #         # cluster_label_rgb_ir = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #         # for l in range(Score_TOPK):
            #         #     intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         #     intersect_count_list.append(intersect_count)
            #         # intersect_count_list = torch.cat(intersect_count_list,1)
            #         # intersect_count, _ = intersect_count_list.max(1)
            #         # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #         # cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #         # if i==0: 
            #         #     mask_rgb_ir =(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
            #         # else:
            #         #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  

            #         # i = 2
            #         # intersect_count_list=[]
            #         # ins_sim_rgb_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #         # Score_TOPK = 20#20#10
            #         # topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #         # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #         # cluster_label_rgb_ir = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #         # for l in range(Score_TOPK):
            #         #     intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         #     intersect_count_list.append(intersect_count)
            #         # intersect_count_list = torch.cat(intersect_count_list,1)
            #         # intersect_count, _ = intersect_count_list.max(1)
            #         # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #         # cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #         # if i==0: 
            #         #     mask_rgb_ir =(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
            #         # else:
            #         #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  

            #         # i = 3
            #         # intersect_count_list=[]
            #         # ins_sim_rgb_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #         # Score_TOPK = 20#20#10
            #         # topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
            #         # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #         # cluster_label_rgb_ir = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            #         # for l in range(Score_TOPK):
            #         #     intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         #     intersect_count_list.append(intersect_count)
            #         # intersect_count_list = torch.cat(intersect_count_list,1)
            #         # intersect_count, _ = intersect_count_list.max(1)
            #         # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #         # cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #         # if i==0: 
            #         #     mask_rgb_ir =(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
            #         # else:
            #         #     mask_rgb_ir = mask_rgb_ir.mul(cluster_label_rgb_ir.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  




            #         num_neighbor_rgb_ir = mask_rgb_ir.sum(dim=1)+1
            #         sim_rgb_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
            #         sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
            #         score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            #         score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
            #         ir_rgb_loss = -score_intra_rgb_ir.log().mul(mask_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            #         ir_rgb_loss = ir_rgb_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##


            # for i in range(part):
            #     intersect_count_list=[]
            #     ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #     Score_TOPK = 20#20#10
            #     topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
            #     # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #     cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
            #     for l in range(Score_TOPK):
            #         intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         intersect_count_list.append(intersect_count)
            #     intersect_count_list = torch.cat(intersect_count_list,1)
            #     intersect_count, _ = intersect_count_list.max(1)
            #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #     cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #     if i==0: 
            #         mask_rgb_rgb = (cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
            #     else:
            #         mask_rgb_rgb = mask_rgb_rgb.mul(cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
                
            # # i=0
            # # intersect_count_list=[]
            # # ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            # # Score_TOPK = 20#20#10
            # # topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
            # # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            # # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            # # for l in range(Score_TOPK):
            # #     intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            # #     intersect_count_list.append(intersect_count)
            # # intersect_count_list = torch.cat(intersect_count_list,1)
            # # intersect_count, _ = intersect_count_list.max(1)
            # # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # # cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            # # if i==0: 
            # #     mask_rgb_rgb = (cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
            # # else:
            # #     mask_rgb_rgb = mask_rgb_rgb.mul(cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  

            # # i=1
            # # intersect_count_list=[]
            # # ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            # # Score_TOPK = 20#20#10
            # # topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
            # # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            # # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            # # for l in range(Score_TOPK):
            # #     intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            # #     intersect_count_list.append(intersect_count)
            # # intersect_count_list = torch.cat(intersect_count_list,1)
            # # intersect_count, _ = intersect_count_list.max(1)
            # # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # # cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            # # if i==0: 
            # #     mask_rgb_rgb = (cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
            # # else:
            # #     mask_rgb_rgb = mask_rgb_rgb.mul(cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
            # # i=2
            # # intersect_count_list=[]
            # # ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            # # Score_TOPK = 20#20#10
            # # topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
            # # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            # # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            # # for l in range(Score_TOPK):
            # #     intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            # #     intersect_count_list.append(intersect_count)
            # # intersect_count_list = torch.cat(intersect_count_list,1)
            # # intersect_count, _ = intersect_count_list.max(1)
            # # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # # cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            # # if i==0: 
            # #     mask_rgb_rgb = (cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
            # # else:
            # #     mask_rgb_rgb = mask_rgb_rgb.mul(cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  
            # # i=3
            # # intersect_count_list=[]
            # # ins_sim_rgb_rgb= F.normalize(f_out_rgb[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_rgb.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            # # Score_TOPK = 20#20#10
            # # topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
            # # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            # # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_ir].detach()#.cpu()
            # # for l in range(Score_TOPK):
            # #     intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            # #     intersect_count_list.append(intersect_count)
            # # intersect_count_list = torch.cat(intersect_count_list,1)
            # # intersect_count, _ = intersect_count_list.max(1)
            # # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # # cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            # # if i==0: 
            # #     mask_rgb_rgb = (cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda() 
            # # else:
            # #     mask_rgb_rgb = mask_rgb_rgb.mul(cluster_label_rgb_rgb.cuda()==self.wise_memory_rgb.labels.cuda()).float().cuda()  


            # num_neighbor_rgb_rgb = mask_rgb_rgb.sum(dim=1)+1
            # sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
            # sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
            # score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
            # rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##


            # for i in range(part):
            #     intersect_count_list=[]
            #     ins_sim_ir_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #     Score_TOPK = 20#20#10
            #     topk, cluster_indices_ir_ir = torch.topk(ins_sim_ir_ir, int(Score_TOPK))#20
            #     # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #     cluster_label_ir_ir = self.wise_memory_ir.labels[cluster_indices_ir_ir].detach()#.cpu()
            #     for l in range(Score_TOPK):
            #         intersect_count=(cluster_label_ir_ir == cluster_label_ir_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #         intersect_count_list.append(intersect_count)
            #     intersect_count_list = torch.cat(intersect_count_list,1)
            #     intersect_count, _ = intersect_count_list.max(1)
            #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
            #     cluster_label_ir_ir = torch.gather(cluster_label_ir_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            #     if i==0: 
            #         mask_ir_ir =(cluster_label_ir_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda()  
            #     else:
            #         mask_ir_ir = mask_ir_ir.mul(cluster_label_ir_ir.cuda()==self.wise_memory_ir.cuda().labels).float().cuda()  
            # # i=0
            # # intersect_count_list=[]
            # # ins_sim_ir_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            # # Score_TOPK = 20#20#10
            # # topk, cluster_indices_ir_ir = torch.topk(ins_sim_ir_ir, int(Score_TOPK))#20
            # # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            # # cluster_label_ir_ir = self.wise_memory_ir.labels[cluster_indices_ir_ir].detach()#.cpu()
            # # for l in range(Score_TOPK):
            # #     intersect_count=(cluster_label_ir_ir == cluster_label_ir_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            # #     intersect_count_list.append(intersect_count)
            # # intersect_count_list = torch.cat(intersect_count_list,1)
            # # intersect_count, _ = intersect_count_list.max(1)
            # # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # # cluster_label_ir_ir = torch.gather(cluster_label_ir_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            # # if i==0: 
            # #     mask_ir_ir =(cluster_label_ir_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda()  
            # # else:
            # #     mask_ir_ir = mask_ir_ir.mul(cluster_label_ir_ir.cuda()==self.wise_memory_ir.cuda().labels).float().cuda()  

            # # i=1
            # # intersect_count_list=[]
            # # ins_sim_ir_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            # # Score_TOPK = 20#20#10
            # # topk, cluster_indices_ir_ir = torch.topk(ins_sim_ir_ir, int(Score_TOPK))#20
            # # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            # # cluster_label_ir_ir = self.wise_memory_ir.labels[cluster_indices_ir_ir].detach()#.cpu()
            # # for l in range(Score_TOPK):
            # #     intersect_count=(cluster_label_ir_ir == cluster_label_ir_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            # #     intersect_count_list.append(intersect_count)
            # # intersect_count_list = torch.cat(intersect_count_list,1)
            # # intersect_count, _ = intersect_count_list.max(1)
            # # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # # cluster_label_ir_ir = torch.gather(cluster_label_ir_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            # # if i==0: 
            # #     mask_ir_ir =(cluster_label_ir_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda()  
            # # else:
            # #     mask_ir_ir = mask_ir_ir.mul(cluster_label_ir_ir.cuda()==self.wise_memory_ir.cuda().labels).float().cuda()  

            # # i=2
            # # intersect_count_list=[]
            # # ins_sim_ir_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            # # Score_TOPK = 20#20#10
            # # topk, cluster_indices_ir_ir = torch.topk(ins_sim_ir_ir, int(Score_TOPK))#20
            # # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            # # cluster_label_ir_ir = self.wise_memory_ir.labels[cluster_indices_ir_ir].detach()#.cpu()
            # # for l in range(Score_TOPK):
            # #     intersect_count=(cluster_label_ir_ir == cluster_label_ir_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            # #     intersect_count_list.append(intersect_count)
            # # intersect_count_list = torch.cat(intersect_count_list,1)
            # # intersect_count, _ = intersect_count_list.max(1)
            # # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # # cluster_label_ir_ir = torch.gather(cluster_label_ir_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            # # if i==0: 
            # #     mask_ir_ir =(cluster_label_ir_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda()  
            # # else:
            # #     mask_ir_ir = mask_ir_ir.mul(cluster_label_ir_ir.cuda()==self.wise_memory_ir.cuda().labels).float().cuda()  

            # # i=3
            # # intersect_count_list=[]
            # # ins_sim_ir_ir = F.normalize(f_out_ir[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_ir.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            # # Score_TOPK = 20#20#10
            # # topk, cluster_indices_ir_ir = torch.topk(ins_sim_ir_ir, int(Score_TOPK))#20
            # # # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            # # cluster_label_ir_ir = self.wise_memory_ir.labels[cluster_indices_ir_ir].detach()#.cpu()
            # # for l in range(Score_TOPK):
            # #     intersect_count=(cluster_label_ir_ir == cluster_label_ir_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            # #     intersect_count_list.append(intersect_count)
            # # intersect_count_list = torch.cat(intersect_count_list,1)
            # # intersect_count, _ = intersect_count_list.max(1)
            # # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # # cluster_label_ir_ir = torch.gather(cluster_label_ir_ir.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1,1) 
            # # if i==0: 
            # #     mask_ir_ir =(cluster_label_ir_ir.cuda()==self.wise_memory_ir.labels.cuda()).float().cuda()  
            # # else:
            # #     mask_ir_ir = mask_ir_ir.mul(cluster_label_ir_ir.cuda()==self.wise_memory_ir.cuda().labels).float().cuda()  


            # num_neighbor_ir_ir = mask_ir_ir.sum(dim=1)+1
            # sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
            # sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
            # score_intra_ir_ir=   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
            # ir_ir_loss = -score_intra_ir_ir.log().mul(mask_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##



            lamda_i = 1
####################
            # lamda_i = 1
            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(ir_ir_loss+rgb_rgb_loss)+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)
            # loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(loss_confusion_rgb+loss_confusion_ir)#+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)

            loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(rgb_ir_loss+ir_rgb_loss)#++(ir_ir_loss+rgb_rgb_loss)(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)



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
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      'ir_rgb_loss_log {:.3f}\t'
                      'rgb_ir_loss_log {:.3f}\t'
                      'ir_ir_loss_log {:.3f}\t'
                      'rgb_rgb_loss_log {:.3f}\t'
                      # 'ir_ir_loss_log {:.3f}\t'
                      # 'rgb_rgb_loss_log {:.3f}\t'
                      # 'loss_ins_ir_log {:.3f}\t'
                      # 'loss_ins_rgb_log {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir_log.val,loss_rgb_log.val,loss_camera_ir_log.val,loss_camera_rgb_log.val,ir_rgb_loss_log.val,rgb_ir_loss_log.val,ir_ir_loss_log.val,rgb_rgb_loss_log.val))

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


class ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_all =  memory
        self.nameMap_all = []
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 0)
        # self.criterion_pa = PredictionAlignmentLoss(lambda_vr=0.5, lambda_rv=0.5)
        self.camstart=0
        self.tri = TripletLoss_WRT()
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        ##########init camera proxy
        # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)
        start_cam=0
        ir_num = len(all_label_ir)
        rgb_num = len(all_label_rgb)-ir_num
        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_all[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_all[name] for name in name_rgb]).cuda()
            indexes_rgb = []#torch.cat((indexes_rgb,indexes_rgb),-1)
            cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)
            # for path,cameraid in  zip(name_ir,cids_ir):
            #     print(path,cameraid)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            # indexes_all = torch.cat((index_rgb,index_ir),-1)
            cid_all=torch.cat((cid_rgb,cid_ir),-1)
            f_out_all=torch.cat((f_out_rgb,f_out_ir),0)
            labels_all = torch.cat((labels_rgb,labels_ir),-1)
#####################################
            loss_all = self.memory_rgb(f_out_all, labels_all) 
            # loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            
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

            # loss_ins_ir = self.wise_memory_all(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_all(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # thresh=0.9

   
    #         # if epoch %2 ==0:
    #         sim_prob_all_rgb_rgb = torch.cat([F.softmax(self.part_sim(self.wise_memory_all.features[:rgb_num].cuda(1).detach(),F.normalize(percam_memory_rgb[i].cuda(1).detach().data, dim=1)),dim=1)/0.01 for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()
    #         sim_prob_B_rgb_rgb = torch.cat([F.softmax(self.part_sim(F.normalize(f_out_rgb, dim=1).cuda(1).detach(),F.normalize(percam_memory_rgb[i].cuda(1).detach().data, dim=1)),dim=1)/0.01 for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()
    #         sim_prob_rgb_rgb = F.normalize(sim_prob_B_rgb_rgb, dim=1).mm(F.normalize(sim_prob_all_rgb_rgb.t(),dim=1))#B N
    #         sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_all.features[:rgb_num].detach().data.t())
    #         sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
    #         nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
    #         nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
    #         mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
    #         mask_neighbor_prob_rgb_rgb = torch.gt(sim_prob_rgb_rgb, nearest_prob_rgb_rgb * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
    #         num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)+1
    #         # print('num_neighbor_rgb_rgb',num_neighbor_rgb_rgb)
    #         score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
    #         # print('score_intra',score_intra)
    #         score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
    #         # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
    #         rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
    #         rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
    #         del sim_prob_B_rgb_rgb,sim_prob_rgb_rgb#,sim_prob_ir_ir
    # # #################ir-ir
    #     # else:
    #         sim_prob_all_ir_ir = torch.cat([F.softmax(self.part_sim(self.wise_memory_all.features[rgb_num:].cuda(1).detach(),F.normalize(percam_memory_rgb[i].cuda(1).detach().data, dim=1)),dim=1)/0.01 for i in range(len(percam_memory_rgb))],dim=1).detach().data
    #         sim_prob_B_ir_ir = torch.cat([F.softmax(self.part_sim(F.normalize(f_out_ir, dim=1).cuda(1).detach(),F.normalize(percam_memory_rgb[i].cuda(1).detach().data, dim=1)),dim=1)/0.01 for i in range(len(percam_memory_rgb))],dim=1).detach().data
    #         sim_prob_ir_ir = F.normalize(sim_prob_B_ir_ir, dim=1).mm(F.normalize(sim_prob_all_ir_ir.t(),dim=1))#B N
    #         sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_all.features[rgb_num:].detach().data.t())
    #         sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
    #         nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
    #         nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
    #         mask_neighbor_prob_ir_ir = torch.gt(sim_prob_ir_ir, nearest_prob_ir_ir * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
    #         mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
    #         num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_prob_ir_ir).sum(dim=1)+1#.mul(sim_wise).
    #         # print('num_neighbor_ir_ir',num_neighbor_ir_ir)
    #         score_intra_ir_ir =   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
    #         # print('score_intra',score_intra)
    #         score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
    #         # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
    #         ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
    #         ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##

    #         del sim_prob_all_ir_ir,sim_prob_all_rgb_rgb,sim_prob_ir_ir
            # concate_mem =  percam_tempV_rgb#torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
            # # concate_mem =  self.memory_rgb.features#torch.cat((self.memory_rgb.features,self.memory_ir.features),dim=0) #C_RGB+C_IR dim
            # sim_concate = F.normalize(f_out_all, dim=1).mm(concate_mem.detach().data.t()) #
            # sim_concate_weight = F.softmax(sim_concate/0.05,dim=1) ##B C_RGB+C_IR
            # confusion_feat = sim_concate_weight.mm(concate_mem)# B dim
            # # loss_confusion_ir = 0.1*self.memory_ir(confusion_feat, labels_ir)
            # # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_ir.features.t())
            # # loss_confusion_ir= F.cross_entropy(confusion_out, labels_ir)


            # loss_confusion_all= self.tri(torch.cat((f_out_all,confusion_feat),dim=0),torch.cat((labels_all,labels_all),dim=-1))
            # # loss_confusion_all= self.tri(torch.cat((F.normalize(f_out_all, dim=1),F.normalize(confusion_feat, dim=1)),dim=0),torch.cat((labels_all,labels_all),dim=-1))
    
            # # loss_tri = self.tri(F.normalize(f_out_all, dim=1),labels_all)
            # loss_tri = self.tri(F.normalize(f_out_all, dim=1),labels_all)
            # # loss_tri = self.tri(torch.cat((F.normalize(f_out_all, dim=1),F.normalize(confusion_feat_ir, dim=1)),dim=0),torch.cat((labels_ir,labels_ir),dim=-1))

            concate_mem = torch.cat(percam_memory_rgb,dim=0)  #C_RGB+C_IR dim
            # concate_mem =  torch.cat((percam_tempV_rgb,percam_tempV_ir),dim=0) #C_RGB+C_IR dim
            sim_concate_rgb = torch.cat([F.softmax(F.normalize(f_out_all, dim=1).mm(percam_memory_rgb[i].detach().data.t()),dim=1) for i in range(len(percam_memory_rgb))],dim=1)
            # sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1) ##B C_RGB+C_IR
            # sim_concate_weight_rgb = torch.cat((F.softmax(sim_concate_rgb[:,:self.memory_rgb.features.size(0)],dim=1),F.softmax(sim_concate_rgb[:,self.memory_rgb.features.size(0):],dim=1)),dim=1)
            sim_concate_weight_rgb = F.softmax(sim_concate_rgb/0.05,dim=1)
            confusion_feat_rgb = sim_concate_weight_rgb.mm(concate_mem)# B dim
            # confusion_out = F.normalize(confusion_feat, dim=1).mm(self.memory_rgb.features.t())
            # loss_confusion_rgb = F.cross_entropy(confusion_out, labels_rgb)
            # loss_confusion_rgb = 0.1*self.memory_rgb(confusion_feat, labels_rgb)
            loss_confusion_all = self.tri(torch.cat((f_out_all,confusion_feat_rgb),dim=0),torch.cat((labels_all,labels_all),dim=-1))
            # loss_confusion_rgb = self.tri(torch.cat((F.normalize(f_out_rgb, dim=1),F.normalize(confusion_feat_rgb, dim=1)),dim=0),torch.cat((labels_rgb,labels_rgb),dim=-1))
   
            # cluster_label_rgb_rgb=[]
            # for i in range(part):
            #     intersect_count_list=[]
            #     ins_sim_rgb_rgb= F.normalize(f_out_all[:,i*768:(i+1)*768], dim=-1).mm(F.normalize(self.wise_memory_all.features[:,i*768:(i+1)*768].detach().t(), dim=-1))
            #     Score_TOPK = 20#20#10
            #     topk, cluster_indices_rgb_rgb = torch.topk(ins_sim_rgb_rgb, int(Score_TOPK))#20
            #     # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
            #     # cluster_label_rgb_rgb = self.wise_memory_rgb.labels[cluster_indices_rgb_rgb].detach()#.cpu()
            #     cluster_label_rgb_rgb.append(self.wise_memory_all.labels[cluster_indices_rgb_rgb].detach())#.cpu()
            # cluster_label_rgb_rgb=torch.cat(cluster_label_rgb_rgb,1)
            # for l in range(Score_TOPK*part):
            #     intersect_count=(cluster_label_rgb_rgb == cluster_label_rgb_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
            #     intersect_count_list.append(intersect_count)
            # intersect_count_list = torch.cat(intersect_count_list,1)
            # intersect_count, _ = intersect_count_list.max(1)
            # topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # cluster_label_rgb_rgb = torch.gather(cluster_label_rgb_rgb.cuda(), dim=1, index=cluster_label_index.view(-1,1).cuda()).view(-1) 
            # no_neg_rgb=torch.where(cluster_label_rgb_rgb>=0)#torch.(cluster_label_rgb_rgb,0)
            # # print(cluster_label_rgb_rgb[no_neg_rgb])
            # all_all_loss = self.memory_rgb(f_out_all[no_neg_rgb], cluster_label_rgb_rgb[no_neg_rgb],training_momentum=0.9)


            lamda_i = 0
            loss = loss_all+lamda_c*loss_camera_all+loss_confusion_all#all_all_loss+lamda_i*(loss_ins_ir+loss_ins_rgb)#+rgb_rgb_loss+ir_ir_loss#+lamda_i*(loss_ins_ir+loss_ins_rgb)#+rgb_rgb_loss+ir_ir_loss#+#+ir_ir_loss #+ loss_tri+loss_rgb_ir_trans+loss_ir_rgb_trans +(loss_rgb_trans+loss_ir_trans)

            # loss = lamda_cc*(loss_ir+loss_rgb)+loss_camera_rgb+loss_camera_ir #+ loss_tri
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
                      'Loss all {:.3f}\t'
                      'Loss all {:.3f}\t'
                      'camera all {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_all,loss_all,loss_camera_all.item(),loss_camera_rgb.item()))
                # if epoch >= start_cam:
                # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
                # print('loss_tri',loss_tri.item())
                print('loss_confusion_all',loss_confusion_all.item())
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



# class ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine(object):
#     def __init__(self, encoder, memory=None):
#         super(ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine, self).__init__()
#         self.encoder = encoder
#         self.memory_ir = memory
#         self.memory_rgb = memory
#         self.wise_memory_all =  memory
#         self.nameMap_all = []
#         self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
#         # self.criterion_pa = PredictionAlignmentLoss(lambda_vr=0.5, lambda_rv=0.5)
#         self.camstart=0

#     def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
#         all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
#                  print_freq=10, train_iters=400):
#         self.encoder.train()

#         batch_time = AverageMeter()
#         data_time = AverageMeter()

#         losses = AverageMeter()
#         ##########init camera proxy
#         # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
#         concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)
#         start_cam=0
#         ir_num = len(all_label_ir)
#         rgb_num = len(all_label_rgb)-ir_num
#         end = time.time()
#         for i in range(train_iters):
#             # load data
#             inputs_ir = data_loader_ir.next()
#             inputs_rgb = data_loader_rgb.next()
#             data_time.update(time.time() - end)

#             # process inputs
#             inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
#             inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
#             # forward
#             inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
#             labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

#             indexes_ir = torch.tensor([self.nameMap_all[name] for name in name_ir]).cuda()
#             indexes_rgb = torch.tensor([self.nameMap_all[name] for name in name_rgb]).cuda()
#             indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
#             cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
#             # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
#             # labels_ir = torch.cat((labels_ir,labels_ir),-1)
#             # for path,cameraid in  zip(name_ir,cids_ir):
#             #     print(path,cameraid)

#             _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
#             cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
#                 cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
#             # indexes_all = torch.cat((index_rgb,index_ir),-1)
#             cid_all=torch.cat((cid_rgb,cid_ir),-1)
#             f_out_all=torch.cat((f_out_rgb,f_out_ir),0)
#             labels_all = torch.cat((labels_rgb,labels_ir),-1)
# #####################################
#             loss_all = self.memory_rgb(f_out_all, labels_all) 
#             # loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            
#             lamda_c=0.1
#             # start=30
#             loss_camera_ir=torch.tensor([0.]).cuda()
#             loss_camera_rgb=torch.tensor([0.]).cuda()
#             loss_camera_all = torch.tensor([0.]).cuda()
#             rgb_rgb_loss = torch.tensor([0.]).cuda()
#             ir_ir_loss = torch.tensor([0.]).cuda()
#             # if epoch >= self.camstart:
#             loss_camera_all = self.camera_loss(f_out_all,cid_all,labels_all,percam_tempV_rgb,concate_intra_class_rgb,percam_tempV_rgb,cross_m=True)#self.camera_loss(f_out_ir,cid_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
#                 # loss_camera_rgb = self.camera_loss(f_out_rgb,cid_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)

#             loss_ins_ir = self.wise_memory_all(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
#             loss_ins_rgb= self.wise_memory_all(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
#             thresh=0.9

   
#     #         # if epoch %2 ==0:
#     #         sim_prob_all_rgb_rgb = torch.cat([F.softmax(self.part_sim(self.wise_memory_all.features[:rgb_num].cuda(1).detach(),F.normalize(percam_memory_rgb[i].cuda(1).detach().data, dim=1)),dim=1)/0.01 for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()
#     #         sim_prob_B_rgb_rgb = torch.cat([F.softmax(self.part_sim(F.normalize(f_out_rgb, dim=1).cuda(1).detach(),F.normalize(percam_memory_rgb[i].cuda(1).detach().data, dim=1)),dim=1)/0.01 for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()
#     #         sim_prob_rgb_rgb = F.normalize(sim_prob_B_rgb_rgb, dim=1).mm(F.normalize(sim_prob_all_rgb_rgb.t(),dim=1))#B N
#     #         sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_all.features[:rgb_num].detach().data.t())
#     #         sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
#     #         nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
#     #         nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
#     #         mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
#     #         mask_neighbor_prob_rgb_rgb = torch.gt(sim_prob_rgb_rgb, nearest_prob_rgb_rgb * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
#     #         num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)+1
#     #         # print('num_neighbor_rgb_rgb',num_neighbor_rgb_rgb)
#     #         score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#     #         # print('score_intra',score_intra)
#     #         score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
#     #         # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
#     #         rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#     #         rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
#     #         del sim_prob_B_rgb_rgb,sim_prob_rgb_rgb#,sim_prob_ir_ir
#     # # #################ir-ir
#     #     # else:
#     #         sim_prob_all_ir_ir = torch.cat([F.softmax(self.part_sim(self.wise_memory_all.features[rgb_num:].cuda(1).detach(),F.normalize(percam_memory_rgb[i].cuda(1).detach().data, dim=1)),dim=1)/0.01 for i in range(len(percam_memory_rgb))],dim=1).detach().data
#     #         sim_prob_B_ir_ir = torch.cat([F.softmax(self.part_sim(F.normalize(f_out_ir, dim=1).cuda(1).detach(),F.normalize(percam_memory_rgb[i].cuda(1).detach().data, dim=1)),dim=1)/0.01 for i in range(len(percam_memory_rgb))],dim=1).detach().data
#     #         sim_prob_ir_ir = F.normalize(sim_prob_B_ir_ir, dim=1).mm(F.normalize(sim_prob_all_ir_ir.t(),dim=1))#B N
#     #         sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_all.features[rgb_num:].detach().data.t())
#     #         sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
#     #         nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
#     #         nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
#     #         mask_neighbor_prob_ir_ir = torch.gt(sim_prob_ir_ir, nearest_prob_ir_ir * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
#     #         mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
#     #         num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_prob_ir_ir).sum(dim=1)+1#.mul(sim_wise).
#     #         # print('num_neighbor_ir_ir',num_neighbor_ir_ir)
#     #         score_intra_ir_ir =   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#     #         # print('score_intra',score_intra)
#     #         score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
#     #         # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
#     #         ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#     #         ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##

#     #         del sim_prob_all_ir_ir,sim_prob_all_rgb_rgb,sim_prob_ir_ir
#             lamda_i = 0
#             loss = loss_all+lamda_c*loss_camera_all+lamda_i*(loss_ins_ir+loss_ins_rgb)#+rgb_rgb_loss+ir_ir_loss#+lamda_i*(loss_ins_ir+loss_ins_rgb)#+rgb_rgb_loss+ir_ir_loss#+#+ir_ir_loss #+ loss_tri+loss_rgb_ir_trans+loss_ir_rgb_trans +(loss_rgb_trans+loss_ir_trans)

#             # loss = lamda_cc*(loss_ir+loss_rgb)+loss_camera_rgb+loss_camera_ir #+ loss_tri
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             losses.update(loss.item())

#             # print log
#             batch_time.update(time.time() - end)
#             end = time.time()

#             if (i + 1) % print_freq == 0:
#                 print('Epoch: [{}][{}/{}]\t'
#                       'Time {:.3f} ({:.3f})\t'
#                       'Data {:.3f} ({:.3f})\t'
#                       'Loss {:.3f} ({:.3f})\t'
#                       'Loss all {:.3f}\t'
#                       'Loss all {:.3f}\t'
#                       'camera all {:.3f}\t'
#                       'camera rgb {:.3f}\t'
#                       #  'adp ir {:.3f}\t'
#                       # 'adp rgb {:.3f}\t'
#                       .format(epoch, i + 1, len(data_loader_rgb),
#                               batch_time.val, batch_time.avg,
#                               data_time.val, data_time.avg,
#                               losses.val, losses.avg,loss_all,loss_all,loss_camera_all.item(),loss_camera_rgb.item()))
#                 # if epoch >= start_cam:
#                 # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
#                 # print('loss_ins_rgb',loss_ins_rgb.item())
#                 print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
#                     # print('loss_intra_ir,loss_inter_ir,intrawise_loss_ir',loss_ins_ir.item(),loss_intra_ir.item(),loss_inter_ir.item(),intrawise_loss_ir.item())
#                     # print('loss_intra_rgb,loss_inter_rgb,intrawise_loss_rgb',loss_ins_rgb.item(),loss_intra_rgb.item(),loss_inter_rgb.item(),intrawise_loss_rgb.item())
#             # pseudo_labels_all=self.wise_memory_all.labels.numpy()
#             # cluster_features_all = self.generate_cluster_features(pseudo_labels_all, self.wise_memory_all.features)

#             # num_cluster_all = len(set(pseudo_labels_all)) - (1 if -1 in pseudo_labels_all else 0)
#             # cam_moment-0.1
#             # for cc in torch.unique(cid_all):
#             #     # print(cc)
#             #     inds = torch.nonzero(cid_all == cc).squeeze(-1)
#             #     percam_targets = labels_all[inds]
#             #     percam_feat = f_out_all[inds].detach().clone()
 
#             #     for k in range(len(percam_feat)):
#             #         ori_asso_ind = torch.nonzero(concate_intra_class_rgb == percam_targets[k]).squeeze(-1)
#             #         percam_tempV_rgb[ori_asso_ind] = (1-cam_moment)*percam_feat[k]+cam_moment*percam_tempV_rgb[ori_asso_ind]

#             # # self.memory_rgb.features = F.normalize(cluster_features_all, dim=1).cuda()

#     def _parse_data_rgb(self, inputs):
#         imgs,imgs1, name, pids, cids, indexes = inputs
#         return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

#     def _parse_data_ir(self, inputs):
#         imgs, name, pids, cids, indexes = inputs
#         return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

#     def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
#         return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)

#     def init_camera_proxy(self,all_img_cams,all_pseudo_label,intra_id_features):
#         all_img_cams = torch.tensor(all_img_cams).cuda()
#         unique_cams = torch.unique(all_img_cams)
#         # print(self.unique_cams)

#         all_pseudo_label = torch.tensor(all_pseudo_label).cuda()
#         init_intra_id_feat = intra_id_features
#         # print(len(self.init_intra_id_feat))

#         # initialize proxy memory
#         percam_memory = []
#         memory_class_mapper = []
#         concate_intra_class = []
#         for cc in unique_cams:
#             percam_ind = torch.nonzero(all_img_cams == cc).squeeze(-1)
#             uniq_class = torch.unique(all_pseudo_label[percam_ind])
#             uniq_class = uniq_class[uniq_class >= 0]
#             concate_intra_class.append(uniq_class)
#             cls_mapper = {int(uniq_class[j]): j for j in range(len(uniq_class))}
#             memory_class_mapper.append(cls_mapper)  # from pseudo label to index under each camera

#             if len(init_intra_id_feat) > 0:
#                 # print('initializing ID memory from updated embedding features...')
#                 proto_memory = init_intra_id_feat[cc]
#                 proto_memory = proto_memory.cuda()
#                 percam_memory.append(proto_memory.detach())
#         concate_intra_class = torch.cat(concate_intra_class)

#         percam_tempV = []
#         for ii in unique_cams:
#             percam_tempV.append(percam_memory[ii].detach().clone())
#         percam_tempV = torch.cat(percam_tempV, dim=0).cuda()
#         return concate_intra_class,percam_tempV,percam_memory#memory_class_mapper
#     def camera_loss(self,f_out_t1,cids,targets,percam_tempV,concate_intra_class,memory_class_mapper,cross_m=False):
#         beta = 0.07#0.07
#         bg_knn = 50#100#50
#         loss_cam = torch.tensor([0.]).cuda()
#         for cc in torch.unique(cids):

#             # print(cc)
#             inds = torch.nonzero(cids == cc).squeeze(-1)
#             percam_targets = targets[inds]
#             # print(percam_targets)
#             percam_feat = f_out_t1[inds]
#             associate_loss = 0
#             # target_inputs = percam_feat.mm(percam_tempV.t().clone())
#             target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
#             temp_sims = target_inputs.detach().clone()
#             target_inputs /= beta

#             for k in range(len(percam_feat)):
#                 ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
#                 temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
#                 sel_ind = torch.sort(temp_sims[k])[1][-bg_knn:]
#                 # sel_ind_2 = torch.sort(temp_sims[k])[1][1:bg_knn*10]
#                 # sel_ind = torch.cat((sel_ind,sel_ind_2), dim=-1)
#                 # nearest_intra = temp_sims[k].max(dim=-1, keepdim=True)[0]
#                 # mask_neighbor_intra = torch.gt(temp_sims[k], nearest_intra * 0.8)
#                 # sel_ind = torch.nonzero(mask_neighbor_intra).squeeze(-1)
#                 # if cross_m == True:
#                 #     concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)#
#                 #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
#                 #     torch.device('cuda'))
#                 #     concated_target[0:len(ori_asso_ind)+len(sel_ind)] = 1.0 / (len(ori_asso_ind)+len(sel_ind)+1e-8)
#                 # else:
#                 concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)#target_inputs[k, ori_asso_ind]#
#                 concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
#                 torch.device('cuda'))
#                 # concated_target[0:len(ori_asso_ind)] = 1.0 / (len(ori_asso_ind)+1e-8)
#                 # print('len(concated_input)',len(concated_input))
#                 # print('len(ori_asso_ind)',len(ori_asso_ind))
#                 # concated_target[0:len(concated_input)] = 1.0 / (len(concated_input)+1e-8)
#                 # concated_target[0:len(ori_asso_ind)+len(sel_ind)] = 1.0 / (len(ori_asso_ind)+len(sel_ind)+1e-8)
#                 concated_target[0:len(ori_asso_ind)] = 1.0 / (len(ori_asso_ind)+1e-8)
#                 associate_loss += -1 * (
#                         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
#                     0)).sum()
#             loss_cam +=  associate_loss / len(percam_feat)
#         return loss_cam
#     @torch.no_grad()
#     def generate_cluster_features(self,labels, features):
#         centers = collections.defaultdict(list)
#         for i, label in enumerate(labels):
#             if label == -1:
#                 continue
#             centers[labels[i]].append(features[i])

#         centers = [
#             torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
#         ]

#         centers = torch.stack(centers, dim=0)
#         return centers

#     def mask(self,ones, labels,ins_label):
#         for i, label in enumerate(labels):
#             ones[i,ins_label==label] = 1
#         return ones

#     def part_sim(self,query_t, key_m):
#         self.seq_len=5
#         q, d_5 = query_t.size() # b d*5,  
#         k, d_5 = key_m.size()

#         z= int(d_5/self.seq_len)
#         d = int(d_5/self.seq_len)        
#         query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
#         key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C
 
#         # query_t = F.normalize(tgt.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
#         # key_m = F.normalize(memory.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
#         score = F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m)/0.01,dim=-1).view(q,-1) # B Q N N
#         # score = F.softmax(score,dim=1)
#         return score


class ClusterContrastTrainer_pretrain_camera_wise_3(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_wise_3, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_all =  memory
        self.nameMap_all = []
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
        # self.criterion_pa = PredictionAlignmentLoss(lambda_vr=0.5, lambda_rv=0.5)
        self.camstart=0

    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        ##########init camera proxy
        # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)
        start_cam=0

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            indexes_ir = []#torch.tensor([self.nameMap_all[name] for name in name_ir]).cuda()
            indexes_rgb = []#torch.tensor([self.nameMap_all[name] for name in name_rgb]).cuda()
            indexes_rgb = []#torch.cat((indexes_rgb,indexes_rgb),-1)
            cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            # indexes_all = torch.cat((index_rgb,index_ir),-1)
            cid_all=torch.cat((cid_rgb,cid_ir),-1)
            f_out_all=torch.cat((f_out_rgb,f_out_ir),0)
            labels_all = torch.cat((labels_rgb,labels_ir),-1)
#####################################
            loss_all = self.memory_rgb(f_out_all, labels_all) 
            # loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            lamda_i = 0
            lamda_c=0.1
            # start=30
            loss_camera_ir=torch.tensor([0.]).cuda()
            loss_camera_rgb=torch.tensor([0.]).cuda()
            loss_camera_all = torch.tensor([0.]).cuda()
            # if epoch >= self.camstart:
            loss_camera_all = self.camera_loss(f_out_all,cid_all,labels_all,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb,cross_m=True)#self.camera_loss(f_out_ir,cid_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)


            loss_ins_ir = self.wise_memory_all(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            loss_ins_rgb= self.wise_memory_all(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#



            loss = loss_all+lamda_i*(loss_ins_ir+loss_ins_rgb)+lamda_c*loss_camera_all#+loss_match #+ loss_tri+loss_rgb_ir_trans+loss_ir_rgb_trans +(loss_rgb_trans+loss_ir_trans)

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
                      'Loss all {:.3f}\t'
                      'Loss all {:.3f}\t'
                      'camera all {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_all,loss_all,loss_camera_all.item(),loss_camera_rgb.item()))

                # print('loss_match',loss_match.item())


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
        return concate_intra_class,percam_tempV,memory_class_mapper
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



class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)

            # forward
            f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)

            # loss_tri, batch_acc = criterion_tri(f_out, labels)
            loss = self.memory(f_out, labels)# + loss_tri

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
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

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

class ClusterContrastTrainer_pretrain(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
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
            inputs_ir, labels_ir, indexes_ir = self._parse_data(inputs_ir)
            inputs_rgb, labels_rgb, indexes_rgb = self._parse_data(inputs_rgb)
            # forward
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            # f_out_rgb = self._forward(inputs_rgb)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)

            # loss_tri_rgb, batch_acc = self.tri(f_out_rgb, labels_rgb)
            # loss_tri_ir, batch_acc = self.tri(f_out_ir, labels_ir)
            # loss_tri = loss_tri_rgb+loss_tri_ir
            loss_ir = self.memory_ir(f_out_ir, labels_ir)# + loss_tri
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss = loss_ir+loss_rgb#+loss_tri
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

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

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
            inputs_ir,labels_ir, indexes_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,pool_rgb,pool_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)

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


class ClusterContrastTrainer_pretrain_camera(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        ##########init camera proxy
        concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1

            # inputs_ir,inputs_ir1,labels_ir, indexes_ir,cids_ir = self._parse_data_rgb(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)
            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)

            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)
            # print('before labels_rgb',labels_rgb)
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            # print('after labels_rgb',labels_rgb)
            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
################camera
            # if epoch >= 40:
            loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
            loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)
            # loss_tri_rgb, batch_acc = self.tri(f_out_rgb, labels_rgb,normalize_feature=True)
            # loss_tri_ir, batch_acc = self.tri(f_out_ir, labels_ir,normalize_feature=True)
            # loss_tri = loss_tri_rgb+loss_tri_ir
##################
            lamda_c = 0.1
            ratio_ir = 1#loss_camera_ir.item()/(loss_camera_ir.item()+loss_camera_rgb.item())
            ratio_rgb = 1#loss_camera_rgb.item()/(loss_camera_ir.item()+loss_camera_rgb.item())

            loss = loss_ir+loss_rgb+lamda_c*(ratio_ir*loss_camera_ir+ratio_rgb*loss_camera_rgb) #+ loss_tri
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
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

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
        return concate_intra_class,percam_tempV,memory_class_mapper
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



class PairwiseMatchingLoss(Module):
    def __init__(self, matcher):
        """
        Inputs:
            matcher: a class for matching pairs of images
        """
        super(PairwiseMatchingLoss, self).__init__()
        self.matcher = matcher

    def reset_running_stats(self):
        self.matcher.reset_running_stats()

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, feature, target):
        self._check_input_dim(feature)
        self.matcher.make_kernel(feature)

        score = self.matcher(feature)  # [b, b]

        target1 = target.unsqueeze(1)
        mask = (target1 == target1.t())
        pair_labels = mask.float()
        loss = F.binary_cross_entropy_with_logits(score, pair_labels, reduction='none')
        loss = loss.sum(-1)

        with torch.no_grad():
            min_pos = torch.min(score * pair_labels + 
                    (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[0]
            max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]
            acc = (min_pos > max_neg).float()

        return loss, acc


def pairwise_distance_matcher_train(matcher, prob_fea, gal_fea, gal_batch_size=4, prob_batch_size=128,label=None):
    with torch.no_grad():
        num_gals = gal_fea.size(0)
        num_probs = prob_fea.size(0)
        score_p = torch.zeros(num_probs, num_gals, device=prob_fea.device)
        score_n = torch.zeros(num_probs, num_gals, device=prob_fea.device)
        return_label=torch.zeros(num_probs, device=prob_fea.device)
        # matcher.eval()
        for i in range(0, num_probs, prob_batch_size):
            j = min(i + prob_batch_size, num_probs)
            # matcher.make_kernel(prob_fea[i: j,  :].cuda())
            # matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            for k in range(0, num_gals, gal_batch_size):
                k2 = min(k + gal_batch_size, num_gals)
                score_p[i: j, k: k2],score_n[i: j, k: k2],return_label[i:j] = matcher(prob_fea[i: j,  :].cuda(),gal_fea[k: k2, :].cuda(),label=label[i:j])
                # score_p[i: j, k: k2],return_label[i:j] = matcher(prob_fea[i: j,  :].cuda(),gal_fea[k: k2, :].cuda(),label=label[i:j])
        # scale matching scores to make them visually more recognizable
        # score = torch.sigmoid(score / 10)#F.softmax(torch.sigmoid(score / 10),dim=1) 
    return score_p,score_n,return_label#.cpu()  # [p, g]

class ClusterContrastTrainer_pretrain_camera_noclustermatch(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_noclustermatch, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir

        self.cmlabel=0
        # self.match_loss = PairwiseMatchingLoss(self.encoder.matcher)
    # def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
    #     all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
    #              print_freq=10, train_iters=400):
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()
        # self.matcher_rgb.train()
        # self.matcher_ir.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        ##########init camera proxy
        # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        # concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        if epoch>=self.cmlabel:
            concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        else:
            concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()


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

            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)


            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            lamda_c = 0.1

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
            loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)

            # target = torch.cat((labels_rgb,labels_ir),-1).unsqueeze(1)
            # mask_all = (target == target.t())
            # pair_labels= mask_all.float()


            # target_rgb = labels_rgb.unsqueeze(1)
            # mask_rgb = (target_rgb == target_rgb.t())
            # pair_labels_rgb= mask_rgb.float()
            # print('score_rgb, pair_labels_rgb',score_rgb.size(), pair_labels_rgb.size())
            # loss_match_rgb = F.binary_cross_entropy_with_logits(score_rgb, pair_labels_rgb, reduction='mean')

            # target_ir = labels_ir.unsqueeze(1)
            # mask_ir = (target_ir == target_ir.t())
            # pair_labels_ir= mask_ir.float()
            # loss_match_ir = F.binary_cross_entropy_with_logits(score_ir, pair_labels_ir, reduction='mean')
            # loss_match = loss_match_rgb+loss_match_ir
            # loss_query_ir= loss_query_ir.mean()
#############loss matcher
            # #########query-query 

            # for l_num in range(len(self.matcher_rgb.decoder.layers)):
            #     self.matcher_rgb.decoder.layers[l_num].qkv = self.encoder.module.base.blocks[l_num-3].attn.qkv
                # self.matcher_ir.decoder.layers[l_num].qkv = self.encoder.module.base.blocks[l_num-3].attn.qkv
            # self.matcher_rgb.make_kernel(f_out_rgb)            
            # rerank_dist_cm,labels_rgb_match = self.matcher_rgb(f_out_rgb)
            # print('labels_rgb',labels_rgb)
            # score_query_rgb_p,score_query_rgb_n,labels_rgb_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_rgb, f_out_rgb, gal_batch_size=64, prob_batch_size=32,label=labels_rgb)
            # print('labels_rgb_match',labels_rgb_match)
            # score_query_ir_p,score_query_ir_n,labels_ir_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_ir, f_out_ir, gal_batch_size=64, prob_batch_size=32,label=labels_ir)
            # score_query_rgb,labels_rgb_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_rgb, f_out_rgb, gal_batch_size=4, prob_batch_size=32,label=labels_rgb)
            # print('labels_rgb_match',labels_rgb_match)
            # score_query_ir,labels_ir_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_ir, f_out_ir, gal_batch_size=4, prob_batch_size=32,label=labels_ir)

            # self.matcher_rgb.make_kernel(f_out_rgb) #matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            # score_query_rgb_p,score_query_rgb_n,labels_rgb_match= self.encoder.module.matcher(f_out_rgb,f_out_rgb,label = labels_rgb)#.detach()
            # # self.matcher_ir.make_kernel(f_out_ir)            
            # score_query_ir_p,score_query_ir_n,labels_ir_match= self.encoder.module.matcher(f_out_ir,f_out_ir,label = labels_ir)
            # self.encoder.module.matcher_ir.make_kernel(f_out_ir)      

            # target_ir = labels_ir_match.unsqueeze(1)
            # mask_query_ir = (target_ir == target_ir.t())
            # pair_labels_query_ir = mask_query_ir.float() 
            # # loss_query_ir = F.binary_cross_entropy_with_logits(score_query_ir_p, pair_labels_query_ir, reduction='mean')+ F.binary_cross_entropy_with_logits(score_query_ir_n, pair_labels_query_ir, reduction='mean')
            # loss_query_ir = F.binary_cross_entropy_with_logits(score_query_ir_n, pair_labels_query_ir, reduction='mean')

            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_query_rgb = (target_rgb == target_rgb.t())
            # pair_labels_query_rgb = mask_query_rgb.float()
            # # loss_query_rgb = F.binary_cross_entropy_with_logits(score_query_rgb_p, pair_labels_query_rgb, reduction='mean')+F.binary_cross_entropy_with_logits(score_query_rgb_p, pair_labels_query_rgb, reduction='mean')
            # loss_query_rgb = F.binary_cross_entropy_with_logits(score_query_rgb_n, pair_labels_query_rgb, reduction='mean')
            # # loss_query_rgb= loss_query_rgb.mean()
            # loss_match = loss_query_ir+loss_query_rgb
##############loss v2
            # target_ir = labels_ir_match.unsqueeze(1)
            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_neighbor_rgb_p=(target_rgb == target_rgb.t())
            # num_neighbor_rgb_p = mask_neighbor_rgb_p.sum(dim=1)
            # score_intra_rgb_p =   F.softmax(score_query_rgb_p,dim=1)
            # score_intra_rgb_p = score_intra_rgb_p.clamp_min(1e-8)
            # loss_query_rgb_p = -score_intra_rgb_p.log().mul(mask_neighbor_rgb_p).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_rgb_p = loss_query_rgb_p.div(num_neighbor_rgb_p).mean()#.mul(mask_neighbor_intra_soft) ##

            # mask_neighbor_rgb_n=(target_rgb != target_rgb.t())
            # num_neighbor_rgb_n = mask_neighbor_rgb_n.sum(dim=1)
            # score_intra_rgb_n =   F.softmax(1-score_query_rgb_n,dim=1)
            # score_intra_rgb_n = score_intra_rgb_n.clamp_min(1e-8)
            # loss_query_rgb_n = -score_intra_rgb_n.log().mul(mask_neighbor_rgb_n).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_rgb_n = loss_query_rgb_n.div(num_neighbor_rgb_n).mean()#.mul(mask_neighbor_intra_soft) ##
            # loss_query_rgb= loss_query_rgb_p+loss_query_rgb_n



            # mask_neighbor_ir_p=(target_ir == target_ir.t())
            # num_neighbor_ir_p = mask_neighbor_ir_p.sum(dim=1)
            # score_intra_ir_p =   F.softmax(score_query_ir_p,dim=1)
            # score_intra_ir_p = score_intra_ir_p.clamp_min(1e-8)
            # loss_query_ir_p = -score_intra_ir_p.log().mul(mask_neighbor_ir_p).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_ir_p = loss_query_ir_p.div(num_neighbor_ir_p).mean()#.mul(mask_neighbor_intra_soft) ##

            # mask_neighbor_ir_n=(target_ir != target_ir.t())
            # num_neighbor_ir_n = mask_neighbor_ir_n.sum(dim=1)
            # score_intra_ir_n =   F.softmax(1-score_query_ir_n,dim=1)
            # score_intra_ir_n = score_intra_ir_n.clamp_min(1e-8)
            # loss_query_ir_n = -score_intra_ir_n.log().mul(mask_neighbor_ir_n).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_ir_n = loss_query_ir_n.div(num_neighbor_ir_n).mean()#.mul(mask_neighbor_intra_soft) ##
            # loss_query_ir= loss_query_ir_p+loss_query_ir_n

            # loss_match = loss_query_ir+loss_query_rgb

###############loss v3
            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_query_rgb = (target_rgb == target_rgb.t())
            # pair_labels_query_rgb = mask_query_rgb.float()
            # loss_query_rgb_p = F.binary_cross_entropy_with_logits(score_query_rgb_p, pair_labels_query_rgb, reduction='mean')
            # # loss_query_rgb_n = F.binary_cross_entropy_with_logits(1-score_query_rgb_n, 1-pair_labels_query_rgb, reduction='mean')
            # loss_query_rgb= loss_query_rgb_p#+loss_query_rgb_n

            # target_ir = labels_ir_match.unsqueeze(1)
            # mask_query_ir = (target_ir == target_ir.t())
            # pair_labels_query_ir = mask_query_ir.float()
            # loss_query_ir_p = F.binary_cross_entropy_with_logits(score_query_ir_p, pair_labels_query_ir, reduction='mean')
            # # loss_query_ir_n = F.binary_cross_entropy_with_logits(1-score_query_ir_n, 1-pair_labels_query_ir, reduction='mean')
            # loss_query_ir= loss_query_ir_p#+loss_query_ir_n
            loss_match = torch.tensor([0.]).cuda() #loss_query_ir+loss_query_rgb

            #######CM
            loss_match_cm = torch.tensor([0.]).cuda() 
            # score_query_rgb_ir,labels_rgb_match = self.encoder.module.matcher(f_out_rgb,self.memory_ir.features,label = labels_rgb)
            if epoch>=10000:#self.cmlabel:
                intersect_count_list=[]
                if epoch %2 ==0:
                    ins_sim_rgb_ir = F.normalize(f_out_rgb, dim=-1).mm(self.wise_memory_ir.features.detach().t())
                    Score_TOPK = 20#20#10
                    topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
                    # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                    cluster_label_rgb_ir = self.wise_memory_ir.labels[cluster_indices_rgb_ir].detach().cpu()
                    for l in range(Score_TOPK):
                        intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                        intersect_count_list.append(intersect_count)
                    intersect_count_list = torch.cat(intersect_count_list,1)
                    intersect_count, _ = intersect_count_list.max(1)
                    topk,cluster_label_index = torch.topk(intersect_count_list,1)
                    # print('ins_label_rgb_ir',ins_label_rgb_ir)
                    # print('cluster_label_rgb_ir',cluster_label_rgb_ir)
                    # print('cluster_label_index',cluster_label_index.view(-1))
                    cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir, dim=1, index=cluster_label_index.view(-1,1)).view(-1)  # cluster_label_rgb_ir[cluster_label_index.reshape(-1,1)]
                    score_query_rgb_ir,labels_rgb_cm = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_rgb, self.wise_memory_ir.features.detach(), gal_batch_size=4, prob_batch_size=32,label=labels_rgb)
                    # print('score_query_rgb_ir',score_query_rgb_ir.size())
                    # target_rgb_ir = labels_rgb_cm#.unsqueeze(1)
                    # print('target_rgb_ir',target_rgb_ir.size(),target_rgb_ir)
                    # label_m_ir=torch.arange(0,self.memory_ir.features.size(0)).cuda()#.unsqueeze(1)
                    label_m_ir=self.wise_memory_ir.labels#.unsqueeze(1)
                    # print(label_m_ir.size())
                    # mask_query_rgb_ir = torch.zero_like(score_query_rgb_ir)#(target_rgb_ir == label_m_ir)
                    # mask_query_rgb_ir[target_rgb_ir]=1.0
                    label_concate=torch.cat((labels_rgb_cm,label_m_ir),dim=-1).view(-1).unsqueeze(1)
                    label_concate_mask = (label_concate == label_concate.t()).float()[:score_query_rgb_ir.size(0),score_query_rgb_ir.size(0):] 
                    # print('label_concate_mask',label_concate_mask.size(),label_concate_mask)
                    pair_labels_query_rgb_ir = label_concate_mask.float()
                    loss_match_cm = F.binary_cross_entropy_with_logits(score_query_rgb_ir, pair_labels_query_rgb_ir, reduction='mean')
                    loss_match_cm =loss_match_cm#+ self.memory_ir(f_out_rgb, cluster_label_rgb_ir.cuda()) 
                else:
                    ins_sim_ir_rgb = F.normalize(f_out_ir, dim=-1).mm(self.wise_memory_rgb.features.detach().t())
                    Score_TOPK = 20#20#10
                    topk, cluster_indices_ir_rgb = torch.topk(ins_sim_ir_rgb, int(Score_TOPK))#20
                    # cluster_label_ir_rgb = cluster_indices_ir_rgb.detach().cpu()#.numpy()#.view(-1)
                    cluster_label_ir_rgb = self.wise_memory_rgb.labels[cluster_indices_ir_rgb].detach().cpu()
                    for l in range(Score_TOPK):
                        intersect_count=(cluster_label_ir_rgb == cluster_label_ir_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                        intersect_count_list.append(intersect_count)
                    intersect_count_list = torch.cat(intersect_count_list,1)
                    intersect_count, _ = intersect_count_list.max(1)
                    topk,cluster_label_index = torch.topk(intersect_count_list,1)
                    # print('ins_label_rgb_ir',ins_label_rgb_ir)
                    # print('cluster_label_rgb_ir',cluster_label_rgb_ir)
                    # print('cluster_label_index',cluster_label_index.view(-1))
                    cluster_label_ir_rgb = torch.gather(cluster_label_ir_rgb, dim=1, index=cluster_label_index.view(-1,1)).view(-1).cuda()  # cluster_label_rgb_ir[cluster_label_index.reshape(-1,1)]
                    score_query_ir_rgb,labels_ir_cm = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_ir, self.memory_rgb.features.detach(), gal_batch_size=4, prob_batch_size=32,label=labels_rgb)
                    # target_ir_rgb = labels_ir_cm.unsqueeze(1)
                    # label_m_rgb=torch.arange(0,self.memory_rgb.features.size(0)).cuda()#.unsqueeze(1)
                    label_m_rgb=self.wise_memory_rgb.labels#.unsqueeze(1)
                    # mask_query_ir_rgb = (target_ir_rgb == label_m_rgb.t())
                    # print(mask_query_ir_rgb.size())

                    label_concate=torch.cat((cluster_label_ir_rgb,label_m_rgb),dim=-1).view(-1).unsqueeze(1)
                    label_concate_mask = (label_concate == label_concate.t()).float()[:score_query_ir_rgb.size(0),score_query_ir_rgb.size(0):] 

                    pair_labels_query_ir_rgb = label_concate_mask.float()
                    loss_match_cm = F.binary_cross_entropy_with_logits(score_query_ir_rgb, pair_labels_query_ir_rgb, reduction='mean')
                    loss_match_cm =loss_match_cm#+ self.memory_rgb(f_out_ir, cluster_label_ir_rgb.cuda()) 
############v1
                # intersect_count_list=[]
                # if epoch %2 ==0:
                #     ins_sim_rgb_ir = F.normalize(f_out_rgb, dim=-1).mm(self.memory_ir.features.detach().t())
                #     Score_TOPK = 10#20#10
                #     topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
                #     cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                #     for l in range(Score_TOPK):
                #         intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                #         intersect_count_list.append(intersect_count)
                #     intersect_count_list = torch.cat(intersect_count_list,1)
                #     intersect_count, _ = intersect_count_list.max(1)
                #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
                #     # print('ins_label_rgb_ir',ins_label_rgb_ir)
                #     # print('cluster_label_rgb_ir',cluster_label_rgb_ir)
                #     # print('cluster_label_index',cluster_label_index.view(-1))
                #     cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir, dim=1, index=cluster_label_index.view(-1,1)).view(-1)  # cluster_label_rgb_ir[cluster_label_index.reshape(-1,1)]
                #     score_query_rgb_ir,labels_rgb_cm = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_rgb, self.memory_ir.features.detach(), gal_batch_size=4, prob_batch_size=32,label=labels_rgb)
                #     # print('score_query_rgb_ir',score_query_rgb_ir.size())
                #     # target_rgb_ir = labels_rgb_cm#.unsqueeze(1)
                #     # print('target_rgb_ir',target_rgb_ir.size(),target_rgb_ir)
                #     label_m_ir=torch.arange(0,self.memory_ir.features.size(0)).cuda()#.unsqueeze(1)
                #     # print(label_m_ir.size())
                #     # mask_query_rgb_ir = torch.zero_like(score_query_rgb_ir)#(target_rgb_ir == label_m_ir)
                #     # mask_query_rgb_ir[target_rgb_ir]=1.0
                #     label_concate=torch.cat((labels_rgb_cm,label_m_ir),dim=-1).view(-1).unsqueeze(1)
                #     label_concate_mask = (label_concate == label_concate.t()).float()[:score_query_rgb_ir.size(0),score_query_rgb_ir.size(0):] 
                #     # print('label_concate_mask',label_concate_mask.size(),label_concate_mask)
                #     pair_labels_query_rgb_ir = label_concate_mask.float()
                #     loss_match_cm = F.binary_cross_entropy_with_logits(score_query_rgb_ir, pair_labels_query_rgb_ir, reduction='mean')
                #     loss_match_cm =loss_match_cm#+ self.memory_ir(f_out_rgb, cluster_label_rgb_ir.cuda()) 
                # else:
                #     ins_sim_ir_rgb = F.normalize(f_out_ir, dim=-1).mm(self.memory_rgb.features.detach().t())
                #     Score_TOPK = 10#20#10
                #     topk, cluster_indices_ir_rgb = torch.topk(ins_sim_ir_rgb, int(Score_TOPK))#20
                #     cluster_label_ir_rgb = cluster_indices_ir_rgb.detach().cpu()#.numpy()#.view(-1)
                #     for l in range(Score_TOPK):
                #         intersect_count=(cluster_label_ir_rgb == cluster_label_ir_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                #         intersect_count_list.append(intersect_count)
                #     intersect_count_list = torch.cat(intersect_count_list,1)
                #     intersect_count, _ = intersect_count_list.max(1)
                #     topk,cluster_label_index = torch.topk(intersect_count_list,1)
                #     # print('ins_label_rgb_ir',ins_label_rgb_ir)
                #     # print('cluster_label_rgb_ir',cluster_label_rgb_ir)
                #     # print('cluster_label_index',cluster_label_index.view(-1))
                #     cluster_label_ir_rgb = torch.gather(cluster_label_ir_rgb, dim=1, index=cluster_label_index.view(-1,1)).view(-1).cuda()  # cluster_label_rgb_ir[cluster_label_index.reshape(-1,1)]
                #     score_query_ir_rgb,labels_ir_cm = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_ir, self.memory_rgb.features.detach(), gal_batch_size=4, prob_batch_size=32,label=labels_rgb)
                #     # target_ir_rgb = labels_ir_cm.unsqueeze(1)
                #     label_m_rgb=torch.arange(0,self.memory_rgb.features.size(0)).cuda()#.unsqueeze(1)
                #     # mask_query_ir_rgb = (target_ir_rgb == label_m_rgb.t())
                #     # print(mask_query_ir_rgb.size())

                #     label_concate=torch.cat((cluster_label_ir_rgb,label_m_rgb),dim=-1).view(-1).unsqueeze(1)
                #     label_concate_mask = (label_concate == label_concate.t()).float()[:score_query_ir_rgb.size(0),score_query_ir_rgb.size(0):] 

                #     pair_labels_query_ir_rgb = label_concate_mask.float()
                #     loss_match_cm = F.binary_cross_entropy_with_logits(score_query_ir_rgb, pair_labels_query_ir_rgb, reduction='mean')
                #     loss_match_cm =loss_match_cm#+ self.memory_rgb(f_out_ir, cluster_label_ir_rgb.cuda()) 

##################

            loss = loss_match+loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+loss_match_cm #+ loss_tri
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
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))
                print('loss_match,loss_query_cm',loss_match.item(),loss_match_cm.item())
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
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV,memory_class_mapper
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


class ClusterContrastTrainer_pretrain_camera_cmrefine(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_cmrefine, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir

        self.cmlabel=0
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

        if epoch>=self.cmlabel:
            concate_intra_class_ir,percam_tempV_ir,percam_memory_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        else:
            concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()


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

            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)


            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            lamda_c = 0.1

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
            thresh=0.8
            lamda_i = 0
################cmrefine
####################V1
#             if epoch>=0:#self.cmlabel:
#                 if epoch %2 ==0:
#                     sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
#                     sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
#                     nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
#                     # nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                     #######part sim
#                     z = 768
#                     q, k = f_out_rgb.size(0),self.wise_memory_ir.features.size(0)
#                     query_t = F.normalize(f_out_rgb.view(q, -1, z).detach(), dim=-1)  #B N C tgt.view(q, -1, z)#
#                     key_m = self.wise_memory_ir.features.view(k, -1, z).detach()#Q N C memory.view(k, -1, d)#
#                     score = einsum('q t d, k s d -> q k s t', query_t, key_m) # B Q N N
#                     score_p = score.max(dim=3)[0].view(q,k,-1)
# ##########################

#                     part_sim_0 = score_p[:,:,0].view(q,k)
#                     nearest_part_sim_0= part_sim_0.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_0, nearest_part_sim_0 * 0.8))

#                     part_sim_1 = score_p[:,:,1].view(q,k)
#                     nearest_part_sim_1= part_sim_1.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_1, nearest_part_sim_1 * 0.8))

#                     part_sim_2 = score_p[:,:,2].view(q,k)
#                     nearest_part_sim_2= part_sim_2.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_2, nearest_part_sim_2 * 0.8))

#                     part_sim_3 = score_p[:,:,3].view(q,k)
#                     nearest_part_sim_3= part_sim_3.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_3, nearest_part_sim_3 * 0.8))

#                     part_sim_4 = score_p[:,:,4].view(q,k)
#                     nearest_part_sim_4= part_sim_4.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_4, nearest_part_sim_4 * 0.8))

#                     ########part sim
#                     # mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * 0.8)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                     num_neighbor_rgb_ir = mask_neighbor_rgb_ir.sum(dim=1)+1
#                     score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#                     # print('num_neighbor_rgb_ir',num_neighbor_rgb_ir)
#                     # print('score_intra',score_intra)
#                     score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
#                     # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
#                     rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#                     loss_cm = rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##
#                 else:
#                 #     ##################ir-rgb
#                     # sim_prob_ir_rgb = F.normalize(sim_prob_B_ir_rgb, dim=1).mm(F.normalize(sim_prob_all_ir_rgb.t(),dim=1))#B N
#                     sim_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
#                     sim_ir_rgb_exp =sim_ir_rgb /0.05  # 64*13638
#                     nearest_ir_rgb = sim_ir_rgb.max(dim=1, keepdim=True)[0]
#                     # nearest_prob_ir_rgb = sim_prob_ir_rgb.max(dim=1, keepdim=True)[0]
#                     # mask_neighbor_prob_ir_rgb = torch.gt(sim_prob_ir_rgb, nearest_prob_ir_rgb * 0.8)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                     mask_neighbor_ir_rgb = torch.gt(sim_ir_rgb, nearest_ir_rgb * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps
#                     #######part sim
#                     z = 768
#                     q, k = f_out_ir.size(0),self.wise_memory_rgb.features.size(0)
#                     query_t = F.normalize(f_out_ir.view(q, -1, z).detach(), dim=-1)  #B N C tgt.view(q, -1, z)#
#                     key_m = self.wise_memory_rgb.features.view(k, -1, z).detach()#Q N C memory.view(k, -1, d)#
#                     score = einsum('q t d, k s d -> q k s t', query_t, key_m) # B Q N N
#                     # print(score.size())
#                     # score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1).mean(-1).view(q, k)
#                     # score_n = torch.cat((score.min(dim=2)[0], score.min(dim=3)[0]), dim=-1).mean(-1).view(q, k)
#                     score_p = score.max(dim=3)[0].view(q,k,-1)
# ###########################
#                     part_sim_0 = score_p[:,:,0].view(q,k)
#                     nearest_part_sim_0= part_sim_0.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_0, nearest_part_sim_0 * 0.8))

#                     part_sim_1 = score_p[:,:,1].view(q,k)
#                     nearest_part_sim_1= part_sim_1.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_1, nearest_part_sim_1 * 0.8))

#                     part_sim_2 = score_p[:,:,2].view(q,k)
#                     nearest_part_sim_2= part_sim_2.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_2, nearest_part_sim_2 * 0.8))

#                     part_sim_3 = score_p[:,:,3].view(q,k)
#                     nearest_part_sim_3= part_sim_3.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_3, nearest_part_sim_3 * 0.8))

#                     part_sim_4 = score_p[:,:,4].view(q,k)
#                     nearest_part_sim_4= part_sim_4.max(dim=1, keepdim=True)[0]
#                     mask_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(torch.gt(part_sim_4, nearest_part_sim_4 * 0.8))
#                     ########part sim
#                     num_neighbor_ir_rgb = mask_neighbor_ir_rgb.sum(dim=1)+1#.mul(sim_wise).
#                     # score_intra_ir_rgb =   F.softmax(sim_ir_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
#                     score_intra_ir_rgb =   F.softmax(sim_ir_rgb_exp,dim=1)
#                     # print('num_neighbor_ir_rgb',num_neighbor_ir_rgb)
#                     score_intra_ir_rgb = score_intra_ir_rgb.clamp_min(1e-8)
#                     # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
#                     ir_rgb_loss = -score_intra_ir_rgb.log().mul(mask_neighbor_ir_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
#                     loss_cm = ir_rgb_loss.div(num_neighbor_ir_rgb).mean()#.mul(mask_neighbor_intra_soft) ##

################V2
            # if epoch>=5:#self.cmlabel:
            #     if epoch %2 ==0:
            #         ############rgb-ir
            #         # sim_prob_all_rgb_ir = torch.cat((F.softmax(self.wise_memory_ir.features.detach().data.mm(F.normalize(self.memory_rgb.features.detach().data, dim=1).t())/0.01,dim=1),\
            #         # F.softmax(self.wise_memory_ir.features.detach().mm(self.memory_ir.features.detach().t())/0.01,dim=1)),dim=1).detach().data  #N C/0.05
            #         # sim_prob_B_rgb_ir = torch.cat((F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.memory_rgb.features.detach().t())/0.01,dim=1),\
            #         # F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.memory_ir.features.detach().t())/0.01,dim=1)),dim=1).detach().data

            #         #######full domain sim
            #         sim_prob_all_rgb_ir_1 = torch.cat([F.softmax(self.wise_memory_ir.features.detach().mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
            #         sim_prob_all_rgb_ir_2 = torch.cat([F.softmax(self.wise_memory_ir.features.detach().mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05
            #         sim_prob_all_rgb_ir = torch.cat((sim_prob_all_rgb_ir_1,sim_prob_all_rgb_ir_2),dim=1)
            #         sim_prob_B_rgb_ir_1 = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()
            #         sim_prob_B_rgb_ir_2 = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()
            #         sim_prob_B_rgb_ir = torch.cat((sim_prob_B_rgb_ir_1,sim_prob_B_rgb_ir_2),dim=1)

            #         sim_prob_rgb_ir = F.normalize(sim_prob_B_rgb_ir, dim=1).mm(F.normalize(sim_prob_all_rgb_ir.t(),dim=1))#B N
            #         sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
            #         sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
            #         nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
            #         nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
            #         mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps
            #         mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * 0.8)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
            #         num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_prob_rgb_ir).sum(dim=1)+1
            #         score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            #         # print('score_intra',score_intra)
            #         score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
            #         # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
            #         rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            #         rgb_ir_loss = rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##

            #     else:
            #     #     ##################ir-rgb
            #         # sim_prob_all_ir_rgb = torch.cat((F.softmax(self.wise_memory_rgb.features.mm(F.normalize(self.memory_rgb.features.detach().data, dim=1).t())/0.01,dim=1),\
            #         # F.softmax(self.wise_memory_rgb.features.mm(self.memory_ir.features.detach().data.t())/0.01,dim=1)),dim=1).detach().data  #N C/0.05
            #         # sim_prob_B_ir_rgb = torch.cat((F.softmax(F.normalize(f_out_ir, dim=1).mm(self.memory_rgb.features.detach().data.t())/0.01,dim=1),\
            #         # F.softmax(F.normalize(f_out_ir, dim=1).mm(self.memory_ir.features.detach().data.t())/0.01,dim=1)),dim=1).detach().data
                    

            #         sim_prob_all_ir_rgb_1 = torch.cat([F.softmax(self.wise_memory_rgb.features.detach().mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
            #         sim_prob_all_ir_rgb_2 = torch.cat([F.softmax(self.wise_memory_rgb.features.detach().mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05
            #         sim_prob_all_ir_rgb = torch.cat((sim_prob_all_ir_rgb_1,sim_prob_all_ir_rgb_2),dim=1)
            #         sim_prob_B_ir_rgb_1 = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()
            #         sim_prob_B_ir_rgb_2 = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()
            #         sim_prob_B_ir_rgb = torch.cat((sim_prob_B_ir_rgb_1,sim_prob_B_ir_rgb_2),dim=1)


            #         sim_prob_ir_rgb = F.normalize(sim_prob_B_ir_rgb, dim=1).mm(F.normalize(sim_prob_all_ir_rgb.t(),dim=1))#B N
            #         sim_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
            #         sim_ir_rgb_exp =sim_ir_rgb /0.05  # 64*13638
            #         nearest_ir_rgb = sim_ir_rgb.max(dim=1, keepdim=True)[0]
            #         nearest_prob_ir_rgb = sim_prob_ir_rgb.max(dim=1, keepdim=True)[0]
            #         mask_neighbor_prob_ir_rgb = torch.gt(sim_prob_ir_rgb, nearest_prob_ir_rgb * 0.8)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
            #         mask_neighbor_ir_rgb = torch.gt(sim_ir_rgb, nearest_ir_rgb * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps
            #         num_neighbor_ir_rgb = mask_neighbor_ir_rgb.mul(mask_neighbor_prob_ir_rgb).sum(dim=1)+1#.mul(sim_wise).
            #         score_intra_ir_rgb =   F.softmax(sim_ir_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            #         # print('score_intra',score_intra)
            #         score_intra_ir_rgb = score_intra_ir_rgb.clamp_min(1e-8)
            #         # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
            #         ir_rgb_loss = -score_intra_ir_rgb.log().mul(mask_neighbor_ir_rgb).mul(mask_neighbor_prob_ir_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            #         ir_rgb_loss = ir_rgb_loss.div(num_neighbor_ir_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
            
            # sim_prob_all_rgb_rgb = torch.cat([F.softmax(self.wise_memory_rgb.features.detach().mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data.cpu()  #N C/0.05
            # sim_prob_B_rgb_rgb = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data.cpu()
            


            # sim_prob_rgb_rgb = F.normalize(sim_prob_B_rgb_rgb, dim=1).mm(F.normalize(sim_prob_all_rgb_rgb.t(),dim=1))#B N
            # sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
            # sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
            # nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
            # nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
            # mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * 0.8).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
            # mask_neighbor_prob_rgb_rgb = torch.gt(sim_prob_rgb_rgb, nearest_prob_rgb_rgb * 0.8).cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
            # num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)+1
            # # print('num_neighbor_rgb_rgb',num_neighbor_rgb_rgb)
            # score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # # print('score_intra',score_intra)
            # score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
            # # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
            # rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
            # # #################ir-ir
            # # sim_prob_all_ir_ir = F.softmax((self.wise_memory_ir.features.detach().mm(self.memory_ir.features.detach().data.t()))/0.01,dim=1).detach().data #N C/0.05
            # # sim_prob_B_ir_ir = F.softmax((F.normalize(f_out_ir, dim=1).mm(self.memory_ir.features.detach().data.t()))/0.01,dim=1).detach().data

            # sim_prob_all_ir_ir = torch.cat([F.softmax(self.wise_memory_ir.features.detach().mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data.cpu()  #N C/0.05
            # sim_prob_B_ir_ir = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data.cpu()
            

            # sim_prob_ir_ir = F.normalize(sim_prob_B_ir_ir, dim=1).mm(F.normalize(sim_prob_all_ir_ir.t(),dim=1))#B N
            # sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
            # sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
            # nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
            # nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
            # mask_neighbor_prob_ir_ir = torch.gt(sim_prob_ir_ir, nearest_prob_ir_ir * 0.8).cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
            # mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * 0.8).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
            # num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_prob_ir_ir).sum(dim=1)+1#.mul(sim_wise).
            # # print('num_neighbor_ir_ir',num_neighbor_ir_ir)
            # score_intra_ir_ir =   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # # print('score_intra',score_intra)
            # score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
            # # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
            # ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##

#################part hthm
            temper=1
            if epoch>=0:#self.cmlabel:
                if epoch %2 ==0:
                ############rgb-ir
                # sim_prob_all_rgb_ir_1 = torch.cat([F.softmax(self.part_sim(self.wise_memory_ir.features.cuda(0).detach(),F.normalize(self.wise_memory_rgb.cam_mem[i].cuda(0).detach().data, dim=1)),dim=1)/0.01 for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                # sim_prob_all_rgb_ir_2 = torch.cat([F.softmax(self.part_sim(self.wise_memory_ir.features.cuda(0).detach(),F.normalize(self.wise_memory_ir.cam_mem[i].cuda(0).detach().data, dim=1)),dim=1)/0.01 for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                # sim_prob_all_rgb_ir = torch.cat((sim_prob_all_rgb_ir_1,sim_prob_all_rgb_ir_2),dim=1)
                # sim_prob_B_rgb_ir_1 = torch.cat([F.softmax(self.part_sim(F.normalize(f_out_rgb.cuda(0).detach(), dim=1),F.normalize(self.wise_memory_rgb.cam_mem[i].cuda(0).detach().data, dim=1)),dim=1)/0.01 for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05    
                # sim_prob_B_rgb_ir_2 = torch.cat([F.softmax(self.part_sim(F.normalize(f_out_rgb.cuda(0).detach(), dim=1),F.normalize(self.wise_memory_ir.cam_mem[i].cuda(0).detach().data, dim=1)),dim=1)/0.01 for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05  
                #######full domain sim
                    if epoch>=self.cmlabel:
                        sim_prob_all_rgb_ir = torch.cat([self.part_sim(self.wise_memory_ir.features.detach(),percam_memory_rgb[i].detach().data) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_B_rgb_ir = torch.cat([self.part_sim(f_out_rgb.detach(),percam_memory_rgb[i].detach().data,) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05
                    else:
                        sim_prob_all_rgb_ir_1 = torch.cat([self.part_sim(self.wise_memory_ir.features.detach(),percam_memory_rgb[i].detach().data) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_all_rgb_ir_2 = torch.cat([self.part_sim(self.wise_memory_ir.features.detach(),percam_memory_ir[i].detach().data) for i in range(len(percam_memory_ir))],dim=1).detach().data#.cpu()  #N C/0.05  
                        sim_prob_all_rgb_ir = torch.cat((sim_prob_all_rgb_ir_1,sim_prob_all_rgb_ir_2),dim=1).cuda()
                        sim_prob_B_rgb_ir_1 = torch.cat([self.part_sim(f_out_rgb.detach(),percam_memory_rgb[i].detach().data,) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05    
                        sim_prob_B_rgb_ir_2 = torch.cat([self.part_sim(f_out_rgb.detach(),percam_memory_ir[i].detach().data,) for i in range(len(percam_memory_ir))],dim=1).detach().data#.cpu()  #N C/0.05    
                        sim_prob_B_rgb_ir = torch.cat((sim_prob_B_rgb_ir_1,sim_prob_B_rgb_ir_2),dim=1)
                    sim_prob_rgb_ir = F.normalize(sim_prob_B_rgb_ir, dim=1).mm(F.normalize(sim_prob_all_rgb_ir.t(),dim=1))#B N
                    sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
                    sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
                    nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
                    nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
                    mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * thresh)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_prob_rgb_ir).sum(dim=1)+1
                    # print('num_neighbor_rgb_ir',num_neighbor_rgb_ir)
                    score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                    # print('score_intra',score_intra)
                    score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                    # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                    rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                    rgb_ir_loss = rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##
                else:
                #     ##################ir-rgb
                    # sim_prob_all_ir_rgb = torch.cat((F.softmax(self.wise_memory_rgb.features.mm(F.normalize(self.memory_rgb.features.detach().data, dim=1).t())/0.01,dim=1),\
                    # F.softmax(self.wise_memory_rgb.features.mm(self.memory_ir.features.detach().data.t())/0.01,dim=1)),dim=1).detach().data  #N C/0.05
                    # sim_prob_B_ir_rgb = torch.cat((F.softmax(F.normalize(f_out_ir, dim=1).mm(self.memory_rgb.features.detach().data.t())/0.01,dim=1),\
                    # F.softmax(F.normalize(f_out_ir, dim=1).mm(self.memory_ir.features.detach().data.t())/0.01,dim=1)),dim=1).detach().data
                    # sim_prob_all_ir_rgb_1 = torch.cat([F.softmax(self.part_sim(self.wise_memory_rgb.features.detach(),F.normalize(percam_memory_rgb[i].detach().data, dim=1)),dim=1)/temper for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05
                    if epoch>=self.cmlabel:
                        sim_prob_all_ir_rgb = torch.cat([self.part_sim(self.wise_memory_rgb.features.cuda(1).detach(),percam_memory_rgb[i].cuda(1).detach().data) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_B_ir_rgb = torch.cat([self.part_sim(f_out_ir.cuda(1).detach(),percam_memory_rgb[i].cuda(1).detach().data) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05  
                    else:
                        sim_prob_all_ir_rgb_1 = torch.cat([self.part_sim(self.wise_memory_rgb.features.cuda(1).detach(),percam_memory_rgb[i].cuda(1).detach().data) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_all_ir_rgb_2 = torch.cat([self.part_sim(self.wise_memory_rgb.features.cuda(1).detach(),percam_memory_ir[i].cuda(1).detach().data) for i in range(len(percam_memory_ir))],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_all_ir_rgb = torch.cat((sim_prob_all_ir_rgb_1,sim_prob_all_ir_rgb_2),dim=1)#.cuda(1)
                        sim_prob_B_ir_rgb_1 = torch.cat([self.part_sim(f_out_ir.cuda(1).detach(),percam_memory_rgb[i].cuda(1).detach().data) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05    
                        sim_prob_B_ir_rgb_2 = torch.cat([self.part_sim(f_out_ir.cuda(1).detach(),percam_memory_ir[i].cuda(1).detach().data) for i in range(len(percam_memory_ir))],dim=1).detach().data#.cpu()  #N C/0.05    
                        sim_prob_B_ir_rgb = torch.cat((sim_prob_B_ir_rgb_1,sim_prob_B_ir_rgb_2),dim=1)
                        
                    sim_prob_ir_rgb = F.normalize(sim_prob_B_ir_rgb, dim=1).mm(F.normalize(sim_prob_all_ir_rgb.t(),dim=1))#B N
                    sim_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
                    sim_ir_rgb_exp =sim_ir_rgb /0.05  # 64*13638
                    nearest_ir_rgb = sim_ir_rgb.max(dim=1, keepdim=True)[0]
                    nearest_prob_ir_rgb = sim_prob_ir_rgb.max(dim=1, keepdim=True)[0]
                    mask_neighbor_prob_ir_rgb = torch.gt(sim_prob_ir_rgb, nearest_prob_ir_rgb * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_ir_rgb = torch.gt(sim_ir_rgb, nearest_ir_rgb * thresh)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_ir_rgb = mask_neighbor_ir_rgb.mul(mask_neighbor_prob_ir_rgb).sum(dim=1)+1#.mul(sim_wise).
                    score_intra_ir_rgb =   F.softmax(sim_ir_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                    # print('score_intra',score_intra)
                    score_intra_ir_rgb = score_intra_ir_rgb.clamp_min(1e-8)
                    # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
                    ir_rgb_loss = -score_intra_ir_rgb.log().mul(mask_neighbor_ir_rgb).mul(mask_neighbor_prob_ir_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                    ir_rgb_loss = ir_rgb_loss.div(num_neighbor_ir_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
            if epoch>=0:
                # if epoch %2 ==0:
                sim_prob_all_rgb_rgb = torch.cat([self.part_sim(self.wise_memory_rgb.features.cuda(1).detach(),percam_memory_rgb[i].cuda(1).detach().data) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()
                sim_prob_B_rgb_rgb = torch.cat([self.part_sim(f_out_rgb.cuda(1).detach(),percam_memory_rgb[i].cuda(1).detach().data) for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()
                # print('sim_prob_all_rgb_rgb',sim_prob_all_rgb_rgb.size())
                sim_prob_rgb_rgb = F.normalize(sim_prob_B_rgb_rgb, dim=1).mm(F.normalize(sim_prob_all_rgb_rgb.t(),dim=1))#B N
                sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
                sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
                nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
                nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
                mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
                mask_neighbor_prob_rgb_rgb = torch.gt(sim_prob_rgb_rgb, nearest_prob_rgb_rgb * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)+1
                # print('num_neighbor_rgb_rgb',num_neighbor_rgb_rgb)
                score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                # print('score_intra',score_intra)
                score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
                # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
    #################ir-ir
                del sim_prob_all_rgb_rgb,sim_prob_B_rgb_rgb,sim_prob_rgb_rgb
            # else:
                sim_prob_all_ir_ir = torch.cat([self.part_sim(self.wise_memory_ir.features.cuda(1).detach(),percam_memory_ir[i].cuda(1).detach().data) for i in range(len(percam_memory_ir))],dim=1).detach().data
                sim_prob_B_ir_ir = torch.cat([self.part_sim(f_out_ir.cuda(1).detach(),percam_memory_ir[i].cuda(1).detach().data) for i in range(len(percam_memory_ir))],dim=1).detach().data
                sim_prob_ir_ir = F.normalize(sim_prob_B_ir_ir, dim=1).mm(F.normalize(sim_prob_all_ir_ir.t(),dim=1))#B N
                sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
                sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
                nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
                nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
                mask_neighbor_prob_ir_ir = torch.gt(sim_prob_ir_ir, nearest_prob_ir_ir * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
                num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_prob_ir_ir).sum(dim=1)+1#.mul(sim_wise).
                # print('num_neighbor_ir_ir',num_neighbor_ir_ir)
                score_intra_ir_ir =   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                # print('score_intra',score_intra)
                score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
                # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
                ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##
                del sim_prob_all_ir_ir,sim_prob_B_ir_ir,sim_prob_ir_ir
            lamda_i = 1
####################
            # lamda_i = 1
            loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(ir_ir_loss+rgb_rgb_loss)+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)


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
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))
                print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
                print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
                print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
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








class ClusterContrastTrainer_pretrain_camera_cmrefine_agw(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_cmrefine_agw, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir

        self.cmlabel=0
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

        if epoch>=self.cmlabel:
            concate_intra_class_ir,percam_tempV_ir,_ = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,_  = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        else:
            concate_intra_class_ir,percam_tempV_ir,_  = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,_  = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()


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

            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            # cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)


            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            lamda_c = 0.1

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
            thresh=0.8
            lamda_i = 0
################cmrefine

#################part hthm
            temper=1
            if epoch>=1000:#self.cmlabel:
                if epoch %2 ==0:
                ############rgb-ir
                # sim_prob_all_rgb_ir_1 = torch.cat([F.softmax(self.part_sim(self.wise_memory_ir.features.cuda(0).detach(),F.normalize(self.wise_memory_rgb.cam_mem[i].cuda(0).detach().data, dim=1)),dim=1)/0.01 for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                # sim_prob_all_rgb_ir_2 = torch.cat([F.softmax(self.part_sim(self.wise_memory_ir.features.cuda(0).detach(),F.normalize(self.wise_memory_ir.cam_mem[i].cuda(0).detach().data, dim=1)),dim=1)/0.01 for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                # sim_prob_all_rgb_ir = torch.cat((sim_prob_all_rgb_ir_1,sim_prob_all_rgb_ir_2),dim=1)
                # sim_prob_B_rgb_ir_1 = torch.cat([F.softmax(self.part_sim(F.normalize(f_out_rgb.cuda(0).detach(), dim=1),F.normalize(self.wise_memory_rgb.cam_mem[i].cuda(0).detach().data, dim=1)),dim=1)/0.01 for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05    
                # sim_prob_B_rgb_ir_2 = torch.cat([F.softmax(self.part_sim(F.normalize(f_out_rgb.cuda(0).detach(), dim=1),F.normalize(self.wise_memory_ir.cam_mem[i].cuda(0).detach().data, dim=1)),dim=1)/0.01 for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05  
                #######full domain sim
                    if epoch>=self.cmlabel:
                        sim_prob_all_rgb_ir = torch.cat([self.part_sim(self.wise_memory_ir.features.detach(),self.wise_memory_rgb.cam_mem[i].detach().data) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_B_rgb_ir = torch.cat([self.part_sim(f_out_rgb.detach(),self.wise_memory_rgb.cam_mem[i].detach().data,) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                    else:
                        sim_prob_all_rgb_ir_1 = torch.cat([self.part_sim(self.wise_memory_ir.features.detach(),self.wise_memory_rgb.cam_mem[i].detach().data) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_all_rgb_ir_2 = torch.cat([self.part_sim(self.wise_memory_ir.features.detach(),self.wise_memory_ir.cam_mem[i].detach().data) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05  
                        sim_prob_all_rgb_ir = torch.cat((sim_prob_all_rgb_ir_1,sim_prob_all_rgb_ir_2),dim=1).cuda()
                        sim_prob_B_rgb_ir_1 = torch.cat([self.part_sim(f_out_rgb.detach(),self.wise_memory_rgb.cam_mem[i].detach().data,) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05    
                        sim_prob_B_rgb_ir_2 = torch.cat([self.part_sim(f_out_rgb.detach(),self.wise_memory_ir.cam_mem[i].detach().data,) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05    
                        sim_prob_B_rgb_ir = torch.cat((sim_prob_B_rgb_ir_1,sim_prob_B_rgb_ir_2),dim=1)
                    sim_prob_rgb_ir = F.normalize(sim_prob_B_rgb_ir, dim=1).mm(F.normalize(sim_prob_all_rgb_ir.t(),dim=1))#B N
                    sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
                    sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
                    nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
                    nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
                    mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * thresh)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_prob_rgb_ir).sum(dim=1)+1
                    # print('num_neighbor_rgb_ir',num_neighbor_rgb_ir)
                    score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                    # print('score_intra',score_intra)
                    score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                    # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                    rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                    rgb_ir_loss = rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##
                else:
                #     ##################ir-rgb
                    # sim_prob_all_ir_rgb = torch.cat((F.softmax(self.wise_memory_rgb.features.mm(F.normalize(self.memory_rgb.features.detach().data, dim=1).t())/0.01,dim=1),\
                    # F.softmax(self.wise_memory_rgb.features.mm(self.memory_ir.features.detach().data.t())/0.01,dim=1)),dim=1).detach().data  #N C/0.05
                    # sim_prob_B_ir_rgb = torch.cat((F.softmax(F.normalize(f_out_ir, dim=1).mm(self.memory_rgb.features.detach().data.t())/0.01,dim=1),\
                    # F.softmax(F.normalize(f_out_ir, dim=1).mm(self.memory_ir.features.detach().data.t())/0.01,dim=1)),dim=1).detach().data
                    # sim_prob_all_ir_rgb_1 = torch.cat([F.softmax(self.part_sim(self.wise_memory_rgb.features.detach(),F.normalize(percam_memory_rgb[i].detach().data, dim=1)),dim=1)/temper for i in range(len(percam_memory_rgb))],dim=1).detach().data#.cpu()  #N C/0.05
                    if epoch>=self.cmlabel:
                        sim_prob_all_ir_rgb = torch.cat([self.part_sim(self.wise_memory_rgb.features.cuda(1).detach(),self.wise_memory_rgb.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_B_ir_rgb = torch.cat([self.part_sim(f_out_ir.cuda(1).detach(),self.wise_memory_rgb.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05  
                    else:
                        sim_prob_all_ir_rgb_1 = torch.cat([self.part_sim(self.wise_memory_rgb.features.cuda(1).detach(),self.wise_memory_rgb.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_all_ir_rgb_2 = torch.cat([self.part_sim(self.wise_memory_rgb.features.cuda(1).detach(),self.wise_memory_ir.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                        sim_prob_all_ir_rgb = torch.cat((sim_prob_all_ir_rgb_1,sim_prob_all_ir_rgb_2),dim=1)#.cuda(1)
                        sim_prob_B_ir_rgb_1 = torch.cat([self.part_sim(f_out_ir.cuda(1).detach(),self.wise_memory_rgb.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05    
                        sim_prob_B_ir_rgb_2 = torch.cat([self.part_sim(f_out_ir.cuda(1).detach(),self.wise_memory_ir.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05    
                        sim_prob_B_ir_rgb = torch.cat((sim_prob_B_ir_rgb_1,sim_prob_B_ir_rgb_2),dim=1)
                        
                    sim_prob_ir_rgb = F.normalize(sim_prob_B_ir_rgb, dim=1).mm(F.normalize(sim_prob_all_ir_rgb.t(),dim=1))#B N
                    sim_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
                    sim_ir_rgb_exp =sim_ir_rgb /0.05  # 64*13638
                    nearest_ir_rgb = sim_ir_rgb.max(dim=1, keepdim=True)[0]
                    nearest_prob_ir_rgb = sim_prob_ir_rgb.max(dim=1, keepdim=True)[0]
                    mask_neighbor_prob_ir_rgb = torch.gt(sim_prob_ir_rgb, nearest_prob_ir_rgb * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_ir_rgb = torch.gt(sim_ir_rgb, nearest_ir_rgb * thresh)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_ir_rgb = mask_neighbor_ir_rgb.mul(mask_neighbor_prob_ir_rgb).sum(dim=1)+1#.mul(sim_wise).
                    score_intra_ir_rgb =   F.softmax(sim_ir_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                    # print('score_intra',score_intra)
                    score_intra_ir_rgb = score_intra_ir_rgb.clamp_min(1e-8)
                    # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
                    ir_rgb_loss = -score_intra_ir_rgb.log().mul(mask_neighbor_ir_rgb).mul(mask_neighbor_prob_ir_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                    ir_rgb_loss = ir_rgb_loss.div(num_neighbor_ir_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
            if epoch>=1000:
                # if epoch %2 ==0:
                sim_prob_all_rgb_rgb = torch.cat([self.part_sim(self.wise_memory_rgb.features.cuda(1).detach(),self.wise_memory_rgb.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()
                sim_prob_B_rgb_rgb = torch.cat([self.part_sim(f_out_rgb.cuda(1).detach(),self.wise_memory_rgb.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()
                # print('sim_prob_all_rgb_rgb',sim_prob_all_rgb_rgb.size())
                sim_prob_rgb_rgb = F.normalize(sim_prob_B_rgb_rgb, dim=1).mm(F.normalize(sim_prob_all_rgb_rgb.t(),dim=1))#B N
                sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
                sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
                nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
                nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
                mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
                mask_neighbor_prob_rgb_rgb = torch.gt(sim_prob_rgb_rgb, nearest_prob_rgb_rgb * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)+1
                # print('num_neighbor_rgb_rgb',num_neighbor_rgb_rgb)
                score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                # print('score_intra',score_intra)
                score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
                # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
    #################ir-ir
                del sim_prob_all_rgb_rgb,sim_prob_B_rgb_rgb,sim_prob_rgb_rgb
            # else:
                sim_prob_all_ir_ir = torch.cat([self.part_sim(self.wise_memory_ir.features.cuda(1).detach(),self.wise_memory_ir.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_ir.allcam],dim=1).detach().data
                sim_prob_B_ir_ir = torch.cat([self.part_sim(f_out_ir.cuda(1).detach(),self.wise_memory_ir.cam_mem[i].cuda(1).detach().data) for i in self.wise_memory_ir.allcam],dim=1).detach().data
                sim_prob_ir_ir = F.normalize(sim_prob_B_ir_ir, dim=1).mm(F.normalize(sim_prob_all_ir_ir.t(),dim=1))#B N
                sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
                sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
                nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
                nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
                mask_neighbor_prob_ir_ir = torch.gt(sim_prob_ir_ir, nearest_prob_ir_ir * thresh).cuda(0)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * thresh).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
                num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_prob_ir_ir).sum(dim=1)+1#.mul(sim_wise).
                # print('num_neighbor_ir_ir',num_neighbor_ir_ir)
                score_intra_ir_ir =   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                # print('score_intra',score_intra)
                score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
                # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
                ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##
                del sim_prob_all_ir_ir,sim_prob_B_ir_ir,sim_prob_ir_ir
            lamda_i = 1
####################
            # lamda_i = 1
            # loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            # loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)+(ir_ir_loss+rgb_rgb_loss)+(rgb_ir_loss+ir_rgb_loss)+ lamda_i*(loss_ins_ir+loss_ins_rgb)


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
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))
                print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
                print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
                print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
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





class ClusterContrastTrainer_pretrain_camera_wise_3_nomatch(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_wise_3_nomatch, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_all =  memory
        self.nameMap_all = []
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
        # self.criterion_pa = PredictionAlignmentLoss(lambda_vr=0.5, lambda_rv=0.5)
        self.camstart=0

    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        ##########init camera proxy
        # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)
        start_cam=0

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            indexes_ir = torch.tensor([self.nameMap_all[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.nameMap_all[name] for name in name_rgb]).cuda()
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)
            cids_rgb = torch.cat((cids_rgb,cids_rgb),-1)
            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)
            # for path,cameraid in  zip(name_ir,cids_ir):
            #     print(path,cameraid)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)
            # indexes_all = torch.cat((index_rgb,index_ir),-1)
            cid_all=torch.cat((cid_rgb,cid_ir),-1)
            f_out_all=torch.cat((f_out_rgb,f_out_ir),0)
            labels_all = torch.cat((labels_rgb,labels_ir),-1)
#####################################
            loss_all = self.memory_rgb(f_out_all, labels_all) 
            # loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            lamda_i = 0
            lamda_c=0.1
            # start=30
            loss_camera_ir=torch.tensor([0.]).cuda()
            loss_camera_rgb=torch.tensor([0.]).cuda()
            loss_camera_all = torch.tensor([0.]).cuda()
            # if epoch >= self.camstart:
            loss_camera_all = self.camera_loss(f_out_all,cid_all,labels_all,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb,cross_m=True)#self.camera_loss(f_out_ir,cid_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
                # loss_camera_rgb = self.camera_loss(f_out_rgb,cid_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)


            # mids_rgb = torch.zeros_like(cids_rgb)
            # mids_ir = torch.ones_like(cids_ir)
            # f_out_all = torch.cat((f_out_rgb,f_out_ir),0)
            # mid_all = torch.cat((mids_rgb,mids_ir),-1)
            # all_label = torch.as_tensor(all_label).cuda()
            # labels_all = torch.cat((all_label[index_rgb],all_label[index_ir+int(len(all_label_rgb))]),-1)
            # if (epoch %2 ==0) and (epoch>=self.crosscamstart):

            #     loss_camera_rgb_all = self.camera_loss(f_out_all,mid_all,labels_all,percam_tempV_all,concate_intra_class_all,memory_class_mapper_all,cross_m=False)
                



            # print('labels_ir',labels_ir)
            # print('tran_label_ir',tran_label_ir)
            # loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            # loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_ins_ir = self.wise_memory_all(f_out_ir,index_ir,cid_ir)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            loss_ins_rgb= self.wise_memory_all(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#


            # # print('labels_rgb',labels_rgb)
            # score_query_rgb,labels_rgb_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_all, f_out_all, gal_batch_size=4, prob_batch_size=32,label=labels_all)
            # # print('labels_rgb_match',labels_rgb_match)
            # # score_query_ir,labels_ir_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_ir, f_out_ir, gal_batch_size=4, prob_batch_size=32,label=labels_ir)

            # # self.matcher_rgb.make_kernel(f_out_rgb) #matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            # # score_query_rgb,labels_rgb_match = self.encoder.module.matcher(f_out_rgb,f_out_rgb,label = labels_rgb)#.detach()
            # # # self.matcher_ir.make_kernel(f_out_ir)            
            # # score_query_ir,labels_ir_match = self.encoder.module.matcher(f_out_ir,f_out_ir,label = labels_ir)
            # # # self.encoder.module.matcher_ir.make_kernel(f_out_ir)      

            # # target_ir = labels_ir_match.unsqueeze(1)
            # # mask_query_ir = (target_ir == target_ir.t())
            # # pair_labels_query_ir = mask_query_ir.float() 
            # # loss_query_ir = F.binary_cross_entropy_with_logits(score_query_ir, pair_labels_query_ir, reduction='mean')
            # # # loss_query_ir = F.binary_cross_entropy_with_logits(score_query_ir, pair_labels_query_ir, reduction='mean')
            # # # loss_query_ir= loss_query_ir.mean()

            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_query_rgb = (target_rgb == target_rgb.t())
            # pair_labels_query_rgb = mask_query_rgb.float()
            # loss_query_rgb = F.binary_cross_entropy_with_logits(score_query_rgb, pair_labels_query_rgb, reduction='mean')
            # # loss_query_rgb= loss_query_rgb.mean()
            # loss_match = loss_query_rgb
            loss_match=torch.tensor([0.]).cuda()

            # score_query_rgb_p,score_query_rgb_n,labels_rgb_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_all, f_out_all, gal_batch_size=4, prob_batch_size=32,label=labels_all)
            # score_query_rgb_p,score_query_rgb_n,labels_rgb_match= self.encoder.module.matcher(f_out_all,f_out_all,label = labels_all)
            # # target_ir = labels_ir_match.unsqueeze(1)
            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_neighbor_rgb_p=(target_rgb == target_rgb.t())
            # num_neighbor_rgb_p = mask_neighbor_rgb_p.sum(dim=1)
            # score_intra_rgb_p =   F.softmax(score_query_rgb_p,dim=1)
            # score_intra_rgb_p = score_intra_rgb_p.clamp_min(1e-8)
            # loss_query_rgb_p = -score_intra_rgb_p.log().mul(mask_neighbor_rgb_p).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_rgb_p = loss_query_rgb_p.div(num_neighbor_rgb_p).mean()#.mul(mask_neighbor_intra_soft) ##

            # mask_neighbor_rgb_n=(target_rgb != target_rgb.t())
            # num_neighbor_rgb_n = mask_neighbor_rgb_n.sum(dim=1)
            # score_intra_rgb_n =   F.softmax(1-score_query_rgb_n,dim=1)
            # score_intra_rgb_n = score_intra_rgb_n.clamp_min(1e-8)
            # loss_query_rgb_n = -score_intra_rgb_n.log().mul(mask_neighbor_rgb_n).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_rgb_n = loss_query_rgb_n.div(num_neighbor_rgb_n).mean()#.mul(mask_neighbor_intra_soft) ##
            # loss_match= loss_query_rgb_p+loss_query_rgb_n
            ########loss v3
            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_query_rgb = (target_rgb == target_rgb.t())
            # pair_labels_query_rgb = mask_query_rgb.float()
            # loss_query_rgb_p = F.binary_cross_entropy_with_logits(score_query_rgb_p, pair_labels_query_rgb, reduction='mean')
            # # loss_query_rgb_n = F.binary_cross_entropy_with_logits(1-score_query_rgb_n, 1-pair_labels_query_rgb, reduction='mean')
            # loss_match= loss_query_rgb_p#+loss_query_rgb_n


################camera
# ##################ins sample
            # if epoch >=0 and epoch<300:
            #     TOPK=1
            #     sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_all.features.t())
            #     # selfjd =  compute_jaccard_distance_cm(F.normalize(f_out_ir, dim=1).detach(),F.normalize(f_out_ir, dim=1).detach(), k1=30, k2=6,search_option=3)
            #     # print('selfjd',selfjd)
            #     # cmjd =  compute_jaccard_distance_cm(F.normalize(f_out_ir, dim=1).detach(),self.wise_memory_ir.features.detach(), k1=30, k2=6,search_option=3)
            #     # print('cmjd',cmjd)
            #     topk_ir_rgb, indices_sim_ir_ir = torch.topk(sim_ir_ir, TOPK)#20
            #     # print('indices_sim_ir_ir',indices_sim_ir_ir)
            #     # topk_ir_rgb, indicesjd_sim_ir_ir = torch.topk(torch.from_numpy(cmjd), TOPK)#20
            #     # print('indicesjd_sim_ir_ir',indicesjd_sim_ir_ir)
            #     topk_feat_ir_ir=self.wise_memory_all.features[indices_sim_ir_ir]
            #     # mix_feat_ir = torch.cat((f_out_ir.view(-1,16,2048),topk_feat_ir_rgb.view(-1,16,2048)),dim=1)#.permute(1,0,2)#seqlenth  batch  dim #########
            #     # mix_feat_ir=self.encoder.module.encoder(F.normalize(mix_feat_ir, dim=-1),F.normalize(mix_feat_ir, dim=-1))
            #     mix_feat_ir=self.encoder.module.encoder(f_out_ir.view(-1,16,2048),topk_feat_ir_ir.view(-1,TOPK*16,2048))#F.normalize(f_out_ir.view(-1,16,2048), dim=-1)
            #     # for blk in self.encoder.module.encoder:
            #     #     mix_feat_ir = blk(mix_feat_ir)
            #     trans_feat_ir = mix_feat_ir#self.encoder.module.encoder(mix_feat_rgb)#.permute(1,0,2)#seqlenth  batch  dim #########
            #     # trans_feat_ir = self.encoder.module.encoder(mix_feat_ir)#.permute(1,0,2)
            #     # print(trans_feat.size())
            #     trans_ir =trans_feat_ir.contiguous().view(-1,2048)#[:,:16,:].contiguous().view(-1,2048)
            #     # trans_ir_rgb = trans_feat_ir[:,16:,:].contiguous().view(-1,2048)
            #     sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_all.features.t())
            #     topk_ir_rgb, indices_sim_rgb_rgb = torch.topk(sim_rgb_rgb, TOPK)#20
            #     topk_feat_rgb_rgb=self.wise_memory_all.features[indices_sim_rgb_rgb]
            #     # mix_feat_rgb = torch.cat((f_out_rgb.view(-1,16,2048),topk_feat_rgb_ir.view(-1,16,2048)),dim=1)#.permute(1,0,2)
            #     # mix_feat_rgb=self.encoder.module.encoder(F.normalize(mix_feat_rgb, dim=-1),F.normalize(mix_feat_rgb, dim=-1))
            #     mix_feat_rgb=self.encoder.module.encoder(f_out_rgb.view(-1,16,2048),topk_feat_rgb_rgb.view(-1,TOPK*16,2048))#F.normalize(f_out_rgb.view(-1,16,2048), dim=-1)
            #     # for blk in self.encoder.module.encoder:
            #     #     mix_feat_rgb = blk(mix_feat_rgb)
            #     trans_feat_rgb = mix_feat_rgb#self.encoder.module.encoder(mix_feat_rgb)#.permute(1,0,2)#seqlenth  batch  dim #########
            #     # print(trans_feat.size())
            #     trans_rgb =trans_feat_rgb.contiguous().view(-1,2048)#[:,:16,:].contiguous().view(-1,2048)
            #     # trans_rgb_ir =trans_feat_rgb[:,16:,:].contiguous().view(-1,2048)
            #     tem=1
            #     # loss_ir_trans = self.criterion_pa(f_out_ir,topk_feat_ir_ir.view(-1,2048))#+loss_ir_trans_self
            #     # loss_rgb_trans = self.criterion_pa(f_out_rgb,topk_feat_rgb_rgb.view(-1,2048))#+loss_rgb_trans_self

            #     loss_ir_trans = self.criterion_pa(f_out_ir,trans_ir)#+loss_ir_trans_self
            #     loss_rgb_trans = self.criterion_pa(f_out_rgb,trans_rgb)#+loss_rgb_trans_self
###########################CM
                # sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_all.features.t())
                # topk_rgb_ir, indices_sim_rgb_ir = torch.topk(sim_rgb_ir, TOPK)#20
                # topk_feat_rgb_ir=self.wise_memory_all.features[indices_sim_rgb_ir]
                # # mix_feat_rgb = torch.cat((f_out_rgb.view(-1,16,2048),topk_feat_rgb_ir.view(-1,16,2048)),dim=1)#.permute(1,0,2)
                # # mix_feat_rgb=self.encoder.module.encoder(F.normalize(mix_feat_rgb, dim=-1),F.normalize(mix_feat_rgb, dim=-1))
                # mix_feat_rgb_ir=self.encoder.module.encoder(f_out_rgb.view(-1,16,2048),topk_feat_rgb_ir.view(-1,TOPK*16,2048))#F.normalize(f_out_rgb.view(-1,16,2048), dim=-1)
                # # for blk in self.encoder.module.encoder:
                # #     mix_feat_rgb = blk(mix_feat_rgb)
                # trans_feat_rgb_ir = mix_feat_rgb_ir#self.encoder.module.encoder(mix_feat_rgb)#.permute(1,0,2)#seqlenth  batch  dim #########
                # # print(trans_feat.size())
                # trans_rgb_ir =trans_feat_rgb_ir.contiguous().view(-1,2048)#[:,:16,:].contiguous().view(-1,2048)
                # # trans_rgb_ir =trans_feat_rgb[:,16:,:].contiguous().view(-1,2048)

                # sim_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_all.features.t())
                # topk_ir_rgb, indices_sim_ir_rgb = torch.topk(sim_ir_rgb, TOPK)#20
                # topk_feat_ir_rgb=self.wise_memory_all.features[indices_sim_ir_rgb]
                # # mix_feat_rgb = torch.cat((f_out_rgb.view(-1,16,2048),topk_feat_rgb_ir.view(-1,16,2048)),dim=1)#.permute(1,0,2)
                # # mix_feat_rgb=self.encoder.module.encoder(F.normalize(mix_feat_rgb, dim=-1),F.normalize(mix_feat_rgb, dim=-1))
                # mix_feat_ir_rgb=self.encoder.module.encoder(f_out_ir.view(-1,16,2048),topk_feat_ir_rgb.view(-1,TOPK*16,2048))#F.normalize(f_out_rgb.view(-1,16,2048), dim=-1)
                # # for blk in self.encoder.module.encoder:
                # #     mix_feat_rgb = blk(mix_feat_rgb)
                # trans_feat_ir_rgb = mix_feat_ir_rgb#self.encoder.module.encoder(mix_feat_rgb)#.permute(1,0,2)#seqlenth  batch  dim #########
                # # print(trans_feat.size())
                # trans_ir_rgb =trans_feat_ir_rgb.contiguous().view(-1,2048)#[:,:16,:].contiguous().view(-1,2048)

                # tem=1
                # # loss_ir_trans = self.criterion_pa(f_out_ir,topk_feat_ir_ir.view(-1,2048))#+loss_ir_trans_self
                # # loss_rgb_trans = self.criterion_pa(f_out_rgb,topk_feat_rgb_rgb.view(-1,2048))#+loss_rgb_trans_self

                # loss_ir_rgb_trans = self.criterion_pa(f_out_ir,trans_ir_rgb)#+loss_ir_trans_self
                # loss_rgb_ir_trans = self.criterion_pa(f_out_rgb,trans_rgb_ir)#+loss_rgb_trans_self




##################
            
            # if epoch < self.camstart:
            #     loss = loss_ir+loss_rgb
            # # loss = loss_ir+loss_rgb+(loss_rgb_trans+loss_ir_trans)+lamda_cc*(loss_camera_ir+loss_camera_rgb) #+ loss_tri
            # else:
            #     loss = loss_ir+loss_rgb+lamda_cc*(loss_camera_ir+loss_camera_rgb) #+ loss_tri+loss_rgb_ir_trans+loss_ir_rgb_trans(loss_rgb_trans+loss_ir_trans)
##################
            # if epoch >= self.camstart:
            #     loss = loss_ir+loss_rgb+lamda_cc*(loss_camera_ir+loss_camera_rgb)
            # else:
            # # loss = loss_ir+loss_rgb+(loss_rgb_trans+loss_ir_trans)+lamda_cc*(loss_camera_ir+loss_camera_rgb) #+ loss_tri
            # loss = loss_ir+loss_rgb+lamda_i*(loss_camera_ir+loss_camera_rgb) #+ loss_tri+loss_rgb_ir_trans+loss_ir_rgb_trans +(loss_rgb_trans+loss_ir_trans)

            loss = loss_all+lamda_i*(loss_ins_ir+loss_ins_rgb)+lamda_c*loss_camera_all+loss_match #+ loss_tri+loss_rgb_ir_trans+loss_ir_rgb_trans +(loss_rgb_trans+loss_ir_trans)

            # loss = lamda_cc*(loss_ir+loss_rgb)+loss_camera_rgb+loss_camera_ir #+ loss_tri
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
                      'Loss all {:.3f}\t'
                      'Loss all {:.3f}\t'
                      'camera all {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_all,loss_all,loss_camera_all.item(),loss_camera_rgb.item()))
                # if epoch >= start_cam:
                # print('loss_ins_ir',loss_ins_ir.item())
                # print('loss_ins_rgb',loss_ins_rgb.item())
                print('loss_match',loss_match.item())
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
        return concate_intra_class,percam_tempV,memory_class_mapper
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





class ClusterContrastTrainer_pretrain_camera_clustermatch(object):
    def __init__(self, encoder, memory=None,matcher_rgb = None,matcher_ir = None):
        super(ClusterContrastTrainer_pretrain_camera_clustermatch, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.wise_memory_ir =  memory
        self.wise_memory_rgb =  memory
        self.nameMap_ir =[]
        self.nameMap_rgb = []
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
        self.matcher_rgb = matcher_rgb
        self.matcher_ir = matcher_ir

        self.cmlabel=0
        # self.match_loss = PairwiseMatchingLoss(self.encoder.matcher)
    # def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
    #     all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
    #              print_freq=10, train_iters=400):
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_all=None,intra_id_labels_ir=None, intra_id_features_ir=None,intra_id_features_all=None,
        all_label_rgb=None,all_label_ir=None,all_label=None,cams_ir=None,cams_rgb=None,cams_all=None,cross_cam=None,intra_id_features_crosscam=None,intra_id_labels_crosscam=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()
        # self.matcher_rgb.train()
        # self.matcher_ir.train()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        ##########init camera proxy
        # concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        # concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        if epoch>=self.cmlabel:
            concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label,intra_id_features_rgb)
        else:
            concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
            concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()


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

            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)


            # _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,score_query_rgb,score_query_ir,pair_labels_query_rgb,pair_labels_query_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)
            
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            lamda_c = 0.1

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
            loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
            loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)

            # target = torch.cat((labels_rgb,labels_ir),-1).unsqueeze(1)
            # mask_all = (target == target.t())
            # pair_labels= mask_all.float()


            # target_rgb = labels_rgb.unsqueeze(1)
            # mask_rgb = (target_rgb == target_rgb.t())
            # pair_labels_rgb= mask_rgb.float()
            # print('score_rgb, pair_labels_rgb',score_rgb.size(), pair_labels_rgb.size())
            # loss_match_rgb = F.binary_cross_entropy_with_logits(score_rgb, pair_labels_rgb, reduction='mean')

            # target_ir = labels_ir.unsqueeze(1)
            # mask_ir = (target_ir == target_ir.t())
            # pair_labels_ir= mask_ir.float()
            # loss_match_ir = F.binary_cross_entropy_with_logits(score_ir, pair_labels_ir, reduction='mean')
            # loss_match = loss_match_rgb+loss_match_ir
            # loss_query_ir= loss_query_ir.mean()
#############loss matcher
            # #########query-query 

            # for l_num in range(len(self.matcher_rgb.decoder.layers)):
            #     self.matcher_rgb.decoder.layers[l_num].qkv = self.encoder.module.base.blocks[l_num-3].attn.qkv
                # self.matcher_ir.decoder.layers[l_num].qkv = self.encoder.module.base.blocks[l_num-3].attn.qkv
            # self.matcher_rgb.make_kernel(f_out_rgb)            
            # rerank_dist_cm,labels_rgb_match = self.matcher_rgb(f_out_rgb)
            # print('labels_rgb',labels_rgb)
            # score_query_rgb_p,score_query_rgb_n,labels_rgb_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_rgb, f_out_rgb, gal_batch_size=64, prob_batch_size=32,label=labels_rgb)
            # print('labels_rgb_match',labels_rgb_match)
            # score_query_ir_p,score_query_ir_n,labels_ir_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_ir, f_out_ir, gal_batch_size=64, prob_batch_size=32,label=labels_ir)
            # score_query_rgb,labels_rgb_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_rgb, f_out_rgb, gal_batch_size=4, prob_batch_size=32,label=labels_rgb)
            # print('labels_rgb_match',labels_rgb_match)
            # score_query_ir,labels_ir_match = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_ir, f_out_ir, gal_batch_size=4, prob_batch_size=32,label=labels_ir)

            # self.matcher_rgb.make_kernel(f_out_rgb) #matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            score_query_rgb_p,score_query_rgb_n,labels_rgb_match= self.encoder.module.matcher(f_out_rgb,f_out_rgb,label = labels_rgb)#.detach()
            # # self.matcher_ir.make_kernel(f_out_ir)            
            score_query_ir_p,score_query_ir_n,labels_ir_match= self.encoder.module.matcher(f_out_ir,f_out_ir,label = labels_ir)
            # self.encoder.module.matcher_ir.make_kernel(f_out_ir)      

            # target_ir = labels_ir_match.unsqueeze(1)
            # mask_query_ir = (target_ir == target_ir.t())
            # pair_labels_query_ir = mask_query_ir.float() 
            # # loss_query_ir = F.binary_cross_entropy_with_logits(score_query_ir_p, pair_labels_query_ir, reduction='mean')+ F.binary_cross_entropy_with_logits(score_query_ir_n, pair_labels_query_ir, reduction='mean')
            # loss_query_ir = F.binary_cross_entropy_with_logits(score_query_ir_n, pair_labels_query_ir, reduction='mean')

            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_query_rgb = (target_rgb == target_rgb.t())
            # pair_labels_query_rgb = mask_query_rgb.float()
            # # loss_query_rgb = F.binary_cross_entropy_with_logits(score_query_rgb_p, pair_labels_query_rgb, reduction='mean')+F.binary_cross_entropy_with_logits(score_query_rgb_p, pair_labels_query_rgb, reduction='mean')
            # loss_query_rgb = F.binary_cross_entropy_with_logits(score_query_rgb_n, pair_labels_query_rgb, reduction='mean')
            # # loss_query_rgb= loss_query_rgb.mean()
            # loss_match = loss_query_ir+loss_query_rgb
##############loss v2
            # target_ir = labels_ir_match.unsqueeze(1)
            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_neighbor_rgb_p=(target_rgb == target_rgb.t())
            # num_neighbor_rgb_p = mask_neighbor_rgb_p.sum(dim=1)
            # score_intra_rgb_p =   F.softmax(score_query_rgb_p,dim=1)
            # score_intra_rgb_p = score_intra_rgb_p.clamp_min(1e-8)
            # loss_query_rgb_p = -score_intra_rgb_p.log().mul(mask_neighbor_rgb_p).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_rgb_p = loss_query_rgb_p.div(num_neighbor_rgb_p).mean()#.mul(mask_neighbor_intra_soft) ##

            # mask_neighbor_rgb_n=(target_rgb != target_rgb.t())
            # num_neighbor_rgb_n = mask_neighbor_rgb_n.sum(dim=1)
            # score_intra_rgb_n =   F.softmax(1-score_query_rgb_n,dim=1)
            # score_intra_rgb_n = score_intra_rgb_n.clamp_min(1e-8)
            # loss_query_rgb_n = -score_intra_rgb_n.log().mul(mask_neighbor_rgb_n).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_rgb_n = loss_query_rgb_n.div(num_neighbor_rgb_n).mean()#.mul(mask_neighbor_intra_soft) ##
            # loss_query_rgb= loss_query_rgb_p+loss_query_rgb_n



            # mask_neighbor_ir_p=(target_ir == target_ir.t())
            # num_neighbor_ir_p = mask_neighbor_ir_p.sum(dim=1)
            # score_intra_ir_p =   F.softmax(score_query_ir_p,dim=1)
            # score_intra_ir_p = score_intra_ir_p.clamp_min(1e-8)
            # loss_query_ir_p = -score_intra_ir_p.log().mul(mask_neighbor_ir_p).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_ir_p = loss_query_ir_p.div(num_neighbor_ir_p).mean()#.mul(mask_neighbor_intra_soft) ##

            # mask_neighbor_ir_n=(target_ir != target_ir.t())
            # num_neighbor_ir_n = mask_neighbor_ir_n.sum(dim=1)
            # score_intra_ir_n =   F.softmax(1-score_query_ir_n,dim=1)
            # score_intra_ir_n = score_intra_ir_n.clamp_min(1e-8)
            # loss_query_ir_n = -score_intra_ir_n.log().mul(mask_neighbor_ir_n).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # loss_query_ir_n = loss_query_ir_n.div(num_neighbor_ir_n).mean()#.mul(mask_neighbor_intra_soft) ##
            # loss_query_ir= loss_query_ir_p+loss_query_ir_n

            # loss_match = loss_query_ir+loss_query_rgb

###############loss v3
            # target_rgb = labels_rgb_match.unsqueeze(1)
            # mask_query_rgb = (target_rgb == target_rgb.t())
            # pair_labels_query_rgb = mask_query_rgb.float()
            # loss_query_rgb_p = F.binary_cross_entropy_with_logits(score_query_rgb_p, pair_labels_query_rgb, reduction='mean')
            # # loss_query_rgb_n = F.binary_cross_entropy_with_logits(1-score_query_rgb_n, 1-pair_labels_query_rgb, reduction='mean')
            # loss_query_rgb= loss_query_rgb_p#+loss_query_rgb_n

            # target_ir = labels_ir_match.unsqueeze(1)
            # mask_query_ir = (target_ir == target_ir.t())
            # pair_labels_query_ir = mask_query_ir.float()
            # loss_query_ir_p = F.binary_cross_entropy_with_logits(score_query_ir_p, pair_labels_query_ir, reduction='mean')
            # # loss_query_ir_n = F.binary_cross_entropy_with_logits(1-score_query_ir_n, 1-pair_labels_query_ir, reduction='mean')
            # loss_query_ir= loss_query_ir_p#+loss_query_ir_n
            # loss_match = loss_query_ir+loss_query_rgb#torch.tensor([0.]).cuda() #

            #######CM
            loss_match_cm = torch.tensor([0.]).cuda() 
            # score_query_rgb_ir,labels_rgb_match = self.encoder.module.matcher(f_out_rgb,self.memory_ir.features,label = labels_rgb)
            if epoch>=10000:#self.cmlabel:
                intersect_count_list=[]
                if epoch %2 ==0:
                    ins_sim_rgb_ir = F.normalize(f_out_rgb, dim=-1).mm(self.wise_memory_ir.features.detach().t())
                    Score_TOPK = 20#20#10
                    topk, cluster_indices_rgb_ir = torch.topk(ins_sim_rgb_ir, int(Score_TOPK))#20
                    # cluster_label_rgb_ir = cluster_indices_rgb_ir.detach().cpu()#.numpy()#.view(-1)
                    cluster_label_rgb_ir = self.wise_memory_ir.labels[cluster_indices_rgb_ir].detach().cpu()
                    for l in range(Score_TOPK):
                        intersect_count=(cluster_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                        intersect_count_list.append(intersect_count)
                    intersect_count_list = torch.cat(intersect_count_list,1)
                    intersect_count, _ = intersect_count_list.max(1)
                    topk,cluster_label_index = torch.topk(intersect_count_list,1)
                    # print('ins_label_rgb_ir',ins_label_rgb_ir)
                    # print('cluster_label_rgb_ir',cluster_label_rgb_ir)
                    # print('cluster_label_index',cluster_label_index.view(-1))
                    cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir, dim=1, index=cluster_label_index.view(-1,1)).view(-1)  # cluster_label_rgb_ir[cluster_label_index.reshape(-1,1)]
                    score_query_rgb_ir,labels_rgb_cm = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_rgb, self.wise_memory_ir.features.detach(), gal_batch_size=4, prob_batch_size=32,label=labels_rgb)
                    # print('score_query_rgb_ir',score_query_rgb_ir.size())
                    # target_rgb_ir = labels_rgb_cm#.unsqueeze(1)
                    # print('target_rgb_ir',target_rgb_ir.size(),target_rgb_ir)
                    # label_m_ir=torch.arange(0,self.memory_ir.features.size(0)).cuda()#.unsqueeze(1)
                    label_m_ir=self.wise_memory_ir.labels#.unsqueeze(1)
                    # print(label_m_ir.size())
                    # mask_query_rgb_ir = torch.zero_like(score_query_rgb_ir)#(target_rgb_ir == label_m_ir)
                    # mask_query_rgb_ir[target_rgb_ir]=1.0
                    label_concate=torch.cat((labels_rgb_cm,label_m_ir),dim=-1).view(-1).unsqueeze(1)
                    label_concate_mask = (label_concate == label_concate.t()).float()[:score_query_rgb_ir.size(0),score_query_rgb_ir.size(0):] 
                    # print('label_concate_mask',label_concate_mask.size(),label_concate_mask)
                    pair_labels_query_rgb_ir = label_concate_mask.float()
                    loss_match_cm = F.binary_cross_entropy_with_logits(score_query_rgb_ir, pair_labels_query_rgb_ir, reduction='mean')
                    loss_match_cm =loss_match_cm#+ self.memory_ir(f_out_rgb, cluster_label_rgb_ir.cuda()) 
                else:
                    ins_sim_ir_rgb = F.normalize(f_out_ir, dim=-1).mm(self.wise_memory_rgb.features.detach().t())
                    Score_TOPK = 20#20#10
                    topk, cluster_indices_ir_rgb = torch.topk(ins_sim_ir_rgb, int(Score_TOPK))#20
                    # cluster_label_ir_rgb = cluster_indices_ir_rgb.detach().cpu()#.numpy()#.view(-1)
                    cluster_label_ir_rgb = self.wise_memory_rgb.labels[cluster_indices_ir_rgb].detach().cpu()
                    for l in range(Score_TOPK):
                        intersect_count=(cluster_label_ir_rgb == cluster_label_ir_rgb[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                        intersect_count_list.append(intersect_count)
                    intersect_count_list = torch.cat(intersect_count_list,1)
                    intersect_count, _ = intersect_count_list.max(1)
                    topk,cluster_label_index = torch.topk(intersect_count_list,1)
                    # print('ins_label_rgb_ir',ins_label_rgb_ir)
                    # print('cluster_label_rgb_ir',cluster_label_rgb_ir)
                    # print('cluster_label_index',cluster_label_index.view(-1))
                    cluster_label_ir_rgb = torch.gather(cluster_label_ir_rgb, dim=1, index=cluster_label_index.view(-1,1)).view(-1).cuda()  # cluster_label_rgb_ir[cluster_label_index.reshape(-1,1)]
                    score_query_ir_rgb,labels_ir_cm = pairwise_distance_matcher_train(self.encoder.module.matcher, f_out_ir, self.memory_rgb.features.detach(), gal_batch_size=4, prob_batch_size=32,label=labels_rgb)
                    # target_ir_rgb = labels_ir_cm.unsqueeze(1)
                    # label_m_rgb=torch.arange(0,self.memory_rgb.features.size(0)).cuda()#.unsqueeze(1)
                    label_m_rgb=self.wise_memory_rgb.labels#.unsqueeze(1)
                    # mask_query_ir_rgb = (target_ir_rgb == label_m_rgb.t())
                    # print(mask_query_ir_rgb.size())

                    label_concate=torch.cat((cluster_label_ir_rgb,label_m_rgb),dim=-1).view(-1).unsqueeze(1)
                    label_concate_mask = (label_concate == label_concate.t()).float()[:score_query_ir_rgb.size(0),score_query_ir_rgb.size(0):] 

                    pair_labels_query_ir_rgb = label_concate_mask.float()
                    loss_match_cm = F.binary_cross_entropy_with_logits(score_query_ir_rgb, pair_labels_query_ir_rgb, reduction='mean')
                    loss_match_cm =loss_match_cm#+ self.memory_rgb(f_out_ir, cluster_label_ir_rgb.cuda()) 

            loss = loss_ir+loss_rgb+lamda_c*(loss_camera_ir+loss_camera_rgb)#
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
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))
                print('loss_match,loss_query_cm',loss_match.item(),loss_match_cm.item())
                print('loss_query_ir_p,',loss_query_ir_p.item())
                # print('loss_query_ir_n,',loss_query_ir_n.item())
                print('loss_query_rgb_p,',loss_query_rgb_p.item())
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
        concate_intra_class = torch.cat(concate_intra_class)

        percam_tempV = []
        for ii in unique_cams:
            percam_tempV.append(percam_memory[ii].detach().clone())
        percam_tempV = torch.cat(percam_tempV, dim=0).cuda()
        return concate_intra_class,percam_tempV,memory_class_mapper
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














class ClusterContrastTrainer_pretrain_camera_sie(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_sie, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        ##########init camera proxy
        concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1

            # inputs_ir,inputs_ir1,labels_ir, indexes_ir,cids_ir = self._parse_data_rgb(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)
            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)

            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)
            if epoch%2 ==0:
                reverse_control=True
            else:
                reverse_control=False
            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir = self._forward(inputs_rgb,inputs_ir,cids_rgb,cids_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,reverse=reverse_control)

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
################camera
            # if epoch >= 40:
            loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
            loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)
            # loss_tri_rgb, batch_acc = self.tri(f_out_rgb, labels_rgb,normalize_feature=True)
            # loss_tri_ir, batch_acc = self.tri(f_out_ir, labels_ir,normalize_feature=True)
            # loss_tri = loss_tri_rgb+loss_tri_ir
##################
            lamda_c = 0.1
            ratio_ir = 1#loss_camera_ir.item()/(loss_camera_ir.item()+loss_camera_rgb.item())
            ratio_rgb = 1#loss_camera_rgb.item()/(loss_camera_ir.item()+loss_camera_rgb.item())

            loss = loss_ir+loss_rgb+lamda_c*(ratio_ir*loss_camera_ir+ratio_rgb*loss_camera_rgb) #+ loss_tri
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
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _forward(self, x1, x2,cids_rgb,cids_ir, label_1=None,label_2=None,modal=0,reverse=False):
        return self.encoder(x1, x2,cids_rgb,cids_ir, modal=modal,label_1=label_1,label_2=label_2,reverse=reverse)

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
        return concate_intra_class,percam_tempV,memory_class_mapper
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

class ClusterContrastTrainer_pretrain_nocamera(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_nocamera, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        ##########init camera proxy
        concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir = self._parse_data_ir(inputs_ir) #inputs_ir1
            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)

            # inputs_ir = torch.cat((inputs_ir,inputs_ir1),0)
            # labels_ir = torch.cat((labels_ir,labels_ir),-1)

            _,f_out_rgb,f_out_ir,labels_rgb,labels_ir = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()
################camera
            # if epoch >= 40:
            # loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
            # loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)
            # loss_tri_rgb, batch_acc = self.tri(f_out_rgb, labels_rgb,normalize_feature=True)
            # loss_tri_ir, batch_acc = self.tri(f_out_ir, labels_ir,normalize_feature=True)
            # loss_tri = loss_tri_rgb+loss_tri_ir
##################
            lamda_c = 0.1
            ratio_ir = 1#loss_camera_ir.item()/(loss_camera_ir.item()+loss_camera_rgb.item())
            ratio_rgb = 1#loss_camera_rgb.item()/(loss_camera_ir.item()+loss_camera_rgb.item())

            loss = loss_ir+loss_rgb#+lamda_c*(ratio_ir*loss_camera_ir+ratio_rgb*loss_camera_rgb) #+ loss_tri
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
                      'camera ir {:.3f}\t'
                      'camera rgb {:.3f}\t'
                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb,loss_camera_ir.item(),loss_camera_rgb.item()))

    def _parse_data_rgb(self, inputs):
        imgs,imgs1, _, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda()

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2)

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
        return concate_intra_class,percam_tempV,memory_class_mapper
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



class ClusterContrastTrainer_pretrain_camera_token(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_pretrain_camera_token, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
        self.wise_memory_ir=memory
        self.wise_memory_rgb=memory

        self.nameMap_ir =[]
        self.nameMap_rgb = []
    def train(self, epoch, data_loader_ir,data_loader_rgb, optimizer,intra_id_labels_rgb=None, intra_id_features_rgb=None,intra_id_labels_ir=None, intra_id_features_ir=None,
        all_label_rgb=None,all_label_ir=None,cams_ir=None,cams_rgb=None,
                 print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()


        ##########init camera proxy
        concate_intra_class_ir,percam_tempV_ir,memory_class_mapper_ir = self.init_camera_proxy(cams_ir,all_label_ir,intra_id_features_ir)
        concate_intra_class_rgb,percam_tempV_rgb,memory_class_mapper_rgb = self.init_camera_proxy(cams_rgb,all_label_rgb,intra_id_features_rgb)


        end = time.time()
        for i in range(train_iters):

            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir,labels_ir, indexes_ir,cids_ir,name_ir = self._parse_data_ir(inputs_ir) #inputs_ir1


            inputs_rgb,inputs_rgb1, labels_rgb, indexes_rgb,cids_rgb,name_rgb = self._parse_data_rgb(inputs_rgb)
            # forward
            inputs_rgb = torch.cat((inputs_rgb,inputs_rgb1),0)
            labels_rgb = torch.cat((labels_rgb,labels_rgb),-1)
            cids_rgb =  torch.cat((cids_rgb,cids_rgb),-1)

            indexes_ir = torch.tensor([self.nameMap_ir[name] for name in name_ir]).cuda()
            indexes_rgb = torch.tensor([self.nameMap_rgb[name] for name in name_rgb]).cuda()
            indexes_rgb = torch.cat((indexes_rgb,indexes_rgb),-1)


            _,f_out_rgb,f_out_ir,gf_rgb,gf_ir,labels_rgb,labels_ir,\
            cid_rgb,cid_ir,index_rgb,index_ir  = self._forward(inputs_rgb,inputs_ir,label_1=labels_rgb,label_2=labels_ir,modal=0,\
                cid_rgb=cids_rgb,cid_ir=cids_ir,index_rgb=indexes_rgb,index_ir=indexes_ir)

            loss_ir = self.memory_ir(f_out_ir, labels_ir) 
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            # loss_tri_rgb, batch_acc_rgb = self.tri(gf_rgb, labels_rgb)
            # loss_tri_ir, batch_acc_ir = self.tri(gf_ir, labels_ir)
            ir_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            rgb_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            loss_ins_ir = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            loss_ins_rgb = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            rgb_rgb_loss = torch.tensor([0.]).cuda()#+loss_ir_trans_self
            ir_ir_loss = torch.tensor([0.]).cuda()#+loss_rgb_trans_self
            if epoch>=1000:#self.cmlabel:
                if epoch %2 ==0:
                    ############rgb-ir
                    # sim_prob_all_rgb_ir = torch.cat((F.softmax(self.wise_memory_ir.features.detach().data.mm(F.normalize(self.memory_rgb.features.detach().data, dim=1).t())/0.01,dim=1),\
                    # F.softmax(self.wise_memory_ir.features.detach().mm(self.memory_ir.features.detach().t())/0.01,dim=1)),dim=1).detach().data  #N C/0.05
                    # sim_prob_B_rgb_ir = torch.cat((F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.memory_rgb.features.detach().t())/0.01,dim=1),\
                    # F.softmax(F.normalize(f_out_rgb, dim=1).mm(self.memory_ir.features.detach().t())/0.01,dim=1)),dim=1).detach().data

                    #######full domain sim
                    sim_prob_all_rgb_ir_1 = torch.cat([F.softmax(self.wise_memory_ir.features.detach().mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                    sim_prob_all_rgb_ir_2 = torch.cat([F.softmax(self.wise_memory_ir.features.detach().mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                    sim_prob_all_rgb_ir = torch.cat((sim_prob_all_rgb_ir_1,sim_prob_all_rgb_ir_2),dim=1)
                    sim_prob_B_rgb_ir_1 = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()
                    sim_prob_B_rgb_ir_2 = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()
                    sim_prob_B_rgb_ir = torch.cat((sim_prob_B_rgb_ir_1,sim_prob_B_rgb_ir_2),dim=1)

                    sim_prob_rgb_ir = F.normalize(sim_prob_B_rgb_ir, dim=1).mm(F.normalize(sim_prob_all_rgb_ir.t(),dim=1))#B N
                    sim_rgb_ir = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
                    sim_rgb_ir_exp =sim_rgb_ir /0.05  # 64*13638
                    nearest_rgb_ir = sim_rgb_ir.max(dim=1, keepdim=True)[0]
                    nearest_prob_rgb_ir = sim_prob_rgb_ir.max(dim=1, keepdim=True)[0]
                    mask_neighbor_rgb_ir = torch.gt(sim_rgb_ir, nearest_rgb_ir * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_prob_rgb_ir = torch.gt(sim_prob_rgb_ir, nearest_prob_rgb_ir * 0.8)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_rgb_ir = mask_neighbor_rgb_ir.mul(mask_neighbor_prob_rgb_ir).sum(dim=1)+1
                    score_intra_rgb_ir =   F.softmax(sim_rgb_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                    # print('score_intra',score_intra)
                    score_intra_rgb_ir = score_intra_rgb_ir.clamp_min(1e-8)
                    # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
                    rgb_ir_loss = -score_intra_rgb_ir.log().mul(mask_neighbor_rgb_ir).mul(mask_neighbor_prob_rgb_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                    rgb_ir_loss = rgb_ir_loss.div(num_neighbor_rgb_ir).mean()#.mul(mask_neighbor_intra_soft) ##

                else:
                #     ##################ir-rgb
                    # sim_prob_all_ir_rgb = torch.cat((F.softmax(self.wise_memory_rgb.features.mm(F.normalize(self.memory_rgb.features.detach().data, dim=1).t())/0.01,dim=1),\
                    # F.softmax(self.wise_memory_rgb.features.mm(self.memory_ir.features.detach().data.t())/0.01,dim=1)),dim=1).detach().data  #N C/0.05
                    # sim_prob_B_ir_rgb = torch.cat((F.softmax(F.normalize(f_out_ir, dim=1).mm(self.memory_rgb.features.detach().data.t())/0.01,dim=1),\
                    # F.softmax(F.normalize(f_out_ir, dim=1).mm(self.memory_ir.features.detach().data.t())/0.01,dim=1)),dim=1).detach().data
                    

                    sim_prob_all_ir_rgb_1 = torch.cat([F.softmax(self.wise_memory_rgb.features.detach().mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                    sim_prob_all_ir_rgb_2 = torch.cat([F.softmax(self.wise_memory_rgb.features.detach().mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()  #N C/0.05
                    sim_prob_all_ir_rgb = torch.cat((sim_prob_all_ir_rgb_1,sim_prob_all_ir_rgb_2),dim=1)
                    sim_prob_B_ir_rgb_1 = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data#.cpu()
                    sim_prob_B_ir_rgb_2 = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data#.cpu()
                    sim_prob_B_ir_rgb = torch.cat((sim_prob_B_ir_rgb_1,sim_prob_B_ir_rgb_2),dim=1)


                    sim_prob_ir_rgb = F.normalize(sim_prob_B_ir_rgb, dim=1).mm(F.normalize(sim_prob_all_ir_rgb.t(),dim=1))#B N
                    sim_ir_rgb = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
                    sim_ir_rgb_exp =sim_ir_rgb /0.05  # 64*13638
                    nearest_ir_rgb = sim_ir_rgb.max(dim=1, keepdim=True)[0]
                    nearest_prob_ir_rgb = sim_prob_ir_rgb.max(dim=1, keepdim=True)[0]
                    mask_neighbor_prob_ir_rgb = torch.gt(sim_prob_ir_rgb, nearest_prob_ir_rgb * 0.8)#.cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    mask_neighbor_ir_rgb = torch.gt(sim_ir_rgb, nearest_ir_rgb * 0.8)#nearest_intra * self.neighbor_eps)self.neighbor_eps
                    num_neighbor_ir_rgb = mask_neighbor_ir_rgb.mul(mask_neighbor_prob_ir_rgb).sum(dim=1)+1#.mul(sim_wise).
                    score_intra_ir_rgb =   F.softmax(sim_ir_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
                    # print('score_intra',score_intra)
                    score_intra_ir_rgb = score_intra_ir_rgb.clamp_min(1e-8)
                    # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
                    ir_rgb_loss = -score_intra_ir_rgb.log().mul(mask_neighbor_ir_rgb).mul(mask_neighbor_prob_ir_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
                    ir_rgb_loss = ir_rgb_loss.div(num_neighbor_ir_rgb).mean()#.mul(mask_neighbor_intra_soft) ##


                #################rgb-rgb


            # sim_prob_all_rgb_rgb = torch.cat([F.softmax(self.wise_memory_rgb.features.detach().mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data.cpu()  #N C/0.05
            # sim_prob_B_rgb_rgb = torch.cat([F.softmax(F.normalize(f_out_rgb, dim=1).mm(F.normalize(self.wise_memory_rgb.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_rgb.allcam],dim=1).detach().data.cpu()
            


            # sim_prob_rgb_rgb = F.normalize(sim_prob_B_rgb_rgb, dim=1).mm(F.normalize(sim_prob_all_rgb_rgb.t(),dim=1))#B N
            # sim_rgb_rgb = F.normalize(f_out_rgb, dim=1).mm(self.wise_memory_rgb.features.detach().data.t())
            # sim_rgb_rgb_exp =sim_rgb_rgb /0.05  # 64*13638
            # nearest_rgb_rgb = sim_rgb_rgb.max(dim=1, keepdim=True)[0]
            # nearest_prob_rgb_rgb = sim_prob_rgb_rgb.max(dim=1, keepdim=True)[0]
            # mask_neighbor_rgb_rgb = torch.gt(sim_rgb_rgb, nearest_rgb_rgb * 0.8).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
            # mask_neighbor_prob_rgb_rgb = torch.gt(sim_prob_rgb_rgb, nearest_prob_rgb_rgb * 0.8).cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
            # num_neighbor_rgb_rgb = mask_neighbor_rgb_rgb.mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)+1
            # # print('num_neighbor_rgb_rgb',num_neighbor_rgb_rgb)
            # score_intra_rgb_rgb =   F.softmax(sim_rgb_rgb_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # # print('score_intra',score_intra)
            # score_intra_rgb_rgb = score_intra_rgb_rgb.clamp_min(1e-8)
            # # count_rgb_ir = (mask_neighbor_rgb_ir).sum(dim=1)
            # rgb_rgb_loss = -score_intra_rgb_rgb.log().mul(mask_neighbor_rgb_rgb).mul(mask_neighbor_prob_rgb_rgb).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # rgb_rgb_loss = rgb_rgb_loss.div(num_neighbor_rgb_rgb).mean()#.mul(mask_neighbor_intra_soft) ##
            # # #################ir-ir
            # # sim_prob_all_ir_ir = F.softmax((self.wise_memory_ir.features.detach().mm(self.memory_ir.features.detach().data.t()))/0.01,dim=1).detach().data #N C/0.05
            # # sim_prob_B_ir_ir = F.softmax((F.normalize(f_out_ir, dim=1).mm(self.memory_ir.features.detach().data.t()))/0.01,dim=1).detach().data

            # sim_prob_all_ir_ir = torch.cat([F.softmax(self.wise_memory_ir.features.detach().mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data.cpu()  #N C/0.05
            # sim_prob_B_ir_ir = torch.cat([F.softmax(F.normalize(f_out_ir, dim=1).mm(F.normalize(self.wise_memory_ir.cam_mem[i].detach().data, dim=1).t())/0.01,dim=1) for i in self.wise_memory_ir.allcam],dim=1).detach().data.cpu()
            

            # sim_prob_ir_ir = F.normalize(sim_prob_B_ir_ir, dim=1).mm(F.normalize(sim_prob_all_ir_ir.t(),dim=1))#B N
            # sim_ir_ir = F.normalize(f_out_ir, dim=1).mm(self.wise_memory_ir.features.detach().data.t())
            # sim_ir_ir_exp =sim_ir_ir /0.05  # 64*13638
            # nearest_ir_ir = sim_ir_ir.max(dim=1, keepdim=True)[0]
            # nearest_prob_ir_ir = sim_prob_ir_ir.max(dim=1, keepdim=True)[0]
            # mask_neighbor_prob_ir_ir = torch.gt(sim_prob_ir_ir, nearest_prob_ir_ir * 0.8).cuda()#nearest_intra * self.neighbor_eps)self.neighbor_eps
            # mask_neighbor_ir_ir = torch.gt(sim_ir_ir, nearest_ir_ir * 0.8).detach().data#nearest_intra * self.neighbor_eps)self.neighbor_eps
            # num_neighbor_ir_ir = mask_neighbor_ir_ir.mul(mask_neighbor_prob_ir_ir).sum(dim=1)+1#.mul(sim_wise).
            # # print('num_neighbor_ir_ir',num_neighbor_ir_ir)
            # score_intra_ir_ir =   F.softmax(sim_ir_ir_exp,dim=1)##sim_exp_intra / sim_exp_intra.sum(dim=1, keepdim=True)# 
            # # print('score_intra',score_intra)
            # score_intra_ir_ir = score_intra_ir_ir.clamp_min(1e-8)
            # # count_ir_rgb = (mask_neighbor_ir_rgb).sum(dim=1)
            # ir_ir_loss = -score_intra_ir_ir.log().mul(mask_neighbor_ir_ir).mul(mask_neighbor_prob_ir_ir).sum(dim=1)#.mul(sim_wise) mul(mask_neighbor_intra) .mul(mask_neighbor_intra)
            # ir_ir_loss = ir_ir_loss.div(num_neighbor_ir_ir).mean()#.mul(mask_neighbor_intra_soft) ##





            loss_camera_ir = torch.tensor([0.]).cuda()
            loss_camera_rgb = torch.tensor([0.]).cuda()

            loss_camera_ir = self.camera_loss(f_out_ir,cids_ir,labels_ir,percam_tempV_ir,concate_intra_class_ir,memory_class_mapper_ir)
            loss_camera_rgb = self.camera_loss(f_out_rgb,cids_rgb,labels_rgb,percam_tempV_rgb,concate_intra_class_rgb,memory_class_mapper_rgb)
            lamda_i=0
            loss_ins_ir = self.wise_memory_ir(f_out_ir,index_ir,cid_ir)#torch.  tensor([0.]).cuda(),torch.tensor([0.]).cuda()#
            loss_ins_rgb= self.wise_memory_rgb(f_out_rgb,index_rgb,cid_rgb)#torch.tensor([0.]).cuda(),torch.tensor([0.]).cuda()#

            loss=loss_ir+loss_rgb+0*(ir_ir_loss+rgb_rgb_loss+rgb_ir_loss+ir_rgb_loss)+lamda_i*(loss_ins_ir+loss_ins_rgb)+0.1*(loss_camera_ir+loss_camera_rgb)#+loss_ir_rgb_trans+loss_rgb_ir_trans

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

                      #  'adp ir {:.3f}\t'
                      # 'adp rgb {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_ir,loss_rgb))
                # print('adp_ir,adp_rgb',loss_tri_ir.item(),loss_tri_rgb.item())
                print('ir_rgb_loss,rgb_ir_loss',ir_rgb_loss.item(),rgb_ir_loss.item())
                # print('ir_ir_loss,rgb_rgb_loss',ir_ir_loss.item(),rgb_rgb_loss.item())
                # print('loss_ins_ir,loss_ins_rgb',loss_ins_ir.item(),loss_ins_rgb.item())
    def _parse_data_rgb(self, inputs):
        imgs,imgs1, name, pids, cids, indexes = inputs
        return imgs.cuda(),imgs1.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _parse_data_ir(self, inputs):
        imgs, name, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(),cids.cuda(),name

    def _forward(self, x1, x2, label_1=None,label_2=None,modal=0,cid_rgb=None,cid_ir=None,index_rgb=None,index_ir=None):
        return self.encoder(x1, x2, modal=modal,label_1=label_1,label_2=label_2,cid_rgb=cid_rgb,cid_ir=cid_ir,index_rgb=index_rgb,index_ir=index_ir)

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
        return concate_intra_class,percam_tempV,memory_class_mapper
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
        return concate_intra_class,percam_tempV,memory_class_mapper
