# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
from solver import make_optimizer, WarmupMultiStepLR
from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import cfg
from clustercontrast import datasets
# from clustercontrast import models
from clustercontrast.model_vit_cmrefine import make_model
from torch import einsum
# from clustercontrast.model_vit_cmrefine.make_model import TransMatcher

from clustercontrast.models.cm import ClusterMemory,ClusterMemory_all,Memory_wise_v3
from clustercontrast.trainers_s123 import ClusterContrastTrainer_pretrain_camera_confusionrefine_noice#ClusterContrastTrainer_pretrain_camera_confusionrefine# as ClusterContrastTrainer_pretrain_joint
# from clustercontrast.trainers import ClusterContrastTrainer_pretrain_camera_cpsrefine as ClusterContrastTrainer_pretrain_joint
from clustercontrast.trainers_s123 import ClusterContrastTrainer_pretrain_joint# as ClusterContrastTrainer_pretrain_joint_intrac
from clustercontrast.trainers_s123 import ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine# as ClusterContrastTrainer_pretrain_camera_wise_3
# from clustercontrast.trainers_s123 import ClusterContrastTrainer_pretrain_camera_wise_3_noddcma

from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam,MoreCameraSampler
import os
import torch.utils.data as data
from torch.autograd import Variable
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter
from solver.scheduler_factory import create_scheduler
from typing import Tuple, List, Optional
from torch import Tensor
import numbers
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import cv2

import copy
import os.path as osp
import errno
import shutil
start_epoch = best_mAP = 0
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
part=1
torch.backends.cudnn.enable =True,
torch.backends.cudnn.benchmark = True


# l2norm = Normalize(2)

def process_test_regdb(img_dir, trial = 1, modal = 'visible'):
    if modal=='visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal=='thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)
def eval_regdb(distmat, q_pids, g_pids, max_rank = 20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP
class channel_jitter(object):
    def __init__(self,channel=0):
        self.jitter = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        self.trans = T.Compose([
        self.jitter
        ])
    def __call__(self, img):
        img_np=np.array(self.trans(img))
        # idx = random.randint(0, 21)
        channel_1 = cv2.applyColorMap(img_np,  random.randint(0, 21))

        channel_2 = cv2.applyColorMap(img_np,  random.randint(0, 21))
        channel_3 = cv2.applyColorMap(img_np,  random.randint(0, 21))
        img_np[0, :,:] = channel_1[0,:,:]
        img_np[1, :,:] = channel_2[1,:,:]
        img_np[2, :,:] = channel_3[2,:,:]
        img = Image.fromarray(img_np, 'RGB')
        idx = random.randint(0, 100)
        img.save('figs/channel_jitter_'+str(idx)+'.jpg')
        print(img)
        return img



def get_data(name, data_dir,trial=0):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root,trial=trial)
    return dataset


class channel_select(object):
    def __init__(self,channel=0):
        self.channel = channel

    def __call__(self, img):
        if self.channel == 3:
            img_gray = img.convert('L')
            np_img = np.array(img_gray, dtype=np.uint8)
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        else:
            np_img = np.array(img, dtype=np.uint8)
            np_img = np_img[:,:,self.channel]
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL=Image.fromarray(img_aug, 'RGB')
        return img_PIL



def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None):




    # train_transformer = T.Compose([
    #     T.Resize((height, width), interpolation=3),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop((height, width)),
    #     T.ToTensor(),
    #     normalizer,
    #     T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    # ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            # sampler = MoreCameraSampler(train_set, num_instances)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):




    # train_transformer = T.Compose([
    #     T.Resize((height, width), interpolation=3),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop((height, width)),
    #     T.ToTensor(),
    #     normalizer,
    #     T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    # ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            # sampler = MoreCameraSampler(train_set, num_instances)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model




class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip
def extract_gall_feat(model,gall_loader,ngall):
    pool_dim=768*part
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input,input, 1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc


def extract_query_feat(model,query_loader,nquery):
    pool_dim=768*part
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net( input, input,2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net( flip_input,flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc



def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP

def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()

def select_merge_data(u_feas, label, label_to_images,  ratio_n,  dists,rgb_num,ir_num):

    dists = torch.from_numpy(dists)
    # homo_mask = torch.zeros(len(u_feas), len(u_feas))
    # homo_mask[:rgb_num,:rgb_num] = 9900000 #100000
    # homo_mask[rgb_num:,rgb_num:] = 9900000
    # homo_mask[rgb_num:,:rgb_num] = 9900000
    print(dists.size())
    # dists.add_(torch.tril(900000 * torch.ones(len(u_feas), len(u_feas))))
    # print(dists.size())
    # dists.add_(homo_mask)
    # cnt = torch.FloatTensor([ len(label_to_images[label[idx]]) for idx in range(len(u_feas))])
    # dists += ratio_n * (cnt.view(1, len(cnt)) + cnt.view(len(cnt), 1))
    
    # for idx in range(len(u_feas)):
    #     for j in range(idx + 1, len(u_feas)):
    #         if label[idx] == label[j]:
    #             dists[idx, j] = 900000
    # print('rgb_num',rgb_num)
    # print('ir_num',ir_num)
    dists = dists.numpy()

    # dists=dists[:rgb_num,rgb_num:]
    ind = np.unravel_index(np.argsort(dists, axis=None)[::-1], dists.shape) #np.argsort(dists, axis=1)#
    idx1 = ind[0]
    idx2 = ind[1]
    dist_list = dists[idx1,idx2] #[dists[i,j] for i,j in zip(idx1,idx2)]
    # print(ind.shape)
    # print(ind)
    return idx1, idx2, dist_list

def select_merge_data_jacard(u_feas, label, label_to_images,  ratio_n,  dists,rgb_num,ir_num):

    dists = torch.from_numpy(dists)

    print(dists.size())

    dists = dists.numpy()

    ind = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
    idx1 = ind[0]
    idx2 = ind[1]
    dist_list = dists[idx1,idx2] 
    return idx1, idx2, dist_list


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def camera(cams,features,labels):
    cf = features
    intra_id_features = []
    intra_id_labels = []
    for cc in np.unique(cams):
        percam_ind = np.where(cams == cc)[0]
        percam_feature = cf[percam_ind].numpy()
        percam_label = labels[percam_ind]
        percam_class_num = len(np.unique(percam_label[percam_label >= 0]))
        percam_id_feature = np.zeros((percam_class_num, percam_feature.shape[1]), dtype=np.float32)
        cnt = 0
        for lbl in np.unique(percam_label):
            if lbl >= 0:
                ind = np.where(percam_label == lbl)[0]
                id_feat = np.mean(percam_feature[ind], axis=0)
                percam_id_feature[cnt, :] = id_feat
                intra_id_labels.append(lbl)
                cnt += 1
        percam_id_feature = percam_id_feature / np.linalg.norm(percam_id_feature, axis=1, keepdims=True)
        intra_id_features.append(torch.from_numpy(percam_id_feature))
    return intra_id_features, intra_id_labels

def pairwise_distance_matcher(matcher, prob_fea, gal_fea, gal_batch_size=4, prob_batch_size=4096):
    with torch.no_grad():
        num_gals = gal_fea.size(0)
        num_probs = prob_fea.size(0)
        score = torch.zeros(num_probs, num_gals, device=prob_fea.device)
        score_2 = torch.zeros(num_probs, num_gals, device=prob_fea.device)
        matcher.eval()
        for i in range(0, num_probs, prob_batch_size):
            j = min(i + prob_batch_size, num_probs)
            # matcher.make_kernel(prob_fea[i: j,  :].cuda())
            # matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
            for k in range(0, num_gals, gal_batch_size):
                k2 = min(k + gal_batch_size, num_gals)
                score[i: j, k: k2],score_2[i: j, k: k2] = matcher(prob_fea[i: j,  :].cuda(),gal_fea[k: k2, :].cuda())
                # print(score[i: j, k: k2])
                # print(torch.sigmoid(score[i: j, k: k2]/10 ))
        # scale matching scores to make them visually more recognizable
        # score = torch.sigmoid(score/10 )#F.softmax(torch.sigmoid(score / 10),dim=1) 
    return score.cpu(), score_2.cpu() # [p, g]
    # score = torch.sigmoid(score / 10)
    # return (1. - score).cpu()

def pairwise_part(prob_fea, gal_fea,percam_memory_all, gal_batch_size=4, prob_batch_size=4096):
    
    num_gals = gal_fea.size(0)
    num_probs = prob_fea.size(0)
    score = torch.zeros(num_probs, num_gals, device=prob_fea.device)

    for i in range(0, num_probs, prob_batch_size):
        j = min(i + prob_batch_size, num_probs)
        # matcher.make_kernel(prob_fea[i: j,  :].cuda())
        # matcher.make_kernel(prob_fea[i: j, :, :, :].cuda())
        for k in range(0, num_gals, gal_batch_size):
            k2 = min(k + gal_batch_size, num_gals)
            score[i: j, k: k2],score_2[i: j, k: k2] = matcher(prob_fea[i: j,  :].cuda(),gal_fea[k: k2, :].cuda())
    return score.cpu()





def part_sim(query_t, key_m):
    seq_len=part
    q, d_5 = query_t.size() # b d*5,  
    k, d_5 = key_m.size()

    z= int(d_5/seq_len)
    d = int(d_5/seq_len)        
    # query_t =  query_t.detach().view(q, -1, z)#self.bn3(tgt.view(q, -1, z))  #B N C
    # key_m = key_m.detach().view(k, -1, d)#self.bn3(memory.view(k, -1, d)) #B N C

    query_t = F.normalize(query_t.view(q, -1, z), dim=-1)  #B N C tgt.view(q, -1, z)#
    key_m = F.normalize(key_m.view(k, -1, d), dim=-1) #Q N C memory.view(k, -1, d)#
    # score = einsum('q t d, k s d -> q k s t', query_t, key_m)#F.softmax(einsum('q t d, k s d -> q k s t', query_t, key_m),dim=-1).view(q,-1) # B Q N N
    score = einsum('q t d, k s d -> q k t s', query_t, key_m)
    score = torch.cat((score.max(dim=2)[0], score.max(dim=3)[0]), dim=-1) #####score.max(dim=3)[0]#q k 10
    score = F.softmax(score.permute(0,2,1)/0.01,dim=-1).reshape(q,-1)

    return score




def init_camera_proxy(all_img_cams,all_pseudo_label,intra_id_features):
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


def save_checkpoint_match(state, is_best, fpath='checkpoint.pth.tar',match=''):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), match+'match_best.pkl'))
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def select_merge_data(dists):
    dists = torch.from_numpy(dists)
    print(dists.size())
    dists = dists.numpy()
    ind = np.unravel_index(np.argsort(dists, axis=None)[::-1], dists.shape) #np.argsort(dists, axis=1)#
    idx1 = ind[0]
    idx2 = ind[1]
    dist_list = dists[idx1,idx2]
    return idx1, idx2, dist_list



def main():
    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    # main_worker(args,cfg)
    log_s1_name = 'regdb_2p_384_g_s2_20cmav1_30cmav2top20_50_iter_dist0.4_lamdacm0.1'
    # log_s2_name = 'sysu_2p_384_5glpartv2_fine_0camcmav1_10cmcmav1_30cmcmav2_a100'
    # log_s2_name = 'sysu_2p_384_g_s23_0cam_5cmav1_l0.3_20cmv2_fine_a100'
    # log_s2_name = 'sysu_2p_384_g_s12_10cam_l0.3_20cmav1lamda0.1_l1.0_30cmv2_noload_a100'
    # log_s2_name = 'sysu_2p_384_5g_s123_5cam_10cmav1lamda0.1_l0.5_15cmv2_noload_a100'
    # log_s2_name = 'sysu_2p_384_5g_s12_10cam_20cmav1_l0.3_30cmv2_noload_a100'
    # log_s2_name = 'sysu_2p_384_g_s12_0cam_10cmav1_20cmv2_fine_a100'
    # log_s2_name = 'sysu_2p_384_g_s12_10cam_20cmav1_30cmv2_fine_a100'
    # log_s2_name = 'sysu_2p_384_5glpartv2_fine_20camcmav2_30cmcmav1_40cmcmav2'
    # main_worker_stage1(args,log_s1_name)
    main_worker_stage1(args,log_s1_name) #add CMA 

def main_worker_stage1(args,log_s1_name):
# def main_worker_stage2(args,log_s1_name):
# def main_worker(args,cfg):
    l2norm = Normalize(2)
    ir_batch=192
    rgb_batch=128
    global start_epoch, best_mAP

    # trial = args.trial
    # log_name='sysu_2p_288_5glpart_10cps_cmav2_v100' # _0.8cmrefinehthm0
    # log_name='sysu_2p_288_5glpart_confusionwrtv1_cmav3_a100' # _0.8cmrefinehthm0
    # log_name='sysu_2p_288_3lpart_cmav2_s23_a100' # _0.8cmrefinehthm0
    # log_name='sysu_2p_288_5glpartgem_cmav2_s23_a100'
    # log_name='sysu_2p_384_5glpartv2_cmpl_7camcmav1_10cmav1_15cmcmav2_a100'
    args.logs_dir = osp.join('logs'+'/'+log_s1_name)
    # args.logs_dir = osp.join(args.logs_dir,str(trial))
    # args.logs_dir = osp.join(args.logs_dir+'/'+log_name)
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'test_log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()
    print(config_str)
    # Create datasets
    iters = args.iters if (args.iters > 0) else None

    # Create model
    # model = create_model(args)
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0)

    model.cuda()

    model = nn.DataParallel(model)#,output_device=1)
    trainer_intrac = ClusterContrastTrainer_pretrain_joint(model)

    trainer = ClusterContrastTrainer_pretrain_camera_confusionrefine_noice(model)
    trainer.cmlabel=2000#30#30#1000
    trainer_interm = ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine(model)
    cam_cma = 1000#15
    s2_cmav1 = 20#10
    # s3_cma = 30
    s3_cma =1000#30#20
    s3=300#30#20
    # s2_cmcamcmav1 = 10

    # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()
    # matcher_ir = TransMatcher(5, 768, 3, 768).cuda()
    # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()

    # matcher_ir = nn.DataParallel(matcher_ir)
    # matcher_rgb = nn.DataParallel(matcher_rgb)

    # matcher_ir = TransMatcher(5, 768, 3, 768).cuda()
    # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()

    # optimizer = make_optimizer(cfg, model)
    # checkpoint = load_checkpoint(osp.join('/dat01/yangbin/cluster-contrast-reid-camera/logs/sysu_all_resnet_pretrain_camera0.1_cma', 'model_best.pth.tar'))
    # checkpoint = load_checkpoint(osp.join('logs/sysu_2p_288_4part_cmav2', 'model_best.pth.tar'))
    # # # print(checkpoint)
    for trial in range(1,11): 
        checkpoint = load_checkpoint(osp.join('./logs/'+log_s1_name+'/'+str(trial), 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])

    ##############################
        args.test_batch=64
        args.img_w=args.width
        args.img_h=args.height
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform_test = T.Compose([
            T.ToPILImage(),
            T.Resize((args.img_h,args.img_w)),
            T.ToTensor(),
            normalize,
        ])
        mode='all'
        data_path='/dat01/yangbin/data/RegDB/'
        query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery = len(query_label)
        ngall = len(gall_label)
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        query_feat_fc = extract_gall_feat(model,query_loader,nquery)
        # for trial in range(1):
        ngall = len(gall_label)
        gall_feat_fc = extract_query_feat(model,gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

        if trial == 1:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP

        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('V2T All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))



    for trial in range(1,11): 
        checkpoint = load_checkpoint(osp.join('./logs/'+log_s1_name+'/'+str(trial), 'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])

    ##############################
        args.test_batch=64
        args.img_w=args.width
        args.img_h=args.height
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform_test = T.Compose([
            T.ToPILImage(),
            T.Resize((args.img_h,args.img_w)),
            T.ToTensor(),
            normalize,
        ])
        mode='all'
        data_path='/dat01/yangbin/data/RegDB/'
        query_img, query_label = process_test_regdb(data_path, trial=trial, modal='thermal')
        gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='visible')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        nquery = len(query_label)
        ngall = len(gall_label)
        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        query_feat_fc = extract_query_feat(model,query_loader,nquery)
        # for trial in range(1):
        ngall = len(gall_label)
        gall_feat_fc = extract_gall_feat(model,gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

        if trial == 1:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP

        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('T2V All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))




def main_worker_stage2312(args,log_s1_name):
# def main_worker(args,cfg):
    l2norm = Normalize(2)
    ir_batch=128
    rgb_batch=128
    global start_epoch, best_mAP

    log_name=log_s1_name#'sysu_2p_384_5glpartv2_cmpl_7camcmav1_10cmav1_15cmcmav2_a100'

    args.logs_dir = osp.join(args.logs_dir+'/'+log_name)
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))
    print("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()
    print(config_str)
    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir)
    dataset_rgb = get_data('sysu_rgb', args.data_dir)

    test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
    test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
    # Create model
    # model = create_model(args)
    model = make_model(cfg, num_class=0, camera_num=0, view_num = 0)

    model.cuda()

    model = nn.DataParallel(model)#,output_device=1)
    trainer_intrac = ClusterContrastTrainer_pretrain_joint(model)

    trainer = ClusterContrastTrainer_pretrain_camera_confusionrefine(model)
    trainer.cmlabel=150#30#30#1000
    trainer_interm = ClusterContrastTrainer_pretrain_camera_wise_3_cmrefine(model)
    cam_cma = 700
    s2_cmav1 =1000


    # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()
    # matcher_ir = TransMatcher(5, 768, 3, 768).cuda()
    # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()

    # matcher_ir = nn.DataParallel(matcher_ir)
    # matcher_rgb = nn.DataParallel(matcher_rgb)

    # matcher_ir = TransMatcher(5, 768, 3, 768).cuda()
    # matcher_rgb = TransMatcher(5, 768, 3, 768).cuda()

    # optimizer = make_optimizer(cfg, model)
    # checkpoint = load_checkpoint(osp.join('/dat01/yangbin/cluster-contrast-reid-camera/logs/sysu_all_resnet_pretrain_camera0.1_cma', 'model_best.pth.tar'))
    # checkpoint = load_checkpoint(osp.join('logs/sysu_2p_288_4part_cmav2', 'model_best.pth.tar'))
    # # print(checkpoint)
    # model.load_state_dict(checkpoint['state_dict'])

    # matcher_ir = torch.load(osp.join('logs/sysu_all_vit_base_2p_288_4part_matchv2_1_nodet', 'irmatch_best.pkl'))
    # # # matcher_ir.load_state_dict(checkpoint_match_ir['state_dict'])

    # matcher_rgb = torch.load(osp.join('logs/sysu_all_vit_base_2p_288_4part_matchv2_1_nodet', 'rgbmatch_best.pkl'))
    # matcher_rgb.load_state_dict(checkpoint_match_rgb['state_dict'])

    # checkpoint_match_ir = load_checkpoint(osp.join('logs/sysu_all_vit_base_2p_288_4part_matchv2_1_nodet', 'irmatch_best.pth.tar'))
    # matcher_ir.load_state_dict(checkpoint_match_ir['state_dict'])

    # checkpoint_match_rgb = load_checkpoint(osp.join('logs/sysu_all_vit_base_2p_288_4part_matchv2_1_nodet', 'rgbmatch_best.pth.tar'))
    # matcher_rgb.load_state_dict(checkpoint_match_rgb['state_dict'])

    # Evaluator
    # params = [
    #     {'params': model.parameters()},
    #     {'params': matcher_ir.parameters()},
    #     {'params': matcher_rgb.parameters()}]

    # trainer.matcher_ir=matcher_ir
    # trainer.matcher_rgb=matcher_rgb
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]



    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    evaluator = Evaluator(model)



    @torch.no_grad()
    def generate_cluster_features(labels, features):
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

    color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)#T.
    # color_aug_ir = T.ColorJitter(brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5)#T.
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    height=args.height
    width=args.width
    train_transformer_rgb = T.Compose([
    color_aug,
    T.Resize((height, width)),#, interpolation=3
    T.Pad(10),
    T.RandomCrop((height, width)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    normalizer,
    ChannelRandomErasing(probability = 0.5)
    ])

    train_transformer_rgb1 = T.Compose([
    color_aug,
    T.Resize((height, width)),
    T.Pad(10),
    T.RandomCrop((height, width)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    normalizer,
    ChannelRandomErasing(probability = 0.5),
    ChannelExchange(gray = 2)
    ])

    transform_thermal = T.Compose( [
        color_aug,
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)
        ])
    transform_thermal1 = T.Compose( [
        color_aug,
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        ChannelRandomErasing(probability = 0.5),
        ChannelAdapGray(probability =0.5)])

    rgb_cluster_num = {}
    ir_cluster_num = {}
    for epoch in range(args.epochs):
        if epoch < 100:
            rgb_cams = np.unique([x[2] for x in dataset_rgb.train])
            # if epoch < 1:
            #     loop_iter = 5
            # else:
            loop_iter = 1
            for rgb_cam_id in sorted(rgb_cams):
                ir_cams = np.unique([x[2] for x in dataset_ir.train])
                for ir_cam_id in sorted(ir_cams):
                    print('==> Create pseudo labels for camera unlabeled RGB data')
                    # print('rgb, camera ',rgb_cam_id)
                    cam_data_rgb = [x for x in dataset_rgb.train if x[2] == rgb_cam_id]
                    cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                     256, args.workers, 
                                                     testset=sorted(cam_data_rgb))
                    features_rgb_dict, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
                    features_rgb = torch.cat([features_rgb_dict[f].unsqueeze(0) for f, _, _ in sorted(cam_data_rgb)], 0)
                    features_rgb_=F.normalize(features_rgb, dim=1) 
                    rerank_dist_rgb = compute_jaccard_distance(features_rgb_, k1=args.k1, k2=args.k2,search_option=3)
                    rgb_eps= 0.55#0.55
                    print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                    cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                    pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
                    cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
                    pseudo_labeled_dataset_rgb = []
                    modality_rgb = []
                    cams_rgb=[]
                    for i, ((fname, _, cid), label) in enumerate(zip(sorted(cam_data_rgb), pseudo_labels_rgb)):
                        cams_rgb.append(cid)
                        modality_rgb.append(0)
                        if label != -1:
                            pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                        # if epoch%100 == 0:
                        #     print(fname,label.item())
                    num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                    memory_rgb = ClusterMemory(model.module.in_planes*part, num_cluster_rgb, temp=args.temp,
                                       momentum=args.momentum, use_hard=args.use_hard).cuda()
                    memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
                    print('==> Statistics for RGB  epoch {}: camera {} {} clusters,cluster data {} '.format(epoch,rgb_cam_id, num_cluster_rgb,len(pseudo_labeled_dataset_rgb)))

                    print('==> Create pseudo labels for camera unlabeled ir data ')
                    # print('ir, camera ',ir_cam_id)
                    cam_data_ir = [x for x in dataset_ir.train if x[2] == ir_cam_id]
                    cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                     256, args.workers, 
                                                     testset=sorted(cam_data_ir))
                    features_ir_dict, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
                    features_ir = torch.cat([features_ir_dict[f].unsqueeze(0) for f, _, _ in sorted(cam_data_ir)], 0)
                    features_ir_=F.normalize(features_ir, dim=1)
                    rerank_dist_ir = compute_jaccard_distance(features_ir_, k1=args.k1, k2=args.k2,search_option=3)
                    ir_eps = 0.55 #0.55 #0.55
                    print('ir Clustering criterion: eps: {:.3f}'.format(ir_eps))
                    cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                    pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
                    cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
                    pseudo_labeled_dataset_ir = []
                    modality_ir = []
                    cams_ir=[]
                    for i, ((fname, _, cid), label) in enumerate(zip(sorted(cam_data_ir), pseudo_labels_ir)):
                        cams_ir.append(cid)
                        modality_ir.append(1)
                        if label != -1:
                            pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                        # if epoch%100 == 0:
                        #     print(fname,label.item())
                    num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                    memory_ir = ClusterMemory(model.module.in_planes*part, num_cluster_ir, temp=args.temp,
                                       momentum=args.momentum, use_hard=args.use_hard).cuda()
                    memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
                    print('==> Statistics for RGB  epoch {}: camera {} {} clusters,cluster data {} '.format(epoch,rgb_cam_id, num_cluster_rgb,len(pseudo_labeled_dataset_rgb)))
                    print('==> Statistics for ir  epoch {}: camera {} {} clusters,cluster data {} '.format(epoch,ir_cam_id, num_cluster_ir,len(pseudo_labeled_dataset_ir)))
                    
                    # if epoch%2 == 0:

                    train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                                ir_batch, args.workers, args.num_instances, 50,
                                                trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)
                    train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            rgb_batch, args.workers, args.num_instances, 50,
                                            trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
                    # else:
                    #     train_loader_ir = get_train_loader_color(args, dataset_ir, args.height, args.width,
                    #                                 ir_batch, args.workers, args.num_instances, 50,
                    #                                 trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal,train_transformer1=transform_thermal1)
                    #     train_loader_rgb = get_train_loader_ir(args, dataset_rgb, args.height, args.width,
                    #                             rgb_batch, args.workers, args.num_instances, 50,
                    #                             trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb_1)


                    trainer_intrac.memory_ir = memory_ir
                    trainer_intrac.memory_rgb = memory_rgb
                    train_loader_ir.new_epoch()
                    time.sleep(1)
                    train_loader_rgb.new_epoch()
                    time.sleep(1)
                    trainer_intrac.train(epoch, train_loader_ir,train_loader_rgb, optimizer, print_freq=10, train_iters=len(train_loader_rgb))
                    ir_cluster_num[ir_cam_id]=num_cluster_ir
                    rgb_cluster_num[rgb_cam_id]=num_cluster_rgb

        with torch.no_grad():
            if epoch < 7:
                # DBSCAN cluster

                ir_eps = 0.6#0.6
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.6#0.6#+0.1
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            else:
                ir_eps = 0.6#0.6
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.6#0.6#+0.1
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_rgb.train))
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_=F.normalize(features_rgb, dim=1)
            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             256, args.workers, 
                                             testset=sorted(dataset_ir.train))
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_=F.normalize(features_ir, dim=1)

            all_feature = []#torch.cat([features_rgb,features_ir], 0)
            # rerank_dist_all_jacard = compute_jaccard_distance(all_feature, k1=args.k1, k2=args.k2,search_option=3)
            # rerank_dist_cm = rerank_dist_all_jacard[:features_rgb.size(0),features_rgb.size(0):]


            rerank_dist_ir = compute_jaccard_distance(features_ir_, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb_, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            del rerank_dist_rgb
            del rerank_dist_ir
            pseudo_labels_all = []
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            # print("epoch: {} \n pseudo_labels: {}".format(epoch, pseudo_labels.tolist()[:100]))

        # generate new dataset and calculate cluster centers


        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        # del features, cluster_loader_1,cluster_loader_2,cluster_loader_3,cluster_loader_4,features_1,features_2,features_3,features_4
        # del features_ir,features_rgb
        # Create hybrid memory
        # if epoch >= 25:
        #     memory_ir = ClusterMemory_all(model.module.num_features, num_cluster_ir, temp=args.temp,
        #                            momentum=args.momentum, use_hard=args.use_hard).cuda()
        #     memory_rgb = ClusterMemory_all(model.module.num_features, num_cluster_rgb, temp=args.temp,
        #                            momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir = ClusterMemory(model.module.in_planes*part, num_cluster_ir, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_rgb = ClusterMemory(model.module.in_planes*part, num_cluster_rgb, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb







        wise_f_ir=[]
        wise_name_ir = []
        pseudo_labeled_dataset_ir = []
        ir_label=[]
        pseudo_real_ir = {}
        cams_ir = []
        modality_ir = []
        outlier=0
        cross_cam=[]
        ir_cluster=collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            cams_ir.append(cid)
            modality_ir.append(1)
            cross_cam.append(int(cid+4))
            ir_label.append(label.item())
            ir_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                
                pseudo_real_ir[label.item()] = pseudo_real_ir.get(label.item(),[])+[_]
                pseudo_real_ir[label.item()] = list(set(pseudo_real_ir[label.item()]))
                wise_f_ir.append(features_ir[i,:].unsqueeze(0))
                wise_name_ir.append((fname, _, cid))
                # if epoch%10 == 0:
                #     print(fname,label.item())
            else:
                outlier=outlier+1
        wise_f_ir=torch.cat(wise_f_ir,dim=0)

        print('==> Statistics for IR epoch {}: {} clusters outlier {}'.format(epoch, num_cluster_ir,outlier))
        wise_f_rgb=[]
        wise_name_rgb = []
        pseudo_labeled_dataset_rgb = []
        rgb_label=[]
        pseudo_real_rgb = {}
        cams_rgb = []
        modality_rgb = []
        outlier=0
        rgb_cluster=collections.defaultdict(list)

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            cams_rgb.append(cid)
            modality_rgb.append(0)
            cross_cam.append(int(cid))
            rgb_label.append(label.item())
            rgb_cluster[cid].append(label.item())
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                
                pseudo_real_rgb[label.item()] = pseudo_real_rgb.get(label.item(),[])+[_]
                pseudo_real_rgb[label.item()] = list(set(pseudo_real_rgb[label.item()]))
                wise_f_rgb.append(features_rgb[i,:].unsqueeze(0))
                wise_name_rgb.append((fname, _, cid))

                # if epoch%10 == 0:
                #     print(fname,label.item())
            else:
                outlier=outlier+1
        wise_f_rgb=torch.cat(wise_f_rgb,dim=0)
        # print(wise_f_rgb.size())

        print('==> Statistics for RGB epoch {}: {} clusters outlier {} '.format(epoch, num_cluster_rgb,outlier))



        # ir_cluster_num={}
        # rgb_cluster_num={}

        # rgb_cams = np.unique([x[2] for x in dataset_rgb.train])
        # ir_cams = np.unique([x[2] for x in dataset_ir.train])

        # for cam in rgb_cams:
        #     rgb_cluster_num[cam]=np.unique(rgb_cluster[cam]).shape[0]
        # for cam in ir_cams:
        #     ir_cluster_num[cam]=np.unique(ir_cluster[cam]).shape[0]

        max_cid_rgb = max(rgb_cluster_num, key=lambda x: rgb_cluster_num[x])
        max_cid_ir = max(ir_cluster_num, key=lambda x: ir_cluster_num[x])
        print('max_cid_rgb,',max_cid_rgb,rgb_cluster_num)
        print('max_cid_ir,',max_cid_ir,ir_cluster_num)
############camera cma v1 
        if epoch>=cam_cma:#5:
            feat_maxcid_rgb = features_rgb_[cams_rgb==max_cid_rgb]
            feat_othercid_rgb = features_rgb_[cams_rgb!=max_cid_rgb]
            pseudo_labels_maxcid_rgb = pseudo_labels_rgb[cams_rgb==max_cid_rgb]
            pseudo_labels_othercid_rgb = pseudo_labels_rgb[cams_rgb!=max_cid_rgb]
            print(feat_maxcid_rgb.size(),feat_othercid_rgb.size())

            print('merge ir and rgb momery'.format(epoch, num_cluster_rgb))
            print('select_merge_data')
            label_to_images = {}
            dist_cm = np.matmul(feat_maxcid_rgb.numpy(), np.transpose(feat_othercid_rgb.numpy()))
            idx1, idx2,dist_list = select_merge_data(dist_cm)
            # del features_ir,features_rgb

            # del dist_cm
            # rgb_label_cnt = Counter(rgb_label) 
            # ir_label_cnt = Counter(ir_label)
            idx_lenth = np.sum(dist_list>=0.6)
            dist_list = dist_list[:idx_lenth]
            rgb2ir_label = [(i,j) for i,j in zip(np.array(pseudo_labels_maxcid_rgb)[idx1[:idx_lenth]],np.array(pseudo_labels_othercid_rgb)[idx2[:idx_lenth]])]
            # print('rgb2ir_label',rgb2ir_label)
            rgb2ir_label_cnt = Counter(rgb2ir_label)
            rgb2ir_label_cnt_sorted = sorted(rgb2ir_label_cnt.items(),key = lambda x:x[1],reverse = True)
            lenth = len(rgb2ir_label_cnt_sorted)
            lamda_cm=0.1
            in_rgb_label=[]
            in_ir_label=[]
            match_cnt = 0
            right = 0
            lenth_ratio = 0.4
            for i in range(int(lenth*lenth_ratio)):
                key = rgb2ir_label_cnt_sorted[i][0] 
                value = rgb2ir_label_cnt_sorted[i][1]
                if key[0] == -1 or key[1] == -1:
                    continue
                if key[0] in in_rgb_label or key[1] in in_ir_label:
                    continue
                if key[0] == key[1]:
                    continue
                print('rgb-ir label{}, real rgb label{}, real ir label{}'.format(key,pseudo_real_rgb[key[0]],pseudo_real_rgb[key[1]]))
                for real_ir in pseudo_real_rgb[key[1]]:
                    if real_ir in pseudo_real_rgb[key[0]]:
                        right = right+1
                    match_cnt=match_cnt+1
                update_memory = trainer.memory_rgb.features[key[0]]
                # pseudo_labels_rgb[pseudo_labels_rgb==key[1]]=key[0]
                trainer.memory_rgb.features[key[0]] =  F.normalize( (lamda_cm*trainer.memory_rgb.features[key[0]] + (1-lamda_cm)*(update_memory) ),dim=-1)
                trainer.memory_rgb.features[key[1]] =  F.normalize( (lamda_cm*trainer.memory_rgb.features[key[1]] + (1-lamda_cm)*(update_memory) ),dim=-1)
                in_rgb_label.append(key[0])
                in_ir_label.append(key[1])
                # match_cnt=match_cnt+1
                label_acc = right/match_cnt
            print('rgb-rgb match_cnt acc,lenth_ratio',match_cnt,label_acc,lenth_ratio)
###################

            feat_maxcid_ir = features_ir_[cams_ir==max_cid_ir]
            feat_othercid_ir = features_ir_[cams_ir!=max_cid_ir]
            pseudo_labels_maxcid_ir = pseudo_labels_ir[cams_ir==max_cid_ir]
            pseudo_labels_othercid_ir = pseudo_labels_ir[cams_ir!=max_cid_ir]


            print('merge ir and rgb momery'.format(epoch, num_cluster_rgb))
            print('select_merge_data')
            label_to_images = {}
            dist_cm = np.matmul(feat_maxcid_ir.numpy(), np.transpose(feat_othercid_ir.numpy()))
            idx1, idx2,dist_list = select_merge_data(dist_cm)
            # del features_ir,features_rgb

            # del dist_cm
            # rgb_label_cnt = Counter(rgb_label) 
            # ir_label_cnt = Counter(ir_label)
            idx_lenth = np.sum(dist_list>=0.7)
            dist_list = dist_list[:idx_lenth]
            rgb2ir_label = [(i,j) for i,j in zip(np.array(pseudo_labels_maxcid_ir)[idx1[:idx_lenth]],np.array(pseudo_labels_othercid_ir)[idx2[:idx_lenth]])]
            # print('rgb2ir_label',rgb2ir_label)
            rgb2ir_label_cnt = Counter(rgb2ir_label)
            rgb2ir_label_cnt_sorted = sorted(rgb2ir_label_cnt.items(),key = lambda x:x[1],reverse = True)
            lenth = len(rgb2ir_label_cnt_sorted)
            lamda_cm=0.1
            in_rgb_label=[]
            in_ir_label=[]
            match_cnt = 0
            right = 0
            lenth_ratio = 0.3 ####0.6 0.3->0.4
            label_acc=0
            for i in range(int(lenth*lenth_ratio)):
                key = rgb2ir_label_cnt_sorted[i][0] 
                value = rgb2ir_label_cnt_sorted[i][1]
                if key[0] == -1 or key[1] == -1:
                    continue
                if key[0] in in_rgb_label or key[1] in in_ir_label:
                    continue
                if key[0] == key[1]:
                    continue
                print('rgb-ir label{}, real rgb label{}, real ir label{}'.format(key,pseudo_real_ir[key[0]],pseudo_real_ir[key[1]]))
                for real_ir in pseudo_real_ir[key[1]]:
                    if real_ir in pseudo_real_ir[key[0]]:
                        right = right+1
                    match_cnt=match_cnt+1
                update_memory = trainer.memory_ir.features[key[0]]
                trainer.memory_ir.features[key[0]] = F.normalize(( lamda_cm*trainer.memory_ir.features[key[0]] + (1-lamda_cm)*(update_memory) ),dim=-1)
                trainer.memory_ir.features[key[1]] = F.normalize(( lamda_cm*trainer.memory_ir.features[key[1]] + (1-lamda_cm)*(update_memory) ),dim=-1)
                # pseudo_labels_ir[pseudo_labels_ir==key[1]]=key[0]
                in_rgb_label.append(key[0])
                in_ir_label.append(key[1])
                # match_cnt=match_cnt+1
                label_acc = right/match_cnt
            print('ir-ir match_cnt acc,lenth_ratio',match_cnt,label_acc,lenth_ratio)




################
        cams_rgb = np.asarray(cams_rgb)
        cams_ir = np.asarray(cams_ir)
        modality_rgb = np.asarray(modality_rgb+modality_ir)
        modality_ir = np.asarray(modality_ir) 
        cross_cam = np.asarray(cross_cam)

        intra_id_features_rgb,intra_id_labels_rgb = camera(cams_rgb,features_rgb,pseudo_labels_rgb)
        intra_id_features_ir,intra_id_labels_ir = camera(cams_ir,features_ir,pseudo_labels_ir)
        # concate_intra_class_ir,percam_tempV_ir,percam_memory_ir  = init_camera_proxy(cams_ir,pseudo_labels_ir,intra_id_features_ir)
        # concate_intra_class_rgb,percam_tempV_rgb,percam_memory_rgb  = init_camera_proxy(cams_rgb,pseudo_labels_rgb,intra_id_features_rgb)
        # del concate_intra_class_ir,percam_tempV_ir,concate_intra_class_rgb,percam_tempV_rgb
        # percam_memory_all = percam_memory_rgb+percam_memory_ir
###################################matcher+cma
        # for l_num in range(len(trainer.matcher_rgb.decoder.layers)):
        #     trainer.matcher_rgb.decoder.layers[l_num].qkv = model.module.base.blocks[l_num-3].attn.qkv#copy.deepcopy(model.module.base.blocks[l_num-3].attn.qkv)
        pseudo_labels_rgb_ori = torch.from_numpy(pseudo_labels_rgb)
        if epoch >= trainer.cmlabel:
            merge_time = time.time()
            print('merge ir and rgb momery'.format(epoch, num_cluster_ir))
            print('select_merge_data')

            ##########################cmav2
            intersect_count_list=[]
            # ins_sim_rgb_ir_2 = trainer.wise_memory_rgb.features.mm(trainer.wise_memory_ir.features.t())
            ins_sim_rgb_ir_2 =     F.normalize(features_rgb, dim=1).mm(F.normalize(features_ir, dim=1).t())         
            # initial_rank_ins = np.argsort(-ins_sim_rgb_ir.detach().cpu().numpy(), axis=1)
            Score_TOPK = 20#20#10
            topk, ins_indices_rgb_ir_2 = torch.topk(ins_sim_rgb_ir_2, int(Score_TOPK))#20
            ins_label_rgb_ir = torch.from_numpy(pseudo_labels_ir)[ins_indices_rgb_ir_2].detach().cpu()#.numpy()#.view(-1)
            cluster_label_rgb_ir = ins_label_rgb_ir#ins_label_rgb_ir_2#ins_label_rgb_ir_2

            
            for l in range(20):
                intersect_count=(ins_label_rgb_ir == cluster_label_rgb_ir[:,l].view(-1,1)).int().sum(1).view(-1,1).detach().cpu()
                intersect_count_list.append(intersect_count)
            intersect_count_list = torch.cat(intersect_count_list,1)
            intersect_count, _ = intersect_count_list.max(1)
            topk,cluster_label_index = torch.topk(intersect_count_list,1)
            # print('ins_label_rgb_ir',ins_label_rgb_ir)
            # print('cluster_label_rgb_ir',cluster_label_rgb_ir)
            # print('cluster_label_index',cluster_label_index.view(-1))
            cluster_label_rgb_ir = torch.gather(cluster_label_rgb_ir, dim=1, index=cluster_label_index.view(-1,1))  # cluster_label_rgb_ir[cluster_label_index.reshape(-1,1)]
            label_pair=[] #[(i,j) for i,j in zip(labels_rgb,cluster_label_rgb_ir.view(-1))]
            cluster_label_rgb_ir=cluster_label_rgb_ir.cuda()



            # index_rgb_ir=index_rgb_ir.cuda()
            # update_memory1 = trainer_interc.memory_ir.features[cluster_label_rgb_ir.view(-1)]#rgb_ratio*memory_rgb.features[key[0]] + ir_ratio*memory_ir.features[key[1]]#memory_rgb.features[key[0]]#
            # update_memory = F.normalize(update_memory, dim=-1).cuda()
            lamda_cm=0.1
            # trainer_interc.memory_rgb.features[trainer_interc.wise_memory_rgb.labels]= lamda_cm*trainer_interc.memory_rgb.features[trainer_interc.wise_memory_rgb.labels] + (1-lamda_cm)*(update_memory1)
            # trainer_interc.memory_ir.features[cluster_label_rgb_ir.view(-1)] = lamda_cm*trainer_interc.memory_ir.features[cluster_label_rgb_ir.view(-1)] + (1-lamda_cm)*(update_memory1)
            pseudo_labels_rgb=cluster_label_rgb_ir.view(-1).cpu().numpy()

            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            # print('stage2 ==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb)

            pseudo_labeled_dataset_ir = []
            # ir_label=[]
            # pseudo_real_ir = {}
            cams_ir = []
            modality_ir = []
            cross_cam=[]
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
                cams_ir.append(int(cid+4))
                modality_ir.append(int(1))
                cross_cam.append(int(cid+4))
                if label != -1:
                    pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                    # if epoch%10 == 0:
                    #     print(fname,label.item())
            print('stage2 ==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

            pseudo_labeled_dataset_rgb = []
            # rgb_label=[]
            # pseudo_real_rgb = {}
            cams_rgb = []
            modality_rgb = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
                cams_rgb.append(int(cid))
                modality_rgb.append(int(0))
                cross_cam.append(int(cid))
                if label != -1:
                    pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                    # if epoch%10 == 0:
                    #     print(fname,label.item())
            print('stage2 ==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

            rgb_label_cnt = Counter(rgb_label) #Counter(pseudo_labeled_all[idx1])
            ir_label_cnt = Counter(ir_label)#Counter(pseudo_labeled_all[idx2])
            rgb2ir_label = [(i,j) for i,j in zip(np.array(pseudo_labels_rgb_ori),np.array(pseudo_labels_rgb))]

            rgb2ir_label_cnt = Counter(rgb2ir_label)
            rgb2ir_label_cnt_sorted = sorted(rgb2ir_label_cnt.items(),key = lambda x:x[1],reverse = True)
            # print("merge_time: {}".format(time.time()-merge_time))
            in_rgb_label=[]
            in_ir_label=[]
            match_cnt = 0
            right = 0
            lenth_ratio = 1
            for i in range(int(lenth_ratio*len(rgb2ir_label_cnt_sorted))):
                key = rgb2ir_label_cnt_sorted[i][0] 
                value = rgb2ir_label_cnt_sorted[i][1]
                if key[0] == -1 or key[1] == -1:
                    continue
                if key[0] in in_rgb_label or key[1] in in_ir_label:
                    continue
                if key[0] == key[1]:
                    continue
                print('rgb-ir label{}, real rgb label{}, real ir label{}'.format(key,pseudo_real_rgb[key[0]],pseudo_real_ir[key[1]]))
                for real_ir in pseudo_real_ir[key[1]]:
                    if real_ir in pseudo_real_rgb[key[0]]:
                        right = right+1
                    match_cnt=match_cnt+1
                
                in_rgb_label.append(key[0])
                in_ir_label.append(key[1])
                # match_cnt=match_cnt+1
                label_acc = right/match_cnt
            print('match_cnt acc,lenth_ratio',match_cnt,label_acc,lenth_ratio)

            # trainer.wise_memory_rgb.labels = torch.from_numpy(pseudo_labels_rgb)
            # trainer.wise_memory_rgb.cam2uid_()
            # trainer.wise_memory_rgb.cam_mem_gen()
            cams_rgb = np.asarray(cross_cam)
            cams_ir = np.asarray(cross_cam)
            # # modality_rgb = np.asarray(modality_rgb+modality_ir)
            # # modality_ir = np.asarray(modality_ir)
            cross_cam = np.asarray(cross_cam)


            features_all = torch.cat((features_rgb,features_ir),dim=0)
            # features_all = torch.cat((trainer.wise_memory_rgb.features,trainer.wise_memory_ir.features),dim=0)
            pseudo_labels_all = torch.cat((torch.from_numpy(pseudo_labels_rgb),torch.from_numpy(pseudo_labels_ir)),dim=-1).view(-1).cpu().numpy()
            intra_id_features_rgb,intra_id_labels_rgb = camera(cross_cam,features_all,pseudo_labels_all)
            intra_id_features_ir,intra_id_labels_ir = camera(cross_cam,features_all,pseudo_labels_all)

            cluster_features_ir = generate_cluster_features(pseudo_labels_all, features_all)
            # cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
            memory_ir = ClusterMemory(model.module.in_planes*part , num_cluster_ir, temp=args.temp,
                                   momentum=args.momentum, use_hard=args.use_hard).cuda()
            # memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp,
            #                        momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            # memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

            trainer.memory_ir = memory_ir
            trainer.memory_rgb = memory_ir
            rgb_label = torch.tensor(rgb_label).view(-1,1)
            ir_label = torch.tensor(ir_label).view(1,-1)
            pos_matrix = rgb_label.eq(ir_label).view(-1)
            # print('pos_matrix',pos_matrix.size(),pos_matrix)

            rgb_pse = torch.from_numpy(pseudo_labels_rgb).view(-1,1)
            ir_pse = torch.from_numpy(pseudo_labels_ir).view(1,-1)
            pse_pos_matrix = rgb_pse.eq(ir_pse).view(-1)
            TP=0
            TN=0
            FN=0
            FP=0
            # for pred_choice,target in zip(pse_pos_matrix,pos_matrix):
            #     TP += ((pred_choice == True) & (target == True)).cpu().sum()
            #     TN += ((pred_choice == False) & (target == False)).cpu().sum()
            #     FN += ((pred_choice == False) & (target == True)).cpu().sum()
            #     FP += ((pred_choice == True) & (target == False)).cpu().sum()
            # for pred_choice,target in zip(pse_pos_matrix,pos_matrix):
            TP += ((pse_pos_matrix == True) & (pos_matrix == True)).cpu().sum()
            TN += ((pse_pos_matrix == False) & (pos_matrix == False)).cpu().sum()
            FN += ((pse_pos_matrix == False) & (pos_matrix == True)).cpu().sum()
            FP += ((pse_pos_matrix == True) & (pos_matrix == False)).cpu().sum()

            p = TP / (TP + FP)
            r = TP / (TP + FN)
            F1 = 2 * r * p / (r + p)
            acc = (TP + TN) / (TP + TN+FN+FP)
            # print('pse_pos_matrix',pse_pos_matrix.size(),pse_pos_matrix)
            # acc_cm = pos_matrix.eq(pse_pos_matrix).sum()/(pos_matrix.size(0)*pos_matrix.size(1))
            print('p',p)
            print('r',r)
            print('F1',F1)
            print('acc',acc)

        # del percam_memory_ir, percam_memory_rgb, percam_memory_all#,ins_sim_rgb_ir,sim_prob_all_rgb_ir,sim_prob_B_rgb_ir

        if epoch >= s2_cmav1:
            merge_time = time.time()
            print('merge ir and rgb momery'.format(epoch, num_cluster_ir))
            print('select_merge_data')
            label_to_images = {}
            ######jacard

            print('merge ir and rgb momery'.format(epoch, num_cluster_ir))
            print('select_merge_data')
            label_to_images = {}
            dist_cm = np.matmul(features_rgb.numpy(), np.transpose(features_ir.numpy()))
            idx1, idx2,dist_list = select_merge_data(dist_cm)
            del features_ir,features_rgb

            del dist_cm
            rgb_label_cnt = Counter(rgb_label) 
            ir_label_cnt = Counter(ir_label)
            idx_lenth = np.sum(dist_list>=0.5)
            dist_list = dist_list[:idx_lenth]
            rgb2ir_label = [(i,j) for i,j in zip(np.array(pseudo_labels_rgb)[idx1[:idx_lenth]],np.array(pseudo_labels_ir)[idx2[:idx_lenth]])]
            # print('rgb2ir_label',rgb2ir_label)
            rgb2ir_label_cnt = Counter(rgb2ir_label)
            rgb2ir_label_cnt_sorted = sorted(rgb2ir_label_cnt.items(),key = lambda x:x[1],reverse = True)
            lenth = len(rgb2ir_label_cnt_sorted)
            lamda_cm=0.1
            in_rgb_label=[]
            in_ir_label=[]
            match_cnt = 0
            right = 0
            lenth_ratio = 1.0
            for i in range(int(lenth*lenth_ratio)):
                key = rgb2ir_label_cnt_sorted[i][0] 
                value = rgb2ir_label_cnt_sorted[i][1]
                if key[0] == -1 or key[1] == -1:
                    continue
                if key[0] in in_rgb_label or key[1] in in_ir_label:
                    continue
                print('simliar:{:.3%},rgb-ir label{}, real rgb label{}, real ir label{}'.format(dist_list[i],key,pseudo_real_rgb[key[0]],pseudo_real_ir[key[1]]))
                for real_ir in pseudo_real_ir[key[1]]:
                    if real_ir in pseudo_real_rgb[key[0]]:
                        right = right+1

                update_memory = trainer.memory_ir.features[key[1]]
                trainer.memory_rgb.features[key[0]] = lamda_cm*trainer.memory_rgb.features[key[0]] + (1-lamda_cm)*(update_memory)
                trainer.memory_ir.features[key[1]] = lamda_cm*trainer.memory_ir.features[key[1]] + (1-lamda_cm)*(update_memory)
                in_rgb_label.append(key[0])
                in_ir_label.append(key[1])
                match_cnt=match_cnt+1
            label_acc = right/match_cnt
            print('s2 cma v1 match_cnt acc,lenth_ratio',match_cnt,label_acc,lenth_ratio)
            # print("merge_time: {}".format(time.time()-merge_time))

        # memory_ir.features = F.normalize(memory_ir.features, dim=1).cuda()
        # memory_rgb.features = F.normalize(memory_rgb.features, dim=1).cuda()







        # train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
        #                                 ir_batch, args.workers, args.num_instances, iters,
        #                                 trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        # train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
        #                                 rgb_batch, args.workers, args.num_instances, iters,
        #                                 trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
        
        # if epoch%2 == 0:

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                    ir_batch, args.workers, args.num_instances, iters,
                                    trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)
        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                rgb_batch, args.workers, args.num_instances, iters,
                                trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
        # else:
        #     train_loader_ir = get_train_loader_color(args, dataset_ir, args.height, args.width,
        #                                 ir_batch, args.workers, args.num_instances, iters,
        #                                 trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal,train_transformer1=transform_thermal1)
        #     train_loader_rgb = get_train_loader_ir(args, dataset_rgb, args.height, args.width,
        #                             rgb_batch, args.workers, args.num_instances, iters,
        #                             trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb_1)


        # cams_rgb = np.asarray(cams_rgb)
        # cams_ir = np.asarray(cams_ir)
        # modality_rgb = np.asarray(modality_rgb)
        # modality_ir = np.asarray(modality_ir)


        # intra_id_features_modality_rgb,intra_id_labels_modality_rgb = camera(modality_rgb,features_rgb,pseudo_labels_rgb)
        # intra_id_features_modality_ir,intra_id_labels_modality_ir = camera(modality_ir,features_ir,pseudo_labels_ir)
        del features_ir,features_rgb
        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()
        intra_id_features_all,intra_id_labels_all = [],[] #camera(modality_rgb,features_all,pseudo_labels_all)

        intra_id_features_ccam,intra_id_labels_ccam = [],[]#camera(cross_cam,features_all,pseudo_labels_all)
        # trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
        #     intra_id_labels_rgb=intra_id_labels_rgb, intra_id_features_rgb=intra_id_features_rgb,intra_id_labels_ir=intra_id_labels_ir, intra_id_features_ir=intra_id_features_ir,
        #     all_label_rgb=pseudo_labels_rgb,all_label_ir=pseudo_labels_ir,cams_ir=cams_ir,cams_rgb=cams_rgb,
        #               print_freq=args.print_freq, train_iters=len(train_loader_ir))
        # torch.cuda.empty_cache()
        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,all_label=pseudo_labels_all,intra_id_labels_all=intra_id_labels_all,cams_all=modality_rgb,intra_id_features_all=intra_id_features_all,
            intra_id_labels_rgb=intra_id_labels_rgb, intra_id_features_rgb=intra_id_features_rgb,intra_id_labels_ir=intra_id_labels_ir, intra_id_features_ir=intra_id_features_ir,
            all_label_rgb=pseudo_labels_rgb,all_label_ir=pseudo_labels_ir,cams_ir=cams_ir,cams_rgb=cams_rgb,cross_cam=cross_cam,intra_id_features_crosscam=intra_id_features_ccam,intra_id_labels_crosscam=intra_id_labels_ccam,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

############################stage3
        # if epoch>1145:#30:
        # if epoch<=1140:#30:
        if epoch>=s3:#30:
            # model.module.rgb_softmax_dim=[trainer.memory_rgb.features.data.size(0)]
            # model.module.ir_softmax_dim=[trainer.memory_ir.features.data.size(0)]

            # distribute_cm_map_rgb = F.normalize(trainer.memory_rgb.features.data,dim=1)
            # distribute_cm_map_ir = F.normalize(trainer.memory_ir.features.data,dim=1)#cluster_features_ir#
            # distribute_cm_map = torch.cat((distribute_cm_map_rgb, distribute_cm_map_ir), dim=0)
            # # distribute_cm_map = F.normalize(distribute_cm_map) 
            # model.module.classifier_rgb = nn.Linear(768*part, distribute_cm_map.size(0), bias=False).cuda()
            # model.module.classifier_rgb.weight.data.copy_(distribute_cm_map.cuda())

            # model.module.classifier_ir = nn.Linear(768*part, distribute_cm_map.size(0), bias=False).cuda()
            # model.module.classifier_ir.weight.data.copy_(distribute_cm_map.cuda())


            with torch.no_grad():
                if epoch == 0:
                    # DBSCAN cluster
                    # ir_eps = 0.6
                    # print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                    # cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                    # rgb_eps = 0.6#+0.1
                    # print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                    # cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                    print('all Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                    cluster_all = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)


                print('==> Create pseudo labels for unlabeled RGB data')

                cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                 256, args.workers, 
                                                 testset=sorted(dataset_rgb.train))
                features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)

                features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

                
                print('==> Create pseudo labels for unlabeled IR data')
                cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                 256, args.workers, 
                                                 testset=sorted(dataset_ir.train))
                features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)

                features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

                features_all = torch.cat([features_rgb,features_ir], 0)
                features_all_ = F.normalize(features_all, dim=1)
                features_rgb_ =F.normalize(features_rgb, dim=1)
                features_ir_ =F.normalize(features_ir, dim=1)

                rerank_dist_all_jacard = compute_jaccard_distance(features_all_, k1=args.k1, k2=args.k2,search_option=3)#args.k1
                pseudo_labels_all = cluster_all.fit_predict(rerank_dist_all_jacard)
                # rerank_dist_cm = rerank_dist_all_jacard[:features_rgb.size(0),features_rgb.size(0):]
                pseudo_labels_rgb=pseudo_labels_all[:features_rgb.size(0)]
                pseudo_labels_ir=pseudo_labels_all[features_rgb.size(0):]

                del rerank_dist_all_jacard
                num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
                num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
                # print("epoch: {} \n pseudo_labels: {}".format(epoch, pseudo_labels.tolist()[:100]))
            # generate new dataset and calculate cluster centers
            @torch.no_grad()
            def generate_cluster_features(labels, features):
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

            cluster_features_all = generate_cluster_features(pseudo_labels_all, features_all)

            num_cluster_all = len(set(pseudo_labels_all)) - (1 if -1 in pseudo_labels_all else 0)
            
            memory_all = ClusterMemory(model.module.in_planes*part, num_cluster_all, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()
            memory_all.features = F.normalize(cluster_features_all, dim=1).cuda()

            # trainer_interm.memory_ir = memory_all
            trainer_interm.memory_rgb = memory_all



    #############memory wise

            # wise_memory_all = Memory_wise_v3(model.module.in_planes*part, len(dataset_rgb.train)+len(dataset_ir.train),num_cluster_all,temp=args.temp, momentum=args.momentum).cuda()
            # wise_memory_all.features = F.normalize(features_all, dim=1).cuda()


            # # # pseudo_labels_ir = generate_pseudo_labels(pseudo_labels_ir, features_ir.clone())
            # # nameMap_all = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_rgb.train)+sorted(dataset_ir.train))}

            # wise_memory_all.labels =  torch.from_numpy(pseudo_labels_all)#.cuda() #pseudo_labels_rgb.cuda()#

            # trainer_interm.wise_memory_all = wise_memory_all

            # wise_memory_rgb = Memory_wise_v3(model.module.in_planes*part, len(dataset_rgb.train),num_cluster_all,temp=args.temp, momentum=args.momentum).cuda()
            # wise_memory_rgb.features = F.normalize(features_rgb, dim=1).cuda()
            # # pseudo_labels_ir = generate_pseudo_labels(pseudo_labels_ir, features_ir.clone())
            # nameMap_rgb = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_rgb.train))}
            # wise_memory_rgb.labels =  torch.from_numpy(pseudo_labels_rgb)#.cuda() #pseudo_labels_rgb.cuda()#
            # trainer_interm.wise_memory_rgb = wise_memory_rgb


            # wise_memory_ir = Memory_wise_v3(model.module.in_planes*part, len(dataset_ir.train),num_cluster_all,temp=args.temp, momentum=args.momentum).cuda()
            # wise_memory_ir.features = F.normalize(features_ir, dim=1).cuda()
            # # # pseudo_labels_ir = generate_pseudo_labels(pseudo_labels_ir, features_ir.clone())
            # nameMap_ir = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_ir.train))}
            # wise_memory_ir.labels =  torch.from_numpy(pseudo_labels_ir)#.cuda() #pseudo_labels_rgb.cuda()#
            # trainer_interm.wise_memory_ir = wise_memory_ir


            # trainer_interm.nameMap_rgb=nameMap_rgb
            # trainer_interm.nameMap_ir=nameMap_ir

    ###################################

            pseudo_labeled_dataset_rgb = []
            rgb_label=[]
            pseudo_real_rgb = {}
            cams_rgb = []
            modality_rgb = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
                cams_rgb.append(int(0))
                # cams_rgb.append(int(cid))
                modality_rgb.append(0)
                if label != -1:
                    pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                    rgb_label.append(label.item())
                    pseudo_real_rgb[label.item()] = pseudo_real_rgb.get(label.item(),[])+[_]
                    pseudo_real_rgb[label.item()] = list(set(pseudo_real_rgb[label.item()]))
                    # if epoch%10 == 0:
                    #     print(fname,label.item())
            print('stage3 ==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))


            pseudo_labeled_dataset_ir = []
            ir_label=[]
            pseudo_real_ir = {}
            cams_ir = []
            modality_ir = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
                cams_ir.append(int(1))
                # cams_ir.append(int(cid+4))
                modality_ir.append(1)
                if label != -1:
                    pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                    ir_label.append(label.item())
                    pseudo_real_ir[label.item()] = pseudo_real_ir.get(label.item(),[])+[_]
                    pseudo_real_ir[label.item()] = list(set(pseudo_real_ir[label.item()]))
                    # if epoch%10 == 0:
                    #     print(fname,label.item())
            print('stage3 ==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))




            modality_rgb = np.asarray(modality_rgb+modality_ir)



#############cma v1
            if epoch>=s3_cma:#15:

                feat_rgb = features_all_[modality_rgb==0]
                feat_ir = features_all_[modality_rgb!=0]
                print(feat_rgb.size(),feat_ir.size())
                pseudo_labels_rgb_ = pseudo_labels_all[modality_rgb==0]
                pseudo_labels_ir_= pseudo_labels_all[modality_rgb!=0]

                # print('merge ir and rgb momery'.format(epoch, num_cluster_ir))
                # print('select_merge_data')
                label_to_images = {}
                dist_cm = np.matmul(feat_rgb.numpy(), np.transpose(feat_ir.numpy()))
                idx1, idx2,dist_list = select_merge_data(dist_cm)
                # del features_ir,features_rgb

                # del dist_cm
                # rgb_label_cnt = Counter(rgb_label) 
                # ir_label_cnt = Counter(ir_label)
                idx_lenth = np.sum(dist_list>=0.5)
                dist_list = dist_list[:idx_lenth]
                rgb2ir_label = [(i,j) for i,j in zip(np.array(pseudo_labels_rgb_)[idx1[:idx_lenth]],np.array(pseudo_labels_ir_)[idx2[:idx_lenth]])]
                # print('rgb2ir_label',rgb2ir_label)
                rgb2ir_label_cnt = Counter(rgb2ir_label)
                rgb2ir_label_cnt_sorted = sorted(rgb2ir_label_cnt.items(),key = lambda x:x[1],reverse = True)
                lenth = len(rgb2ir_label_cnt_sorted)
                lamda_cm=0.1
                in_rgb_label=[]
                in_ir_label=[]
                match_cnt = 0
                right = 0
                lenth_ratio = 1.0
                for i in range(int(lenth*lenth_ratio)):
                    key = rgb2ir_label_cnt_sorted[i][0] 
                    value = rgb2ir_label_cnt_sorted[i][1]
                    if key[0] == -1 or key[1] == -1:
                        continue
                    if key[0] in in_rgb_label or key[1] in in_ir_label:
                        continue
                    print('rgb-ir label{}, real rgb label{}, real ir label{}'.format(key,pseudo_real_rgb[key[0]],pseudo_real_ir[key[1]]))
                    for real_ir in pseudo_real_ir[key[1]]:
                        if real_ir in pseudo_real_rgb[key[0]]:
                            right = right+1
                        match_cnt=match_cnt+1
                    update_memory = trainer_interm.memory_rgb.features[key[1]]
                    trainer_interm.memory_rgb.features[key[0]] = F.normalize(( lamda_cm*trainer_interm.memory_rgb.features[key[0]] + (1-lamda_cm)*(update_memory)),dim=-1 )
                    trainer_interm.memory_rgb.features[key[1]] = F.normalize(( lamda_cm*trainer_interm.memory_rgb.features[key[1]] + (1-lamda_cm)*(update_memory)),dim=-1 )
                    # pseudo_labels_all[pseudo_labels_all==key[0]]=key[1]
                    in_rgb_label.append(key[0])
                    in_ir_label.append(key[1])
                    # match_cnt=match_cnt+1
                label_acc = right/match_cnt
                print('s3 match_cnt acc,lenth_ratio',match_cnt,label_acc,lenth_ratio)


            trainer_interm.wise_memory_all.cam = torch.Tensor(cams_rgb+cams_ir)
            trainer_interm.wise_memory_all.cam2uid_()
            trainer_interm.wise_memory_all.cam_mem_gen()
    ##################################


            train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,#128
                                            ir_batch, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)#256

            train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                            rgb_batch, args.workers, args.num_instances, iters,
                                            trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
            


        # if epoch%2 == 0: 

            train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        ir_batch, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)
            train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                    rgb_batch, args.workers, args.num_instances, iters,
                                    trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
        # else:
        #     train_loader_ir = get_train_loader_color(args, dataset_ir, args.height, args.width,
        #                                 ir_batch, args.workers, args.num_instances, iters,
        #                                 trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal,train_transformer1=transform_thermal1)
        #     train_loader_rgb = get_train_loader_ir(args, dataset_rgb, args.height, args.width,
        #                             rgb_batch, args.workers, args.num_instances, iters,
        #                             trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb_1)



            cams_rgb = np.asarray(cams_rgb+cams_ir)
            cams_ir = np.asarray(cams_ir)
            
            modality_ir = np.asarray(modality_ir)
            intra_id_features_rgb,intra_id_labels_rgb =camera(cams_rgb,features_all,pseudo_labels_all)#camera(cams_rgb,features_rgb,pseudo_labels_rgb)
            intra_id_features_ir,intra_id_labels_ir = [],[]#camera(modality_ir,features_ir,pseudo_labels_ir)#camera(cams_rgb,features_rgb,pseudo_labels_rgb)

            intra_id_features_modality_rgb,intra_id_labels_modality_rgb = _,_#camera(modality_rgb,features_rgb,pseudo_labels_rgb)
            intra_id_features_modality_ir,intra_id_labels_modality_ir =_,_#camera(modality_ir,features_ir,pseudo_labels_ir)
            # del features_ir,features_rgb,features_all
            train_loader_ir.new_epoch()
            time.sleep(1)
            train_loader_rgb.new_epoch()
            time.sleep(1)
            torch.cuda.empty_cache()
            trainer_interm.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                intra_id_labels_rgb=intra_id_labels_rgb, intra_id_features_rgb=intra_id_features_rgb,intra_id_labels_ir=intra_id_labels_ir, intra_id_features_ir=intra_id_features_ir,
                all_label_rgb=pseudo_labels_rgb,all_label_ir=pseudo_labels_ir,cams_ir=cams_ir,cams_rgb=cams_rgb,all_label=pseudo_labels_all,
                          print_freq=args.print_freq, train_iters=len(train_loader_ir))
            del memory_all 

        if epoch>=0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
            _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path='/dat01/yangbin/data/sysu'
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(10):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
                ######match
                # match_score_min,match_score_max = pairwise_distance_matcher(model.module.matcher, torch.from_numpy(query_feat_fc).cuda() , torch.from_numpy(gall_feat_fc).cuda(), gal_batch_size=64, prob_batch_size=2048)
                # cmc_match_min, mAP_match_min, mINP_match_min = eval_sysu(-match_score_min, query_label, gall_label, query_cam, gall_cam)
                # cmc_match_max, mAP_match_max, mINP_match_max = eval_sysu(-match_score_max, query_label, gall_label, query_cam, gall_cam)
                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
                # print(
                #     'cmc_match_min:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                #         cmc_match_min[0], cmc_match_min[4], cmc_match_min[9], cmc_match_min[19], mAP_match_min, mINP_match_min))
                # print(
                #     'cmc_match_max:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                #         cmc_match_max[0], cmc_match_max[4], cmc_match_max[9], cmc_match_max[19], mAP_match_max, mINP_match_max))


            cmc = all_cmc / 10
            mAP = all_mAP / 10
            mINP = all_mINP / 10
            print('All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
#################################
            is_best = (cmc[0] > best_mAP)
            best_mAP = max(cmc[0], best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': cmc[0],
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            # is_best = (mAP > best_mAP)
            # best_mAP = max(mAP, best_mAP)
            # save_checkpoint({
            #     'state_dict': model.state_dict(),
            #     'epoch': epoch + 1,
            #     'best_mAP': best_mAP,
            # }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            # save_checkpoint_match(matcher_rgb, is_best, fpath=osp.join(args.logs_dir, 'matcher_rgb_checkpoint.pkl'),match='rgb')

            # save_checkpoint_match(matcher_ir, is_best, fpath=osp.join(args.logs_dir, 'matcher_ir_checkpoint.pkl'),match='ir')
            # save_checkpoint_match({
            #     'state_dict': matcher_rgb.state_dict(),
            #     'epoch': epoch + 1,
            #     'best_mAP': best_mAP,
            # }, is_best, fpath=osp.join(args.logs_dir, 'matcher_rgb_checkpoint.pth.tar'),match='rgb')

            # save_checkpoint_match({
            #     'state_dict': matcher_ir.state_dict(),
            #     'epoch': epoch + 1,
            #     'best_mAP': best_mAP,
            # }, is_best, fpath=osp.join(args.logs_dir, 'matcher_ir_checkpoint.pth.tar'),match='ir')


            print('\n * Finished epoch {:3d}  model r1: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, cmc[0], best_mAP, ' *' if is_best else ''))
############################
        lr_scheduler.step()
        # if epoch >=21:
        #     break
    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
    _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
    mode='all'
    data_path='/dat01/yangbin/data/sysu'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('all search All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))


    mode='indoor'
    data_path='/dat01/yangbin/data/sysu'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
    nquery = len(query_label)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
        ngall = len(gall_label)
        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP

        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP


        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    cmc = all_cmc / 10
    mAP = all_mAP / 10
    mINP = all_mINP / 10
    print('indoor All Average:')
    print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

#################################
    # is_best = (mAP > best_mAP)
    # best_mAP = max(mAP, best_mAP)
    # save_checkpoint({
    #     'state_dict': model.state_dict(),
    #     'epoch': epoch + 1,
    #     'best_mAP': best_mAP,
    # }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

    # print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
    #       format(epoch, mAP, best_mAP, ' *' if is_best else ''))
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))






#####ori
# def main_worker(args,cfg):
#     global start_epoch, best_mAP
#     log_name='sysu_all_vit_base_2p_288_4part_matchv1_nodet'#_querymatch
#     args.logs_dir = osp.join(args.logs_dir+'/'+log_name)
#     start_time = time.monotonic()

#     cudnn.benchmark = True

#     sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
#     print("==========\nArgs:{}\n==========".format(args))
#     print("Loaded configuration file {}".format(args.config_file))
#     with open(args.config_file, 'r') as cf:
#         config_str = "\n" + cf.read()
#     print(config_str)
#     # Create datasets
#     iters = args.iters if (args.iters > 0) else None
#     print("==> Load unlabeled dataset")
#     dataset_ir = get_data('sysu_ir', args.data_dir)
#     dataset_rgb = get_data('sysu_rgb', args.data_dir)

#     test_loader_ir = get_test_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers)
#     test_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers)
#     # Create model
#     # model = create_model(args)
#     model = make_model(cfg, num_class=0, camera_num=0, view_num = 0)

#     model.cuda()
#     model = nn.DataParallel(model)#,output_device=1)

#     # optimizer = make_optimizer(cfg, model)
#     # checkpoint = load_checkpoint(osp.join('/dat01/yangbin/cluster-contrast-reid-camera/logs/sysu_all_resnet_pretrain_camera0.1_cma', 'model_best.pth.tar'))
#     # checkpoint = load_checkpoint(osp.join('/dat01/yangbin/cluster-contrast-reid-camera/logs/sysu_all_resnet_pretrain_camera0.1', 'model_best.pth.tar'))
#     # model.load_state_dict(checkpoint['state_dict'])
#     # Evaluator
#     params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
#     # print('optimizer: %s'%(args.optimizer))
#     # if args.optimizer == 'Adam':
#     #     optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
#     # if args.optimizer == 'AdamW':
#     #     optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
#     # elif args.optimizer == 'SGD':
#     optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

#     # if cfg.SOLVER.WARMUP_METHOD == 'cosine':
#     #     print('===========using cosine learning rate=======')
#     #     lr_scheduler = create_scheduler(cfg, optimizer)
#     # else:
#     #     print('===========using normal learning rate=======')
#     #     lr_scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
#     #                                   cfg.SOLVER.WARMUP_FACTOR,
#     #                                   cfg.SOLVER.WARMUP_EPOCHS, cfg.SOLVER.WARMUP_METHOD)
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

#     evaluator = Evaluator(model)

#     # # Optimizer
#     # params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
#     # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
#     # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

#     # lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
#     #                                  warmup_iters=args.warmup_step)

#     # Trainer
#     trainer = ClusterContrastTrainer_pretrain_joint(model)

#     for epoch in range(args.epochs):
#         # if epoch == 50:
#         #     optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
#         #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
#         with torch.no_grad():
#             if epoch == 0:
#                 # DBSCAN cluster
#                 ir_eps = 0.6
#                 print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
#                 cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
#                 rgb_eps = 0.6#+0.1
#                 print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
#                 cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

#             print('==> Create pseudo labels for unlabeled RGB data')

#             cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
#                                              256, args.workers, 
#                                              testset=sorted(dataset_rgb.train))
#             features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
#             del cluster_loader_rgb,
#             features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            
#             print('==> Create pseudo labels for unlabeled IR data')
#             cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
#                                              256, args.workers, 
#                                              testset=sorted(dataset_ir.train))
#             features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
#             del cluster_loader_ir
#             features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)


#             all_feature = []#torch.cat([features_rgb,features_ir], 0)
#             # rerank_dist_all_jacard = compute_jaccard_distance(all_feature, k1=args.k1, k2=args.k2,search_option=3)
#             # rerank_dist_cm = rerank_dist_all_jacard[:features_rgb.size(0),features_rgb.size(0):]


#             rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
#             pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
#             rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
#             pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
#             del rerank_dist_rgb
#             del rerank_dist_ir
#             num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
#             num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
#             # print("epoch: {} \n pseudo_labels: {}".format(epoch, pseudo_labels.tolist()[:100]))

#         # generate new dataset and calculate cluster centers
#         @torch.no_grad()
#         def generate_cluster_features(labels, features):
#             centers = collections.defaultdict(list)
#             for i, label in enumerate(labels):
#                 if label == -1:
#                     continue
#                 centers[labels[i]].append(features[i])

#             centers = [
#                 torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
#             ]

#             centers = torch.stack(centers, dim=0)
#             return centers

#         cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
#         cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
#         # del features, cluster_loader_1,cluster_loader_2,cluster_loader_3,cluster_loader_4,features_1,features_2,features_3,features_4
#         # del features_ir,features_rgb
#         # Create hybrid memory
#         # if epoch >= 25:
#         #     memory_ir = ClusterMemory_all(model.module.num_features, num_cluster_ir, temp=args.temp,
#         #                            momentum=args.momentum, use_hard=args.use_hard).cuda()
#         #     memory_rgb = ClusterMemory_all(model.module.num_features, num_cluster_rgb, temp=args.temp,
#         #                            momentum=args.momentum, use_hard=args.use_hard).cuda()
#         memory_ir = ClusterMemory(model.module.in_planes*5, num_cluster_ir, temp=args.temp,
#                                momentum=args.momentum, use_hard=args.use_hard).cuda()
#         memory_rgb = ClusterMemory(model.module.in_planes*5, num_cluster_rgb, temp=args.temp,
#                                momentum=args.momentum, use_hard=args.use_hard).cuda()
#         memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
#         memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

#         trainer.memory_ir = memory_ir
#         trainer.memory_rgb = memory_rgb


#         wise_memory_rgb = Memory_wise_v3(768*5, len(dataset_rgb.train),num_cluster_rgb,temp=args.temp, momentum=args.momentum).cuda()
#         wise_memory_ir = Memory_wise_v3(768*5, len(dataset_ir.train),num_cluster_ir,temp=args.temp, momentum=args.momentum).cuda()
#         wise_memory_ir.features = F.normalize(features_ir, dim=1).cuda()
#         wise_memory_rgb.features = F.normalize(features_rgb, dim=1).cuda()

#         # pseudo_labels_ir = generate_pseudo_labels(pseudo_labels_ir, features_ir.clone())
#         nameMap_ir = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_ir.train))}

#         # pseudo_labels_rgb = generate_pseudo_labels(pseudo_labels_rgb, features_rgb.clone())
#         nameMap_rgb = {val[0]: idx for (idx, val) in enumerate(sorted(dataset_rgb.train))}

#         wise_memory_rgb.labels =  torch.from_numpy(pseudo_labels_rgb)#.cuda() #pseudo_labels_rgb.cuda()#
#         wise_memory_ir.labels = torch.from_numpy(pseudo_labels_ir)#.cuda() #pseudo_labels_ir.cuda()#

#         trainer.wise_memory_ir = wise_memory_ir
#         trainer.wise_memory_rgb = wise_memory_rgb
#         trainer.nameMap_ir=nameMap_ir
#         trainer.nameMap_rgb=nameMap_rgb



#         pseudo_labeled_dataset_ir = []
#         ir_label=[]
#         pseudo_real_ir = {}
#         cams_ir = []
#         modality_ir = []
#         outlier=0
#         for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
#             cams_ir.append(cid)
#             modality_ir.append(1)
#             if label != -1:
#                 pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
#                 ir_label.append(label.item())
#                 pseudo_real_ir[label.item()] = pseudo_real_ir.get(label.item(),[])+[_]
#                 pseudo_real_ir[label.item()] = list(set(pseudo_real_ir[label.item()]))
#                 if epoch%10 == 0:
#                     print(fname,label.item())
#             else:
#                 outlier=outlier+1
#         print('==> Statistics for IR epoch {}: {} clusters outlier {}'.format(epoch, num_cluster_ir,outlier))

#         pseudo_labeled_dataset_rgb = []
#         rgb_label=[]
#         pseudo_real_rgb = {}
#         cams_rgb = []
#         modality_rgb = []
#         outlier=0
#         for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
#             cams_rgb.append(cid)
#             modality_rgb.append(0)
#             if label != -1:
#                 pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
#                 rgb_label.append(label.item())
#                 pseudo_real_rgb[label.item()] = pseudo_real_rgb.get(label.item(),[])+[_]
#                 pseudo_real_rgb[label.item()] = list(set(pseudo_real_rgb[label.item()]))
#                 if epoch%10 == 0:
#                     print(fname,label.item())
#             else:
#                 outlier=outlier+1
#         print('==> Statistics for RGB epoch {}: {} clusters outlier {} '.format(epoch, num_cluster_rgb,outlier))

# ###################################matcher+cma
#         if epoch >= 1000:
#             merge_time = time.time()
#             print('merge ir and rgb momery'.format(epoch, num_cluster_ir))
#             print('select_merge_data')
#             label_to_images = {}
#             ######jacard
            
#             # idx1, idx2,dist_list = select_merge_data_jacard(all_feature, pseudo_labels_ir, label_to_images, 0.005,rerank_dist_cm,features_rgb.size(0),features_ir.size(0))
#             ##### cosin
#             rerank_dist_cm = np.matmul(features_rgb.numpy(), np.transpose(features_ir.numpy()))
#             idx1, idx2,dist_list = select_merge_data(all_feature, pseudo_labels_ir, label_to_images, 0.005,rerank_dist_cm,features_rgb.size(0),features_ir.size(0))
#             print("merge_time: {}".format(time.time()-merge_time))
#             # del features_ir,features_rgb
#             print(idx1.shape,idx2.shape)
#             del rerank_dist_cm,all_feature
#             rgb_label_cnt = Counter(rgb_label) #Counter(pseudo_labeled_all[idx1])
#             ir_label_cnt = Counter(ir_label)#Counter(pseudo_labeled_all[idx2])
#             ##cosin
#             idx_lenth = np.sum(dist_list>=0.5)
#             ############jacard
#             # idx_lenth = np.sum(dist_list<=1.0)
#             ############
#             print(idx_lenth)
#             dist_list = dist_list[:idx_lenth]
#             print('dist_list > 0.0',dist_list[:idx_lenth])
#             print("simliar len",idx_lenth)
#             rgb2ir_label = [(i,j) for i,j in zip(np.array(pseudo_labels_rgb)[idx1[:idx_lenth]],np.array(pseudo_labels_ir)[idx2[:idx_lenth]])]
#             # print('rgb2ir_label',rgb2ir_label)
#             rgb2ir_label_cnt = Counter(rgb2ir_label)
#             rgb2ir_label_cnt_sorted = sorted(rgb2ir_label_cnt.items(),key = lambda x:x[1],reverse = True)
#             print("merge_time: {}".format(time.time()-merge_time))

#             lenth = len(rgb2ir_label_cnt_sorted)
#             lamda_cm=0.1
#             print(lenth)

#             in_rgb_label=[]
#             in_ir_label=[]
#             match_cnt = 0
#             right = 0
#             lenth_ratio = 1.0
#             # label_acc =0
#             # if label_acc >= 0.95:
#             #     lenth_ratio = lenth_ratio+0.1
#             # lenth_ratio = max(1,lenth_ratio)
#             for i in range(int(lenth*lenth_ratio)):
#                 key = rgb2ir_label_cnt_sorted[i][0] 
#                 value = rgb2ir_label_cnt_sorted[i][1]
#                 if key[0] == -1 or key[1] == -1:
#                     continue
#                 if key[0] in in_rgb_label or key[1] in in_ir_label:
#                     continue
#                 print('simliar:{:.3%},rgb-ir label{}, real rgb label{}, real ir label{}'.format(dist_list[i],key,pseudo_real_rgb[key[0]],pseudo_real_ir[key[1]]))
#                 for real_ir in pseudo_real_ir[key[1]]:
#                     if real_ir in pseudo_real_rgb[key[0]]:
#                         right = right+1
#                 rgb_ratio = rgb_label_cnt[key[0]] / rgb_label_cnt[key[0]]+ir_label_cnt[key[1]]
#                 ir_ratio = ir_label_cnt[key[1]] / rgb_label_cnt[key[0]]+ir_label_cnt[key[1]]
#                 # print(value,rgb_label_cnt[key[0]],ir_label_cnt[key[1]])
#                 # print(rgb_ratio,ir_ratio)
#                 update_memory = memory_ir.features[key[1]]#rgb_ratio*memory_rgb.features[key[0]] + ir_ratio*memory_ir.features[key[1]]#memory_rgb.features[key[0]]#
#                 # update_memory = F.normalize(update_memory, dim=-1).cuda()
#                 memory_rgb.features[key[0]] = lamda_cm*memory_rgb.features[key[0]] + (1-lamda_cm)*(update_memory)
#                 memory_ir.features[key[1]] = lamda_cm*memory_ir.features[key[1]] + (1-lamda_cm)*(update_memory)
#                 in_rgb_label.append(key[0])
#                 in_ir_label.append(key[1])
#                 match_cnt=match_cnt+1
#                 label_acc = right/match_cnt
#             print('match_cnt acc,lenth_ratio',match_cnt,label_acc,lenth_ratio)
#             print("merge_time: {}".format(time.time()-merge_time))

#         # memory_ir.features = F.normalize(memory_ir.features, dim=1).cuda()
#         # memory_rgb.features = F.normalize(memory_rgb.features, dim=1).cuda()
#         color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)#T.

#         normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#         height=args.height
#         width=args.width
#         train_transformer_rgb = T.Compose([
#         color_aug,
#         T.Resize((height, width), interpolation=3),
#         T.Pad(10),
#         T.RandomCrop((height, width)),
#         T.RandomHorizontalFlip(p=0.5),
#         T.ToTensor(),
#         normalizer,
#         ChannelRandomErasing(probability = 0.5)
#         ])

#         train_transformer_rgb1 = T.Compose([
#         color_aug,
#         T.Resize((height, width), interpolation=3),
#         T.Pad(10),
#         T.RandomCrop((height, width)),
#         T.RandomHorizontalFlip(p=0.5),
#         T.ToTensor(),
#         normalizer,
#         ChannelRandomErasing(probability = 0.5),
#         ChannelExchange(gray = 2)
#         ])

#         transform_thermal = T.Compose( [
#             color_aug,

#             T.Resize((height, width), interpolation=3),
#             T.Pad(10),
#             T.RandomCrop((height, width)),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             normalizer,
#             ChannelRandomErasing(probability = 0.5),
#             ChannelAdapGray(probability =0.5)
#             ])
#         transform_thermal1 = T.Compose( [
#             color_aug,
#             T.Resize((height, width), interpolation=3),
#             T.Pad(10),
#             T.RandomCrop((height, width)),
#             T.RandomHorizontalFlip(),
#             T.ToTensor(),
#             normalizer,
#             ChannelRandomErasing(probability = 0.5),
#             ChannelAdapGray(probability =0.5)])

#         # train_loader_ir = get_train_loader_color(args, dataset_ir, args.height, args.width,
#         #                                 128, args.workers, args.num_instances, iters,
#         #                                 trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal,train_transformer1=transform_thermal1)
#         train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
#                                         128, args.workers, args.num_instances, iters,
#                                         trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

#         train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
#                                         128, args.workers, args.num_instances, iters,
#                                         trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)
#         cams_rgb = np.asarray(cams_rgb)
#         cams_ir = np.asarray(cams_ir)
#         modality_rgb = np.asarray(modality_rgb)
#         modality_ir = np.asarray(modality_ir)
#         intra_id_features_rgb,intra_id_labels_rgb = camera(cams_rgb,features_rgb,pseudo_labels_rgb)
#         intra_id_features_ir,intra_id_labels_ir = camera(cams_ir,features_ir,pseudo_labels_ir)

#         intra_id_features_modality_rgb,intra_id_labels_modality_rgb = camera(modality_rgb,features_rgb,pseudo_labels_rgb)
#         intra_id_features_modality_ir,intra_id_labels_modality_ir = camera(modality_ir,features_ir,pseudo_labels_ir)
#         del features_ir,features_rgb
#         train_loader_ir.new_epoch()
#         train_loader_rgb.new_epoch()

#         trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
#             intra_id_labels_rgb=intra_id_labels_rgb, intra_id_features_rgb=intra_id_features_rgb,intra_id_labels_ir=intra_id_labels_ir, intra_id_features_ir=intra_id_features_ir,
#             all_label_rgb=pseudo_labels_rgb,all_label_ir=pseudo_labels_ir,cams_ir=cams_ir,cams_rgb=cams_rgb,
#                       print_freq=args.print_freq, train_iters=len(train_loader_ir))

#         if epoch>=0 and ( (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
#             _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
#             _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
# ##############################
#             args.test_batch=64
#             args.img_w=args.width
#             args.img_h=args.height
#             normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#             transform_test = T.Compose([
#                 T.ToPILImage(),
#                 T.Resize((args.img_h,args.img_w)),
#                 T.ToTensor(),
#                 normalize,
#             ])
#             mode='all'
#             data_path='/dat01/yangbin/data/sysu'
#             query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
#             nquery = len(query_label)
#             queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
#             query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
#             query_feat_fc = extract_query_feat(model,query_loader,nquery)
#             for trial in range(1):
#                 gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
#                 ngall = len(gall_label)
#                 trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
#                 trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

#                 gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

#                 # fc feature
#                 distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
#                 cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

#                 if trial == 0:
#                     all_cmc = cmc
#                     all_mAP = mAP
#                     all_mINP = mINP

#                 else:
#                     all_cmc = all_cmc + cmc
#                     all_mAP = all_mAP + mAP
#                     all_mINP = all_mINP + mINP

#                 print('Test Trial: {}'.format(trial))
#                 print(
#                     'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#                         cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

#             cmc = all_cmc / 1
#             mAP = all_mAP / 1
#             mINP = all_mINP / 1
#             print('All Average:')
#             print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#                     cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
# #################################
#             is_best = (mAP > best_mAP)
#             best_mAP = max(mAP, best_mAP)
#             save_checkpoint({
#                 'state_dict': model.state_dict(),
#                 'epoch': epoch + 1,
#                 'best_mAP': best_mAP,
#             }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

#             print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
#                   format(epoch, mAP, best_mAP, ' *' if is_best else ''))
# ############################
#         lr_scheduler.step()

#     print('==> Test with the best model:')
#     checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
#     model.load_state_dict(checkpoint['state_dict'])
#     _,mAP_homo = evaluator.evaluate(test_loader_ir, dataset_ir.query, dataset_ir.gallery, cmc_flag=True,modal=2)
#     _,mAP_homo = evaluator.evaluate(test_loader_rgb, dataset_rgb.query, dataset_rgb.gallery, cmc_flag=True,modal=1)
#     mode='all'
#     data_path='/dat01/yangbin/data/sysu'
#     query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
#     nquery = len(query_label)
#     queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
#     query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
#     query_feat_fc = extract_query_feat(model,query_loader,nquery)
#     for trial in range(10):
#         gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
#         ngall = len(gall_label)
#         trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
#         trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

#         gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
#         # fc feature
#         distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

#         cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
#         if trial == 0:
#             all_cmc = cmc
#             all_mAP = mAP
#             all_mINP = mINP

#         else:
#             all_cmc = all_cmc + cmc
#             all_mAP = all_mAP + mAP
#             all_mINP = all_mINP + mINP


#         print('Test Trial: {}'.format(trial))
#         print(
#             'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#                 cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
#     cmc = all_cmc / 10
#     mAP = all_mAP / 10
#     mINP = all_mINP / 10
#     print('all search All Average:')
#     print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#             cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))


#     mode='indoor'
#     data_path='/dat01/yangbin/data/sysu'
#     query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
#     nquery = len(query_label)
#     queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
#     query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
#     query_feat_fc = extract_query_feat(model,query_loader,nquery)
#     for trial in range(10):
#         gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
#         ngall = len(gall_label)
#         trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
#         trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

#         gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)
#         # fc feature
#         distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

#         cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
#         if trial == 0:
#             all_cmc = cmc
#             all_mAP = mAP
#             all_mINP = mINP

#         else:
#             all_cmc = all_cmc + cmc
#             all_mAP = all_mAP + mAP
#             all_mINP = all_mINP + mINP


#         print('Test Trial: {}'.format(trial))
#         print(
#             'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#                 cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
#     cmc = all_cmc / 10
#     mAP = all_mAP / 10
#     mINP = all_mINP / 10
#     print('indoor All Average:')
#     print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
#             cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

# #################################
#     # is_best = (mAP > best_mAP)
#     # best_mAP = max(mAP, best_mAP)
#     # save_checkpoint({
#     #     'state_dict': model.state_dict(),
#     #     'epoch': epoch + 1,
#     #     'best_mAP': best_mAP,
#     # }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

#     # print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
#     #       format(epoch, mAP, best_mAP, ' *' if is_best else ''))
#     end_time = time.monotonic()
#     print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    parser.add_argument(
        "--config_file", default="vit_base_ics_384.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")#288 384
    parser.add_argument('--width', type=int, default=128, help="input width")#144 128
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,#30
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        )
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")
    parser.add_argument('--trial', type=int, default=1)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--warmup-step', type=int, default=0)
    parser.add_argument('--milestones', nargs='+', type=int, default=[20,40],
                        help='milestones for the learning rate decay')


    main()
