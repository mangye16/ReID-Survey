from __future__ import print_function, absolute_import
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

import data_manager
from video_loader import VideoDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, WeightedRegularizedTriplet, CenterLoss
from utils import AverageMeter, Logger, save_checkpoint, mkdir_if_missing
from eval_metrics import evaluate
from samplers import RandomIdentitySampler
from lr_scheduler import WarmupMultiStepLR

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('--train-dataset', type=str, default='mars',
                    choices=data_manager.get_names())
parser.add_argument( '--test-dataset', type=str, default='duke',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
parser.add_argument('--seq-len', type=int, default=6, help="number of images to sample in a tracklet")
# Optimization options
parser.add_argument('--max-epoch', default=400, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--train-batch', default=64, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=1, type=int, help="has to be 1")
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    help="initial learning rate, use 0.0001 for rnn, use 0.0003 for pooling and attention")
parser.add_argument('--stepsize', default=[100,200,300], nargs='+', type=int,
                    help="stepsize to decay learning rate")
parser.add_argument('--gamma', default=0.1, type=float,
                    help="learning rate decay")
parser.add_argument('--weight-decay', default=5e-04, type=float,
                    help="weight decay (default: 5e-04)")
parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
parser.add_argument('--num-instances', type=int, default=4,
                    help="number of instances per identity")
parser.add_argument('--htri-only', action='store_true', default=False,
                    help="if this is True, only htri loss is used in training")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50tp', help="resnet503d, resnet50tp, resnet50ta, resnetrnn")
parser.add_argument('--pool', type=str, default='avg', choices=['avg', 'max'])

# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--load-model', type=str, default='', help='path to pretrained model')
parser.add_argument('--stage', type=int, default=0, help="print frequency")
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--start-eval', type=int, default=0,
                    help="start to evaluate after specific epoch")
parser.add_argument('--eval-step', type=int, default=50,
                    help="run evaluation for every N epochs (set to -1 to test after training)")
parser.add_argument('--save-dir', type=str, default='log')

args = parser.parse_args()


def main():
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    use_gpu = torch.cuda.is_available()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    print("Initializing train dataset {}".format(args.train_dataset))
    train_dataset = data_manager.init_dataset(name=args.train_dataset)
    print("Initializing test dataset {}".format(args.test_dataset))
    test_dataset = data_manager.init_dataset(name=args.test_dataset)

    # print("Initializing train dataset {}".format(args.train_dataset, split_id=6))
    # train_dataset = data_manager.init_dataset(name=args.train_dataset)
    # print("Initializing test dataset {}".format(args.test_dataset, split_id=6))
    # test_dataset = data_manager.init_dataset(name=args.test_dataset)

    transform_train = T.Compose([
        T.Resize([args.height, args.width]),
        T.RandomHorizontalFlip(),
        T.Pad(10),
        T.RandomCrop([args.height, args.width]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    pin_memory = True if use_gpu else False

    # random_snip  first_snip constrain_random evenly
    trainloader = DataLoader(
        VideoDataset(train_dataset.train, seq_len=args.seq_len, sample='constrain_random',transform=transform_train),
        sampler=RandomIdentitySampler(train_dataset.train, num_instances=args.num_instances),
        batch_size=args.train_batch, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=True,
    )

    queryloader = DataLoader(
        VideoDataset(test_dataset.query, seq_len=args.seq_len, sample='evenly', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        VideoDataset(test_dataset.gallery, seq_len=args.seq_len, sample='evenly', transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, num_classes=train_dataset.num_train_pids, loss={'xent', 'htri'})
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    print("load model {0} from {1}".format(args.arch, args.load_model))
    if args.load_model != '':
        pretrained_model = torch.load(args.load_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        start_epoch = pretrained_model['epoch'] + 1
        best_rank1 = pretrained_model['rank1']
    else:
        start_epoch = args.start_epoch
        best_rank1 = -np.inf

    criterion = dict()
    criterion['triplet'] = WeightedRegularizedTriplet()
    criterion['xent'] = CrossEntropyLabelSmooth(num_classes=train_dataset.num_train_pids)
    criterion['center'] = CenterLoss(num_classes=train_dataset.num_train_pids, feat_dim=512,
                                     use_gpu=True)
    print(criterion)

    optimizer = dict()
    optimizer['model'] = model.get_optimizer(args)
    optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=0.5)

    scheduler = lr_scheduler.MultiStepLR(optimizer['model'], milestones=args.stepsize, gamma=args.gamma)

    print(model)
    model = nn.DataParallel(model).cuda()

    if args.evaluate:
        print("Evaluate only")
        distmat = test(model, queryloader, galleryloader, args.pool, use_gpu, return_distmat=True)
        return


    start_time = time.time()
    train_time = 0
    best_epoch = args.start_epoch
    print("==> Start training")
    for epoch in range(start_epoch, args.max_epoch):

        scheduler.step()
        print('Epoch',epoch,'lr', scheduler.get_lr()[0])

        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu)
        train_time += round(time.time() - start_train_time)

        if (epoch + 1) > args.start_eval and args.eval_step > 0 and (epoch + 1) % args.eval_step == 0 or (
                epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, args.pool, use_gpu)
            is_best = rank1 > best_rank1

            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


def train(epoch, model, criterion, optimizer, trainloader, use_gpu):
    model.train()

    losses = AverageMeter()
    losses_htri = AverageMeter()
    losses_xent = AverageMeter()
    losses_center = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for batch_idx, sample in enumerate(trainloader):

        optimizer['model'].zero_grad()
        optimizer['center'].zero_grad()

        (imgs, pids, cids) = sample
        data_time.update(time.time() - end)
        if use_gpu:
            imgs, pids, cids= imgs.cuda(), pids.cuda(), cids.cuda()
        outputs, features = model(imgs)

        xent_loss = criterion['xent'](outputs, pids)
        htri_loss = criterion['triplet'](features, pids)
        center_loss = criterion['center'](features, pids)

        loss = xent_loss + htri_loss + 0.0005*center_loss
        loss.backward()

        optimizer['model'].step()
        for param in criterion['center'].parameters():
            param.grad.data *= (1. / 0.0005)
        optimizer['center'].step()

        batch_time.update(time.time() - end)

        losses.update(loss.item(), pids.size(0))
        losses_htri.update(htri_loss.item(), pids.size(0))
        losses_xent.update(xent_loss.item(), pids.size(0))
        losses_center.update(center_loss.item(), pids.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_htri {loss_htri.val:.4f} Loss_xent {loss_xent.avg:.4f}\t Loss_cnet {loss_cent.avg:.4f}\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                data_time=data_time, loss=losses, loss_htri=losses_htri, loss_xent=losses_xent, loss_cent=losses_center))

        end = time.time()


def test(model, queryloader, galleryloader, pool, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()
    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()
            b, n, s, c, h, w = imgs.size()
            assert (b == 1)
            imgs = imgs.view(b * n, s, c, h, w)

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.view(n, -1)
            features = torch.mean(features, 0)
            features = features.cpu()
            qf.append(features.numpy())
            q_pids.extend(pids.numpy())
            q_camids.extend(camids.numpy())
        qf = np.stack(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.shape[0], qf.shape[1]))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()
            b, n, s, c, h, w = imgs.size()
            assert (b == 1)
            imgs = imgs.view(b * n, s, c, h, w)

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)
            features = features.view(n, -1)
            features = torch.mean(features, 0)
            features = features.cpu()
            gf.append(features.numpy())
            g_pids.extend(pids.numpy())
            g_camids.extend(camids.numpy())
        gf = np.stack(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.shape[0], gf.shape[1]))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch * args.seq_len))

    m, n = qf.shape[0], gf.shape[0]
    distmat = np.tile((qf**2).sum(axis=1, keepdims=True),(1, n)) + \
              np.tile((gf**2).sum(axis=1, keepdims=True),(1, m)).transpose()
    distmat = distmat - 2*np.dot(qf,gf.transpose())

    print("Computing CMC and mAP")
    cmc, mAP, mINP= evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mINP: {:.1%} mAP: {:.1%} CMC curve {:.1%} {:.1%} {:.1%} {:.1%}".format(mINP, mAP,cmc[1 - 1],cmc[5 - 1],cmc[10 - 1],cmc[20 - 1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]

if __name__ == '__main__':
    main()
