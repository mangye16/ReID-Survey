# encoding: utf-8
import logging
import torchvision
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from utils.reid_metric import r1_mAP_mINP, r1_mAP_mINP_reranking
from ignite.handlers import Timer
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

global ITER
ITER = 0
feat_ls = []
cam_ls = []
label_ls = []
from torch.utils.tensorboard import SummaryWriter
global writer
writer = SummaryWriter('./log/market1501/embedding_projector/')

def create_supervised_evaluator(model, metrics, device=None):
    def _inference(engine, batch):
        global ITER
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids
    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine

def do_embedding_projector(
    cfg, 
    model, 
    data_loader,
    num_query
):
    global feat_ls, cam_ls, label_ls
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("embedding projector")
    logger.info("Enter embedding images")
    
    if cfg.TEST.RE_RANKING == 'off':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'on':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for on or off, but got {}.".format(cfg.TEST.RE_RANKING))

    @evaluator.on(Events.ITERATION_COMPLETED)
    def append_embedding_query(engine):
        global feat_ls, cam_ls, label_ls
        global ITER
        ITER += 1
        feat_ls.append(evaluator.state.output[0])
        cam_ls.extend(evaluator.state.output[2])
        label_ls.extend(evaluator.state.output[1])

    evaluator.run(data_loader['eval'])

    features = (torch.cat(feat_ls)).to(device)
    # ? RGB
    # features = features.view(-1, cfg.INPUT.IMG_SIZE[0] * cfg.INPUT.IMG_SIZE[1])
    print(label_ls)
    writer.add_embedding(features, 
        metadata=label_ls,
        global_step=1)
    writer.close()

    cmc, mAP, mINP = evaluator.state.metrics['r1_mAP_mINP']
    logger.info('Validation Results')
    logger.info("mINP: {:.1%}".format(mINP))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

