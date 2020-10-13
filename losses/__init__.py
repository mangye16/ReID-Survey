import torch 
import torch.nn as nn

from .id_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .circle_loss import CircleLoss
from .cosine_loss import CosFace, AdaCos, ArcFace
from .triplet_loss import TripletLoss, WeightedRegularizedTriplet
from .smooth_ap_loss import SmoothAP

def build_loss_fn(cfg, num_classes):
    if cfg.MODEL.ID_LOSS_TYPE == 'none':
        def id_loss_fn(score, target):
            return 0
    else:
        id_loss_fn = CrossEntropyLabelSmooth(num_classes=num_classes)
    
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        metric_loss_fn = TripletLoss(margin=0.0)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'circle':
        metric_loss_fn = CircleLoss(m=cfg.MODEL.METRIC_LOSS.MARGIN, s=cfg.MODEL.METRIC_LOSS.SCALE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'smoothAP':
        assert(cfg.SOLVER.BATCH_SIZE % cfg.DATALOADER.NUM_INSTANCE == 0)
        metric_loss_fn = SmoothAP(anneal=0.01, batch_size=cfg.SOLVER.BATCH_SIZE, 
                                num_id=cfg.SOLVER.BATCH_SIZE // cfg.DATALOADER.NUM_INSTANCE, feat_dims=2048
        )
    else:
        def metric_loss_fn(feat, target, feat_t, target_t):
            return 0
    def loss_func(score, feat, target, feat_t, target_t):
        return id_loss_fn(score, target), metric_loss_fn(feat, target, feat_t, target_t)
        
    return loss_func
