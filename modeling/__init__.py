# encoding: utf-8

from .baseline import Baseline

def build_model(cfg, num_classes):
    model = Baseline(num_classes=num_classes, 
                    last_stride=cfg.MODEL.LAST_STRIDE, 
                    model_path=cfg.MODEL.PRETRAIN_PATH,
                    backbone=cfg.MODEL.BACKBONE,
                    pool_type=cfg.MODEL.POOL_TYPE,
                    use_dropout=cfg.COSINE_HEAD.USE_DROPOUT,
                    cosine_loss_type=cfg.COSINE_HEAD.COSINE_LOSS_TYPE,
                    s=cfg.COSINE_HEAD.SCALING_FACTOR,
                    m=cfg.COSINE_HEAD.MARGIN,
                    use_bnbias=cfg.COSINE_HEAD.USE_BNBIAS,
                    use_sestn=cfg.COSINE_HEAD.USE_SESTN,
                    pretrain_choice=cfg.COSINE_HEAD.PRETRAIN_CHOICE)
    return model