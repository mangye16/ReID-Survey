# encoding: utf-8

from .baseline import Baseline

def build_model(cfg, num_classes):
    model = Baseline(num_classes=num_classes, 
                    last_stride=cfg.MODEL.LAST_STRIDE, 
                    model_path=cfg.MODEL.PRETRAIN_PATH,
                    backbone=cfg.MODEL.BACKBONE,
                    pool_type=cfg.MODEL.BASELINE.POOL_TYPE,
                    use_dropout=cfg.MODEL.USE_DROPOUT,
                    cosine_loss_type=cfg.MODEL.BASELINE.COSINE_LOSS_TYPE,
                    s=cfg.MODEL.BASELINE.SCALING_FACTOR,
                    m=cfg.MODEL.BASELINE.MARGIN,
                    use_bnbias=cfg.MODEL.BASELINE.USE_BNBIAS,
                    use_sestn=cfg.MODEL.BASELINE.USE_SESTN,
                    pretrain_choice=cfg.MODEL.PRETRAIN_CHOICE)
    return model