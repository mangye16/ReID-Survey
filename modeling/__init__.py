# encoding: utf-8

from .baseline import Baseline

def build_model(cfg, num_classes):
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NAME,
                     cfg.MODEL.GENERALIZED_MEAN_POOL, cfg.MODEL.PRETRAIN_CHOICE)
    return model