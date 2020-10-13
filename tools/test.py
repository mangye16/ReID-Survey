# encoding: utf-8
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import r1_mAP_mINP, r1_mAP_mINP_reranking


def create_supervised_evaluator(model, metrics, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to evaluate
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    
    def _inference(engine, batch):
        model.eval()
        # setting requires_grad flag to false
        with torch.no_grad():
            data, pids, camids = batch
            # 
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids
    # Engine is the abstraction refering to the loop provided data and execute func and return result
    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_test(
        cfg,
        model,
        data_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline")
    logger.info("Enter inferencing")
    # TRY : By default reranking is off
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

    evaluator.run(data_loader['eval'])
    cmc, mAP, mINP = evaluator.state.metrics['r1_mAP_mINP']
    logger.info('Validation Results')
    logger.info("mINP: {:.1%}".format(mINP))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
