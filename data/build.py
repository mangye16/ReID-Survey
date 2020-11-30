# encoding: utf-8

import torch
from torch.utils.data import DataLoader

from .datasets import init_dataset, ImageDataset
from .triplet_sampler import RandomIdentitySampler
from .transforms import build_transforms


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def make_data_loader(cfg):
    transforms = build_transforms(cfg)
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_set = ImageDataset(dataset.train, transforms['train'])
    data_loader={}
    if cfg.DATALOADER.PK_SAMPLER == 'on':
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )
    else:
        data_loader['train'] = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )

    if cfg.TEST.PARTIAL_REID == 'off':
        eval_set = ImageDataset(dataset.query + dataset.gallery, transforms['eval'])
        data_loader['eval'] = DataLoader(
            eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
    else:
        dataset_reid = init_dataset('partial_reid', root=cfg.DATASETS.ROOT_DIR)
        dataset_ilids = init_dataset('partial_ilids', root=cfg.DATASETS.ROOT_DIR)
        eval_set_reid = ImageDataset(dataset_reid.query + dataset_reid.gallery, transforms['eval'])
        data_loader['eval_reid'] = DataLoader(
            eval_set_reid, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        eval_set_ilids = ImageDataset(dataset_ilids.query + dataset_ilids.gallery, transforms['eval'])
        data_loader['eval_ilids'] = DataLoader(
            eval_set_ilids, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
    return data_loader, len(dataset.query), num_classes
