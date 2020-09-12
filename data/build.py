# encoding: utf-8

import torch
from torch.utils.data import DataLoader

from .datasets import init_dataset, ImageDataset, ImageNoLabelDataset
from .triplet_sampler import RandomIdentitySampler
from .transforms import build_transforms

# ASK :
def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def val_no_label_collate_fn(batch) :
    imgs, camids, dates, _ = zip(*batch)
    return torch.stack(imgs, dim=0), camids, dates

def make_data_loader(cfg):
    transforms = build_transforms(cfg)
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if cfg.VISUALIZE.OPTION == "on_no_label" :
        gallery_set = ImageNoLabelDataset( dataset.gallery, transforms['eval'])
        data_loader={}
        data_loader['gallery'] = DataLoader(
            gallery_set, batch_size=cfg.VISUALIZE.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_no_label_collate_fn
        )
        return data_loader
    # number of identities
    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, transforms['train'])
    data_loader={}
    # ASK : what is PK_SAMPLER, collate_fm
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

    eval_set = ImageDataset(dataset.query + dataset.gallery, transforms['eval'])
    data_loader['eval'] = DataLoader(
        eval_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    if cfg.VISUALIZE.OPTION == "on" :
        query_set = ImageDataset(dataset.query , transforms['eval'])
        gallery_set = ImageDataset( dataset.gallery, transforms['eval'])
        data_loader['query'] = DataLoader(
            query_set, batch_size=cfg.VISUALIZE.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
        data_loader['gallery'] = DataLoader(
            gallery_set, batch_size=cfg.VISUALIZE.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn
        )
    return data_loader, len(dataset.query), num_classes