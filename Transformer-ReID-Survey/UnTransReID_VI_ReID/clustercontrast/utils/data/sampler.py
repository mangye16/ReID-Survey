from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if not select_indexes:
                    continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)


class RandomMultipleGallerySamplerNoCam(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)

        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            index = self.pid_index[pid_i]

            select_indexes = No_index(index, i)
            if not select_indexes:
                continue
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

            for kk in ind_indexes:
                ret.append(index[kk])

        return iter(ret)

class MoreCameraSampler(Sampler):
    def __init__(self, data_source, num_instances=4, video=False):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances
        self.video = video

        if self.video:
            for index, (_, pid, cam, _) in enumerate(data_source):
                if (pid < 0): continue
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)
        else:
            for index, (_, pid, cam) in enumerate(data_source):
                if (pid < 0): continue
                self.index_pid[index] = pid
                self.pid_cam[pid].append(cam)
                self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            if self.video:
                _, i_pid, i_cam, _ = self.data_source[i]
            else:
                _, i_pid, i_cam = self.data_source[i]

            cams = self.pid_cam[i_pid]
            index = self.pid_index[i_pid]

            unique_cams = set(cams)
            cams = np.array(cams)
            index = np.array(index)
            select_indexes = []
            for cam in unique_cams:
                select_indexes.append(np.random.choice(index[cams==cam], size=1, replace=False))
            select_indexes = np.concatenate(select_indexes)
            if len(select_indexes)< self.num_instances:
                diff_indexes = np.setdiff1d(index, select_indexes)
                if len(diff_indexes) == 0:
                    select_indexes = np.random.choice(select_indexes, size=self.num_instances, replace=True)
                elif len(diff_indexes) >= (self.num_instances-len(select_indexes)):
                    diff_indexes = np.random.choice(diff_indexes, size=(self.num_instances-len(select_indexes)), replace=False)
                else:
                    diff_indexes = np.random.choice(diff_indexes, size=(self.num_instances-len(select_indexes)), replace=True)
                select_indexes = np.concatenate([select_indexes, diff_indexes])
            else:
                select_indexes = np.random.choice(select_indexes, size=self.num_instances, replace=False)
            ret.extend(select_indexes)
        return iter(ret)