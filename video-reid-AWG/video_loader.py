from __future__ import print_function, absolute_import
import os
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import random


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['constrain_random', 'evenly']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        if self.sample == 'constrain_random':
            """
            Evenly and randomly sample seq_len items from num items .
            """
            def random_int_list(start, stop, length):
                start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
                length = int(abs(length)) if length else 0
                random_list = []
                for i in range(length):
                    random_list.append(random.randint(start, stop))
                return random_list
            if num >= self.seq_len:
                r = num % self.seq_len
                stride = num // self.seq_len
                if r != 0:
                    stride += 1
                bias_indices=random_int_list(0,stride-1,self.seq_len)
                ach_indices=np.arange(0,self.seq_len)*stride
                indices = bias_indices + ach_indices
                indices = indices.clip(max=num - 1)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])

            imgs = []
            for index in indices:
                img_path = img_paths[int(index)]
                img = read_image(img_path)
                if self.transform is not None:
                    img = self.transform(img)
                img = img.unsqueeze(0)
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0)
            return imgs, pid, camid
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            indices_list = []
            if num >= self.seq_len:
                r = num % self.seq_len
                stride = num // self.seq_len
                if r != 0:
                    stride += 1
                for i in range(stride):
                    indices = np.arange(i, stride*self.seq_len, stride)
                    indices = indices.clip(max=num-1)
                    indices_list.append(indices)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
                indices_list.append(indices)

            if len(indices_list) > 50:
                indices_list = indices_list[:50]

            imgs_list=[]
            for indices in indices_list:
                imgs = []
                for index in indices:
                    img_path = img_paths[int(index)]
                    img = read_image(img_path)
                    if self.transform is not None:
                        img = self.transform(img)
                    img = img.unsqueeze(0)
                    imgs.append(img)
                imgs = torch.cat(imgs, dim=0)
                imgs_list.append(imgs)
            imgs_array = torch.stack(imgs_list)
            return imgs_array, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))







