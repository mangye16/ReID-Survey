# encoding: utf-8

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class PartialILIDS(BaseImageDataset):

    dataset_dir = 'partial_ilids'

    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(PartialILIDS, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'Probe')
        self.gallery_dir = osp.join(self.dataset_dir, 'Gallery')

        self._check_before_run()

        query = self._process_dir(self.query_dir,camid=1)
        gallery = self._process_dir(self.gallery_dir,camid=2)

        if verbose:
            print("=> partial_ilids loaded")
            self.print_dataset_statistics(query, query, gallery)

        self.query = query
        self.gallery = gallery

        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, camid):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        dataset = []
        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = int(osp.splitext(img_name)[0])
            pid_container.add(pid)
            dataset.append((img_path, pid, camid))

        return dataset