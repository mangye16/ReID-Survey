# encoding: utf-8

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class PartialREID(BaseImageDataset):

    dataset_dir = 'partial_reid'

    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(PartialREID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'partial_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')

        self._check_before_run()

        query, gallery = self._process(self.query_dir, self.gallery_dir)

        if verbose:
            print("=> partial_reid loaded")
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

    def _process(self, query_path, gallery_path):
        query_img_paths = glob.glob(osp.join(query_path, '*.jpg'))
        gallery_img_paths = glob.glob(osp.join(gallery_path, '*.jpg'))
        query_paths = []
        pattern = re.compile(r'([-\d]+)_(\d*)')
        for img_path in query_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            query_paths.append([img_path, pid, camid])
        gallery_paths = []
        for img_path in gallery_img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            gallery_paths.append([img_path, pid, camid])
        return query_paths, gallery_paths