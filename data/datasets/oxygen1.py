# encoding: utf-8

import glob
import re
import os
import os.path as osp
from os.path import join

from .bases import BaseImageDataset

class Oxygen_1(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'oxygen1'

    def __init__(self, root='./toDataset', verbose=True, **kwargs):
        super(Oxygen_1, self).__init__()
        # global print_dataset_statistics
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery/')

        self._check_before_run()

        # debug here
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Oxygen Gallery loaded")

        self.gallery = gallery
        self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_dates = self._get_imagedata_info(self.gallery)
    
    def _get_imagedata_info(self, data):
        cams, dates = [], []
        for _, cam, date in data:
            cams += [cam]
            dates += [date]
        cams = set(cams)
        dates = set(dates)
        num_dates = len(dates)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_imgs , num_cams ,  num_dates 

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    # TODO : 1 mbk-(/d+)-(/d+)
    # data loader without label or pid
    def _process_dir(self, dir_path, relabel=False):
        # print(dir_path)
        all_folder = os.listdir(dir_path)
        img_paths = [fs for files in [glob.glob(osp.join(dir_path+folder, '*.jpg')) for folder in all_folder] for fs in files]
        
        # (path, cam, date)
        # without validate cam id or date format
        dataset = [(img_path, img_path.split("/")[-2], img_path.split("/")[-1][:10]) for img_path in img_paths]
        return dataset
