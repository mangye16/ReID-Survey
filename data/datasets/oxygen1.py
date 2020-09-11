# encoding: utf-8

import glob
import re

import os.path as osp

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

    def __init__(self, root='./Oxygen_1', verbose=True, **kwargs):
        super(Oxygen_1, self).__init__()
        # global print_dataset_statistics
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')
        # self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        # self.query_dir = osp.join(self.dataset_dir, 'query')
        # self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # train = self._process_dir(self.train_dir, relabel=True)
        # query = self._process_dir(self.query_dir, relabel=False)
        # gallery = self._process_dir(self.gallery_dir, relabel=False)

        # debug here
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Oxygen Gallery loaded")
            # print_dataset_statistics(gallery)

        # self.train = train
        # self.query = query
        self.gallery = gallery

        # self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        # self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
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
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        # if not osp.exists(self.query_dir):
        #     raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    # TODO : 1 mbk-(/d+)-(/d+)
    # data loader without label or pid
    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        # pattern = re.compile(r'mbk-(/d+)-(/d+)')
        # pid_container = set()
        # for img_path in img_paths:
        #     pid, _ = map(int, pattern.search(img_path).groups())
        #     if pid == -1: continue  # junk images are just ignored
        #     pid_container.add(pid)
        # pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # dataset = []
        # for img_path in img_paths:
        #     pid, camid = map(int, pattern.search(img_path).groups())
        #     if pid == -1: continue  # junk images are just ignored
        #     assert 0 <= pid <= 1501  # pid == 0 means background
        #     assert 1 <= camid <= 6
        #     camid -= 1  # index starts from 0
        #     if relabel: pid = pid2label[pid]
        #     dataset.append((img_path, pid, camid))
        
        # (path, cam, date)
        # without validate cam id or date format
        dataset = [(img_path, img_path.split("/")[-2], img_path.split("/")[-1][:10]) for img_path in img_paths]

        return dataset
