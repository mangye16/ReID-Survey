# encoding: utf-8


import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        # while it needs to transform everytime that want to get item
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

class ImageNoLabelDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        print("init dataset")
        self.dataset = dataset
        # TODO compute new transform
        self.transform = transform
        # self.transform = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, camid, date = self.dataset[index]
        img = read_image(img_path)

        # while it needs to transform everytime that want to get item
        if self.transform is not None:
            img = self.transform(img)

        return img, camid, date, img_path
