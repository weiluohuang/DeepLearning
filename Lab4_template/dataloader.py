
import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
def get_key(fp):
    return int(''.join(filter(str.isdigit, os.path.basename(fp))))

class Dataset_Dance(torchData):
    """
        Args:
            root (str)      : The path of your Dataset
            transform       : Transformation to your dataset
            mode (str)      : train, val, test
            partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """
    def __init__(self, root, transform, mode='train', video_len=7, partial=1.0):
        super().__init__()
        assert mode in ['train', 'val'], "There is no such mode !!!"
        if mode == 'train':
            self.img_folder = sorted(glob(os.path.join(root, 'train', 'train_img', '*.png')), key=get_key)
            self.prefix = 'train'
        elif mode == 'val':
            self.img_folder = sorted(glob(os.path.join(root, 'val', 'val_img', '*.png')), key=get_key)
            self.prefix = 'val'
        else:
            raise NotImplementedError
        
        self.transform = transform
        self.partial = partial
        self.video_len = video_len
        self.root = root

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):

        imgs = []
        labels = []
        for i in range(self.video_len):
            img_name = self.img_folder[index * self.video_len + i]
            img_name = os.path.normpath(img_name)
            
            label_name = os.path.join(self.root, self.prefix, f'{self.prefix}_label', os.path.basename(img_name))
            label_name = os.path.normpath(label_name)

            imgs.append(self.transform(imgloader(img_name)))
            labels.append(self.transform(imgloader(label_name)))
        
        return stack(imgs), stack(labels)