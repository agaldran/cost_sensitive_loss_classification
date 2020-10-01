import sys
import math
import random
import copy
import numbers

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from torchvision.transforms import Normalize
from PIL import Image

def scale_to_mu_sigma_to_0_1(tensor):
    m = tensor.mean()
    s = tensor.std()
    return tensor.sub(m).div(s)

class DRDataset(Dataset):
    def __init__(self, csv_path, transforms=None, mean=None, std=None):
        self.csv_path = csv_path
        df = pd.read_csv(self.csv_path)
        self.im_list = df.image_id
        self.dr = df.dr.values
        self.transforms = transforms
        if mean is not None and std is not None:
            self.normalize = Normalize(mean, std)
        else:
            # self.normalize = scale_to_mu_sigma_to_0_1
            self.normalize = lambda x: x
    def __getitem__(self, index):
        # load image and labels
        img = Image.open(self.im_list[index])
        label = self.dr[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return self.normalize(img), label

    def __len__(self):
        return len(self.im_list)


def get_train_val_datasets(csv_path_train, csv_path_val, mean=None, std=None, tg_size=(512,512)):

    train_dataset = DRDataset(csv_path_train, mean=mean, std=std)

    val_dataset = DRDataset(csv_path_val, mean=mean, std=std)

    size = tg_size
    # required transforms
    resize = tr.Resize(size)
    tensorizer = tr.ToTensor()
    # geometric transforms
    h_flip = tr.RandomHorizontalFlip()
    v_flip = tr.RandomVerticalFlip()
    rotate = tr.RandomRotation(degrees=45)
    scale = tr.RandomAffine(degrees=0, scale=(0.95, 1.20))
    transl = tr.RandomAffine(degrees=0, translate=(0.05, 0))
    # either translate, rotate, or scale
    scale_transl_rot = tr.RandomChoice([scale, transl, rotate])
    # intensity transforms
    brightness, contrast, saturation, hue = 0.10, 0.10, 0.10, 0.

    jitter = tr.ColorJitter(brightness, contrast, saturation, hue)

    train_transforms = tr.Compose([resize, jitter, scale_transl_rot, h_flip, v_flip, tensorizer])
    val_transforms = tr.Compose([resize, tensorizer])
    train_dataset.transforms = train_transforms
    val_dataset.transforms = val_transforms

    for c in range(len(np.unique(train_dataset.dr))):
        exs_train = np.count_nonzero(train_dataset.dr == c)
        exs_val = np.count_nonzero(val_dataset.dr == c)
        print('Found {:d}/{:d} train/val examples of class {:d}'.format(exs_train, exs_val, c))

    return train_dataset, val_dataset

def get_test_dataset(csv_path_test, mean=None, std=None, tg_size=(512,512)):
    test_dataset = DRDataset(csv_path_test, mean=mean, std=std)

    size = tg_size
    # required transforms
    resize = tr.Resize(size)
    h_flip = tr.RandomHorizontalFlip(p=0)
    v_flip = tr.RandomVerticalFlip(p=0)
    tensorizer = tr.ToTensor()
    test_transforms = tr.Compose([resize, h_flip, v_flip, tensorizer])
    test_dataset.transforms = test_transforms

    return test_dataset

def get_train_val_loaders(csv_path_train, csv_path_val, batch_size=8,
                          mean=None, std=None):
    train_dataset, val_dataset = get_train_val_datasets(csv_path_train, csv_path_val, mean=mean, std=std)


    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size * torch.cuda.device_count(),
    #                           num_workers=8, pin_memory=True, shuffle=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size * torch.cuda.device_count(),
    #                         num_workers=8, pin_memory=True, shuffle=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              num_workers=8, pin_memory=True, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            num_workers=8, pin_memory=True, shuffle=False)
    return train_loader, val_loader

def get_test_loader(csv_path_test, batch_size=8, mean=None, std=None):
    test_dataset = get_test_dataset(csv_path_test, mean=mean, std=std)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size * torch.cuda.device_count(),
                            num_workers=8, pin_memory=True, shuffle=False)
    return test_loader

def modify_dataset(train_loader, csv_train_path, keep_samples=2000):
    train_loader_new = copy.deepcopy(train_loader)  # note, otherwise we modify underlying dataset
    train_dr = pd.read_csv(csv_train_path)

    r0_ims = train_dr.loc[train_dr['dr'] == 0]
    nr_r0 = r0_ims.shape[0]
    r1_ims = train_dr.loc[train_dr['dr'] == 1]
    nr_r1 = r1_ims.shape[0]
    r2_ims = train_dr.loc[train_dr['dr'] == 2]
    nr_r2 = r2_ims.shape[0]
    r3_ims = train_dr.loc[train_dr['dr'] == 3]
    nr_r3 = r3_ims.shape[0]
    r4_ims = train_dr.loc[train_dr['dr'] == 4]
    nr_r4 = r4_ims.shape[0]
    if isinstance(keep_samples, numbers.Number):
        r0_subset = r0_ims.sample(n=keep_samples, replace=nr_r0 < keep_samples)
        r1_subset = r1_ims.sample(n=keep_samples, replace=nr_r1 < keep_samples)
        r2_subset = r2_ims.sample(n=keep_samples, replace=nr_r2 < keep_samples)
        r3_subset = r3_ims.sample(n=keep_samples, replace=nr_r3 < keep_samples)
        r4_subset = r4_ims.sample(n=keep_samples, replace=r4_ims.shape[0] < keep_samples)
    elif isinstance(keep_samples, (list, tuple)):
        r0_subset = r0_ims.sample(n=int(keep_samples[0]*nr_r0), replace=nr_r0 < keep_samples[0]*nr_r0)
        r1_subset = r1_ims.sample(n=int(keep_samples[1]*nr_r1), replace=nr_r1 < keep_samples[1]*nr_r1)
        r2_subset = r2_ims.sample(n=int(keep_samples[2]*nr_r2), replace=nr_r2 < keep_samples[2]*nr_r2)
        r3_subset = r3_ims.sample(n=int(keep_samples[3]*nr_r3), replace=nr_r3 < keep_samples[3]*nr_r3)
        r4_subset = r4_ims.sample(n=int(keep_samples[4]*nr_r4), replace=nr_r4 < keep_samples[4]*nr_r4)
    else:
        sys.exit('keep_samples should be number, list, or tuple')

    duplicate = r0_subset[r0_subset.duplicated()]
    print('R0 nr samples (duplicated): {:d}({:d})'.format(r0_subset.shape[0], duplicate.shape[0]))
    duplicate = r1_subset[r1_subset.duplicated()]
    print('R1 nr samples (duplicated): {:d}({:d})'.format(r1_subset.shape[0], duplicate.shape[0]))
    duplicate = r2_subset[r2_subset.duplicated()]
    print('R2 nr samples (duplicated): {:d}({:d})'.format(r2_subset.shape[0], duplicate.shape[0]))
    duplicate = r3_subset[r3_subset.duplicated()]
    print('R3 nr samples (duplicated): {:d}({:d})'.format(r3_subset.shape[0], duplicate.shape[0]))
    duplicate = r4_subset[r4_subset.duplicated()]
    print('R4 nr samples (duplicated): {:d}({:d})'.format(r4_subset.shape[0], duplicate.shape[0]))

    dfs = [r0_subset, r1_subset, r2_subset, r3_subset, r4_subset]
    train_dr_under_oversampled = pd.concat(dfs)
    train_loader_new.dataset.im_list = train_dr_under_oversampled['image_id'].values
    train_loader_new.dataset.dr = train_dr_under_oversampled['dr'].values
    return train_loader_new

if __name__ == '__main__':
    # to be run from root folder
    csv_path_train = 'data/train_0.csv'
    csv_path_val = 'data/val_0.csv'
    train_dl, val_dl = get_train_val_loaders(csv_path_train, csv_path_val)
    batch = next(iter(train_dl))
    print(batch[0].shape, batch[1])



