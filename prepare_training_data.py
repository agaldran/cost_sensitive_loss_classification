import pandas as pd
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
from skimage.measure import regionprops, label

from skimage import io
from skimage.filters import threshold_li

from PIL import Image
from torchvision import transforms as tr
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

tg_size = (512, 512)
rsz = tr.Resize(tg_size)
rsz_fov = tr.Resize(tg_size, interpolation=Image.NEAREST)

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def process_im(i, im_list, path_in, path_out):
    im_name = osp.join(path_in, im_list[i])
    im_name_out = osp.join(path_out, im_list[i][:-4] + 'jpg')

    im = io.imread(im_name)
    t = threshold_li(im[:, :, 1])
    binary = im[:, :, 1] > t
    binary = getLargestCC(binary)

    label_img = label(binary)
    regions = regionprops(label_img)
    areas = [r.area for r in regions]
    largest_cc_idx = np.argmax(areas)
    fov = regions[largest_cc_idx]

    cropped = im[fov.bbox[0]:fov.bbox[2], fov.bbox[1]: fov.bbox[3], :]
    cropped = rsz(Image.fromarray(cropped))
    cropped.save(im_name_out)


if __name__ == "__main__":
    # handle paths
    path_data_in = '/home/agaldran/Desktop/data/eyepacs_data/'
    path_ims_in = osp.join(path_data_in, 'images/train')
    path_csv_in = osp.join(path_data_in, 'trainLabels.csv')

    path_data_out = 'data/'
    path_ims_out = osp.join(path_data_out, 'images')

    os.makedirs(path_ims_out, exist_ok=True)

    df_all = pd.read_csv(path_csv_in)
    all_im_names = list(df_all['image'].values)
    all_im_names = [osp.join(path_ims_out, n + '.jpg') for n in all_im_names]

    dr_grades = df_all.level.values
    df_all['image_id'] = all_im_names
    df_all['dr'] = dr_grades
    df_all.drop(['image', 'level'], axis=1, inplace=True)

    df_train, df_val = train_test_split(df_all, test_size=0.10, random_state=42, stratify=df_all.dr)
    df_train.to_csv('data/train.csv', index=None)
    df_val.to_csv('data/val.csv', index=None)

    im_list = os.listdir(path_ims_in)
    num_ims = len(im_list)
    Parallel(n_jobs=6)(delayed(process_im)(i, im_list, path_ims_in, path_ims_out)
                       for i in tqdm(range(num_ims)))