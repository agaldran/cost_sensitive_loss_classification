import pandas as pd
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import warnings
from skimage.measure import regionprops, label
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage import io
from skimage.filters import threshold_li
from skimage import img_as_float, img_as_ubyte
from PIL import Image
from torchvision import transforms as tr
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from utils.image_preprocessing import correct_illumination

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

def illuminate_im(i, im_list, path_in, path_out):
    im_name = osp.join(path_in, im_list[i])
    im_name_out = osp.join(path_out, im_list[i])

    im = io.imread(im_name)
    im = Image.fromarray(correct_illumination(im))
    im.save(im_name_out)

def getLargestCC(segmentation):
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def get_fov(im):
    im = correct_illumination(im)
    t = threshold_li(im[:, :, 1])
    binary = im[:, :, 1] > t
    binary = getLargestCC(binary)

    label_img = label(binary)
    regions = regionprops(label_img)
    areas = [r.area for r in regions]
    largest_cc_idx = np.argmax(areas)
    fov = regions[largest_cc_idx].filled_image
    return np.array(rsz_fov(Image.fromarray(fov)))

def transfer_color_im(i, im_list, path_in, path_out, source_im):
    im_name = osp.join(path_in, im_list[i])
    im_name_out = osp.join(path_out, im_list[i])

    x1 = img_as_float(io.imread(im_name))
    x2 = img_as_float(source_im)
    x1_fov = get_fov(x1)
    x2_fov = get_fov(x2)

    mr_source, mg_source, mb_source = x1[:, :, 0][x1_fov].mean(), x1[:, :, 1][x1_fov].mean(), x1[:, :, 2][x1_fov].mean()
    sr_source, sg_source, sb_source = x1[:, :, 0][x1_fov].std(), x1[:, :, 1][x1_fov].std(), x1[:, :, 2][x1_fov].std()

    mr_target, mg_target, mb_target = x2[:, :, 0][x2_fov].mean(), x2[:, :, 1][x2_fov].mean(), x2[:, :, 2][x2_fov].mean()
    sr_target, sg_target, sb_target = x2[:, :, 0][x2_fov].std(), x2[:, :, 1][x2_fov].std(), x2[:, :, 2][x2_fov].std()

    x12 = x1.copy()  # x1 is target, x2 is source, x21 holds result
    x12[:, :, 0] = (x12[:, :, 0] - mr_source) * (sr_target / sr_source) + mr_target
    x12[:, :, 0][~x1_fov] = 0
    x12[:, :, 1] = (x12[:, :, 1] - mg_source) * (sg_target / sg_source) + mg_target
    x12[:, :, 1][~x1_fov] = 0
    x12[:, :, 2] = (x12[:, :, 2] - mb_source) * (sb_target / sb_source) + mb_target
    x12[:, :, 2][~x1_fov] = 0
    x12 = np.clip(x12, 0, 1)
    io.imsave(im_name_out, img_as_ubyte(x12))

if __name__ == "__main__":
    # handle paths
    path_data_in = '/home/agaldran/Desktop/data/eyepacs_data/'
    path_ims_in = osp.join(path_data_in, 'images/test')
    path_csv_in = osp.join(path_data_in, 'retinopathy_solution.csv')

    path_data_out = 'data/'
    path_ims_out = osp.join(path_data_out, 'test_eyepacs')

    os.makedirs(path_ims_out, exist_ok=True)

    df_all = pd.read_csv(path_csv_in)
    all_im_names = list(df_all['image'].values)
    all_im_names = [osp.join(path_ims_out, n + '.jpg') for n in all_im_names]

    dr_grades = df_all.level.values
    df_all['image_id'] = all_im_names
    df_all['dr'] = dr_grades
    df_private = df_all[df_all.Usage == 'Private']
    df_public = df_all[df_all.Usage == 'Public']
    df_all.drop(['image', 'level','Usage'], axis=1, inplace=True)
    df_public.drop(['image', 'level', 'Usage'], axis=1, inplace=True)
    df_private.drop(['image', 'level', 'Usage'], axis=1, inplace=True)

    df_all.to_csv('data/test_eyepacs.csv', index=None)
    df_public.to_csv('data/test_eyepacs_public.csv', index=None)
    df_private.to_csv('data/test_eyepacs_private.csv', index=None)

    im_list = os.listdir(path_ims_in)
    num_ims = len(im_list)
    # Parallel(n_jobs=6)(delayed(process_im)(i, im_list, path_ims_in, path_ims_out)
    #                    for i in tqdm(range(num_ims)))


    path_ims_in = path_ims_out
    path_ims_out = osp.join(path_data_out, 'images_pre')
    im_list = os.listdir(path_ims_in)
    Parallel(n_jobs=6)(delayed(illuminate_im)(i, im_list, path_ims_in, path_ims_out)
                       for i in tqdm(range(num_ims)))

    all_im_names = [n.replace('images', 'images_pre') for n in all_im_names]
    df_all['image_id'] = all_im_names
    df_all.to_csv('data/test_eyepacs_pre.csv', index=None)


    path_ims_in = path_ims_out
    path_ims_out = osp.join(path_data_out, 'images_pre_color')
    im_list = os.listdir(path_ims_in)
    source_im = io.imread(osp.join(path_ims_in, '13_left.jpg'))
    Parallel(n_jobs=6)(delayed(transfer_color_im)(i, im_list, path_ims_in, path_ims_out, source_im)
                       for i in tqdm(range(num_ims)))

    all_im_names = [n.replace('images_pre', 'images_pre_color') for n in all_im_names]
    df_all['image_id'] = all_im_names
    df_all.to_csv('data/test_eyepacs_pre_color.csv', index=None)

