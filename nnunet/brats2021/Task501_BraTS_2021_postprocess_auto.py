import argparse
from copy import deepcopy
from multiprocessing.pool import Pool

import numpy as np

from batchgenerators.utilities.file_and_folder_operations import *
import SimpleITK as sitk
from medpy.metric import dc
from nnunet.dataset_conversion.Task501_BraTS_2021 import determine_brats_postprocessing


def load_niftis_threshold_compute_dice(gt_file, pred_file, threshold):
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_file))
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_file))
    mask_pred = pred == 3
    mask_gt = gt == 3
    num_pred = np.sum(mask_pred)

    num_gt = np.sum(mask_gt)
    dice = dc(mask_pred, mask_gt)

    res_dice = {}
    res_was_smaller = {}
    t = threshold
    was_smaller = False

    if num_pred < t:
        was_smaller = True
        if num_gt == 0:
            dice_here = 1.
        else:
            dice_here = 0.
    else:
        dice_here = deepcopy(dice)

    res_dice[t] = dice_here
    res_was_smaller[t] = was_smaller

    return res_was_smaller, res_dice


def apply_brats_threshold(fname, out_dir, threshold, replace_with):
    img_itk = sitk.ReadImage(fname)
    img_npy = sitk.GetArrayFromImage(img_itk)
    s = np.sum(img_npy == 3)
    if s < threshold:
        # print(s, fname)
        img_npy[img_npy == 3] = replace_with
    img_itk_postprocessed = sitk.GetImageFromArray(img_npy)
    img_itk_postprocessed.CopyInformation(img_itk)
    sitk.WriteImage(img_itk_postprocessed, join(out_dir, fname.split("/")[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--pred-folder', help="模型预测结果目录", required=True)
    parser.add_argument('-g', "--folder-gt", help="真值目录", required=True)
    parser.add_argument('-r', "--result-folder", help="结果目录", required=True)

    args = parser.parse_args()
    processes = 4
    data_dir_pred = args.pred_folder
    result_folder = args.result_folder
    determine_brats_postprocessing(args.pred_folder, args.folder_gt, args.result_folder, processes=processes)