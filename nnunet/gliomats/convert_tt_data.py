import ants
from copy import deepcopy
from multiprocessing.pool import Pool

import numpy as np
from collections import OrderedDict

from batchgenerators.utilities.file_and_folder_operations import *
import scipy.stats as ss

from nnunet.dataset_conversion.Task032_BraTS_2018 import convert_labels_back_to_BraTS_2018_2019_convention
from nnunet.dataset_conversion.Task043_BraTS_2019 import copy_BraTS_segmentation_and_convert_labels
from nnunet.evaluation.region_based_evaluation import get_brats_regions, evaluate_regions
from nnunet.paths import nnUNet_raw_data
import SimpleITK as sitk
import shutil
from medpy.metric import dc, hd95

from nnunet.postprocessing.consolidate_postprocessing import collect_cv_niftis
from typing import Tuple


def convert_seg(in_file, out_file):
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    for i in [2, 3]:
        img_npy[img_npy == i] = 1
    img_corr = sitk.GetImageFromArray(img_npy)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)
    print(f"out_file : {out_file}")


def print_seg(in_file):
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    if len(np.unique(img_npy)) > 2:
        print(f"{np.unique(img_npy)} : {in_file}")
        print(f"{(img_npy == 2).sum()}, {(img_npy == 3).sum()}")


def copy_and_compress_ttbrats_data(in_file, out_file):
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)
    img_corr = sitk.GetImageFromArray(img_npy)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)
    print(f"out_file : {out_file}")


def convert_data(in_file, target_file, out_file):
    image_info_in = ants.image_header_info(in_file)
    image_info_target = ants.image_header_info(target_file)

    in_file_resample = out_file + "resample.nii"

    if not (image_info_in["dimensions"] == image_info_target["dimensions"] and
            image_info_in["origin"] == image_info_target["origin"]):
        image_in = ants.image_read(in_file)
        image_target = ants.image_read(target_file)
        image_in_resample = ants.resample_image_to_target(image_in, image_target)
        ants.image_write(image_in_resample, in_file_resample)

    if os.path.isfile(in_file_resample):
        copy_and_compress_ttbrats_data(in_file_resample, out_file)
        os.system(f"rm {in_file_resample}")
    else:
        copy_and_compress_ttbrats_data(in_file, out_file)


def convet_data_floder(p_dir, p_name):
    t1 = join(p_dir, "t1.nii")
    t1c = join(p_dir, "ce.nii")
    t2 = join(p_dir, "t2.nii")
    seg = join(p_dir, "tu_mask.nii")

    # print(os.path.isfile(t1), t1)
    # print(os.path.isfile(t1c), t1c)
    # print(os.path.isfile(t2), t2)
    # print(os.path.isfile(seg), seg)

    assert all([
        isfile(t1),
        isfile(t1c),
        isfile(t2),
        isfile(seg)
    ]), "%s" % p_name

    # print_seg(seg)

    convert_data(t1, seg, join(target_imagesTr, p_name + "_0000.nii.gz"))
    convert_data(t1c, seg, join(target_imagesTr, p_name + "_0001.nii.gz"))
    convert_data(t2, seg, join(target_imagesTr, p_name + "_0002.nii.gz"))
    convert_seg(seg, join(target_labelsTr, p_name + ".nii.gz"))


if __name__ == "__main__":
    """
    THIS CODE IS A MESS. IT IS PROVIDED AS IS WITH NO GUARANTEES. YOU HAVE TO DIG THROUGH IT YOURSELF. GOOD LUCK ;-)

    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task510_TTBraTS20210706"
    downloaded_data_dir = "/mnt/ngshare/OrigData/TianTan_Glioma/TT_DATA_20210706/Tiantan_LGG_1162"
    downloaded_data_dir_val = "/mnt/ngshare/OrigData/TianTan_Glioma/TT_DATA_20210706/Tiantan_LGG_139"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    patient_names = []
    patdirs = []
    cur = join(downloaded_data_dir)

    for p in subdirs(cur, join=False):
        if p == "1876966":  # 这个数据是坏的，跳过
            print(p)
            continue
        patdir = join(cur, p)
        patient_name = p
        patdirs.append(patdir)
        patient_names.append(patient_name)

        # convet_data_floder(patdir, patient_name)

    thread = 4
    pool = Pool(thread)
    pool.starmap(convet_data_floder, zip(patdirs, patient_names))
    pool.close()
    pool.join()

    json_dict = OrderedDict()
    json_dict['name'] = "TTBraTS"
    json_dict['description'] = "nothing"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see TTBraTS"
    json_dict['licence'] = "see TTBraTS license"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "T1",
        "1": "T1ce",
        "2": "T2",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "tumor",
    }
    json_dict['numTraining'] = len(patient_names)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in
                             patient_names]
    json_dict['test'] = []

    save_json(json_dict, join(target_base, "dataset.json"))

