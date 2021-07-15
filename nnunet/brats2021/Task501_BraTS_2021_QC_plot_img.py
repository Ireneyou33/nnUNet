#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import pandas as pd

from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.paths import nnUNet_raw_data
from nnunet.brats2021.misc import plot_voxel_enhance_brats
import SimpleITK as sitk
import cv2


def plot_mri_img_along_z(seg_file, t2_file, out_file_origin, out_file_fillpoly=None, out_file_contours=None):
    # 读取标签数据
    img_seg = sitk.ReadImage(seg_file)
    img_npy_seg = sitk.GetArrayFromImage(img_seg)

    # 读取T2数据
    img_t2 = sitk.ReadImage(t2_file)
    img_npy_t2 = sitk.GetArrayFromImage(img_t2)
    # 归一化
    img_npy_t2 = cv2.normalize(img_npy_t2, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    # 只截取具有肿瘤标签的位置
    z_range = [np.where((img_npy_seg != 0))[0].min(), np.where((img_npy_seg != 0))[0].max()]
    z_range = [max([0, z_range[0] - 3]), min([z_range[1] + 3, 155])]
    img_npy_t2 = img_npy_t2[z_range[0]:z_range[1]]
    img_npy_seg = img_npy_seg[z_range[0]:z_range[1]]

    # img_npy_seg[img_npy_seg == 2] = 3  # >=5，G整个肿瘤 WT = ET+Net+ED = 4 + 1 + 2
    # img_npy_seg[img_npy_seg == 4] = 2  # >=7，R增强肿瘤 ET = 4
    # img_npy_seg[img_npy_seg == 1] = 6  # >=6，B坏死和非增强的肿瘤核心 TC = ET+Net = 4 + 1

    plot_voxel_enhance_brats(img_npy_t2, img_npy_seg, out_file=out_file_origin, plot_type='origin')
    print(f'>>> {out_file_origin}')

    plot_voxel_enhance_brats(img_npy_t2, img_npy_seg, out_file=out_file_fillpoly, plot_type='fillpoly')
    print(f'>>> {out_file_fillpoly}')

    if out_file_contours is not None:
        plot_voxel_enhance_brats(img_npy_t2, img_npy_seg, out_file=out_file_contours, plot_type='contours')
    print(f'>>> {out_file_contours}')


if __name__ == "__main__":
    """
    THIS CODE IS A MESS. IT IS PROVIDED AS IS WITH NO GUARANTEES. YOU HAVE TO DIG THROUGH IT YOURSELF. GOOD LUCK ;-)

    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task501_BraTS2021"
    downloaded_data_dir = "/home/anning/Dataset/RawData/MICCAI_BraTS2021_TrainingData"
    # downloaded_data_dir_val = "/home/fabian/Downloads/MICCAI_BraTS2021_ValidationData"
    downloaded_data_dir_val = None

    img_dir = "/home/anning/Dataset/ProjData/BraTS2021/QC_IMG_MUL"

    target_base = join(nnUNet_raw_data, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesVal = join(target_base, "imagesVal")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_imagesVal)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)
    maybe_mkdir_p(img_dir)

    patient_names = []
    cur = join(downloaded_data_dir)

    grading_file = 'grading_res.csv'
    if os.path.isfile(grading_file):
        df_grad = pd.read_csv(grading_file, index_col='name')
    else:
        df_grad = None

    count = 1
    for p in subdirs(cur, join=False):
        print(p)
        patdir = join(cur, p)
        patient_name = p
        patient_names.append(patient_name)
        t1 = join(patdir, p + "_t1.nii.gz")
        t1c = join(patdir, p + "_t1ce.nii.gz")
        t2 = join(patdir, p + "_t2.nii.gz")
        flair = join(patdir, p + "_flair.nii.gz")
        seg = join(patdir, p + "_seg.nii.gz")

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
            isfile(seg)
        ]), "%s" % patient_name

        if df_grad is not None:
            grading_pvalue = df_grad.loc[p]['grading_pvalue']
            LGG, HGG = grading_pvalue[1:-1].split()
            if np.isclose(float(LGG), 1):
                grading = 'LGG'
            else:
                grading = 'HGG'
        else:
            grading = 'None'
        out_img_origin = os.path.join(img_dir, f'{p}_{grading}_origin.png')
        out_img_fillpoly = os.path.join(img_dir, f'{p}_{grading}_fillpoly.png')
        out_img_contours = os.path.join(img_dir, f'{p}_{grading}_contours.png')
        plot_mri_img_along_z(seg, t2, out_img_origin, out_img_fillpoly, out_img_contours)
        count += 1
        # if count >= 4:
        #     break
