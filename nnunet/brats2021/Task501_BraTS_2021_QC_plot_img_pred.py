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


if __name__ == "__main__":
    """
    THIS CODE IS A MESS. IT IS PROVIDED AS IS WITH NO GUARANTEES. YOU HAVE TO DIG THROUGH IT YOURSELF. GOOD LUCK ;-)

    REMEMBER TO CONVERT LABELS BACK TO BRATS CONVENTION AFTER PREDICTION!
    """

    task_name = "Task501_BraTS2021"
    downloaded_data_dir = "/home/anning/Dataset/RawData/MICCAI_BraTS2021_TrainingData"
    # downloaded_data_dir_val = "/home/fabian/Downloads/MICCAI_BraTS2021_ValidationData"
    downloaded_data_dir_val = None

    data_dir_pred = "/home/anning/Dataset/ProjData/nnunet/RESULTS_FOLDER/nnUNet/3d_fullres/Task501_BraTS2021/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/validation_raw_best"
    data_dir_gt = "/home/anning/Dataset/ProjData/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTS2021/labelsTr"
    img_dir = "/home/anning/Dataset/ProjData/BraTS2021/QC_IMG_EVAL_BASELINE"  # 输出图片的文件夹

    evaluate_file = os.path.join(data_dir_pred, "results.csv")  # 预测结果的评价指标文件
    grading_file = '/home/anning/project/TTbraTS/nnUNet/nnunet/brats2021/grading_res.csv'  # HGG_LGG结果文件

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

    cur = join(downloaded_data_dir)

    if os.path.isfile(grading_file):
        df_grad = pd.read_csv(grading_file, index_col='name')
    else:
        df_grad = None

    if os.path.isfile(evaluate_file):
        df_eval = pd.read_csv(evaluate_file, index_col='name')
    else:
        df_eval = None

    count = 0
    for p in subdirs(cur, join=False):
        patdir = join(cur, p)
        patient_name = p
        t1 = join(patdir, p + "_t1.nii.gz")
        t1c = join(patdir, p + "_t1ce.nii.gz")
        t2 = join(patdir, p + "_t2.nii.gz")
        flair = join(patdir, p + "_flair.nii.gz")
        seg = join(patdir, p + "_seg.nii.gz")

        # 调整标注序号以后的label
        label_gt = join(data_dir_gt, p + ".nii.gz")
        label_pred = join(data_dir_pred, p + ".nii.gz")

        print(p)

        assert all([
            isfile(t1),
            isfile(t1c),
            isfile(t2),
            isfile(flair),
            isfile(seg)
        ]), "%s" % patient_name

        # 分级Label
        if df_grad is not None:
            grading_pvalue = df_grad.loc[p]['grading_pvalue']
            LGG, HGG = grading_pvalue[1:-1].split()
            if np.isclose(float(LGG), 1):
                grading = 'LGG'
            else:
                grading = 'HGG'
        else:
            grading = 'None'

        # DICE和HD95评分
        if df_eval is not None:
            key = p+".nii.gz"
            try:
                eval_row = df_eval.loc[key]
                dc_mean = eval_row.iloc[:3].mean()
                hd95_mean = eval_row.iloc[3:].mean()
            except KeyError:
                continue
        else:
            dc_mean = None
            hd95_mean = None

        # 读取gt标签数据
        img_seg = sitk.ReadImage(seg)
        img_npy_seg = sitk.GetArrayFromImage(img_seg)

        # 只截取gt中具有肿瘤标签的位置
        z_range = [np.where((img_npy_seg != 0))[0].min(), np.where((img_npy_seg != 0))[0].max()]
        z_range = [max([0, z_range[0] - 3]), min([z_range[1] + 3, 155])]

        # 绘制MRI的4种模态图
        for f, m_n in zip([t1, t1c, t2, flair], ["t1", "t1c", "t2", "flair"]):
            out_img_origin = os.path.join(img_dir, f'{patient_name}_{grading}_a_{m_n}_{dc_mean:02.4f}_{hd95_mean:03.2f}.png')
            img = sitk.ReadImage(f)
            img_npy = sitk.GetArrayFromImage(img)[z_range[0]:z_range[1]]
            # 归一化
            img_npy = cv2.normalize(img_npy, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

            plot_voxel_enhance_brats(img_npy, None, out_file=out_img_origin, plot_type='origin')
            print(f'>>> {out_img_origin}')

        # 以t2为底图绘制label
        img = sitk.ReadImage(t2)
        img_npy_t2 = sitk.GetArrayFromImage(img)[z_range[0]:z_range[1]]
        img_npy_t2 = cv2.normalize(img_npy_t2, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

        for f, m_n in zip([label_pred, label_gt], ["pred", "gt"]):
            out_img_fillpoly = os.path.join(img_dir, f'{patient_name}_{grading}_b_{m_n}_fillpoly_{dc_mean:02.4f}_{hd95_mean:03.2f}.png')
            out_img_contours = os.path.join(img_dir, f'{patient_name}_{grading}_c_{m_n}_contours_{dc_mean:02.4f}_{hd95_mean:03.2f}.png')

            img = sitk.ReadImage(f)
            img_npy_label = sitk.GetArrayFromImage(img)[z_range[0]:z_range[1]]

            plot_voxel_enhance_brats(img_npy_t2, img_npy_label, out_file=out_img_fillpoly, plot_type='fillpoly')
            print(f'>>> {out_img_fillpoly}')

            plot_voxel_enhance_brats(img_npy_t2, img_npy_label, out_file=out_img_contours, plot_type='contours')
            print(f'>>> {out_img_contours}')

        count += 1
        # if count >= 1:
        #     break
