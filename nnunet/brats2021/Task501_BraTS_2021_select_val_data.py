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

from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.Task043_BraTS_2019 import copy_BraTS_segmentation_and_convert_labels
import shutil


if __name__ == "__main__":
    """
    软连接需要验证的数据
    """
    task_name = "Task501_BraTS2021"
    downloaded_data_dir = "/home/anning/Dataset/RawData/MICCAI_BraTS2021_TrainingData"
    # downloaded_data_dir_val = "/home/fabian/Downloads/MICCAI_BraTS2021_ValidationData"
    downloaded_data_dir_val = None
    split_final_file = "/home/anning/Dataset/ProjData/nnunet/nnUNet_preprocessed/Task501_BraTS2021/splits_final.pkl"
    result_dir = "/home/anning/Dataset/ProjData/nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTS2021/imageTr_val_0"

    flod = 0
    with open(split_final_file, 'rb') as f:
        split_data = pickle.load(f)

    val_ids = split_data[flod]['val']

    train_images_val = result_dir
    # train_images_val = join(result_dir, f'images_val_{flod}')
    # train_label_pred = join(result_dir, f'label_pred_{flod}')
    # train_label_val = join(result_dir, f'label_val_{flod}')
    maybe_mkdir_p(train_images_val)
    # maybe_mkdir_p(train_label_pred)
    # maybe_mkdir_p(train_label_val)

    patient_names = []
    cur = join(downloaded_data_dir)

    count = 1
    for p in val_ids:
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

        if not os.path.isfile(join(train_images_val, patient_name + "_0000.nii.gz")):
            shutil.copy(t1, join(train_images_val, patient_name + "_0000.nii.gz"))
        if not os.path.isfile(join(train_images_val, patient_name + "_0001.nii.gz")):
            shutil.copy(t1c, join(train_images_val, patient_name + "_0001.nii.gz"))
        if not os.path.isfile(join(train_images_val, patient_name + "_0002.nii.gz")):
            shutil.copy(t2, join(train_images_val, patient_name + "_0002.nii.gz"))
        if not os.path.isfile(join(train_images_val, patient_name + "_0003.nii.gz")):
            shutil.copy(flair, join(train_images_val, patient_name + "_0003.nii.gz"))
        # if not os.path.isfile(join(train_label_val, patient_name + ".nii.gz")):
        #     copy_BraTS_segmentation_and_convert_labels(seg, join(train_label_val, patient_name + ".nii.gz"))
