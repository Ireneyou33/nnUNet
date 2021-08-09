from batchgenerators.utilities.file_and_folder_operations import *
import os
import shutil
cotr_dir = "/mnt/ngshare/PersonData/qingyu/dataset/nnUNet/RESULTS_FOLDER/Predict_Result/Predict_Train/CoTr_IN_LeakyReLU/Out"
swin_dir = "/mnt/ngshare/PersonData/qingyu/dataset/nnUNet/RESULTS_FOLDER/Predict_Result/Predict_Train/nnUNetTrainerV2BraTS_SwinUnet_Adam/Out"
out_dir = "/mnt/ngshare/PersonData/qingyu/dataset/nnUNet/RESULTS_FOLDER/Predict_Result/Predict_Train/nnUNetTrainerV2BraTS_SwinUnet_Adam_SELECT/Out"

cotr_filelist = subfiles(cotr_dir)
swin_filelist = subfiles(swin_dir)

# print(cotr_filelist)


cotr_filename_list = [os.path.basename(i) for i in cotr_filelist]
cotr_filename_set = set(cotr_filename_list)


for swin_file in swin_filelist:
    swin_filename = os.path.basename(swin_file)
    if swin_filename in cotr_filename_set:
        print(swin_filename)
        shutil.copy(swin_file, join(out_dir, swin_filename))


