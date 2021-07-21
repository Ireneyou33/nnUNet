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


import torch

from nnunet.training.network_training.nnUNetPlusPlusTrainerV2 import nnUNetPlusPlusTrainerV2
from nnunet.training.network_training.competitions_with_custom_Trainers.BraTS2021.nnUNetPlusPlusTrainerV2BraTSRegions_moreDA import \
    nnUNetPlusPlusTrainerV2BraTSRegions_DA3_BN_BD, nnUNetPlusPlusTrainerV2BraTSRegions_DA4_BN, nnUNetPlusPlusTrainerV2BraTSRegions_DA4_BN_BD


class nnUNetPlusPlusTrainerV2BraTS_Adam(nnUNetPlusPlusTrainerV2):
    """
    Info for Fabian: same as internal nnUNetPlusPlusTrainerV2_2
    """

    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.patience = 30  # 如果 50 个轮次MA没有减低，停止训练
        self.max_num_epochs = 160  # anning 2021-07-13 from 1000 to 160 40000 iterations
        self.initial_lr = 1e-3  # anning 2021-07-13 from 1e-2 to 1e-3

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr)
        self.lr_scheduler = None


class nnUNetPlusPlusTrainerV2BraTSRegions_DA3_BN_BD_Adam(nnUNetPlusPlusTrainerV2BraTSRegions_DA3_BN_BD):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.patience = 30  # 如果 50 个轮次MA没有减低，停止训练
        self.max_num_epochs = 160  # anning 2021-07-13 from 1000 to 160 40000 iterations
        self.initial_lr = 1e-3  # anning 2021-07-13 from 1e-2 to 1e-3

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr)
        self.lr_scheduler = None


class nnUNetPlusPlusTrainerV2BraTSRegions_DA4_BN_Adam(nnUNetPlusPlusTrainerV2BraTSRegions_DA4_BN):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.patience = 30  # 如果 50 个轮次MA没有减低，停止训练
        self.max_num_epochs = 160  # anning 2021-07-13 from 1000 to 160 40000 iterations
        self.initial_lr = 1e-3  # anning 2021-07-13 from 1e-2 to 1e-3

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr)
        self.lr_scheduler = None


class nnUNetPlusPlusTrainerV2BraTSRegions_DA4_BN_BD_Adam(nnUNetPlusPlusTrainerV2BraTSRegions_DA4_BN_BD):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.patience = 30  # 如果 50 个轮次MA没有减低，停止训练
        self.max_num_epochs = 160  # anning 2021-07-13 from 1000 to 160 40000 iterations
        self.initial_lr = 1e-3  # anning 2021-07-13 from 1e-2 to 1e-3

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr)
        self.lr_scheduler = None
