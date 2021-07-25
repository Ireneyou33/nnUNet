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

import os
import numpy as np
import torch
from torch import nn
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.evaluation.region_based_evaluation import get_brats_regions
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *

from nnunet.training.network_training.competitions_with_custom_Trainers.BraTS2020.nnUNetTrainerV2BraTSRegions_moreDA import \
    nnUNetTrainerV2BraTSRegions_DA3_BN_BD, nnUNetTrainerV2BraTSRegions_DA4_BN, nnUNetTrainerV2BraTSRegions_DA4_BN_BD
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.network_architecture.seg_hrnet_ocr_3d import HighResolutionNet
from nnunet.network_architecture.seg_hrnet_ocr_config_default import _C as config_default

nnunet_path = os.path.dirname(os.path.abspath(__file__))
for i in range(4):
    nnunet_path = os.path.dirname(nnunet_path)
print(f"nnunet_path: {nnunet_path}")

CONFIG = os.path.join(nnunet_path, "brats2021", "seg_hrnet_ocr_w18_128_128_128.yaml")


def update_config(cfg, config_file):
    cfg.defrost()

    cfg.merge_from_file(config_file)

    cfg.freeze()
    return cfg


class nnUNetHRNetOCRTrainerV2BraTS_Adam(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
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

    def initialize_network(self):
        """
        :return:
        """
        config = update_config(config_default, CONFIG)
        self.network = HighResolutionNet(config=config)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            if force_load_plans or (self.plans is None):
                self.load_plans_file()
            self.plans['plans_per_stage'][0]['patch_size'] = [64, 64, 64]
            self.plans['plans_per_stage'][0]['batch_size'] = 1
            self.process_plans(self.plans)

            self.setup_DA_params()
            self.deep_supervision_scales = None

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights
            # now wrap the loss
            self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:
                    print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
            else:
                pass

            self.initialize_network()
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                print("output.shape, target.shape: ", output.shape, target.shape)
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()


class nnUNetHRNetOCRTrainerV2BraTSRegions_DA3_BN_BD_Adam(nnUNetTrainerV2BraTSRegions_DA3_BN_BD):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.regions = get_brats_regions()
        self.regions_class_order = (1, 3, 2)  # 如果是(1, 2, 3)，和baseline的标签不同，需要改为(1, 3, 2)
        self.patience = 30  # 如果 50 个轮次MA没有减低，停止训练
        self.max_num_epochs = 160  # anning 2021-07-13 from 1000 to 160 40000 iterations
        self.initial_lr = 1e-3  # anning 2021-07-13 from 1e-2 to 1e-3

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr)
        self.lr_scheduler = None


class nnUNetHRNetOCRTrainerV2BraTSRegions_DA4_BN_Adam(nnUNetTrainerV2BraTSRegions_DA4_BN):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.regions = get_brats_regions()
        self.regions_class_order = (1, 3, 2)  # 如果是(1, 2, 3)，和baseline的标签不同，需要改为(1, 3, 2)
        self.patience = 30  # 如果 50 个轮次MA没有减低，停止训练
        self.max_num_epochs = 160  # anning 2021-07-13 from 1000 to 160 40000 iterations
        self.initial_lr = 1e-3  # anning 2021-07-13 from 1e-2 to 1e-3

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr)
        self.lr_scheduler = None


class nnUNetHRNetOCRTrainerV2BraTSRegions_DA4_BN_BD_Adam(nnUNetTrainerV2BraTSRegions_DA4_BN_BD):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        self.regions = get_brats_regions()
        self.regions_class_order = (1, 3, 2)  # 如果是(1, 2, 3)，和baseline的标签不同，需要改为(1, 3, 2)
        self.patience = 30  # 如果 50 个轮次MA没有减低，停止训练
        self.max_num_epochs = 160  # anning 2021-07-13 from 1000 to 160 40000 iterations
        self.initial_lr = 1e-3  # anning 2021-07-13 from 1e-2 to 1e-3

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr)
        self.lr_scheduler = None
