import numpy as np
import torch
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


class nnUNetTrainerTBADResidual(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # TBAD change 2: stronger foreground oversampling
        self.oversample_foreground_percent = 0.5


    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True
    ):

        # TBAD change 1: use ResidualEncoderUNet
        new_arch_class_name = \
            "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet"

        new_kwargs = dict(arch_init_kwargs)

        if "n_conv_per_stage" in new_kwargs and "n_blocks_per_stage" not in new_kwargs:
            new_kwargs["n_blocks_per_stage"] = new_kwargs.pop("n_conv_per_stage")

        return get_network_from_plans(
            new_arch_class_name,
            new_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )


    def _build_loss(self):

        loss = super()._build_loss()

        if hasattr(loss, "weights"):
            weights = np.array(loss.weights)
            weights[0] = weights[0] * 1.5
            weights = weights / weights.sum()
            loss.weights = weights

        return loss