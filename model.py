# model.py
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def get_unet_model(input_channels=1, out_channels=1):
    model = UNet(
        spatial_dims=2,
        in_channels=input_channels,
        out_channels=out_channels,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )
    return model
