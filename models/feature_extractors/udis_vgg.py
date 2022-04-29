import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmdet.models import BACKBONES

def _conv_block(in_channels, out_channels, norm_cfg):
    block = nn.Sequential(
                ConvModule(in_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg),
                ConvModule(out_channels, out_channels, 3, padding=1, norm_cfg=norm_cfg)
            )
    return block

@BACKBONES.register_module()
class UDISVGG(BaseModule):
    def __init__(self,
                 out_indices=[1, 2, 3],
                 block_setting=[64, 64, 128, 128],
                 norm_cfg=None,
                 init_cfg=[dict(type='Kaiming', layer=['Conv2d']),
                           dict(type='Constant', val=1., layer=['_BatchNorm'])]):
        super(UDISVGG, self).__init__(init_cfg)
        self.num_blocks = len(block_setting)
        assert max(out_indices) < self.num_blocks
        self.out_indices = out_indices
        in_channels = 1
        self.layers = []
        for i, out_channels in enumerate(block_setting):
            block = _conv_block(in_channels, out_channels, norm_cfg)
            layer_name = f'block{i + 1}'
            self.add_module(layer_name, block)
            self.layers.append(layer_name)
            in_channels = out_channels

    def forward(self, x):
        x = x.mean(dim=1, keepdim=True)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x) if i==0 else layer(F.max_pool2d(x, kernel_size=2, stride=2))
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

