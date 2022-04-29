import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, force_fp32

from ..builder import H_DECODERS
from core import solve_DLT, spatial_transform_layer, fast_cost_volume, CCL
from ..losses import mask_reprojection_loss


@H_DECODERS.register_module()
class UDISDecoder(BaseModule):
    """Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images"""
    def __init__(self, 
                 feat_size=[(16, 16), (32, 32), (64, 64)],
                 strides=[8, 4, 2],
                 search_range=[16, 8, 4],
                 reg_channels=[512, 256, 128],
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=None,
                 init_cfg=[dict(type='Kaiming', layer=['Conv2d']),
                           dict(type='Constant', val=1., layer=['_BatchNorm']),
                           dict(type='Normal', std=0.01, layer=['Linear'])]):
        super(UDISDecoder, self).__init__(init_cfg)
        assert len(feat_size) == len(strides) \
                              == len(search_range) \
                              == len(reg_channels)
        self.num_levels = len(feat_size)
        self.search_range = search_range
        self.strides = strides
        self.feat_size = feat_size
        self.image_size = strides[0] * feat_size[0][0]
        self.build_reg_net(search_range, feat_size, reg_channels, act_cfg, norm_cfg, init_cfg)
        
    def build_reg_net(self, search_range, feat_size, reg_channels, act_cfg, norm_cfg, init_cfg):
        self.reg_net1 = SubRegressionNet((search_range[0]*2+1)**2, feat_size[0][0], reg_channels[0], [1, 1], act_cfg, norm_cfg, init_cfg)
        self.reg_net2 = SubRegressionNet((search_range[1]*2+1)**2, feat_size[0][0], reg_channels[1], [1, 2], act_cfg, norm_cfg, init_cfg)
        self.reg_net3 = SubRegressionNet((search_range[2]*2+1)**2, feat_size[0][0], reg_channels[2], [2, 2], act_cfg, norm_cfg, init_cfg)

    def forward(self, mlvl_feats1, mlvl_feats2):
        """Compute four points offset from multi-level features."""
        assert len(mlvl_feats1) == len(mlvl_feats2) == self.num_levels
        assert mlvl_feats1[-1].shape[2] == self.feat_size[0][0]
        
        # Regression Net1
        global_correlation = fast_cost_volume(F.normalize(mlvl_feats1[-1], dim=1, p=2), F.normalize(mlvl_feats2[-1], dim=1, p=2), self.search_range[0])
        offset1 = self.reg_net1(global_correlation)
        H1 = solve_DLT(offset1[..., None]/self.strides[1], self.feat_size[1][0])
        
        # Regression Net2
        norm_feat1 = F.normalize(mlvl_feats1[-2], dim=1, p=2)
        warp_feat2 = spatial_transform_layer(F.normalize(mlvl_feats2[-2], dim=1, p=2), H1) 
        local_correlation2 = fast_cost_volume(norm_feat1, warp_feat2, self.search_range[1])
        offset2 = self.reg_net2(local_correlation2)
        H2 = solve_DLT((offset1+offset2)[..., None]/self.strides[2], self.feat_size[2][0])
        
        # Regression Net3
        norm_feat1 = F.normalize(mlvl_feats1[-3], dim=1, p=2)
        warp_feat2 = spatial_transform_layer(F.normalize(mlvl_feats2[-3], dim=1, p=2), H2)
        local_correlation3 = fast_cost_volume(norm_feat1, warp_feat2, self.search_range[2])
        offset3 = self.reg_net3(local_correlation3)
        
        return offset1[..., None], offset2[..., None], offset3[..., None]
        
    def forward_train(self,
                      x1,
                      x2,
                      raw_img1,
                      raw_img2,
                      img_metas,
                      **kwargs):
        """
        Args:
            x1, x2: Features from feature_extractor.
            raw_img1, raw_img2: Images without any preprocessing.
            img_metas: Meta information of each imag.
        Returns:
            losses: A dictionary of loss components.
        """
        offset1, offset2, offset3 = self(x1, x2)
        H1 = solve_DLT(offset1, self.image_size)
        H2 = solve_DLT(offset1+offset2, self.image_size)
        H3 = solve_DLT(offset1+offset2+offset3, self.image_size)
        losses = dict()
        losses['H1_loss'] = 16. * mask_reprojection_loss(raw_img1, raw_img2, H1)
        losses['H2_loss'] = 4. * mask_reprojection_loss(raw_img1, raw_img2, H2)
        losses['H3_loss'] = 1. * mask_reprojection_loss(raw_img1, raw_img2, H3)
        return losses

    def forward_test(self, 
                     x1,
                     x2,
                     img_metas,
                     **kwargs):
        """
        Args:
            x1, x2: Features from feature_extractor.
            img_metas: Meta information of each imag.
        Returns:
            H_mat: Homography matrix. x1 = H_mat@x2
        """
        offset1, offset2, offset3 = self(x1, x2)
        offset = offset1+offset2+offset3
        image_size = []
        scale = []
        device = offset.device
        for meta in img_metas:
            image_size.append(torch.tensor(meta['ori_shape'][1::-1]).to(device))
            scale.append(torch.from_numpy(meta['scale_factor']).to(device))
        scale = torch.stack(scale, dim=0).repeat(1, 2)[..., None]
        rescale_offset = offset / scale
        image_size = torch.stack(image_size, dim=0)[..., None]
        
        H_mat = solve_DLT(rescale_offset, image_size)
        return H_mat


@H_DECODERS.register_module()
class UDISDecoderCCL(UDISDecoder):
    def build_reg_net(self, search_range, feat_size, reg_channels, act_cfg, norm_cfg, init_cfg):
        self.reg_net1 = SubRegressionNet(2, feat_size[0][0], reg_channels[0], [1, 1], act_cfg, norm_cfg, init_cfg)
        self.reg_net2 = SubRegressionNet(2, feat_size[0][0], reg_channels[1], [1, 2], act_cfg, norm_cfg, init_cfg)
        self.reg_net3 = SubRegressionNet(2, feat_size[0][0], reg_channels[2], [2, 2], act_cfg, norm_cfg, init_cfg)
 
    def forward(self, mlvl_feats1, mlvl_feats2):
        """Compute four points offset from multi-level features."""
        assert len(mlvl_feats1) == len(mlvl_feats2) == self.num_levels
        assert mlvl_feats1[-1].shape[2] == self.feat_size[0][0]
        
        # Regression Net1
        global_correlation = CCL(F.normalize(mlvl_feats1[-1], dim=1, p=2), F.normalize(mlvl_feats2[-1], dim=1, p=2))
        offset1 = self.reg_net1(global_correlation)
        H1 = solve_DLT(offset1[..., None]/self.strides[1], self.feat_size[1][0])
        
        # Regression Net2
        norm_feat1 = F.normalize(mlvl_feats1[-2], dim=1, p=2)
        warp_feat2 = spatial_transform_layer(F.normalize(mlvl_feats2[-2], dim=1, p=2), H1) 
        local_correlation2 = CCL(norm_feat1, warp_feat2)
        offset2 = self.reg_net2(local_correlation2)
        H2 = solve_DLT((offset1+offset2)[..., None]/self.strides[2], self.feat_size[2][0])
        
        # Regression Net3
        norm_feat1 = F.normalize(mlvl_feats1[-3], dim=1, p=2)
        warp_feat2 = spatial_transform_layer(F.normalize(mlvl_feats2[-3], dim=1, p=2), H2)
        local_correlation3 = CCL(norm_feat1, warp_feat2)
        offset3 = self.reg_net3(local_correlation3)
        
        return offset1[..., None], offset2[..., None], offset3[..., None]
    

class SubRegressionNet(BaseModule):
    def __init__(self, 
                 in_channels, 
                 base_size,
                 channels,
                 strides,
                 act_cfg=dict(type='ReLU'),
                 norm_cfg=None,
                 init_cfg=None):
        super(SubRegressionNet, self).__init__(init_cfg)
        self.conv1 = ConvModule(in_channels, channels, 3, padding=1, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(channels, channels, 3, stride=strides[0], padding=1, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.conv3 = ConvModule(channels, channels, 3, stride=strides[1], padding=1, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.fc1 = nn.Linear(base_size**2*channels, channels*2)
        self.fc2 = nn.Linear(channels*2, 8)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.drop(x)
        x = self.fc2(x)
        return x

