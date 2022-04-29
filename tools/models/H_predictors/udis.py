import torch
from .base import BasePredictor
from ..builder import H_PREDICTORS, build_feature_extractor, build_H_decoder


@H_PREDICTORS.register_module()
class UDIS_H_Predictor(BasePredictor):
    """Unsupervised Deep Image Stitching: Reconstructing Stitched Features to Images"""
    def __init__(self, 
                 feature_extractor,
                 H_decoder,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(UDIS_H_Predictor, self).__init__(init_cfg)
        if pretrained:
            feature_extractor.pretrained = pretrained
        self.feature_extractor = build_feature_extractor(feature_extractor)
        self.H_decoder = build_H_decoder(H_decoder)
        if neck is not None:
            self.neck = build_neck(neck)

    @property
    def with_neck(self):
        """bool: whether the predictor has a neck"""
        return hasattr(self, 'neck') and self.neck is not None
    
    def forward_train(self, 
                      img1, 
                      img2,
                      img_metas, 
                      raw_img1,
                      raw_img2,
                      **kwargs):
        mlvl_feats1 = self.feature_extractor(img1)
        mlvl_feats2 = self.feature_extractor(img2)
        if self.with_neck:
            mlvl_feats1 = self.neck(mlvl_feats1)
            mlvl_feats2 = self.neck(mlvl_feats2)
        losses = self.H_decoder.forward_train(mlvl_feats1, mlvl_feats2, raw_img1, raw_img2, img_metas, **kwargs)
        return losses

    def forward_test(self, 
                      img1, 
                      img2,
                      img_metas, 
                      **kwargs):
        mlvl_feats1 = self.feature_extractor(img1)
        mlvl_feats2 = self.feature_extractor(img2)
        if self.with_neck:
            mlvl_feats1 = self.neck(mlvl_feats1)
            mlvl_feats2 = self.neck(mlvl_feats2)
        H_mat = self.H_decoder.forward_test(mlvl_feats1, mlvl_feats2, img_metas)
        return H_mat
