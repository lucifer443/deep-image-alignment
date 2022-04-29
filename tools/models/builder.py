from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from mmdet.models import build_backbone

MODELS = Registry('models', parent=MMCV_MODELS)

H_DECODERS = MODELS
H_PREDICTORS = MODELS

build_feature_extractor = build_backbone


def build_H_decoder(cfg):
    """Build H matrix decoder."""
    return H_DECODERS.build(cfg)


def build_H_predictor(cfg):
    """Build H matrix predictor."""
    return H_PREDICTORS.build(cfg)

