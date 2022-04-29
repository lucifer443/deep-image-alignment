import torch
import torch.nn.functional as F

from core import spatial_transform_layer

def mask_reprojection_loss(image1, image2, H, normalize=False):
    """Loss function used in Unsupervised Deep Image Stitching: Reconstructing
       Stitched Features to Images (https://arxiv.org/pdf/2106.12859.pdf).
    Args:
        image1, image2: Two images needed to align.
        H: Homography matrix.
    Returns:
        loss: Reprojection error only computed in overlap area.
    """
    warp_image2 = spatial_transform_layer(image2, H)
    overlap_mask = spatial_transform_layer(torch.ones_like(image2), H)
    if normalize:
        loss = (warp_image2-image1*overlap_mask).abs().sum(dim=[1, 2, 3]) / (overlap_mask.sum(dim=[1, 2, 3])+1e-7)
        loss = loss.mean()
    else:
        loss = (warp_image2-image1*overlap_mask).abs().mean()
    return loss
