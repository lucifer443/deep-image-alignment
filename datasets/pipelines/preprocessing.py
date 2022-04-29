import numpy as np
import os.path as osp
import mmcv
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import LoadImageFromFile, PhotoMetricDistortion


@PIPELINES.register_module()
class LoadImagePairFromFile(LoadImageFromFile):
    """Load image pair from file."""

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        if results['img_prefix'] is not None:
            filename1 = osp.join(results['img_prefix'],
                                 results['img_path'][0])
            filename2 = osp.join(results['img_prefix'],
                                 results['img_path'][1])
        else:
            filename1 = results['img_path'][0]
            filename2 = results['img_path'][1]

        img_bytes1 = self.file_client.get(filename1)
        img_bytes2 = self.file_client.get(filename2)
        img1 = mmcv.imfrombytes(img_bytes1, flag=self.color_type)
        img2 = mmcv.imfrombytes(img_bytes2, flag=self.color_type)
        if self.to_float32:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

        results['filename'] = (filename1, filename2)
        results['ori_filename'] = results['img_path']
        results['img1'] = img1
        results['img2'] = img2
        results['img_shape'] = img2.shape
        results['ori_shape'] = img2.shape
        results['img_fields'] = ['img1', 'img2']
        return results


@PIPELINES.register_module()
class BeforeDataAugment:
    """Save a copy of raw image."""
    def __call__(self, results):
        results['raw_img1'] = results['img1'].copy()
        results['raw_img2'] = results['img2'].copy()
        return results


@PIPELINES.register_module()
class PairPhotoMetricDistortion(PhotoMetricDistortion):
    """Call function to perform photometric distortion on image pairs."""
    def __call__(self, results):
        for key in results['img_fields']:
            results[key] = super(PairPhotoMetricDistortion, self).__call__(dict(img=results[key]))['img']
        return results
