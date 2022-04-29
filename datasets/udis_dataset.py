import mmcv
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose

@DATASETS.register_module()
class UDISDataset(Dataset):
    """UDIS Dataset. (https://github.com/nie-lang/UnsupervisedDeepImageStitching)"""
    def __init__(self,
                 ann_file,
                 pipeline,
                 img_prefix='',
                 file_client_args=dict(backend='disk')):
        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.file_client = mmcv.FileClient(**file_client_args)

        # load annotations
        with self.file_client.get_local_path(self.ann_file) as local_path:
            self.data_infos = self.load_annotations(local_path)

        self.flag = np.ones(len(self), dtype=np.uint8)
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        return mmcv.load(ann_file)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data .
        """
        results = dict(img_path=self.data_infos[idx], img_prefix=self.img_prefix)
        return self.pipeline(results)






