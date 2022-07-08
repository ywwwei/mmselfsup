import os
import os.path as osp
import mmcv
import numpy as np
from PIL import Image

from ..builder import DATASOURCES
from .base import BaseDataSource

def fhas_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_folders(root):
    """Find classes by folders under a root.

    Args:
        root (string): root directory of folders

    Returns:
        folder_to_idx (dict): the map from folder name to class idx
    """
    folders = [d for d in os.listdir(root) if osp.isdir(osp.join(root, d))]
    folders.sort()
    folder_to_idx = {folders[i]: i for i in range(len(folders))}
    return folder_to_idx

def get_samples(root):
    """Make dataset from .npy
    """
    samples = []
    source = np.load(root) # filename list
    for i in range(len(source)):
        path = source[i]
        item = (path, 0) # mannully set gt 0
        samples.append(item)
    return samples

@DATASOURCES.register_module()
class BigearthNet(BaseDataSource):
    def get_img(self, idx):
        #bigearthnrgb filename_.npy
        filename = self.data_infos[idx]['img_info']['filename']
        img = np.load(filename)
        #print(img)
        #print(img.shape)
        img = img.astype(np.uint8)
        return Image.fromarray(img)

    def load_annotations(self):
        samples = get_samples(self.ann_file)
        self.samples = samples

        data_infos = []
        for i, (filename, gt_label) in enumerate(self.samples):
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['idx'] = int(i)
            data_infos.append(info)
        return data_infos
