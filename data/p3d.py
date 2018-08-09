"""
Data loader for pascal VOC categories.

Should output:
    - img: B X 3 X H X W
    - kp: B X nKp X 2
    - mask: B X H X W
    - sfm_pose: B X 7 (s, tr, q)
    (kp, sfm_pose) correspond to image coordinates in [-1, 1]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data.dataloader import default_collate

from . import base as base_data
from ..utils import transformations

# -------------- flags ------------- #
# ---------------------------------- #
curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

flags.DEFINE_string('p3d_dir', '/data1/shubhtuls/cachedir/PASCAL3D+_release1.1', 'PASCAL Data Directory')
flags.DEFINE_string('p3d_anno_path', osp.join(cache_path, 'p3d'), 'Directory where pascal annotations are saved')
flags.DEFINE_string('p3d_class', 'car', 'PASCAL VOC category name')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #
class P3dDataset(base_data.BaseDataset):
    ''' 
    VOC Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super(P3dDataset, self).__init__(opts, filter_key=filter_key)
        self.img_dir = osp.join(opts.p3d_dir, 'Images')
        self.kp_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_kps.mat'.format(opts.p3d_class))
        self.anno_path = osp.join(
            opts.p3d_anno_path, 'data', '{}_{}.mat'.format(opts.p3d_class, opts.split))
        self.anno_sfm_path = osp.join(
            opts.p3d_anno_path, 'sfm', '{}_{}.mat'.format(opts.p3d_class, opts.split))

        # Load the annotation file.
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']
        self.kp_perm = sio.loadmat(
            self.kp_path, struct_as_record=False, squeeze_me=True)['kp_perm_inds'] - 1

        opts.num_kps = len(self.kp_perm)
        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts):
    return base_data.base_loader(P3dDataset, opts.batch_size, opts, filter_key=None)


def kp_data_loader(batch_size, opts):
    return base_data.base_loader(P3dDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    return base_data.base_loader(P3dDataset, batch_size, opts, filter_key='mask')


def sfm_data_loader(batch_size, opts):
    return base_data.base_loader(P3dDataset, batch_size, opts, filter_key='sfm_pose')

