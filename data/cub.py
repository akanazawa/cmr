"""
CUB has 11788 images total, for 200 subcategories.
5994 train, 5794 test images.

After removing images that are truncated:
min kp threshold 6: 5964 train, 5771 test.
min_kp threshold 7: 5937 train, 5747 test.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np

import scipy.io as sio
from absl import flags, app

import torch
from torch.utils.data import Dataset

from . import base as base_data
from ..utils import transformations

# -------------- flags ------------- #
# ---------------------------------- #
if osp.exists('/scratch1/storage'):
    kData = '/scratch1/storage/CUB'
elif osp.exists('/data1/shubhtuls'):
    kData = '/data0/shubhtuls/datasets/CUB'
else:  # Savio
    kData = '/global/home/users/kanazawa/scratch/CUB'
    
flags.DEFINE_string('cub_dir', kData, 'CUB Data Directory')

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')
flags.DEFINE_string('cub_cache_dir', osp.join(cache_path, 'cub'), 'CUB Data Directory')

opts = flags.FLAGS

# -------------- Dataset ------------- #
# ------------------------------------ #
class CUBDataset(base_data.BaseDataset):
    '''
    CUB Data loader
    '''

    def __init__(self, opts, filter_key=None):
        super(CUBDataset, self).__init__(opts, filter_key=filter_key)
        self.data_dir = opts.cub_dir
        self.data_cache_dir = opts.cub_cache_dir

        self.img_dir = osp.join(self.data_dir, 'images')
        self.anno_path = osp.join(self.data_cache_dir, 'data', '%s_cub_cleaned.mat' % opts.split)
        self.anno_sfm_path = osp.join(self.data_cache_dir, 'sfm', 'anno_%s.mat' % opts.split)

        if not osp.exists(self.anno_path):
            print('%s doesnt exist!' % self.anno_path)
            import ipdb; ipdb.set_trace()
        self.filter_key = filter_key

        # Load the annotation file.
        print('loading %s' % self.anno_path)
        self.anno = sio.loadmat(
            self.anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.anno_sfm = sio.loadmat(
            self.anno_sfm_path, struct_as_record=False, squeeze_me=True)['sfm_anno']

        self.num_imgs = len(self.anno)
        print('%d images' % self.num_imgs)
        self.kp_perm = np.array([1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15]) - 1;


#----------- Data Loader ----------#
#----------------------------------#
def data_loader(opts, shuffle=True):
    return base_data.base_loader(CUBDataset, opts.batch_size, opts, filter_key=None, shuffle=shuffle)


def kp_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='kp')


def mask_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='mask')

    
def sfm_data_loader(batch_size, opts):
    return base_data.base_loader(CUBDataset, batch_size, opts, filter_key='sfm_pose')
