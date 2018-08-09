"""
Segmentation using mask rcnn.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import numpy as np
import scipy.io as sio

from caffe2.python import workspace

sys.path.append('/home/shubhtuls/packages/detectron')

import pycocotools.coco as coco
anno_file = '/home/shubhtuls/packages/cocoapi/data/annotations/instances_val2017.json'
coco_data = coco.COCO(anno_file)
coco_classes = coco_data.loadCats(coco_data.getCatIds())
coco_classes = [cat['name'] for cat in coco_classes]

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def coco_to_pascal_name(category):
    if category == 'airplane':
        return 'aeroplane'
    if category == 'dining table':
        return 'diningtable'
    if category == 'motorcycle':
        return 'motorbike'
    if category == 'couch':
        return 'sofa'
    if category == 'tv':
        return 'tvmonitor'
    else:
        return category


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for i, im_name in enumerate(im_list):
        #if i > 10:
        #    continue
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name)[:-len(args.image_ext)] + 'mat')
        )
        
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        boxes, segms, keyps, classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
        if boxes is None:
            continue

        segms = vis_utils.mask_util.decode(segms)

        valid_inds = np.greater(boxes[:, 4], 0.5)
        boxes = boxes[valid_inds, :]
        segms = segms[:, :, valid_inds]
        classes = np.array(classes)[valid_inds]
        class_names = np.asarray(([coco_to_pascal_name(coco_classes[c-1]) for c in classes]), dtype='object')

        sio.savemat(out_name, {'masks': segms, 'boxes': boxes, 'classes': class_names});


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)