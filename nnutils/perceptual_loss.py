"""
Calls Richard's Perceptual Loss.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
from ..external.PerceptualSimilarity.models import dist_model


class PerceptualLoss(object):
    def __init__(self, model='net', net='alex', use_gpu=True):
        print('Setting up Perceptual loss..')
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=True)
        print('Done')

    def __call__(self, pred, target, normalize=True):
        """
        Pred and target are Variables.
        If normalize is on, scales images between [-1, 1]
        Assumes the inputs are in range [0, 1].
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        dist = self.model.forward_pair(target, pred)

        return dist