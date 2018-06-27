"""
Takes an image, returns stuff.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import scipy.misc
import torch
import torchvision
from torch.autograd import Variable
import scipy.io as sio

from ..nnutils import mesh_net
from ..nnutils import geom_utils
from ..nnutils.nmr import NeuralRenderer
from ..utils import bird_vis

# These options are off by default, but used for some ablations reported.
flags.DEFINE_boolean('ignore_pred_delta_v', False, 'Use only mean shape for prediction')
flags.DEFINE_boolean('use_sfm_ms', False, 'Uses sfm mean shape for prediction')
flags.DEFINE_boolean('use_sfm_camera', False, 'Uses sfm mean camera')


class MeshPredictor(object):
    def __init__(self, opts):
        self.opts = opts

        self.symmetric = opts.symmetric

        img_size = (opts.img_size, opts.img_size)
        print('Setting up model..')
        self.model = mesh_net.MeshNet(img_size, opts, nz_feat=opts.nz_feat)

        self.load_network(self.model, 'pred', self.opts.num_train_epoch)
        self.model.eval()
        self.model = self.model.cuda(device=self.opts.gpu_id)

        self.renderer = NeuralRenderer(opts.img_size)

        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        if opts.use_sfm_ms:
            anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_testval.mat')
            anno_sfm = sio.loadmat(
                anno_sfm_path, struct_as_record=False, squeeze_me=True)
            sfm_mean_shape = torch.Tensor(np.transpose(anno_sfm['S'])).cuda(
                device=opts.gpu_id)
            self.sfm_mean_shape = Variable(sfm_mean_shape, requires_grad=False)
            self.sfm_mean_shape = self.sfm_mean_shape.unsqueeze(0).repeat(
                opts.batch_size, 1, 1)
            sfm_face = torch.LongTensor(anno_sfm['conv_tri'] - 1).cuda(
                device=opts.gpu_id)
            self.sfm_face = Variable(sfm_face, requires_grad=False)
            faces = self.sfm_face.view(1, -1, 3)
        else:
            # For visualization
            faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.vis_rend = bird_vis.VisRenderer(opts.img_size,
                                             faces.data.cpu().numpy())
        self.vis_rend.set_bgcolor([1., 1., 1.])

        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '{}_net_{}.pth'.format(network_label, epoch_label)
        network_dir = os.path.join(self.opts.checkpoint_dir, self.opts.name)
        save_path = os.path.join(network_dir, save_filename)
        print('loading {}..'.format(save_path))
        network.load_state_dict(torch.load(save_path))

        return

    def set_input(self, batch):
        opts = self.opts

        # original image where texture is sampled from.
        img_tensor = batch['img'].clone().type(torch.FloatTensor)

        # input_img is the input to resnet
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        if opts.use_sfm_camera:
            cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)
            self.sfm_cams = Variable(
                cam_tensor.cuda(device=opts.gpu_id), requires_grad=False)

    def predict(self, batch):
        """
        batch has B x C x H x W numpy
        """
        self.set_input(batch)
        self.forward()
        return self.collect_outputs()

    def forward(self):
        if self.opts.texture:
            pred_codes, self.textures = self.model.forward(self.input_imgs)
        else:
            pred_codes = self.model.forward(self.input_imgs)

        self.delta_v, scale, trans, quat = pred_codes

        if self.opts.use_sfm_camera:
            self.cam_pred = self.sfm_cams
        else:
            self.cam_pred = torch.cat([scale, trans, quat], 1)

        del_v = self.model.symmetrize(self.delta_v)
        # Deform mean shape:
        self.mean_shape = self.model.get_mean_shape()

        if self.opts.use_sfm_ms:
            self.pred_v = self.sfm_mean_shape
        elif self.opts.ignore_pred_delta_v:
            self.pred_v = self.mean_shape + del_v*0
        else:
            self.pred_v = self.mean_shape + del_v

        # Compute keypoints.
        if self.opts.use_sfm_ms:
            self.kp_verts = self.pred_v
        else:
            self.vert2kp = torch.nn.functional.softmax(
                self.model.vert2kp, dim=1)
            self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts,
                                                    self.cam_pred)
        self.mask_pred = self.renderer.forward(self.pred_v, self.faces,
                                               self.cam_pred)

        # Render texture.
        if self.opts.texture and not self.opts.use_sfm_ms:
            if self.textures.size(-1) == 2:
                # Flow texture!
                self.texture_flow = self.textures
                self.textures = geom_utils.sample_textures(self.textures,
                                                           self.imgs)
            if self.textures.dim() == 5:  # B x F x T x T x 3
                tex_size = self.textures.size(2)
                self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1,
                                                                  tex_size, 1)

            # Render texture:
            self.texture_pred = self.tex_renderer.forward(
                self.pred_v, self.faces, self.cam_pred, textures=self.textures)

            # B x 2 x H x W
            uv_flows = self.model.texture_predictor.uvimage_pred
            # B x H x W x 2
            self.uv_flows = uv_flows.permute(0, 2, 3, 1)
            self.uv_images = torch.nn.functional.grid_sample(self.imgs,
                                                             self.uv_flows)
        else:
            self.textures = None

    def collect_outputs(self):
        outputs = {
            'kp_pred': self.kp_pred.data,
            'verts': self.pred_v.data,
            'kp_verts': self.kp_verts.data,
            'cam_pred': self.cam_pred.data,
            'mask_pred': self.mask_pred.data,
        }
        if self.opts.texture and not self.opts.use_sfm_ms:
            outputs['texture'] = self.textures
            outputs['texture_pred'] = self.texture_pred.data
            outputs['uv_image'] = self.uv_images.data
            outputs['uv_flow'] = self.uv_flows.data

        return outputs
