"""
Script for the bird shape, pose and texture experiment.

The model takes imgs, outputs the deformation to the mesh & camera parameters
Loss consists of:
- keypoint reprojection loss
- mask reprojection loss
- smoothness/laplacian priors on triangles
- texture reprojection losses

example usage : python -m cmr.experiments.shape --name=bird_shape --plot_scalars --save_epoch_freq=1 --batch_size=8 --display_visuals --display_freq=2000
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

import os.path as osp
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import scipy.io as sio
from collections import OrderedDict

from ..data import cub as cub_data
from ..utils import visutil
from ..utils import bird_vis
from ..utils import image as image_utils
from ..nnutils import train_utils
from ..nnutils import loss_utils
from ..nnutils import mesh_net
from ..nnutils.nmr import NeuralRenderer
from ..nnutils import geom_utils

flags.DEFINE_string('dataset', 'cub', 'cub or pascal or p3d')
# Weights:
flags.DEFINE_float('kp_loss_wt', 30., 'keypoint loss weight')
flags.DEFINE_float('mask_loss_wt', 2., 'mask loss weight')
flags.DEFINE_float('cam_loss_wt', 2., 'weights to camera loss')
flags.DEFINE_float('deform_reg_wt', 10., 'reg to deformation')
flags.DEFINE_float('triangle_reg_wt', 30., 'weights to triangle smoothness prior')
flags.DEFINE_float('vert2kp_loss_wt', .16, 'reg to vertex assignment')
flags.DEFINE_float('tex_loss_wt', .5, 'weights to tex loss')
flags.DEFINE_float('tex_dt_loss_wt', .5, 'weights to tex dt loss')
flags.DEFINE_boolean('use_gtpose', True, 'if true uses gt pose for projection, but camera still gets trained.')

opts = flags.FLAGS

curr_path = osp.dirname(osp.abspath(__file__))
cache_path = osp.join(curr_path, '..', 'cachedir')

class ShapeTrainer(train_utils.Trainer):
    def define_model(self):
        opts = self.opts

        # ----------
        # Options
        # ----------
        self.symmetric = opts.symmetric
        anno_sfm_path = osp.join(opts.cub_cache_dir, 'sfm', 'anno_train.mat')
        anno_sfm = sio.loadmat(anno_sfm_path, struct_as_record=False, squeeze_me=True)
        sfm_mean_shape = (np.transpose(anno_sfm['S']), anno_sfm['conv_tri']-1)

        img_size = (opts.img_size, opts.img_size)
        self.model = mesh_net.MeshNet(
            img_size, opts, nz_feat=opts.nz_feat, num_kps=opts.num_kps, sfm_mean_shape=sfm_mean_shape)

        if opts.num_pretrain_epochs > 0:
            self.load_network(self.model, 'pred', opts.num_pretrain_epochs)

        self.model = self.model.cuda(device=opts.gpu_id)

        # Data structures to use for triangle priors.
        edges2verts = self.model.edges2verts
        # B x E x 4
        edges2verts = np.tile(np.expand_dims(edges2verts, 0), (opts.batch_size, 1, 1))
        self.edges2verts = Variable(torch.LongTensor(edges2verts).cuda(device=opts.gpu_id), requires_grad=False)
        # For renderering.
        faces = self.model.faces.view(1, -1, 3)
        self.faces = faces.repeat(opts.batch_size, 1, 1)
        self.renderer = NeuralRenderer(opts.img_size)
        self.renderer_predcam = NeuralRenderer(opts.img_size) #for camera loss via projection

        # Need separate NMR for each fwd/bwd call.
        if opts.texture:
            self.tex_renderer = NeuralRenderer(opts.img_size)
            # Only use ambient light for tex renderer
            self.tex_renderer.ambient_light_only()

        # For visualization
        self.vis_rend = bird_vis.VisRenderer(opts.img_size, faces.data.cpu().numpy())
        return

    def init_dataset(self):
        opts = self.opts
        if opts.dataset == 'cub':
            self.data_module = cub_data
        else:
            print('Unknown dataset %d!' % opts.dataset)

        self.dataloader = self.data_module.data_loader(opts)
        self.resnet_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def define_criterion(self):
        self.projection_loss = loss_utils.kp_l2_loss
        self.mask_loss_fn = torch.nn.MSELoss()
        self.entropy_loss = loss_utils.entropy_loss
        self.deform_reg_fn = loss_utils.deform_l2reg
        self.camera_loss = loss_utils.camera_loss
        self.triangle_loss_fn = loss_utils.LaplacianLoss(self.faces)

        if self.opts.texture:
            self.texture_loss = loss_utils.PerceptualTextureLoss()
            self.texture_dt_loss_fn = loss_utils.texture_dt_loss


    def set_input(self, batch):
        opts = self.opts

        # Image with annotations.
        input_img_tensor = batch['img'].type(torch.FloatTensor)
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        img_tensor = batch['img'].type(torch.FloatTensor)
        mask_tensor = batch['mask'].type(torch.FloatTensor)
        kp_tensor = batch['kp'].type(torch.FloatTensor)
        cam_tensor = batch['sfm_pose'].type(torch.FloatTensor)

        self.input_imgs = Variable(
            input_img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.imgs = Variable(
            img_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.masks = Variable(
            mask_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.kps = Variable(
            kp_tensor.cuda(device=opts.gpu_id), requires_grad=False)
        self.cams = Variable(
            cam_tensor.cuda(device=opts.gpu_id), requires_grad=False)

        # Compute barrier distance transform.
        mask_dts = np.stack([image_utils.compute_dt_barrier(m) for m in batch['mask']])
        dt_tensor = torch.FloatTensor(mask_dts).cuda(device=opts.gpu_id)
        # B x 1 x N x N
        self.dts_barrier = Variable(dt_tensor, requires_grad=False).unsqueeze(1)

    def forward(self):
        opts = self.opts
        if opts.texture:
            pred_codes, self.textures = self.model.forward(self.input_imgs)
        else:
            pred_codes = self.model.forward(self.input_imgs)
        self.delta_v, scale, trans, quat = pred_codes

        self.cam_pred = torch.cat([scale, trans, quat], 1)

        if opts.only_mean_sym:
            del_v = self.delta_v
        else:
            del_v = self.model.symmetrize(self.delta_v)

        # Deform mean shape:
        self.mean_shape = self.model.get_mean_shape()
        self.pred_v = self.mean_shape + del_v

        # Compute keypoints.
        self.vert2kp = torch.nn.functional.softmax(self.model.vert2kp, dim=1)
        self.kp_verts = torch.matmul(self.vert2kp, self.pred_v)

        # Decide which camera to use for projection.
        if opts.use_gtpose:
            proj_cam = self.cams
        else:
            proj_cam = self.cam_pred

        # Project keypoints
        self.kp_pred = self.renderer.project_points(self.kp_verts, proj_cam)

        # Render mask.
        self.mask_pred = self.renderer.forward(self.pred_v, self.faces, proj_cam)

        if opts.texture:
            self.texture_flow = self.textures
            self.textures = geom_utils.sample_textures(self.texture_flow, self.imgs)
            tex_size = self.textures.size(2)
            self.textures = self.textures.unsqueeze(4).repeat(1, 1, 1, 1, tex_size, 1)
  
            self.texture_pred = self.tex_renderer.forward(self.pred_v.detach(), self.faces, proj_cam.detach(), textures=self.textures)
        else:
            self.textures = None

        # Compute losses for this instance.
        self.kp_loss = self.projection_loss(self.kp_pred, self.kps)
        self.mask_loss = self.mask_loss_fn(self.mask_pred, self.masks)
        self.cam_loss = self.camera_loss(self.cam_pred, self.cams, 0)

        if opts.texture:
            self.tex_loss = self.texture_loss(self.texture_pred, self.imgs, self.mask_pred, self.masks)
            self.tex_dt_loss = self.texture_dt_loss_fn(self.texture_flow, self.dts_barrier)

        # Priors:
        self.vert2kp_loss = self.entropy_loss(self.vert2kp)
        self.deform_reg = self.deform_reg_fn(self.delta_v)
        self.triangle_loss = self.triangle_loss_fn(self.pred_v)

        # Finally sum up the loss.
        # Instance loss:
        self.total_loss = opts.kp_loss_wt * self.kp_loss
        self.total_loss += opts.mask_loss_wt * self.mask_loss
        self.total_loss += opts.cam_loss_wt * self.cam_loss
        if opts.texture:
            self.total_loss += opts.tex_loss_wt * self.tex_loss

        # Priors:
        self.total_loss += opts.vert2kp_loss_wt * self.vert2kp_loss
        self.total_loss += opts.deform_reg_wt * self.deform_reg
        self.total_loss += opts.triangle_reg_wt * self.triangle_loss

        self.total_loss += opts.tex_dt_loss_wt * self.tex_dt_loss


    def get_current_visuals(self):
        vis_dict = {}
        mask_concat = torch.cat([self.masks, self.mask_pred], 2)


        if self.opts.texture:
            # B x 2 x H x W
            uv_flows = self.model.texture_predictor.uvimage_pred
            # B x H x W x 2
            uv_flows = uv_flows.permute(0, 2, 3, 1)
            uv_images = torch.nn.functional.grid_sample(self.imgs, uv_flows)

        num_show = min(2, self.opts.batch_size)
        show_uv_imgs = []
        show_uv_flows = []

        for i in range(num_show):
            input_img = bird_vis.kp2im(self.kps[i].data, self.imgs[i].data)
            pred_kp_img = bird_vis.kp2im(self.kp_pred[i].data, self.imgs[i].data)
            masks = bird_vis.tensor2mask(mask_concat[i].data)
            if self.opts.texture:
                texture_here = self.textures[i]
            else:
                texture_here = None

            rend_predcam = self.vis_rend(self.pred_v[i], self.cam_pred[i], texture=texture_here)
            # Render from front & back:
            rend_frontal = self.vis_rend.diff_vp(self.pred_v[i], self.cam_pred[i], texture=texture_here, kp_verts=self.kp_verts[i])
            rend_top = self.vis_rend.diff_vp(self.pred_v[i], self.cam_pred[i], axis=[0, 1, 0], texture=texture_here, kp_verts=self.kp_verts[i])
            diff_rends = np.hstack((rend_frontal, rend_top))

            if self.opts.texture:
                uv_img = bird_vis.tensor2im(uv_images[i].data)
                show_uv_imgs.append(uv_img)
                uv_flow = bird_vis.visflow(uv_flows[i].data)
                show_uv_flows.append(uv_flow)

                tex_img = bird_vis.tensor2im(self.texture_pred[i].data)
                imgs = np.hstack((input_img, pred_kp_img, tex_img))
            else:
                imgs = np.hstack((input_img, pred_kp_img))

            rend_gtcam = self.vis_rend(self.pred_v[i], self.cams[i], texture=texture_here)
            rends = np.hstack((diff_rends, rend_predcam, rend_gtcam))
            vis_dict['%d' % i] = np.hstack((imgs, rends, masks))
            vis_dict['masked_img %d' % i] = bird_vis.tensor2im((self.imgs[i] * self.masks[i]).data)

        if self.opts.texture:
            vis_dict['uv_images'] = np.hstack(show_uv_imgs)
            vis_dict['uv_flow_vis'] = np.hstack(show_uv_flows)

        return vis_dict


    def get_current_points(self):
        return {
            'mean_shape': visutil.tensor2verts(self.mean_shape.data),
            'verts': visutil.tensor2verts(self.pred_v.data),
        }

    def get_current_scalars(self):
        sc_dict = OrderedDict([
            ('smoothed_total_loss', self.smoothed_total_loss),
            ('total_loss', self.total_loss.data[0]),
            ('kp_loss', self.kp_loss.data[0]),
            ('mask_loss', self.mask_loss.data[0]),
            ('vert2kp_loss', self.vert2kp_loss.data[0]),
            ('deform_reg', self.deform_reg.data[0]),
            ('tri_loss', self.triangle_loss.data[0]),
            ('cam_loss', self.cam_loss.data[0]),
        ])
        if self.opts.texture:
            sc_dict['tex_loss'] = self.tex_loss.data[0]
            sc_dict['tex_dt_loss'] = self.tex_dt_loss.data[0]

        return sc_dict


def main(_):
    torch.manual_seed(0)
    trainer = ShapeTrainer(opts)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
