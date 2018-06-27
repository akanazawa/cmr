"""
Mesh net model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import os
import os.path as osp
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable

from ..utils import mesh
from ..utils import geometry as geom_utils
from . import net_blocks as nb

#-------------- flags -------------#
#----------------------------------#
flags.DEFINE_boolean('symmetric', True, 'Use symmetric mesh or not')
flags.DEFINE_integer('nz_feat', 200, 'Encoded feature size')

flags.DEFINE_boolean('texture', True, 'if true uses texture!')
flags.DEFINE_boolean('symmetric_texture', True, 'if true texture is symmetric!')
flags.DEFINE_integer('tex_size', 6, 'Texture resolution per face')

flags.DEFINE_integer('subdivide', 3, '# to subdivide icosahedron, 3=642verts, 4=2562 verts')

flags.DEFINE_boolean('use_deconv', False, 'If true uses Deconv')
flags.DEFINE_string('upconv_mode', 'bilinear', 'upsample mode')

flags.DEFINE_boolean('only_mean_sym', False, 'If true, only the meanshape is symmetric')


#------------- Modules ------------#
#----------------------------------#
class ResNetConv(nn.Module):
    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)
        return x


class Encoder(nn.Module):
    """
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, n_blocks=4, nz_feat=100, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(n_blocks=4)
        self.enc_conv1 = nb.conv2d(batch_norm, 512, 256, stride=2, kernel_size=4)
        nc_input = 256 * (input_shape[0] // 64) * (input_shape[1] // 64)
        self.enc_fc = nb.fc_stack(nc_input, nz_feat, 2)

        nb.net_init(self.enc_conv1)

    def forward(self, img):
        resnet_feat = self.resnet_conv.forward(img)

        out_enc_conv1 = self.enc_conv1(resnet_feat)
        out_enc_conv1 = out_enc_conv1.view(img.size(0), -1)
        feat = self.enc_fc.forward(out_enc_conv1)

        return feat


class TexturePredictorUV(nn.Module):
    """
    Outputs mesh texture
    """

    def __init__(self, nz_feat, uv_sampler, opts, img_H=64, img_W=128, n_upconv=5, nc_init=256, predict_flow=False, symmetric=False, num_sym_faces=624):
        super(TexturePredictorUV, self).__init__()
        self.feat_H = img_H // (2 ** n_upconv)
        self.feat_W = img_W // (2 ** n_upconv)
        self.nc_init = nc_init
        self.symmetric = symmetric
        self.num_sym_faces = num_sym_faces
        self.F = uv_sampler.size(1)
        self.T = uv_sampler.size(2)
        self.predict_flow = predict_flow
        # B x F x T x T x 2 --> B x F x T*T x 2
        self.uv_sampler = uv_sampler.view(-1, self.F, self.T*self.T, 2)

        self.enc = nb.fc_stack(nz_feat, self.nc_init*self.feat_H*self.feat_W, 2)
        if predict_flow:
            nc_final=2
        else:
            nc_final=3
        self.decoder = nb.decoder2d(n_upconv, None, nc_init, init_fc=False, nc_final=nc_final, use_deconv=opts.use_deconv, upconv_mode=opts.upconv_mode)

    def forward(self, feat):
        # pdb.set_trace()
        uvimage_pred = self.enc.forward(feat)
        uvimage_pred = uvimage_pred.view(uvimage_pred.size(0), self.nc_init, self.feat_H, self.feat_W)
        # B x 2 or 3 x H x W
        self.uvimage_pred = self.decoder.forward(uvimage_pred)
        self.uvimage_pred = torch.nn.functional.tanh(self.uvimage_pred)

        tex_pred = torch.nn.functional.grid_sample(self.uvimage_pred, self.uv_sampler)
        tex_pred = tex_pred.view(uvimage_pred.size(0), -1, self.F, self.T, self.T).permute(0, 2, 3, 4, 1)

        if self.symmetric:
            # Symmetrize.
            tex_left = tex_pred[:, -self.num_sym_faces:]
            return torch.cat([tex_pred, tex_left], 1)
        else:
            # Contiguous Needed after the permute..
            return tex_pred.contiguous()



class ShapePredictor(nn.Module):
    """
    Outputs mesh deformations
    """

    def __init__(self, nz_feat, num_verts):
        super(ShapePredictor, self).__init__()
        # self.pred_layer = nb.fc(True, nz_feat, num_verts)
        self.pred_layer = nn.Linear(nz_feat, num_verts * 3)

        # Initialize pred_layer weights to be small so initial def aren't so big
        self.pred_layer.weight.data.normal_(0, 0.0001)

    def forward(self, feat):
        # pdb.set_trace()
        delta_v = self.pred_layer.forward(feat)
        # Make it B x num_verts x 3
        delta_v = delta_v.view(delta_v.size(0), -1, 3)
        # print('shape: ( Mean = {}, Var = {} )'.format(delta_v.mean().data[0], delta_v.var().data[0]))
        return delta_v


class QuatPredictor(nn.Module):
    def __init__(self, nz_feat, nz_rot=4, classify_rot=False):
        super(QuatPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, nz_rot)
        self.classify_rot = classify_rot

    def forward(self, feat):
        quat = self.pred_layer.forward(feat)
        if self.classify_rot:
            quat = torch.nn.functional.log_softmax(quat)
        else:
            quat = torch.nn.functional.normalize(quat)
        return quat


class ScalePredictor(nn.Module):
    def __init__(self, nz):
        super(ScalePredictor, self).__init__()
        self.pred_layer = nn.Linear(nz, 1)

    def forward(self, feat):
        scale = self.pred_layer.forward(feat) + 1  #biasing the scale to 1
        scale = torch.nn.functional.relu(scale) + 1e-12
        # print('scale: ( Mean = {}, Var = {} )'.format(scale.mean().data[0], scale.var().data[0]))
        return scale


class TransPredictor(nn.Module):
    """
    Outputs [tx, ty] or [tx, ty, tz]
    """

    def __init__(self, nz, orth=True):
        super(TransPredictor, self).__init__()
        if orth:
            self.pred_layer = nn.Linear(nz, 2)
        else:
            self.pred_layer = nn.Linear(nz, 3)

    def forward(self, feat):
        trans = self.pred_layer.forward(feat)
        # print('trans: ( Mean = {}, Var = {} )'.format(trans.mean().data[0], trans.var().data[0]))
        return trans


class CodePredictor(nn.Module):
    def __init__(self, nz_feat=100, num_verts=1000):
        super(CodePredictor, self).__init__()
        self.quat_predictor = QuatPredictor(nz_feat)
        self.shape_predictor = ShapePredictor(nz_feat, num_verts=num_verts)
        self.scale_predictor = ScalePredictor(nz_feat)
        self.trans_predictor = TransPredictor(nz_feat)

    def forward(self, feat):
        shape_pred = self.shape_predictor.forward(feat)
        scale_pred = self.scale_predictor.forward(feat)
        quat_pred = self.quat_predictor.forward(feat)
        trans_pred = self.trans_predictor.forward(feat)
        return shape_pred, scale_pred, trans_pred, quat_pred

#------------ Mesh Net ------------#
#----------------------------------#
class MeshNet(nn.Module):
    def __init__(self, input_shape, opts, nz_feat=100, num_kps=15, sfm_mean_shape=None):
        # Input shape is H x W of the image.
        super(MeshNet, self).__init__()
        self.opts = opts
        self.pred_texture = opts.texture
        self.symmetric = opts.symmetric
        self.symmetric_texture = opts.symmetric_texture

        # Mean shape.
        verts, faces = mesh.create_sphere(opts.subdivide)
        num_verts = verts.shape[0]

        if self.symmetric:
            verts, faces, num_indept, num_sym, num_indept_faces, num_sym_faces = mesh.make_symmetric(verts, faces)
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(verts, sfm_mean_shape[0], sfm_mean_shape[1])

            num_sym_output = num_indept + num_sym
            if opts.only_mean_sym:
                print('Only the mean shape is symmetric!')
                self.num_output = num_verts
            else:
                self.num_output = num_sym_output
            self.num_sym = num_sym
            self.num_indept = num_indept
            self.num_indept_faces = num_indept_faces
            self.num_sym_faces = num_sym_faces
            # mean shape is only half.
            self.mean_v = nn.Parameter(torch.Tensor(verts[:num_sym_output]))

            # Needed for symmetrizing..
            self.flip = Variable(torch.ones(1, 3).cuda(), requires_grad=False)
            self.flip[0, 0] = -1
        else:
            if sfm_mean_shape is not None:
                verts = geom_utils.project_verts_on_mesh(verts, sfm_mean_shape[0], sfm_mean_shape[1])            
            self.mean_v = nn.Parameter(torch.Tensor(verts))
            self.num_output = num_verts

        verts_np = verts
        faces_np = faces
        self.faces = Variable(torch.LongTensor(faces).cuda(), requires_grad=False)
        self.edges2verts = mesh.compute_edges2verts(verts, faces)

        vert2kp_init = torch.Tensor(np.ones((num_kps, num_verts)) / float(num_verts))
        # Remember initial vert2kp (after softmax)
        self.vert2kp_init = torch.nn.functional.softmax(Variable(vert2kp_init.cuda(), requires_grad=False), dim=1)
        self.vert2kp = nn.Parameter(vert2kp_init)


        self.encoder = Encoder(input_shape, n_blocks=4, nz_feat=nz_feat)
        self.code_predictor = CodePredictor(nz_feat=nz_feat, num_verts=self.num_output)

        if self.pred_texture:
            if self.symmetric_texture:
                num_faces = self.num_indept_faces + self.num_sym_faces
            else:
                num_faces = faces.shape[0]

            uv_sampler = mesh.compute_uvsampler(verts_np, faces_np[:num_faces], tex_size=opts.tex_size)
            # F' x T x T x 2
            uv_sampler = Variable(torch.FloatTensor(uv_sampler).cuda(), requires_grad=False)
            # B x F' x T x T x 2
            uv_sampler = uv_sampler.unsqueeze(0).repeat(self.opts.batch_size, 1, 1, 1, 1)
            img_H = int(2**np.floor(np.log2(np.sqrt(num_faces) * opts.tex_size)))
            img_W = 2 * img_H
            self.texture_predictor = TexturePredictorUV(
              nz_feat, uv_sampler, opts, img_H=img_H, img_W=img_W, predict_flow=True, symmetric=opts.symmetric_texture, num_sym_faces=self.num_sym_faces)
            nb.net_init(self.texture_predictor)

    def forward(self, img):
        img_feat = self.encoder.forward(img)
        codes_pred = self.code_predictor.forward(img_feat)
        if self.pred_texture:
            texture_pred = self.texture_predictor.forward(img_feat)
            return codes_pred, texture_pred
        else:
            return codes_pred

    def symmetrize(self, V):
        """
        Takes num_indept+num_sym verts and makes it
        num_indept + num_sym + num_sym
        Is identity if model is not symmetric
        """
        if self.symmetric:
            if V.dim() == 2:
                # No batch
                V_left = self.flip * V[-self.num_sym:]
                return torch.cat([V, V_left], 0)
            else:
                # With batch
                V_left = self.flip * V[:, -self.num_sym:]
                return torch.cat([V, V_left], 1)
        else:
            return V

    def get_mean_shape(self):
        return self.symmetrize(self.mean_v)
