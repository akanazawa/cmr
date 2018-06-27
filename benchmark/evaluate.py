"""
Script for testing on CUB.

Sample usage:
python -m cmr.benchmark.evaluate --split val --name <model_name> --num_train_epoch <model_epoch>
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
import scipy.io as sio

from ..nnutils import test_utils
from ..data import cub as cub_data
from ..nnutils import predictor as pred_utils

flags.DEFINE_boolean('visualize', False, 'if true visualizes things')

opts = flags.FLAGS


class ShapeTester(test_utils.Tester):
    def define_model(self):
        opts = self.opts

        self.predictor = pred_utils.MeshPredictor(opts)

        # for visualization
        self.renderer = self.predictor.vis_rend
        self.renderer.set_bgcolor([1., 1., 1.])
        self.renderer.renderer.renderer.renderer.image_size = 512
        self.renderer.set_light_dir([0, 1, -1], 0.38)

    def init_dataset(self):
        opts = self.opts
        self.data_module = cub_data

        torch.manual_seed(0)
        self.dataloader = self.data_module.data_loader(opts)

    def evaluate(self, outputs, batch):
        """
        Compute IOU and keypoint error
        """
        opts = self.opts
        bs = opts.batch_size

        ## compute iou
        mask_gt = batch['mask'].view(bs, -1).numpy()
        mask_pred = outputs['mask_pred'].cpu().view(bs, -1).type_as(
            batch['mask']).numpy()
        intersection = mask_gt * mask_pred
        union = mask_gt + mask_pred - intersection
        iou = intersection.sum(1) / union.sum(1)

        # Compute pck
        padding_frac = opts.padding_frac
        # The [-1,1] coordinate frame in which keypoints corresponds to:
        #    (1+2*padding_frac)*max_bbox_dim in image coords
        # pt_norm = 2* (pt_img - trans)/((1+2*pf)*max_bbox_dim)
        # err_pt = 2*err_img/((1+2*pf)*max_bbox_dim)
        # err_pck_norm = err_img/max_bbox_dim = err_pt*(1+2*pf)/2
        # so the keypoint error in the canonical fram should be multiplied by:
        err_scaling = (1 + 2 * padding_frac) / 2.0
        kps_gt = batch['kp'].cpu().numpy()

        kps_vis = kps_gt[:, :, 2]
        kps_gt = kps_gt[:, :, 0:2]
        kps_pred = outputs['kp_pred'].cpu().type_as(batch['kp']).numpy()
        kps_err = kps_pred - kps_gt
        kps_err = np.sqrt(np.sum(kps_err * kps_err, axis=2)) * err_scaling

        return iou, kps_err, kps_vis

    def visualize(self, outputs, batch):
        vert = outputs['verts'][0]
        cam = outputs['cam_pred'][0]
        texture = outputs['texture'][0]

        img_pred = self.renderer(vert, cam, texture=texture)
        aroundz = []
        aroundy = []
        # for deg in np.arange(0, 180, 30):
        for deg in np.arange(0, 150, 30):
            rendz = self.renderer.diff_vp(
                vert, cam, angle=-deg, axis=[1, 0, 0], texture=texture)
            rendy = self.renderer.diff_vp(
                vert, cam, angle=deg, axis=[0, 1, 0], texture=texture)
            aroundz.append(rendz)
            aroundy.append(rendy)

        aroundz = np.hstack(aroundz)
        aroundy = np.hstack(aroundy)
        vps = np.vstack((aroundz, aroundy))

        img = np.transpose(convert2np(batch['img'][0]), (1, 2, 0))
        import matplotlib.pyplot as plt
        plt.ion()
        fig = plt.figure(1)
        ax = fig.add_subplot(121)
        ax.imshow(img)
        ax.set_title('input')
        ax.axis('off')
        ax = fig.add_subplot(122)
        ax.imshow(img_pred)
        ax.set_title('pred_texture')
        ax.axis('off')
        plt.draw()

        fig = plt.figure(2)
        plt.imshow(vps)
        plt.axis('off')
        plt.draw()
        plt.pause(0.01)
        import ipdb
        ipdb.set_trace()

    def test(self):
        opts = self.opts
        bench_stats = {'ious': [], 'kp_errs': [], 'kp_vis': []}

        if opts.ignore_pred_delta_v:
            result_path = osp.join(opts.results_dir, 'results_meanshape.mat')
        elif opts.use_sfm_ms:
            result_path = osp.join(opts.results_dir,
                                   'results_sfm_meanshape.mat')
        else:
            result_path = osp.join(opts.results_dir, 'results.mat')

        if opts.use_sfm_camera:
            result_path = result_path.replace('.mat', '_sfm_camera.mat')

        print('Writing to %s' % result_path)

        if not osp.exists(result_path):

            n_iter = len(self.dataloader)
            for i, batch in enumerate(self.dataloader):
                if i % 100 == 0:
                    print('{}/{} evaluation iterations.'.format(i, n_iter))
                if opts.max_eval_iter > 0 and (i >= opts.max_eval_iter):
                    break
                outputs = self.predictor.predict(batch)
                if opts.visualize:
                    self.visualize(outputs, batch)
                iou, kp_err, kp_vis = self.evaluate(outputs, batch)

                bench_stats['ious'].append(iou)
                bench_stats['kp_errs'].append(kp_err)
                bench_stats['kp_vis'].append(kp_vis)

                if opts.save_visuals and (i % opts.visuals_freq == 0):
                    self.save_current_visuals(batch, outputs)

            bench_stats['kp_errs'] = np.concatenate(bench_stats['kp_errs'])
            bench_stats['kp_vis'] = np.concatenate(bench_stats['kp_vis'])

            bench_stats['ious'] = np.concatenate(bench_stats['ious'])
            sio.savemat(result_path, bench_stats)
        else:
            bench_stats = sio.loadmat(result_path)

        # Report numbers.

        mean_iou = bench_stats['ious'].mean()

        n_vis_p = np.sum(bench_stats['kp_vis'], axis=0)
        n_correct_p_pt1 = np.sum(
            (bench_stats['kp_errs'] < 0.1) * bench_stats['kp_vis'], axis=0)
        n_correct_p_pt15 = np.sum(
            (bench_stats['kp_errs'] < 0.15) * bench_stats['kp_vis'], axis=0)
        pck1 = (n_correct_p_pt1 / n_vis_p).mean()
        pck15 = (n_correct_p_pt15 / n_vis_p).mean()
        print('%s mean iou %.3g, pck.1 %.3g, pck.15 %.3g' %
              (osp.basename(result_path), mean_iou, pck1, pck15))


def main(_):
    opts.n_data_workers = 0
    opts.batch_size = 1

    opts.results_dir = osp.join(opts.results_dir_base, '%s' % (opts.split),
                                opts.name, 'epoch_%d' % opts.num_train_epoch)
    if not osp.exists(opts.results_dir):
        print('writing to %s' % opts.results_dir)
        os.makedirs(opts.results_dir)

    torch.manual_seed(0)
    tester = ShapeTester(opts)
    tester.init_testing()
    tester.test()


if __name__ == '__main__':
    app.run(main)
