"""
Makes the IOU curve in the paper.

Run run_evals.py before hand.

python -m cmr.benchmark.plot_curves --split test --name {model_name} --num_train_epoch {epoch_num}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import os.path as osp

# For options.
# results_dir_base
from ..nnutils import test_utils
# split
from ..data import cub as cub_data

import scipy.io as sio
import numpy as np

opts = flags.FLAGS


def get_pck_curve(kp_errs, kp_vis, alphas):
    """
    kp_errs: N x 15
    kp_vis: N x 15
    """
    n_vis_p = np.sum(kp_vis, axis=0)
    vals = np.array([(np.sum(
        (kp_errs < alpha) * kp_vis, axis=0) / n_vis_p).mean()
                     for alpha in alphas])

    return vals


def get_iou_curve(ious, alphas):
    """
    ious: N x 1
    """
    N = ious.size
    vals = np.array([np.sum(ious > alpha) / float(N) for alpha in alphas])

    return vals


def load_data(data_path):
    bench_stats = sio.loadmat(data_path)
    return bench_stats['kp_errs'], bench_stats['kp_vis'], bench_stats['ious']


def get_iou_pck(data_path, alphas):
    data = load_data(data_path)

    pck = get_pck_curve(data[0], data[1], [.1])
    iou_curve = get_iou_curve(data[2], alphas)

    return pck, iou_curve


def plot_iou(full_path, ms_path, full_path_gtcam, ms_path_gtcam, fig_path):
    alphas = np.linspace(0, 1., 50)
    pck_full, curve_full = get_iou_pck(full_path, alphas)
    pck_ms, curve_ms = get_iou_pck(ms_path, alphas)
    gtcam_pck_full, gtcam_curve_full = get_iou_pck(full_path_gtcam, alphas)
    gtcam_pck_ms, gtcam_curve_ms = get_iou_pck(ms_path_gtcam, alphas)

    print('PCK@1 full %.2f, sfm mean shape %.2f' % (pck_full, pck_ms))
    print('sfm-cam PCK@1 full %.2f, sfm mean shape %.2f' % (gtcam_pck_full,
                                                            gtcam_pck_ms))

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.clf()
    plt.figure(1)
    with plt.style.context('fivethirtyeight'):
        # ax = fig.add_subplot(111)
        p0 = plt.plot(alphas, curve_full, '-')
        p1 = plt.plot(alphas, curve_ms, '-')
        plt.plot(alphas, gtcam_curve_full, '--', color=p0[0].get_color())
        plt.plot(alphas, gtcam_curve_ms, '--', color=p1[0].get_color())

        plt.ylabel('% of instances (IoU > t)', fontsize=13)
        plt.xlabel('IoU', fontsize=13)

        plt.legend(
            [
                'full model', 'mean shape', 'full model (sfm cam)',
                'mean shape (sfm cam)'
            ],
            framealpha=0.8,
            frameon=True,
            fontsize=13,
            facecolor='w')
        plt.draw()
        plt.title('Mask Reprojection Accuracy', fontsize=13)
        plt.savefig(fig_path, bbox_inches='tight')

def main(_):
    results_dir = osp.join(opts.results_dir_base, '%s' % (opts.split),
                           opts.name, 'epoch_%d' % opts.num_train_epoch)

    # With pred camera.
    full_path = osp.join(results_dir, 'results.mat')
    ms_path = osp.join(results_dir, 'results_sfm_meanshape.mat')

    # With sfm camera
    gtcam_full_path = osp.join(results_dir, 'results_sfm_camera.mat')
    gtcam_ms_path = osp.join(results_dir, 'results_sfm_meanshape_sfm_camera.mat')

    fig_path = osp.join(results_dir, 'iou.png')

    all_paths = [full_path, ms_path, gtcam_full_path, gtcam_ms_path]
    for path in all_paths:
        if not osp.exists(path):
            print('{} does not exist, run run_evals.py'.format(path))
            exit(1)
    plot_iou(full_path, ms_path, gtcam_full_path, gtcam_ms_path, fig_path)


if __name__ == '__main__':
    app.run(main)
