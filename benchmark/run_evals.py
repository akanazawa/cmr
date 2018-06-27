"""
Runs evaluation for ablation study.

- pred shape + pred camera (full model)
- pred shape + SfM camera
- SfM mean shape + pred camera
- SfM mean shape + SfM camera

Sample run:
python -m cmr.benchmark.run_evals --split val  --name CUB_submitted --num_train_epoch 500
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags

from os import system
import itertools

flags.DEFINE_string('name', 'test_model', 'name of the trained model')
flags.DEFINE_integer('num_train_epoch', 400, 'which epochs to use.')
flags.DEFINE_string('split', 'val', 'which split.')
opts = flags.FLAGS


def main(_):
    base_cmd = 'python -m cmr.benchmark.evaluate --split {} --name {} --num_train_epoch {} '.format(
        opts.split, opts.name, opts.num_train_epoch)

    optionA = [' ', ' --use_sfm_ms']
    optionB = [' ', ' --use_sfm_camera']

    all_options = [
        ' '.join(p) for p in list(itertools.product(*[optionA, optionB]))
    ]

    for option in all_options:
        cmd = base_cmd + option
        print('Running {}'.format(cmd))
        res = system(cmd)
        if res > 0:
            print('something went wrong')
            import ipdb
            ipdb.set_trace()


if __name__ == '__main__':
    app.run(main)
