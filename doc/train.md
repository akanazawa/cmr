
## Pre-reqs

### CUB Data
1. Download CUB-200 images somewhere:
```
wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz && tar -xf images.tgz
```

2. Download our CUB annotation mat files and pre-computed SfM outputs.
Do this from the `cmr/` directory, and this should make `cmr/cachedir` directory:
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/cmr/cachedir.tar.gz & tar -vzxf cachedir.tar.gz
```

#### Computing SfM
We provide the computed SfM. If you want to compute them yourself, run via matlab:
```
cd preprocess/cub
main
```

### Model training
Change the `name` to whatever you want to call. Also see `shape.py` to adjust
hyper-parameters (for eg. increase `tex_loss_wt` and `text_dt_loss_wt` if you
want better texture, increase texture resolution with `tex_size`).
See `nnutils/mesh_net.py` and `nnutils/train_utils.py` for more model/training options.

```
cmd='python -m cmr.experiments.shape --name=bird_net --display_port 8087'
```

### Evaluation
We provide evaluation code to compute the IOU curves in the paper.
Command below runs the model with different ablation settings.
Run it from one directory above the `cmr` directory.
```
python -m cmr.benchmark.run_evals --split val  --name CUB_submitted --num_train_epoch 500
```

Then, run 
```
python -m cmr.benchmark.plot_curvess --split val  --name CUB_submitted --num_train_epoch 500
```
in order to see the IOU curve.
