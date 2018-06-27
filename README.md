# Learning Category-Specific Mesh Reconstruction from Image Collections

Angjoo Kanazawa<sup>\*</sup>, Shubham Tulsiani<sup>\*</sup>, Alexei A. Efros, Jitendra Malik

University of California, Berkeley

[Project Page](https://akanazawa.github.io/cmr/)
![Teaser Image](https://akanazawa.github.io/cmr/resources/images/teaser.png)

### Requirements
- Python 2.7
- [PyTorch](https://pytorch.org/) tested on version `0.3.0.post4`

### Installation

#### Setup virtualenv
```
virtualenv venv_cmr
source venv_cmr/bin/activate
pip install -U pip
deactivate
source venv_cmr/bin/activate
pip install -r requirements.txt
```

### Demo
1. From the `cmr` directory, download the trained model:
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/cmr/model.tar.gz & tar -vzxf model.tar.gz
```
You should see `cmr/cachedir/snapshots/bird_net/`

2. Run the demo:
```
python -m cmr.demo --name bird_net --num_train_epoch 500 --img_path cmr/demo_data/img1.jpg
python -m cmr.demo --name bird_net --num_train_epoch 500 --img_path cmr/demo_data/birdie.jpg
```

### Training
Please see [doc/train.md](https://github.com/akanazawa/cmr/blob/master/doc/train.md)

### Citation
If you use this code for your research, please consider citing:
```
@article{cmrKanazawa18,
  title={Learning Category-Specific Mesh Reconstruction
  from Image Collections},
  author = {Angjoo Kanazawa and
  Shubham Tulsiani
  and Alexei A. Efros
  and Jitendra Malik},
  journal={arXiv preprint arXiv:1803.07549},
  year={2018}
}

```

## TODO:
- requirements.txt is way too long
