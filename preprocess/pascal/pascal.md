## Pascal Data Pre-processing Guidelines

### Dowload segmentation masks for PASCAL images
```
mkdir BASE_DIR/cachedir/pascal; cd cachedir/pascal;
wget https://people.eecs.berkeley.edu/~shubhtuls/cachedir/vpsKps/segkps.zip
unzip segkps.zip -d ./segkps
```

### Compute Segmentation for Pascal3D ImageNet subset
As the segmentation annotations are not available for the ImageNet images in PASCAL3D+, we use [detectron](https://github.com/facebookresearch/Detectron) to obtain these. Sample command to compute these for the car category:

```
# running detectron
export PYTHONPATH=/usr/local
export PYTHONPATH=$PYTHONPATH:/home/shubhtuls/packages/caffe2
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shubhtuls/packages/caffe2/lib

python2 instance_segment.py --cfg /home/shubhtuls/packages/detectron/configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml     --output-dir /data1/shubhtuls/cachedir/PASCAL3D+_release1.1/masks/car_imagenet --image-ext JPEG --wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl /data1/shubhtuls/cachedir/PASCAL3D+_release1.1/Images/car_imagenet
```

### Obtain Data strctures to use with training
After the above steps, simply run (from matlab):
```
preprocess_p3d
```

### Training
There is a sample code for a Pascal3D data loader in data/p3d.py. Please use this data loader instead of the cub data loader.
