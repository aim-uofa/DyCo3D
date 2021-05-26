# DyCo3d
## DyCo3d: Robust Instance Segmentation of 3D Point Clouds through Dynamic Convolution (CVPR 2021)
![overview](https://github.com/aim-uofa/DyCo3D/blob/main/doc/dyco3d.png)

Code for the paper **DyCo3D: Robust Instance Segmentation of 3D Point Clouds through Dynamic Convolution**, CVPR 2021.

**Authors**: Tong He, Chunhua Shen, Anton van den Hengel

[[arxiv]](https://arxiv.org/abs/2011.13328)

## Introduction
Previous top-performing approaches for point cloud instance segmentation involve a bottom-up strategy, which often includes inefficient operations or complex pipelines, such as grouping over-segmented components, introducing additional steps for refining, or designing complicated loss functions. The inevitable variation in the instance scales can lead bottom-up methods to become particularly sensitive to hyper-parameter values. To this end, we propose instead a dynamic, proposal-free, data-driven approach that generates the appropriate convolution kernels to apply in response to the nature of the instances. To make the kernels discriminative, we explore a large context by gathering homogeneous points that share identical semantic categories and have close votes for the geometric centroids. Instances are then decoded by several simple convolutional layers. Due to the limited receptive field introduced by the sparse convolution, a small light-weight transformer is also devised to capture the long-range dependencies and high-level interactions among point samples. The proposed method achieves promising results on both ScanetNetV2 and S3DIS, and this performance is robust to the particular hyper-parameter values chosen. It also improves inference speed by more than 25% over the current state-of-the-art.

## Installation

### Requirements
* Python 3.7.0
* Pytorch 1.1.0
* CUDA 10.1

### Virtual Environment
```
conda create -n dyco3d python==3.7
conda activate dyco3d
```

### Install `DyCo3d` (Follow the installation steps of [PointGroup](https://github.com/Jia-Research-Lab/PointGroup))

(1) Clone the DyCo3d repository.
```
git clone https://github.com/aim-uofa/DyCo3D.git
cd DyCo3D
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we use the repo from [PointGroup](https://github.com/Jia-Research-Lab/PointGroup)

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and `pip install` the generated `.whl` file.



(4) Compile the `pointgroup_ops` library.
```
cd lib/pointgroup_ops
python setup.py develop
```
If any header files could not be found, run the following commands. 
```
python setup.py build_ext --include-dirs=$INCLUDE_PATH$
python setup.py develop
```
`$INCLUDE_PATH$` is the path to the folder containing the header files that could not be found.


## Data Preparation

(1) Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

(2) Put the data in the corresponding folders. 
* Copy the files `[scene_id]_vh_clean_2.ply`,  `[scene_id]_vh_clean_2.labels.ply`,  `[scene_id]_vh_clean_2.0.010000.segs.json`  and `[scene_id].aggregation.json`  into the `dataset/scannetv2/train` and `dataset/scannetv2/val` folders according to the ScanNet v2 train/val [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Copy the files `[scene_id]_vh_clean_2.ply` into the `dataset/scannetv2/test` folder according to the ScanNet v2 test [split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark). 

* Put the file `scannetv2-labels.combined.tsv` in the `dataset/scannetv2` folder.

The dataset files are organized as follows.
```
DyCo3D
├── dataset
│   ├── scannetv2
│   │   ├── train
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── val
│   │   │   ├── [scene_id]_vh_clean_2.ply & [scene_id]_vh_clean_2.labels.ply & [scene_id]_vh_clean_2.0.010000.segs.json & [scene_id].aggregation.json
│   │   ├── test
│   │   │   ├── [scene_id]_vh_clean_2.ply 
│   │   ├── scannetv2-labels.combined.tsv
```

(3) Generate input files `[scene_id]_inst_nostuff.pth` for instance segmentation.
```
cd dataset/scannetv2
python prepare_data_inst.py --data_split train
python prepare_data_inst.py --data_split val
python prepare_data_inst.py --data_split test
```
You can also download the data [here (about 13G)](https://cloudstor.aarnet.edu.au/plus/s/RtzGnxR6wqPXMSw).
The dataset files are organized as 
```
DyCo3D
├── dataset
│   ├── scannetv2
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```



## Training
```
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --master_port=$((RANDOM + 10000)) train.py --config config/dyco3d_multigpu_scannet.yaml  --output_path OUTPUT_DIR  --use_backbone_transformer
```


## Inference and Evaluation
To test with a pretrained model, run
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/dyco3d_multigpu_scannet.yaml --output_path exp/model --resume MODEL --use_backbone_transforme
```
## Pretrained Model
We provide a pretrained model trained on ScanNet v2 dataset. Download it [here](https://cloudstor.aarnet.edu.au/plus/s/nza0IvigppngfkC). Its performance on ScanNet v2 validation set is 35.5/57.6/72.9 in terms of mAP/mAP50/mAP25. (with a masking head size of 16)



## Results on ScanNet Benchmark 
Quantitative results on ScanNet test set at the submisison time.
![scannet_result](http://kaldir.vc.in.tum.de/scannet_benchmark/semantic_instance_3d)



## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{He2021dyco3d,
  title     =   {{DyCo3d}: Robust Instance Segmentation of 3D Point Clouds through Dynamic Convolution},
  author    =   {Tong He and Chunhua Shen and Anton van den Hengel},
  booktitle =   {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      =   {2021}
}
```

## Acknowledgement
This repo is built upon [PointGroup](https://github.com/Jia-Research-Lab/PointGroup), [spconv](https://github.com/traveller59/spconv), [condinst](https://github.com/aim-uofa/AdelaiDet). 

## Contact
If you have any questions or suggestions about this repo, please feel free to contact me (tonghe90@gmail.com).


