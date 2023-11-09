# 3D Human Pose Estimation with Spatial and Temporal Transformers
This repo is the official implementation for [3D Human Pose Estimation with Spatial and Temporal Transformers](https://arxiv.org/pdf/2103.10455.pdf). The paper is accepted to [ICCV 2021](http://iccv2021.thecvf.com/home). 

- **Welcome to check our Neurips 2023 work:** [Context-Aware PoseFormer](https://arxiv.org/pdf/2311.03312.pdf)
- **Welcome to check our CVPR 2023 work:** [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2)
- Visualization code for in-the-wild videos can be found here [PoseFormer_demo](https://github.com/zczcwh/poseformer_demo)

[Video Demonstration](https://youtu.be/z8HWOdXjGR8)

## PoseFormer Architecture
<p align="left"> <img src="./figure/PoseFormer.gif" width="75%"> </p>


## Video Demo


| <p align="center"> <img src="./figure/H3.6.gif" width="80%"> </p> | 
|:--:| 
| 3D HPE on Human3.6M |

| <p align="center"> <img src="./figure/wild.gif" width="80%"> </p> | 
|:--:| 
| 3D HPE on videos in-the-wild using PoseFormer |



Our code is built on top of [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

### Environment

The code is developed and tested under the following environment

* Python 3.8.2
* PyTorch 1.7.1
* CUDA 11.0

You can create the environment:
```bash
conda env create -f poseformer.yml
```

### Dataset

Our code is compatible with the dataset setup introduced by [Martinez et al.](https://github.com/una-dinosauria/3d-pose-baseline) and [Pavllo et al.](https://github.com/facebookresearch/VideoPose3D). Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset  (./data directory). 

### Evaluating pre-trained models

We provide the pre-trained 81-frame model (CPN detected 2D pose as input) [here](https://drive.google.com/file/d/1oX5H5QpVoFzyD-Qz9aaP3RDWDb1v1sIy/view?usp=sharing). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 81 -c checkpoint --evaluate detected81f.bin
```

We also provide pre-trained 81-frame model (Ground truth 2D pose as input) [here](https://drive.google.com/file/d/18wW4TdNYxF-zdt9oInmwQK9hEdRJnXzu/view?usp=sharing). To evaluate it, put it into the `./checkpoint` directory and run:

```bash
python run_poseformer.py -k gt -f 81 -c checkpoint --evaluate gt81f.bin
```


### Training new models

* To train a model from scratch (CPN detected 2D pose as input), run:

```bash
python run_poseformer.py -k cpn_ft_h36m_dbb -f 27 -lr 0.00004 -lrd 0.99
```

`-f` controls how many frames are used as input. 27 frames achieves 47.0 mm, 81 frames achieves achieves 44.3 mm. 

* To train a model from scratch (Ground truth 2D pose as input), run:

```bash
python run_poseformer.py -k gt -f 81 -lr 0.0004 -lrd 0.99
```

81 frames achieves 31.3 mm (MPJPE). 

### Visualization and other functions

We keep our code consistent with [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). Please refer to their project page for further information. 

### Bibtex
If you find our work useful in your research, please consider citing:

    @article{zheng20213d,
    title={3D Human Pose Estimation with Spatial and Temporal Transformers},
    author={Zheng, Ce and Zhu, Sijie and Mendieta, Matias and Yang, Taojiannan and Chen, Chen and Ding, Zhengming},
    journal={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
    year={2021}
    }

## Acknowledgement

Part of our code is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D). We thank the authors for releasing the codes.
