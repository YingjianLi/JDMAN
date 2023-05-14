# JDMAN
Pytorch code release for ["JDMAN: Joint Discriminative and Mutual Adaptation Networks for Cross-Domain Facial Expression Recognition"](https://dl.acm.org/doi/10.1145/3474085.3475484) (ACM MM'21) and ["Deep Margin-Sensitive Representation Learning for Cross-Domain Facial Expression Recognition"](https://ieeexplore.ieee.org/document/9676449)(IEEE TMM'22)

## Requirements
Python3.6+, Pytorch 1.2.0+

## Training

For example, training using VGG backbone:

```
sh ./raf_sh/raf2ck/raf2ck_vgg16_center_MI.sh
```

We recommend VGG backbone for more stale performance.

## Citation

If you use this code for your research, please consider citing:

```
@inproceedings{10.1145/3474085.3475484,
author = {Li, Yingjian and Gao, Yingnan and Chen, Bingzhi and Zhang, Zheng and Zhu, Lei and Lu, Guangming},
title = {JDMAN: Joint Discriminative and Mutual Adaptation Networks for Cross-Domain Facial Expression Recognition},
year = {2021},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
pages = {3312–3320},
numpages = {9},
series = {MM '21}
}

```
and
```
@ARTICLE{9676449,
  author={Li, Yingjian and Zhang, Zheng and Chen, Bingzhi and Lu, Guangming and Zhang, David},
  journal={IEEE Transactions on Multimedia}, 
  title={Deep Margin-Sensitive Representation Learning for Cross-Domain Facial Expression Recognition}, 
  year={2023},
  volume={25},
  number={},
  pages={1359-1373},
  doi={10.1109/TMM.2022.3141604}}
```