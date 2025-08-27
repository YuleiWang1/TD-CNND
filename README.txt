## TD-CNND

This is the official implementation of the paper "Tucker Decomposition-Based Network Compression for Anomaly Detection With Large-Scale Hyperspectral Images" in  IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTAR). Paper link: https://ieeexplore.ieee.org/abstract/document/10537985

## Abstract

Deep learning methodologies have demonstrated considerable effectiveness in hyperspectral anomaly detection (HAD).However, the practicality of deep learning-based HAD in real-world applications is impeded by challenges arising from limited labeled data, large-scale hyperspectral images (HSIs), and constrained computational resources. In light of these challenges, this article introduces a convolutional neural network (CNN)-based HAD model through the incorporation of Tucker decomposition, named TDCNND. Drawing inspiration from transfer learning, the proposed model initially constructs pixel sample pairs from known labeled HSIs in the source domain, feeding them into the designed CNN to train the network learning spectral feature differences to obtain a CNN containing knowledge from the source domain. Subsequently, to prevent the need for network retraining caused by structural changes and to reduce model parameters for improving detecting timeliness, a general network compression scheme based on Tucker decomposition is applied to the CNN, where the convolutional layers of the above CNN undergo Tucker tensor decomposition to compress the network and alleviate parameter redundancy. Finally, spectral features realignment is used to recover the detection accuracy loss caused by Tucker tensor decomposition. In addition, a dual-windows structure is implemented during the detection phase, incorporating spatial information to the aforementioned spectral-level learning model, facilitating spectral-spatial collaborative HAD. Experimental evaluations using three real hyperspectral datasets and artificially expanded datasets demonstrate that, in comparison with state-of-the-art methods, the proposed TD-CNND method exhibits effectiveness and superiority in terms of both time cost and detection accuracy, where the notable advantages in terms of time cost become more pronounced with an increasing number of pixels.

## Workflow

### 1.  train

Transfer learning is achieved through the labeled hyperspectral classification dataset in the source domain to obtain a deep model for implementing the detection task in the target domain.

### 2.  tucker_adjust

The deep model obtained by 【### 1.train】 is subject to Tucker decomposition and spectral features realignment to obtain the final deep model for target domain detection.

### 3.  eval&dec

Achieve the anomaly detection task of the target domain.

## Citation

If you find the code helpful in your research, please kindly cite our paper:

```bibtex
@article{TD-CNND,
title={Tucker Decomposition-Based Network Compression for Anomaly Detection With Large-Scale Hyperspectral Images},
author={Yulei Wang andHongzhou Wang and Enyu Zhao and Meiping Song andChunhui Zhao },
journal={ IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
year={2024},
publisher={IEEE}
}
```
