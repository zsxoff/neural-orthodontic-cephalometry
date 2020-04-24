# Neural Orthodontic Cephalometry

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> The efficiency of deep learning algorithms for detecting anatomical reference points on radiological images of the head profile.

This repository describes the solution to the problem of detecting anatomical reference points on radiological images of the head profile for orthodontic analysis based on convolution neural networks. In this study, the definition of reference points on the radiological for cephalometry problems used.

The anatomical reference point detection process is defined as the task of localizing position in a reference point area with normal distribution. The maximum value of normal distribution is a desired anatomical reference point.

![Training process animation](<https://github.com/zsxoff/neural-orthodontic-cephalometry/blob/master/assets/anim_train_0.gif>)

As the considered networks were selected convolutional networks of the sequential convolutional layers and U-Net architecture.

See more at:

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## License

This project is licensed under the terms of the [MIT](https://opensource.org/licenses/MIT) license (see [LICENSE](<https://github.com/zsxoff/neural-orthodontic-cephalometry/blob/master/LICENSE>) file).
