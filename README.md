# Neural Orthodontic Cephalometry

> Source code for article "The efficiency of deep learning algorithms for detecting anatomical reference points on radiological images of the head profile".

---

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-204A6A.svg?style=flat-square)](https://www.python.org/)
[![made-with-tensorflow](https://img.shields.io/badge/Made%20with-TensorFlow-FF6C1E.svg?style=flat-square)](https://www.tensorflow.org/)
[![made-with-numpy](https://img.shields.io/badge/Made%20with-NumPy-3885C3.svg?style=flat-square)](https://numpy.org/)
[![made-with-pandas](https://img.shields.io/badge/Made%20with-Pandas-0B0051.svg?style=flat-square)](https://pandas.pydata.org/)
[![made-with-scikit-learn](https://img.shields.io/badge/Made%20with-Scikit--learn-FB9845.svg?style=flat-square)](https://scikit-learn.org/)

---

[![arXiv](https://img.shields.io/badge/arXiv-2005.12110-b31b1b.svg)](https://arxiv.org/abs/2005.12110)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

This repository describes the solution to the problem of detecting anatomical reference points on radiological images of the head profile for orthodontic analysis based on convolution neural networks. In this study, the definition of reference points on the radiological for cephalometry problems used.

The anatomical reference point detection process is defined as the task of localizing position in a reference point area with normal distribution. The maximum value of normal distribution is a desired anatomical reference point.

![Training process animation](<https://github.com/zsxoff/neural-orthodontic-cephalometry/blob/master/assets/anim_train_0.gif>)

As the considered networks were selected convolutional networks of the sequential convolutional layers and U-Net architecture.

See more at:

* [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](https://opensource.org/licenses/MIT)

This project is licensed under the terms of the [MIT](https://opensource.org/licenses/MIT) license (see [LICENSE](<https://github.com/zsxoff/neural-orthodontic-cephalometry/blob/master/LICENSE>) file).
