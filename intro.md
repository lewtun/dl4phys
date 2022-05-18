# Deep Learning for Particle Physicists

> Welcome to the graduate course on deep learning at the University of Bern's [Albert Einstein Center for Fundamental Physics](https://www.einstein.unibe.ch/)!

## Why all the fuss?

```{figure} ./images/jets.png
---
scale: 40%
name: directive-fig
---
Figure reference: [Jet Substructure at the Large Hadron Collider](https://arxiv.org/abs/1709.04464)
```

Deep learning is a subfield of artificial intelligence that focuses on using neural networks to parse data, learn from it, and then make predictions about something in the world. In the last decade, this framework has led to significant advances in [computer vision](https://www.youtube.com/watch?v=kSLJriaOumA&feature=youtu.be), [natural language processing](https://openai.com/blog/better-language-models/), and [reinforcement learning](https://deepmind.com/research/case-studies/alphago-the-story-so-far). More recently, deep learning has begun to attract interest in the physical sciences and is rapidly becoming an important part of the physicist's toolkit, especially in data-rich fields like high-energy particle physics and cosmology.

This course provides students with a _hands-on_ introduction to the methods of deep learning, with an emphasis on applying these methods to solve particle physics problems. A tentative list of the topics covered in this course includes:

* Jet tagging with convolutional neural networks
* Transformer neural networks for sequential data
* Normalizing flows for physics simulations such as lattice QFT

Throughout the course, students will learn how to implement neural networks from scratch, as well as core algorithms such as backpropagation and stochastic gradient descent. If time permits, we'll explore how symmetries can be encoded in graph neural networks, along with symbolic regression techniques to extract equations from data.

## Prerequisites

Although no prior knowledge of deep learning is required, we do recommend having some familiarity with the core concepts of machine learning. This course is _hands on_, which means you can expect to be running a lot of code in [fastai](https://docs.fast.ai/) and [PyTorch](https://pytorch.org/). You don't need to know either of these frameworks, but we assume that you're comfortable programming in Python and data analysis libraries such as NumPy. A useful precursor to the material covered in this course is [_Practical Machine Learning for Physicists_](https://lewtun.github.io/hepml/).

## Getting started

You can run the Jupyter notebooks from this course on cloud platforms like [Google Colab](https://colab.research.google.com/) or your local machine. Note that each notebook requires a GPU to run in a reasonable amount of time, so we recommend one of the cloud platforms as they come pre-installed with CUDA.

### Running on a cloud platform

To run these notebooks on a cloud platform, just click on one of the badges in the table below:

<!--This table is automatically generated, do not fill manually!-->


| Lecture                                        | Colab                                                                                                                                                           | Kaggle                                                                                                                                                               | Gradient                                                                                                                                           | Studio Lab                                                                                                                                                               |
|:-----------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 - Jet tagging with neural networks           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lewtun/dl4phys/blob/main/lecture01.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/lewtun/dl4phys/blob/main/lecture01.ipynb) | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/lewtun/dl4phys/blob/main/lecture01.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/lewtun/dl4phys/blob/main/lecture01.ipynb) |
| 2 - Gradient descent                           | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lewtun/dl4phys/blob/main/lecture02.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/lewtun/dl4phys/blob/main/lecture02.ipynb) | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/lewtun/dl4phys/blob/main/lecture02.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/lewtun/dl4phys/blob/main/lecture02.ipynb) |
| 3 - Neural network deep dive                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lewtun/dl4phys/blob/main/lecture03.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/lewtun/dl4phys/blob/main/lecture03.ipynb) | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/lewtun/dl4phys/blob/main/lecture03.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/lewtun/dl4phys/blob/main/lecture03.ipynb) |
| 4 - Jet images and transfer learning with CNNs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lewtun/dl4phys/blob/main/lecture04.ipynb) | [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/lewtun/dl4phys/blob/main/lecture04.ipynb) | [![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/lewtun/dl4phys/blob/main/lecture04.ipynb) | [![Open In SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/lewtun/dl4phys/blob/main/lecture04.ipynb) |


<!--End of table-->

Nowadays, the GPUs on Colab tend to be K80s (which have limited memory), so we recommend using [Kaggle](https://www.kaggle.com/docs/notebooks), [Gradient](https://gradient.run/notebooks), or [SageMaker Studio Lab](https://studiolab.sagemaker.aws/). These platforms tend to provide more performant GPUs like P100s, all for free!

> Note: some cloud platforms like Kaggle require you to restart the notebook after installing new packages.

### Running on your machine

To run the notebooks on your own machine, first clone the repository and navigate to it:

```bash
$ git clone https://github.com/nlp-with-transformers/notebooks.git
$ cd notebooks
```

Next, run the following command to create a `conda` virtual environment that contains all the libraries needed to run the notebooks:

```bash
$ conda env create -f environment.yml
```

## Recommended references

### Deep learning

* [Deep Learning for Coders with Fastai and PyTorch](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527) by Jeremy Howard and Sylvain Gugger. A highly accessible and practical book that will serve as a guide for these lectures.
* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.amazon.com/Hands-Machine-Learning-Scikit-Learn-TensorFlow-dp-1492032646/dp/1492032646) by Aurélien Géron. An excellent book that covers both machine learning and deep learning.

### Particle physics

* The [Particle Data Group](https://pdg.lbl.gov/2021/reviews/contents_sports.html) has a wonderfully concise review on machine learning. You can find it under _Mathematical Tools > Machine Learning_.
* [_Jet Substructure at the Large Hadron Collider_](https://arxiv.org/abs/1709.04464) by A. Larkowski et al (2017). Although ancient by deep learning standards (most papers are outdated the moment they land on the arXiv 🙃), this review covers all the concepts we'll need when looking at jets and how to tag them with neural networks.
* [_HEPML-LivingReview_](https://iml-wg.github.io/HEPML-LivingReview/). A remarkable project that catalogues loads of papers about machine learning and particles physics in a useful set of categories.
* [_Physics Meets ML_](http://www.physicsmeetsml.org/). A regular online seminar series that brings together researchers from the machine learning and physics communities.
* [_Machine Learning and the Physical Sciences_](https://ml4physicalsciences.github.io/2021/). A recent workshop at the NeurIPS conference that covers the whole gamut of machine learning and physics (not just particle physics).
* [_Graph Neural Networks in Particle Physics_](https://arxiv.org/abs/2007.13681) by J. Shlomi et al (2020). A concise summary of applying graph networks to experimental particle physics - mostly useful if we have time to cover these exciting architectures.