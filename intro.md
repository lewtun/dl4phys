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

Although no prior knowledge of deep learning is required, we do recommend having some familiarity with the core concepts of machine learning. This course is _hands on_, which means you can expect to be running a lot of code in [PyTorch](https://pytorch.org/). You don't need to know PyTorch, but we assume that you're comfortable programming in Python and data analysis libraries such as NumPy. A useful precursor to the material covered in this course is [_Practical Machine Learning for Physicists_](https://lewtun.github.io/hepml/).

## Recommended references

* The [Particle Data Group](https://pdg.lbl.gov/2021/reviews/contents_sports.html) has a wonderfully concise review on machine learning. You can find it under _Mathematical Tools > Machine Learning_.
* [_Jet Substructure at the Large Hadron Collider_](https://arxiv.org/abs/1709.04464) by A. Larkowski et al (2017). Although ancient by deep learning standards (most papers are outdated the moment they land on the arXiv 🙃), this review covers all the concepts we'll need when looking at jets and how to tag them with neural networks.
* [_HEPML-LivingReview_](https://iml-wg.github.io/HEPML-LivingReview/). A remarkable project that catalogues loads of papers about machine learning and particles physics in a useful set of categories.
* [_Physics Meets ML_](http://www.physicsmeetsml.org/). A regular online seminar series that brings together researchers from the machine learning and physics communities.
* [_Machine Learning and the Physical Sciences_](https://ml4physicalsciences.github.io/2021/). A recent workshop at the NeurIPS conference that covers the whole gamut of machine learning and physics (not just particle physics).
* [_Graph Neural Networks in Particle Physics_](https://arxiv.org/abs/2007.13681) by J. Shlomi et al (2020). A concise summary of applying graph networks to experimental particle physics - mostly useful if we have time to cover these exciting architectures.