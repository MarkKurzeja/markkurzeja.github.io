+++
title = "Large Model Norms Violate Training Horizon Invariance"
subtitle = "If you can't train forever, at least pretend you can"
date = 2024-03-27
draft = true
+++

Training dynamics are an underappreciated aspect of pre-training large language models.

Large language models are typically trained with billions of examples. With engineering constraints (the entire dataset cannot fit into memory) and performance considerations (SGD can have a [regularization effect](https://arxiv.org/pdf/2101.12176.pdf)), models are also typically trained in a streaming fashion. Once we decide to train models in a streaming fashion, a natural question arises: in what order should data be presented to our models?

#### Two Ideals

Ideally, we want two ideals to be satisfied with our training recipe:

1. Seed Determinism: If a model is trained twice with the same data order and seed each time, then the _eval loss_ curves should be identical.
2. Permutation Invariance: If a model is trained twice, the first time with a dataset $\mathcal{X}$ and the second time with a permutation of $\mathcal{X}$, the _final eval loss_ should be identical. Note, the curves may be different, and we should expect it to be, but, ideally, we should end at the same place no matter what the data order.

The first ideal is usually implemented in SOTA production stacks. It is an engineering and statistical convenience, since it removes random noise in the training process from random seeds. In some sense, I would argue it is table stakes to reproducible research. Its importance, however, cannot be overstated since, without it, we cannot achieve the second ideal.

The second ideal, in modern pipelines, is not guaranteed because models have training dynamics. In some respects, achieving permutation invariance would require us to understand, fundamentally, how our models interact with our data over the entire span of training which, at the current moment, appears to be out of reach. How much this matters is up for debate. Of course, permutation invariance is an ideal. But until it is proven impossible, is it not worth striving for?

#### Potential sources of training dynamics

In [Rethinking Diffusion Model](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/), the authors highlight one potential source of training dynamics:

> First, we find that the growth of weights is a numerical artifact and not something the training “knowingly” seeks to do for some beneficial end. It’s a statistical tendency caused by noise and numerical approximation errors in gradient updates. Unfortunately, despite this incidental nature, it’s problematic for two distinct reasons:
> 1. The training gets saturated over time because updates made to the weights make a smaller relative impact when the existing weight is already large. It’s like throwing teaspoons of ingredients into a pot. After a while, adding one more teaspoon doesn’t change much. The training slows down to a crawl—at a different rate at each layer—even when there is more to learn. In other words, the training suffers from uncontrolled per-layer learning rate decay.
> 2. The weights act multiplicatively on the activations through convolutions and matrix products. When a weight grows, so do the activations of that layer. Thus, weight growth causes activation growth.

#### Normalization Layers and Training

In most modern transformer recipes, layer norms are used to help normalize the inputs to specific components and improve training stability. For example, Gemma, from Google, uses RMS Norm [pre-attention](https://github.com/google-deepmind/gemma/blob/main/gemma/modules.py#L200) and [pre-mlp](https://github.com/google-deepmind/gemma/blob/main/gemma/modules.py#L207).

I like to think about these layers as "symmetry correctors" in a certain sense. No matter what information may be stored in the magnitudes of the norms of the incoming tokens, the attention and MLP heads are made agnostic to it via the norm regularizes. While this is great for the local learning of the attention and MLP heads, it appears to generate issues globally with the optimization.

While useful, are these normalizers necessary? Or are they band-aids placed on larger bullet wounds?

#### Your Eigenvalues are not happy families

Taking a step back, it's pretty easy to make the norms of a hundred-layer neural net grow without bound. It's easy because *any* dense transform in the entire billion-parameter ensemble can make an unbounded contribution to the global norm, through its eigenvalues, unless extreme care is taken. While it is not sufficient to prevent norm growth, it is necessary that the eigenvalues of every linear transform be reasonably bounded lest the pathology persist.

This implies a certain [Anna Karenina principle](https://en.wikipedia.org/wiki/Anna_Karenina_principle): all "happy" deep nets have decently controlled norms. All "unhappy" deep nets have at least one component which has eigenvalues that are unbounded or some training dynamic which allows for the norms to grow without bound.

#### The two approaches to control

There appear to be two ways forward:

1. We can make changes to the training process to encourage the norms to grow in a more controlled manner. Techniques like those employed in [AdamW](http://go/arxiv/1711.05101), along with other forms of regularization, which penalize large norms, are one such way forward.
2. We can control the structure of the net itself by making it impossible for the norms to grow or shrink in the first place.

When we think about the purpose of RMSNorm or LayerNorm, generally, in a transformer, we are somewhat adhering to approach two. They take a two-step approach, however.

* Step One: Do absolutely nothing to control the norms upstream of a transformer block.
* Step Two: When the time comes, perform a quick normalization right before the attention and MLP blocks

What if we could do away with Step One? Even if we could only do away with it partially, is it possible to prevent norm growth, or shrinkage, through more matrix structure?

The next blog post, about orthonormal matrices, will explore this.
