+++
title = 'Are Token-wise MLPs over-parameterized in Transformers?'
subtitle = "Part II of a two part series on RFFs"
date = 2024-03-23
draft = true
+++

> "The pessimist says the glass is half-full. The optimist says the glass is
> half-empty. The engineer says its mearly 2x larger than it needs to be. <br>
> \\( \approx \\) Thomas Cathcart

To spoil the punchline, there is a curious connection between the token-wise
MLP projections of modern Transformer recipes and Random Fourier Features. 

Some adhoc experiments show a tiny tweak to MLPs allow them to train with far
smaller batch sizes and dramatically reduce the number of parameters required
to learn complex functions.

### Domain and Range

Random Fourier Features (RFF), like an MLP, maps an input vector \\(X\\) of dimension \\(l_x\\)
and to a vector \\(Y\\) of dimension \\(l_y\\). 

\\[
X \in \mathbb{R}^{l_x}
\hspace{0.2cm} 
\underset{RFF}{\rightarrow} 
\hspace{0.2cm} 
Y \in \mathbb{R}^{l_y}
\\]

### Data

In the learning problem, we are given input-output mappings: 
\\( \lbrace X \in \mathbb{R}^{l_x}, Y \in \mathbb{R}^{l_y} \rbrace_{i = 1}^{N_{data}} \\). 
In standard RFF, it is assumed the data fits
in memory. In modern deep-learning stacks, mappings are learned via SGD. We will
assume the data fits in memory to begin and generalize shortly.

### RFF's classic formulation

Given some data,
\\( \lbrace X, Y \rbrace_{i = 1}^{N_{data}} \\)
we want to learn a functional which approximates the data.

RFF requires us to specify some parameters:

| Parameter       | Rank               | Description                                                                                                                                                                                                          |
|-----------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| \\( R \\)       | \\( \mathbb{R} \\) | Number of "features". The more features, the better the approximation, at the cost of more computational overhead                                                                                                    |
| \\( \gamma \\)  | \\( \mathbb{R} \\) | \\( \gamma \\) acts as the "width" or "frequency" of the kernel approximation with larger values favoring more "global" approximations while smaller values of \\( \gamma \\) prefers more "local" approximations.   |
| \\( \lambda \\) | \\( \mathbb{R} \\) | \\( \lambda \\) acts as a regularization term with larger values favoring "smoother" approximations. \\( \lambda \\) has an intimate connection to Ridge Regression and the penalty terms it uses.                   |

Then the learning problem of RFF proceeds as follows:

Algorithm: 


| Step                    | Shape                             |                                                                    |
|-------------------------|-----------------------------------|--------------------------------------------------------------------|
| S1: [Generate a kernel] | \\(\text{kernel} \in [R, l_x] \\) | \\(\text{kernel} \sim \mathcal{N}(\mu=0,\sigma=1) \\) |
| S1: [Generate a kernel] | \\(\text{kernel} \in [R, l_x] \\) | \\(\text{kernel} \sim \mathcal{N}(\mu=0,\sigma=1) \\) |











We use \\(R\\) random
fourier features to approximate the function.

To put some dimensions on things:
\\[ 
X \in [N, l_x] \newline
Y \in [N, l_y] \newline
\text{kernel} \in [R, l_x] \sim \mathcal{N}(\mu=0,\sigma=1) \newline
\text{bias} \in [R, 1] \sim 2\pi\mathcal{U}(\min=0,\max=1) \newline
\\]
<!-- \text{rff\\_projection} \in [N, R] \newline -->
<!-- \text{weights} \in [R, l_y] \newline -->

\\[
\text{bias} \sim 2\pi\mathcal{U}(\min=0,\max=1); 
\\]



<!-- Despite being a great dialog, the blog post also has little teasers like the -->
<!-- following passage near the bottom: -->

<!-- > I sometimes use random features in my job. I like to get creative with -->
<!-- > special-purpose random features. It’s such an easy thing to try. When they -->
<!-- > work and I’m feeling good about life, I say “wow, random features are so -->
<!-- > powerful! They solved this problem!” Or if I’m in a more somber mood, I say -->
<!-- > “that problem was trivial. Even random features cracked it.” It’s the same -->
<!-- > way I think about nearest neighbors. When nearest neighbors cracks a dataset, -->
<!-- > you either marvel at the power of nearest neighbors, or you conclude your -->
<!-- > problem wasn’t hard at all. Regardless, it’s an easy trick to try. -->





