+++
title = 'Random Fourier Features, backstop'
subtitle = 'RFFs deserve to be your method-of-choice for baselines'
date = 2024-03-23
draft = true
+++


### This is Part-One of a Three-Part Series

A few years ago, I stumbled on a short blog post: [Reflections on Random
Kitchen Sinks](https://archives.argmin.net/2017/12/05/kitchen-sinks/). This
blog post turned out to be a transcription of an awards speech: the NIPS 2017
test-of-time award.

The post consumed a great deal of my mental space for quite some time. It
changed the way I actually practice machine learning, how I think about
baselines, and how I think about MLPs in the Transformer Architecture.  And my
goal is to share this adventure with you.

My first posts on TensorTales will be about the road I traveled after reading
this transcript. I will break this out across three blog posts since I have
three main points I want to discuss.

#### How I Read Math

A short digression: I developed a habit a long time ago when reading
mathematics. I've found that when reading math, or any topic which can be
highly technical, a "best-practice" is often to code up the idea for oneself.
The goal of this practice is two fold:

1. Programming requires precision: to code something which is abstract often
   involves resolving a bunch of the hand-wavy details that traditional
   mathematical notation can allow
2. Programming an idea provides a grounding in reality: a lot of ideas are
   "cool", few are "useful", and fewer still are "economic". Proving a concrete
   manifestation of an idea, in code, helps to separate out ideas quickly and
   get a decent proxy for their utility (at least in the context in which you
   test them)

Will this coding practice slow you down? With high probability, its slower than
careful reading. However, to "feel" an idea, sometimes its best to poke it.
Prod it. Find its rough edges... and programming advanced ideas can help one to
do that in an economic way.

### Random Fourier Features are Five Lines of Code

I mention my practice of coding math to emphasize my shock when I learned the
code economy of Random Fourier Features. Its five lines of "productive" code:

```
import numpy as np
import matplotlib.pyplot as plt

# Setup:
R = 10            # Number of RFF samples
gamma = 1         # Kernel "width" parameter
lambda_reg = 0.1  # Regularization parameter

# Data:
N = 1000
np.random.seed(0)
X = np.linspace(-np.pi, np.pi, N).reshape(N, 1)
y = (X * np.sin(2 * X) + np.random.normal(size=(N, 1), scale = 0.25)).reshape(N, 1)

# Here is RFF, in all it's glory...
kernel = np.random.normal(size=(R, 1))
bias = 2 * np.pi * np.random.rand(R, 1)
rff_projection = np.cos(gamma * (X @ kernel.T) + bias.T)
weights = np.linalg.solve(rff_projection.T @ rff_projection + lambda_reg * np.eye(R), rff_projection.T @ y)
y_hat = rff_projection @ weights

# Plot it out.
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
plt.plot(X, y_hat, label='RFF Approximation', linestyle='-', color = "orange", lw= 5)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.suptitle(f'RFF Approximation R={R}')
plt.title(f'Total Learnable Parameters = {np.prod(weights.shape)}')
plt.show()
```

### Random Kitchen Sinks

Random Fourier Features, at first glance, seems a little.. well.. mad..

1. Take a feature set. Multi-dimensional ? Nonlinear interactions ? Doesn't
   really matter.
2. Linearly project it onto some noise distribution.
3. Take a little cosine here... Add a little uniform noise over there...
4. Fit a linear classifier to the strange little mix from (3)

.. And out comes a reasonably performing, easy, economic regressor.

But why?...

In [Reflections on Random Kitchen
Sinks](https://archives.argmin.net/2017/12/05/kitchen-sinks/), Ali Rahimi and
Ben Recht's talk on Random Fourier Features, they provide the intuition behind
the method and how their thinking about RFF has grown over the years. No matter
how many times I've seen the explanations, read the code, read the math,
derived the math... I couldn't quite "feel" why this should work. When I learn
something new, I like to think about putting that new thing "on a hook" of my
existing knoweldge. RFF seemed alien, however. I knew about kernel methods.
I knew about regression. But random regression? Why cosine? The code was
strange too... It felt too simple and seemed to work decently well for a whole
host of problems. Trying to find some way to convince myself it
`{can|could|should|must}` work, I began coding.

Note: This post _will not_ be about the _Why_ of RFF. Instead, it will be about
a curious connection I found when coding with it. I highly recommend reading
Gregory Gundersen's [blog
post](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/)
for more background if you need it.


