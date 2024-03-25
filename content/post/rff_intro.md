+++
title = 'Random Fourier Features, Part I '
subtitle = 'RFFs deserve to be your method-of-choice for baselines'
date = 2024-03-23
+++

Random Fourier Features, in my opinion, have one of the best
performance-to-cost tradeoffs in machine learning techniques today. Simple to
code, cheap to fit, and unreasonably effective, they have been my
bread-and-butter for small-to-medium multi-dimensional learning tasks and serve
as an amazing baseline for more complex systems. 

This post will be the first of a three part series. To give you a roadmap:

* **Post I**: This post will help to build intuition of RFF through
  low-dimensional examples and explain some of the code to come.
* **Post II**: The next post will explore the connection between MLPs, used in
  Transformers, and RFF. I won't spoil the surprise just yet.
* **Post III**: The final post will provide numerical examples comparing MLPs
  and RFFs in higher dimensional problems. We will also explore their potential
  use in language modeling and deep learning stacks. 

Note: This post intends to show how RFF works via examples, code, and visuals.
Great tutorials of RFF, from a mathematical perspective exist and to summarize
their findings here I feel would be a disservice to the topic.
I highly recommend reading Gregory Gundersen's [blog
post](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/)
or Rahimi and Recht's [Reflections on Random Kitchen
Sinks](https://archives.argmin.net/2017/12/05/kitchen-sinks/) if you are
curious about *how* these approximations work.

## Learning by Example: One Dimensional Learning

#### Function Definition

Our goal, today, will be to teach-via-example by fitting the following
(highly-nonlinear) function:

$$
y =
\begin{cases}
x \sin(3x) & \text{if } x < 0, \\\\
1 + x & \text{otherwise}.
\end{cases}
$$

This functional is *intentionally* pathological to showcase the
adaptability of RFF even with traditionally difficult functions.


#### Plots

Let's see what we are working with:

```
import numpy as np
import matplotlib.pyplot as plt

N_data = 1000
X = np.linspace(-np.pi, np.pi, N_data).reshape(N_data, 1)
y = [np.piecewise(x, x < 0, [x * np.sin(3 * x), 1 + x]) for x in X]

# Plot it out.
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('$f(x) = I[X < 0](x \sin(3x)) + I[x\geq 0](1 + x)$')
```

Nothing special here. 
1. We start with $N_{data}$ points
2. Our input, $x$ has domain $x \in [-\pi, \pi]$.
3. We set $y = \mathbb{I}\[X < 0\](x
   \sin(3x)) + \mathbb{I}\[x \geq 0\](1 + x)$. 

$\mathbb{I}[\cdot]$ is the [Iverson
bracket](https://en.wikipedia.org/wiki/Iverson_bracket). For simplicity, in
this blog post, we will assume all the data fits *in memory*. In future blog
posts, we will address the case where this is not possible.

![](/posts/rff/rff_goal.png)

#### RFF is Five Lines of Code

The first amazing fact about RFF is that it is four lines of setup and five
lines of "modeling" code. 

There are three parameters which drive RFF:

| Parameter | Rank               | Description                                                                                                                                                                                                          |
|-----------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $R$       | $\mathbb{R} $ | Number of "features" or "kernels" used by RFF. The more features, the better the approximation, at the cost of more computational overhead                                                                                                    |
| $\gamma$  | $\mathbb{R} $ | $ \gamma $ acts as the "width" or "frequency" of the kernel approximation. Larger values  of $\gamma$ favor more "local" approximations while smaller values of $ \gamma $ prefer more "global" approximations. I've heuristically found increasing $\gamma$ as a function of R can help with fitting fitting.|
| $\lambda$ | $\mathbb{R} $ | $ \lambda $ acts as a regularization term. Larger values of $ \lambda $ favor simpler functions. | 

And now for it all put together:

```
# Setup
np.random.seed(0)  # Set the random seed for reproducibility
R = 100            # Number of RFF samples
gamma = R / 10     # Kernel "width" parameter
llambda = 0.1      # Regularization parameter

# Here is RFF, in all it's glory...
kernel = np.random.normal(size=(R, 1))
bias = 2 * np.pi * np.random.rand(R, 1)
proj = np.cos(gamma * (X @ kernel.T) + bias.T)
weights = np.linalg.solve(proj.T @ proj + llambda * np.eye(R), proj.T @ y)
y_hat = proj @ weights

# And plot the results.
plt.plot(X, y_hat, label='RFF Approximation', linestyle='-', color = "orange", lw= 5)
plt.suptitle(f'RFF Approximation R={R}')
plt.legend()
plt.show()
```

To avoid code without explanation, the modeling code is roughly as
follows. In general, $x$ and $y$ are not limited to scalars and be vectors. We
will let $d_x$ and $d_y$ denote the dimensionality of $x$ and $y$ respectively.
In the problem above, since $x$ and $y$ are scalar quantities, $d_x = d_y = 1$.

| Step                    | Shape                             |                                                                    |
|-------------------------|-----------------------------------|--------------------------------------------------------------------|
| S1: [Sample a kernel, $K$] | $K \in [R, d_x] $ | $K \sim \mathcal{N}(\mu=0,\sigma=1) $ |
| S2: [Sample a phase bias $B$] | $B \in [R, 1] $ | $B\sim 2\pi\text{Uniform}(0, 1) $ |
| S3: [Project onto random map, $P$] | $P \in [N_{data}, R] $ | $P \leftarrow \cos\left(\gamma XK^T + B^T\right) $ |
| S4: [Solve for weights, $W$] | $W \in [R, d_y] $ | Solve for $W$: $$ \left( P^TP + \lambda I(R)\right)W = P^Ty$$ |
| S5: [Predict $\hat{y}$] | $\hat{y} \in [N_{data}, d_y] $ | $\hat{y} = PW$|

#### Plot the fit for various values of $R$

Another small miracle of RFF is the approximation improves by increasing $R$.
Let's plot the fit for a few values of $R$.

{{< gallery caption-effect="fade" >}}
    {{< figure link="/posts/rff/rff_2.png" caption="R=2" >}}
    {{< figure link="/posts/rff/rff_5.png" caption="R=5" >}}
    {{< figure link="/posts/rff/rff_10.png" caption="R=10" >}}
    {{< figure link="/posts/rff/rff_30.png" caption="R=30" >}}
    {{< figure link="/posts/rff/rff_100.png" caption="R=100" >}}
{{< /gallery >}}

We can see a minor ringing effect in the fit, akin to [Gibbs
Phenomenon](https://en.wikipedia.org/wiki/Gibbs_phenomenon) but the
approximation fits decently well and handles the discontinuities easily.

## Underspecification

The function above was difficult to fit, but we had a large amount of data.
What happens if we reduce the data significantly but keep the fit heavily over
parameterized ? A good approximation should (a) become better with more
parameters and (b) self-regularize when it is over-parameterized. This second
point is important in practice: methods which fail to self-regularize can
"blow-up" without other interventions. If you have ever fit a polynomial of
large degree to data, you have likely experienced [Runge
phenomena](https://www.johndcook.com/blog/2017/11/18/runge-phenomena/) which is
one such pathology.

The second amazing fact of RFF is it has regularization built in. The
parameters $\gamma$ and $\lambda$ jointly determine how flexible the fit will
be in the limit. For a fixed $\gamma$ and $\lambda$ we can see this "limiting"
effect by increasing $R$.

{{< gallery caption-effect="fade" >}}
    {{< figure link="/posts/rff/rff_underspecified_2.png" caption="R=2" >}}
    {{< figure link="/posts/rff/rff_underspecified_5.png" caption="R=5" >}}
    {{< figure link="/posts/rff/rff_underspecified_30.png" caption="R=30" >}}
    {{< figure link="/posts/rff/rff_underspecified_100.png" caption="R=100" >}}
    {{< figure link="/posts/rff/rff_underspecified_500.png" caption="R=500" >}}
{{< /gallery >}}

Even with hundreds more interpolates than data points, the function naturally
regularizes itself due to the $\gamma$ and $\lambda$ parameters. What happens
if $\gamma$ or $\lambda$ are misspecified? 

{{< gallery caption-effect="fade" >}}
    {{< figure link="/posts/rff/rff_underspecified_small_gamma_small_lambda.png" caption="Small $\gamma$, Small $\lambda$" >}}
    {{< figure link="/posts/rff/rff_underspecified_big_gamma_small_lambda.png" caption="Big $\gamma$, Small $\lambda$" >}}
    {{< figure link="/posts/rff/rff_underspecified_big_gamma_big_lambda.png" caption="Big $\gamma$, Big $\lambda$" >}}
    {{< figure link="/posts/rff/rff_underspecified_small_gamma_big_lambda.png" caption="Small $\gamma$, Big $\lambda$" >}}
    {{< figure link="/posts/rff/rff_underspecified_small_gamma_zero_lambda.png" caption="Small $\gamma$, Zero $\lambda$" >}}
    {{< figure link="/posts/rff/rff_underspecified_big_gamma_zero_lambda.png" caption="Big $\gamma$, Zero $\lambda$" >}}
{{< /gallery >}}

The worst fits come when $\lambda \rightarrow 0$ and $\gamma$ is larger than
the intrinsic variance of the function. $\lambda$ is a regularization term, so
as $\lambda \rightarrow 0$, the functional is able to fit the data perfectly at
the cost of larger variance.

In practice, I typically start with somewhat larger values of $\lambda$ and
somewhat smaller values of $\gamma$ and adjust them accordingly depending on
the type of fit I want and how it performs on validation data.

## Higher Dimensions

Most analytic methods of curve fits tend to be specified only for one dimension
and quickly become unwieldy in higher dimensions. RFF's third amazing fact is
that it works, just fine, for inputs *and* outputs which are multidimensional.
The code changes are almost unnoticeable as well: where we used to have $d_x$
implicitly specified as $d_x=1$, we now will explicitly parameterise it.

#### Learn-By-Example

As before, we are going to learn-by-example and teach RFF to approximate the function:

$$
y = 2 \sin(x_1) + 4 \sin(x_1x_2)
$$

Unfortunately, we are limited by human biology in this example: with $d_x = 2$
and $d_y = 1$, we exhaust the three dimensions our eyes are capable of seeing.
The dimensions interact however, and the functional looks quite cool.

The plotting code this time is only slightly more difficult:

```
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Generate multivariate data for y = 2 * sin(x1) + 4 * sin(x1 * x2)
N = 1000  # Number of data points along each axis
x1 = np.linspace(-np.pi, np.pi, N)
x2 = np.linspace(-np.pi, np.pi, N)
X1, X2 = np.meshgrid(x1, x2)
Y = 2 * np.sin(X1) + 4 * np.sin(X1 * X2)

# Plotting
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.7)
ax.set_title('$y=2 \sin(x_1) + 4 \sin(x_1 x_2)$')
```

![](/posts/rff/rff_2d.png)

#### Fitting with RFF

The code for RFF is more-or-less the same as before except we have to "flatten"
the $x$ and $y$ dimensions from a grid to a vector of examples. 

```
# Setup
np.random.seed(0)  # Set the random seed for reproducability
R = 10             # Number of RFF samples
gamma = 1          # Kernel "width" parameter
llambda = 0.1      # Regularization parameter

# Flatten the inputs and outputs
X1_flat = X1.flatten().reshape(-1, 1)
X2_flat = X2.flatten().reshape(-1, 1)
X_flat = np.hstack((X1_flat, X2_flat))
Y_flat = Y.flatten()

# Random Fourier feature mapping for Gaussian kernel
kernel = np.random.normal(size=(R, 2))
bias = 2 * np.pi * np.random.rand(R, 1)
proj = np.cos(gamma * (X_flat.dot(kernel.T)) + bias.T)
weights = np.linalg.solve(
  proj.T.dot(proj) + llambda * np.eye(R),
  proj.T.dot(Y_flat),
)
y_hat = proj.dot(weights).reshape(N, N)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X1, X2, y_hat, cmap='viridis', alpha=0.7)
ax2.set_title(f'RFF Approximation; R={R}')
plt.show()
```

which, for different values of $R$ yields:

{{< gallery caption-effect="fade" >}}
    {{< figure link="/posts/rff/rff_2d_2.png" caption="R=2" >}}
    {{< figure link="/posts/rff/rff_2d_5.png" caption="R=5" >}}
    {{< figure link="/posts/rff/rff_2d_10.png" caption="R=10" >}}
    {{< figure link="/posts/rff/rff_2d_100.png" caption="R=100" >}}
    {{< figure link="/posts/rff/rff_2d_200.png" caption="R=200" >}}
{{< /gallery >}}


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


