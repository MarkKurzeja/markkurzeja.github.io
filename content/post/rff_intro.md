+++
title = 'Random Fourier Features, Part I '
subtitle = 'RFFs deserve to be your method-of-choice for baselines'
date = 2024-03-23
+++

Random Fourier Features, in my opinion, have one of the best
performance-to-cost tradeoffs in machine learning techniques today. Simple to
code, cheap to fit, and unreasonably effective, they have been my
bread-and-butter for small-to-medium learning tasks and serve as an amazing
baseline for more complex systems. 

This post will be the first of a three part series. To give you a roadmap:

* **Post I**: This post will be short and introduce RFF by example
* **Post II**: The next post will explore the connection between MLPs, used in
  Transformers, and RFF. I won't spoil the surprise just yet.
* **Post III**: The final post will provide numerical examples comparing MLPs and
  RFFs. We will also explore their potential use in language modeling and deep
  learning stacks. 

Note: This post _will not_ be about the "how" of RFF. I highly recommend
reading Gregory Gundersen's [blog
post](https://gregorygundersen.com/blog/2019/12/23/random-fourier-features/)
for more background if you need it.

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
adaptability of RFF even under strange cases.


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

Let's define the parameters:

| Parameter | Rank               | Description                                                                                                                                                                                                          |
|-----------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $R$       | $\mathbb{R} $ | Number of "features" or "kernels" used by RFF. The more features, the better the approximation, at the cost of more computational overhead                                                                                                    |
| $\gamma$  | $\mathbb{R} $ | $ \gamma $ acts as the "width" or "frequency" of the kernel approximation. Larger values  of $\gamma$ favor more "global" approximations while smaller values of $ \gamma $ prefer more "local" approximations. I've heuristically found increasing $\gamma$ as a function  |
| $\lambda$ | $\mathbb{R} $ | $ \lambda $ acts as a regularization term. Larger values of $ \lambda $ favor simpler functions. | 

To avoid the suspense:

```
# Setup
np.random.seed(0)  # Set the random seed for reproducibility
R = 100            # Number of RFF samples
gamma = R / 10     # Kernel "width" parameter
lambda = 0.1       # Regularization parameter

# Here is RFF, in all it's glory...
kernel = np.random.normal(size=(R, 1))
bias = 2 * np.pi * np.random.rand(R, 1)
proj = np.cos(gamma * (X @ kernel.T) + bias.T)
weights = np.linalg.solve(proj.T @ proj + lambda * np.eye(R), proj.T @ y)
y_hat = proj @ weights

# And plot the results.
plt.plot(X, y_hat, label='RFF Approximation', linestyle='-', color = "orange", lw= 5)
plt.suptitle(f'RFF Approximation R={R}')
plt.legend()
plt.show()
```

#### Plot the fit for various values of $R$

Another small miracle of RFF is the approximation improves by increasing $R$.
Let's plot the fit for a few values of $R$.

{{< gallery caption-effect="fade" >}}
    {{< figure link="/posts/rff/rff_2.png" caption="R=2" >}}
    {{< figure link="/posts/rff/rff_5.png" caption="R=10" >}}
    {{< figure link="/posts/rff/rff_10.png" caption="R=10" >}}
    {{< figure link="/posts/rff/rff_30.png" caption="R=10" >}}
    {{< figure link="/posts/rff/rff_100.png" caption="R=100" >}}
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

### Part II: Five Little Lines

On it's face, RFF is only five lines of Numpy code. To avoid any mystery, I When I code, I want
everything to be easy to copy-paste, so I include the setup code as well.
Translating the matlab code more or less directly from their blog:

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



In RFF, the parameter `R` sets the number of random projections used to build
the kernel estimator. Larger values of `R` yield better approximations, on
average, at the cost of more compute. For example, we can plot the function we
are approximating for a few choices of `R` to get a feel for how `R` increases
the complexity of the fit.

![](/posts/rff/rff_1.png)
![](/posts/rff/rff_2.png)
![](/posts/rff/rff_10.png)
![](/posts/rff/rff_100.png)

### Part III: Multidimensional Generalizations

Being a linear regression method (in the parameters), RFF naturally scales to
multidimensional data. For example:

```
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Adjust parameters for the multivariate case
R = 200  # Dimensionality of random feature space
gamma = 1.0  # Kernel parameter
lambda_reg = 0.1  # Regularization parameter

# Generate multivariate data for y = 2 * sin(x1) + 4 * sin(x1 * x2)
N = 1000  # Number of data points along each axis
x1 = np.linspace(-np.pi, np.pi, N)
x2 = np.linspace(-np.pi, np.pi, N)
X1, X2 = np.meshgrid(x1, x2)
Y = 2 * np.sin(X1) + 4 * np.sin(X1 * X2)

# Flatten the inputs and outputs
X1_flat = X1.flatten().reshape(-1, 1)
X2_flat = X2.flatten().reshape(-1, 1)
X_flat = np.hstack((X1_flat, X2_flat))
Y_flat = Y.flatten()

# Random Fourier feature mapping for Gaussian kernel
kernel = np.random.normal(size=(R, 2))
bias = 2 * np.pi * np.random.rand(R, 1)
rff_projection = np.cos(gamma * (X_flat.dot(kernel.T)) + bias.T)
weights = np.linalg.solve(
  rff_projection.T.dot(rff_projection) + lambda_reg * np.eye(R),
  rff_projection.T.dot(Y_flat),
)
y_hat = rff_projection.dot(weights).reshape(N, N)

# Plotting
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.7)
ax.set_title('$y=2 \sin(x_1) + 4 \sin(x_1 x_2)$')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X1, X2, y_hat, cmap='viridis', alpha=0.7)
ax2.set_title(f'RFF Approximation; R={R}; param_count={np.prod(weights.shape)}')
plt.show()
```

Produces:

![](/posts/rff/rff_2d.png)

