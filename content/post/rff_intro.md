+++
title = 'Random Fourier Features are economic nonlinear regressors'
subtitle = 'Part I of a two part series on RFFs'
date = 2024-03-23
+++

### Part I: Random Kitchen Sinks

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

The `gamma` and `lambda_reg` parameters determine specific properties of the
kernel fit. `gamma` roughly controls the 'width' or 'frequency' of the kernel
and `lambda_reg` roughly controls the amount of potential over-fitting (using
a mechanism similar to ridge regressions).

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

