+++
title = "Learning Orthonormal Matrices for Machine Learning"
subtitle = "How to learn orthonormal projections"
date = 2024-03-27
+++

## Prelude

I have been researching alternative means to initialize and parameterize MLPs for use in deep learning pipelines. Orthonormal matrices continue to come up as a fundamental building block of some methods. This post is about my journey to find the fastest way to parameterize them in modern machine learning stacks.

## Introduction to Orthonormal Matrices

An orthonormal matrix is a square matrix whose rows and columns are orthonormal vectors. In other words, the dot product of any two different rows (or columns) of the matrix is zero, and the dot product of a row (or column) with itself is one. Such a matrix has several properties which make it useful in a machine learning pipeline.

| Property      |  Comment   | 
| ---           | --- |
| Fast Inverses | If $Q$ is orthonormal, then its inverse is $Q^T$, and $QQ^T = Q^TQ = I$. This makes it easy to build a lot of "there-and-back-again" algorithms which have neat properties.|
| Unit Eigenvalues | If $Q$ is orthonormal, then the magnitude of each of its eigenvalues is one. This property is useful since it implies multiplying by $Q$ will not change the norm of each of the incoming vectors. | 

Orthonormal matrices are valuable in machine learning pipelines because they preserve the Euclidean norm of vectors. This property is particularly useful when preprocessing data since it ensures that the scale of the data remains unchanged. 

Despite their usefulness, parametrizing orthonormal matrices for machine learning can be difficult. In this post, we will cover some of the ways they can be generated quickly for use in a deep learning pipeline.

While transforms like the QR decomposition exist, they tend to be slow. To spoil the surprise, there are methods which are hundreds of times faster for large matrices

![](/posts/orthonormal_recipe/benchmarks.png)

## Before we begin

Throughout the post, we will be working with numerical examples. Assume we are given a matrix of parameters $A \in \mathbb{R}^{N\times N}$. Critically, $A$ has no constraints: it is just a learnable set of $N^2$ parameters arranged into a rank-2 tensor. 

The stated goal of this post will be to use $A$ to build an orthonormal matrix $Q$ as efficiently as possible. We will use $Q$ as a replacement for dense projections in parts of the pipeline, and we will need to parameterize it efficiently if we hope to succeed.


```
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import functools
import chex

# Utility function for generating the norms of the eigen values of a matrix
# Note: This function is not available for GPU/TPU backends.
def eigen_value_norms(x):
  return jax.vmap(jnp.linalg.norm)(jnp.linalg.eigvals(x))

# Generate a random matrix
key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (4, 4))
print(f"A: \n{A}")
print(f"eigen_value_norms(A): \n{eigen_value_norms(A)}")
```

Yields: 

$$
A = \begin{bmatrix}
0.0848 & 1.9098 & 0.2956 & 1.1210 \\\\
0.3343 & -0.8261 & 0.6481 & 1.0435 \\\\
-0.7825 & -0.4540 & 0.6298 & 0.8152 \\\\
-0.3279 & -1.1234 & -1.6607 & 0.2729 \\\\
\end{bmatrix}
$$

with eigenvalue norms:

$$
(||\lambda_1||, \ldots, ||\lambda_4||) = 
\begin{bmatrix}
1.3769 & 1.9691 & 1.9691 &  1.1398 
\end{bmatrix}^T
$$

Clearly $A$ is random and not orthonormal since the magnitude of $A$'s eigenvalues is not equal to one.

## Doesn't Jax Already have functions which do this?

Jax already has functions which initialize parameter matrices as orthonormal matrices such as [`jax.nn.initalizers.orthogonal`](https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.orthogonal.html). However, these are initializers. During training, it is easy to "lose" orthogonality unless care is taken with the updates.

Another good question: can't we just take the QR decomposition of a learnable matrix and call it a day? We can ! In fact, the QR decomposition method will be compared against others in this post. 

## Take One: QR Decomposition

One approach is to directly utilize $A$ by computing its QR decomposition. Given any matrix, we can decompose it into an orthogonal matrix $Q$ and an upper triangular matrix $R$. By extracting the $Q$ matrix from this decomposition, we obtain an orthonormal matrix.

```
# Perform QR decomposition
@jax.jit
def qr(x):
  q, _ = jsp.linalg.qr(x)
  return q

Q = qr(A)

print(f"Orthonormal Matrix Q: \n{Q}")
print(f"Eigenvalues Q: \n{eigen_value_norms(Q)}")

---
Orthonormal Matrix Q:
[[-0.09262133  0.7986421   0.12070771  0.5822558 ]
 [-0.36504808 -0.4619358  -0.45298663  0.66944635]
 [ 0.8543949   0.04950267 -0.48875752  0.1693365 ]
 [ 0.35800898 -0.38253853  0.7357642   0.42912114]]
Eigenvalues Q:
[1.0000002 0.9999999 0.9999999 1.       ]
```

## Take Two: Householder Transforms

Another method of transforming $A$ into $Q$ involves Householder transformations. In a Householder transform, we are given a vector $v$. We can generate an orthonormal matrix $Q$ via: 

$$ Q \leftarrow I - 2 v^Tv$$

```
@jax.jit
def householder(x):
  chex.assert_rank(x, 1)
  v = x * jnp.reciprocal(jnp.linalg.norm(x))
  return jnp.eye(len(x)) - 2 * jnp.outer(v, v)

# Example with a 4x4 matrix
H = householder(A[:, 0])
print(f"Orthonormal Matrix H: \n{H}")
print(f"eigen_value_norms H: \n{eigen_value_norms(H)}")

---

Orthonormal Matrix H:
[[ 0.98284256 -0.06762246  0.15827034  0.06631851]
 [-0.06762246  0.7334798   0.62379044  0.26138097]
 [ 0.15827034  0.62379044 -0.45998132 -0.61176205]
 [ 0.06631851  0.26138097 -0.61176205  0.74365914]]
eigen_value_norms H:
[1.         0.99999964 1.         1.        ]
```

This, however, does not fully utilize $A$: it only uses its first column. Luckily, the product of two orthonormal matrices is another orthonormal matrix. This fact can be used to build more flexible matrices by multiplying the Householder transforms of a subset of, or all of, the columns of $A$

```
@functools.partial(jax.jit, static_argnames=["order"])
def chained_householder(A, order = None):
  # chained_householder takes in a matrix, A, and extracts the first _order_
  # column vectors. It then constructs a householder matrix from each column
  # vector extracted and computes their matrix product to create the
  # final orthonormal approximation. If order is None, use all of the columns
  # of A.
  chex.assert_rank(A, 2)
  if order is not None:
    A = A[:, 0:order]
  return jnp.linalg.multi_dot(jax.vmap(householder, in_axes=1)(A))

print(f"Chained Householder Matrix: \n{chained_householder(A, order = 2)}")
print(f"eigen_value_norms: \n{eigen_value_norms(chained_householder(A, order = 2))}")

---

Chained Householder Matrix:
[[-0.19409285  0.44146028  0.4380249   0.75866663]
 [ 0.79662025  0.35965204  0.41836214 -0.24702147]
 [-0.29171947  0.8184333  -0.35301986 -0.34704953]
 [ 0.49256864  0.0770067  -0.7130807   0.49291176]]
eigen_value_norms:
[0.99999994 0.99999976 0.99999976 1.        ]
```

## Take Three: Cayley Transform

The Cayley transform provides another method to generate orthonormal matrices. Given a skew-symmetric matrix $S$, $Q$ can be formed via:

$$
Q \leftarrow (I + S)(I - S)^{-1}
$$

There are two ways to turn $A$ into a skew symmetric matrix: 

| Method | Formulation |
| --- | --- | 
| Use all of $A$ directly | $ S \leftarrow A - A^T $ 
| Use the lower-tri components of $A$ | $$ \begin{align*} A_{\text{lower}} &\leftarrow \text{Tril}(A, \text{diag = false}) \\\\ S &\leftarrow A_{\text{lower}} - A_{\text{lower}}^T \end{align*} $$ | 

This second formulation is preferred, when it is economic, since it allows us to get two orthonormal matrices out of $A$ instead of one if we utilize the upper triangular section of $A$ instead.

With the skew-symmetric formulation of $A$, we can generate $Q$:

$$
\begin{align*}
S &\leftarrow \text{SkewSymmetric}(A) \\\\
Q &\leftarrow \text{CayleyTransform}(S) \\\\
&= (I + S)(I - S)^{-1}
\end{align*}
$$

```python
@jax.jit
def skew_symmetric(A):
  chex.assert_rank(A, 2)
  A_lower = jax.numpy.tril(A, k = -1)
  return A_lower - A_lower.T

@jax.jit
def cayley_transform(A):
  chex.assert_rank(A, 2)
  S = skew_symmetric(A)
  I = jnp.eye(A.shape[0])
  return (I + S) @ jnp.linalg.inv(I - S)

print(f"Skew-symmetric Matrix: \n{skew_symmetric(A)}")
print(f"Cayley Transform: \n{cayley_transform(A)}")
print(f"eigen_value_norms: \n{eigen_value_norms(cayley_transform(A))}")

---

Skew-symmetric Matrix:
[[ 0.         -0.33432344  0.7824839   0.32787678]
 [ 0.33432344  0.          0.4539462   1.1234448 ]
 [-0.7824839  -0.4539462   0.          1.6607416 ]
 [-0.32787678 -1.1234448  -1.6607416   0.        ]]
Cayley Transform:
[[ 0.35566828 -0.8281733   0.3969655   0.1733424 ]
 [ 0.45279822  0.16153368 -0.4076859   0.77632004]
 [-0.7581913  -0.4245323  -0.3563349   0.3434295 ]
 [ 0.30597383 -0.32834178 -0.74110454 -0.4993353 ]]
eigen_value_norms:
[0.9999999 0.9999999 1.        1.       ]
```

## Take Four: Neumann Approximation to Cayley Transform

In the Cayley Transform, taking the inverse of $(I - S)$ can be expensive. To speed up the Cayley transform, we can use the Neumann series approximation:

$$
(I - S)^{-1} \approx \sum_{i = 0}^\infty A^i
$$

The Neumann approximation converges iff $||S|| < 1$ where $||\cdot||$ denotes the matrix norm. If we can ensure $S$ has a small norm, then $Q$ can be expressed as:

$$
\begin{align*}
S &\leftarrow \text{SkewSymmetric}(A) \\\\
Q &\leftarrow \text{CayleyTransform}(S) \\\\
&= (I + S)(I - S)^{-1} \\\\
&\approx (I + S)(I + S + S^2 + S^3 + S^4) \\\\
&\approx (I + S)(I + S + S^2) \\\\
\end{align*}
$$

The second to last approximation is the fourth-order Neumann approximation while the last approximation is a second-order Neumann approximation.
While not perfect, it is often "good enough for government work".

```python
@functools.partial(jax.jit, static_argnames=["expanded"])
def neumann_approx(A, expanded = False):
  S = skew_symmetric(A)
  # Normalize S to reduce its spectral norm
  S *= jnp.reciprocal(jnp.linalg.norm(S))
  I = jnp.eye(A.shape[0])
  IS = I + S
  S2 = jnp.linalg.matrix_power(S, 2)
  if not expanded:
    return IS @ (IS + S2)
  S3 = jnp.linalg.matrix_power(S, 3)
  S4 = jnp.linalg.matrix_power(S, 4)
  return IS @ (IS + S2 + S3 + S4)

print(f"Neumann Matrix: \n{neumann_approx(A)}")
print(f"eigen_value_norms: \n{eigen_value_norms(neumann_approx(A))}")

print(f"Neumann Matrix with expansion: \n{neumann_approx(A, expanded=True)}")
print(f"eigen_value_norms: \n{eigen_value_norms(neumann_approx(A, expanded=True))}")

---

Neumann Matrix:
[[ 0.83558977 -0.36708668  0.27625954  0.31916648]
 [ 0.08097215  0.6876185  -0.11617136  0.7318331 ]
 [-0.5515893  -0.51815355  0.292894    0.6450456 ]
 [ 0.04617295 -0.39037955 -0.9481574   0.1838977 ]]
eigen_value_norms:
[1.0491595 1.0491595 1.0000209 1.0000209]
Neumann Matrix with expansion:
[[ 0.88549775 -0.30075887  0.30205643  0.19132581]
 [ 0.15770163  0.8115121  -0.00212702  0.56809056]
 [-0.4397213  -0.3150354   0.6141498   0.5885297 ]
 [-0.00865609 -0.39736384 -0.7400856   0.55965173]]
eigen_value_norms:
[0.9999999 0.9999999 1.0108453 1.0108453]
```

## Take Five: Matrix Exponentials

Matrix exponentials provide yet another method to generate orthonormal matrices. Given a skew symmetric matrix, $S$, we can generate an orthonormal matrix $Q$ using: $ Q \leftarrow \exp(A) $. The matrix exponential is defined as $ \exp(A) = \sum_{k=0}^{\infty} \frac{A^k}{k!} $

```python
import functools

@functools.partial(jax.jit, static_argnames=["max_squarings"])
def expm(A, max_squarings = 16):
  S = skew_symmetric(A)
  return jsp.linalg.expm(S, max_squarings=max_squarings)

print(f"Orthonormal Matrix from Matrix Exponential: \n{expm(A)}")
print(f"eigen_value_norms: \n{eigen_value_norms(expm(A))}")
```

## Which is fastest?

Lets run some profiling code to determine which is fastest for a given matrix size. The following benchmarks were ran using the benchmarking code at the bottom of this post. All functions were ran under `jax.jit` on an A100 CPU.

![](/posts/orthonormal_recipe/benchmarks.png)

Two questions come out of this:

1. What is best? It appears the best method by almost 10x performance is to use a Householder approximation of order two. This makes sense: if you can parameterize a matrix using two matrix multiplies, then it is best to do so. 
