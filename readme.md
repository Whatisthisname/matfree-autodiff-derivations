A repository where I collect my experiments with JAX.

Implementation of matrix free jacobian-vector products for QR-decomposition, Cholesky factorization, and bidiagonalization. These are needed for efficient forward-mode automatic differentiation.

Next up are vector-jacobian products of the above, needed for reverse-mode automatic differentiation.

# CAS system to assist in finding the adjoint system
Can compute differentials, simplify expressions.
```text
Differentiated constraints:
dc1 = (dA·R + A·dR - dL·B - L·dB) = 0
dc2 = (dAᵀ·L + Aᵀ·dL - dR·Bᵀ - R·dBᵀ) = 0
dc3 = (dLᵀ·L + Lᵀ·dL) = 0
dc4 = (dRᵀ·R + Rᵀ·dR) = 0
dc5 = (dR·e1 - dc·r~ - c·dr~) = 0
dc6 = (dc - r~ᵀ·dr~·(||r~||^-3)) = 0
dc7 = (»dB + ◺dB) = 0

Goal expressions:
(λ1·Rᵀ + L·λ2ᵀ) = ∇Aµ
(- c·λ5 - (||r~||^-3)·λ6·r~) = ∇r~µ

Adjoint system:
(∇Lµ - λ1·Bᵀ + A·λ2 + L·λ3ᵀ + L·λ3) = 0
(∇Bµ - Lᵀ·λ1 - λ2ᵀ·R + »λ7 + ◺λ7) = 0
(∇Rµ + Aᵀ·λ1 - λ2·B + R·λ4ᵀ + R·λ4 + λ5·e1ᵀ) = 0
(∇cµ - r~ᵀ·λ5 + λ6) = 0

A     matrix, A rows x A cols
B     matrix, B rows x B cols
L     matrix, A rows x B rows
R     matrix, A cols x B cols
r~    vector, A cols
λ1    matrix, A rows x B cols
λ2    matrix, A cols x B rows
λ3    matrix, B rows x B rows
λ4    matrix, B cols x B cols
λ5    vector, A cols
λ6    scalar
λ7    matrix, B rows x B cols
```

# Random taylor stuff

![alt text](image.png)
