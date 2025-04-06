A repository where I collect my experiments with JAX.

Implementation of matrix free jacobian-vector products for QR-decomposition, Cholesky factorization, and bidiagonalization. These are needed for efficient forward-mode automatic differentiation.

Next up are vector-jacobian products of the above, needed for reverse-mode automatic differentiation.

# CAS system to assist in finding the adjoint system
Can compute differentials, simplify expressions.
```text
Goal expressions:
(λ0 · Rᵀ + L · λ1ᵀ) = ∇Aµ
- (cᵀ · λ4) = ∇r~µ

Adjoint system:
(∇Rµ + Aᵀ · λ0 - (λ1 · B) + R · λ3ᵀ + R · λ3 + λ4 · e1ᵀ) = 0
(∇Lµ - (λ0 · Bᵀ) + A · λ1 + L · λ2ᵀ + L · λ2) = 0
(∇Bµ - (Lᵀ · λ0) - (λ1ᵀ · R)) = 0
(∇cµ - (r~ᵀ · λ4)) = 0
```

![alt text](image-1.png)
Output below. Next would be isolating within all the inner products and grouping them.

```
dµ = (〈∇Rµ , dR〉 + 〈∇Lµ , dL〉 + 〈∇Bµ , dB〉 + 〈∇cµ , dc〉 + 〈λ0 , dA · R〉 + 〈λ0 , A · dR〉 + 〈λ0 , - dL · B〉 + 〈λ0 , - L · dB〉 + 〈λ1 , dAᵀ · L〉 + 〈λ1 , Aᵀ · dL〉 + 〈λ1 , - dR · Bᵀ〉 + 〈λ1 , - R · dBᵀ〉 + 〈λ2 , dLᵀ · L〉 + 〈λ2 , Lᵀ · dL〉 + 〈λ3 , dRᵀ · R〉 + 〈λ3 , Rᵀ · dR〉 + 〈λ4 , dR · e1〉 + 〈λ4 , - dc · r~〉 + 〈λ4 , - c · dr~〉)
```

# Random taylor stuff

![alt text](image.png)
