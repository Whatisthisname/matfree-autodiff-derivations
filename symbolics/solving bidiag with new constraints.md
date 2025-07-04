| Name | Type | Shape |
|------|------|-------|
| A | matrix | n × m |
| B | matrix | k × k |
| L | matrix | n × k |
| R | matrix | m × k |
| c | scalar | 1 × 1 |
| r~ | vector | m × 1 |
| $\Lambda$ | matrix | n × k |
| P | matrix | m × k |
| $\Sigma$ | matrix | k × k |
| $\Omega$ | matrix | k × k |
| $\kappa$ | vector | m × 1 |
| S | matrix | k × k |


## Goal expressions:
$$(\Lambda·Rᵀ + L·Pᵀ) = ∇Aµ$$
$$- c·\kappa = ∇r~µ$$

## Adjoint system:
$$(∇Lµ - \Lambda·Bᵀ + A·P + L·(⟍\Sigma)ᵀ + L·⟍\Sigma) = 0$$
$$(∇Bµ - Lᵀ·\Lambda - Pᵀ·R + »S) = 0$$
$$(∇Rµ + Aᵀ·\Lambda - P·B + R·(◣\Omega)ᵀ + R·◣\Omega + \kappa·e_1ᵀ) = 0$$
$$(∇cµ - \tilde{r} ^\top \kappa) = 0$$
$$(∇resµ - P·e_k) = 0$$


## Solving:

### $\color{yellow}\text{Initial condition}$:

We are given $\rho_k$ from the last constraint. We also "have" $\lambda_{k+1} := 0.$

### $\color{yellow}\rho_i \to \sigma_i$:

From $dB$, we get $$L^\top \Lambda = \nabla B - P^\top R + »S$$

In equation $$dL, we can left multiply $L^\top$ and get:

$$L^\top \nabla L - L^\top \Lambda B^\top + L^\top A P + 2 ⟍\Sigma = 0$$

Replace the occurence of $L^\top \Lambda$:

$$L^\top \nabla L - (\nabla B - P^\top R + »S) B^\top + L^\top A P + 2 ⟍\Sigma = 0$$

Now, take only the diagonal of this matrix equation, which will remove the occurence of $S$:

$$ - \sigma_i = l_i^\top \nabla {l_i} - \alpha_i \nabla \alpha_i - \beta_i \nabla \beta_i + \rho^\top(\alpha_i r_i + \beta_i r_{i+1}) + l_i^\top A \rho_i$$

### $\color{yellow} \lambda_{i+1}, \rho_i, \sigma_i \to \lambda_i$:

Take $[dL]$ and expand columnwise:

$$\nabla l_i - \alpha_i \lambda_i - \beta_i \lambda_{i+}  + A\rho_i + 2Le_i\sigma_i = 0$$

Isolate for $ \lambda_i$.

### $\color{yellow}\rho_i \to ø_i$:
From $[dB]^\top$, we get $$R^\top P = (\nabla B)^\top - \Lambda^\top L + «S^\top$$

Define $$Ø = ◣\Omega + (◣\Omega)^\top$$

$R^\top [dR]$:
$$R^\top \nabla R + R^\top Aᵀ\Lambda - R^\top PB + Ø + R^\top\kappa e_1^\top = 0$$

Substitute $R^\top P$ from previous equation:

$$\begin{align}
0 &= R^\top \nabla R + R^\top Aᵀ\Lambda - (\nabla B^\top - \Lambda^\top L + «S^\top)B + ◣\Omega + (◣\Omega)^\top + R^\top\kappa e_1^\top
\\ &= R^\top \nabla R + R^\top Aᵀ\Lambda - \nabla B^\top B + \Lambda^\top L B - «S^\top B + ◣\Omega + ◥\Omega^\top + R^\top\kappa e_1^\top
\end{align}$$
Here,  $«S^\top B$ is a strictly lower triangular matrix. If we consider only equations in the strict upper triangle, we get rid of both this term and $◣\Omega $:

$$\begin{align}
◹0 = 0&= ◹(R^\top \nabla R + R^\top Aᵀ\Lambda - \nabla B^\top B + \Lambda^\top L B - «S^\top B + ◣\Omega + ◥\Omega^\top + R^\top\kappa e_1^\top)
\\ &= ◹R^\top \nabla R + ◹R^\top Aᵀ\Lambda - ◹\nabla B^\top B + ◹\Lambda^\top L B + ◹\Omega^\top + ◹R^\top\kappa e_1^\top
\end{align}$$

Taking a columnwise expansion:
$$\begin{align}
0 &= \bigg[◹R^\top \nabla R + ◹R^\top Aᵀ\Lambda - ◹\nabla B^\top B + ◹\Lambda^\top L B + ◹\Omega^\top + ◹R^\top\kappa e_1^\top\bigg]_i 
\\ 
&=◹R^\top \nabla r_i + ◹R^\top Aᵀ\lambda_i - [◹\nabla B^\top B]_i + [◹\Lambda^\top L B]_i + [◹\Omega^\top]_i + [◹R^\top\kappa e_1^\top]_i
\end{align}$$

We will investigate these terms separately:

$$[◹\Lambda^\top L B]_i = \alpha_i ◹_i\Lambda^\top l_i + \beta_{i-1} ◹_i\Lambda^\top l_{i-1}$$

### $\color{yellow} \rho_i, \lambda_i, ø_i \to \rho_{i-1}$:

Take $[dR]$ and expand columnwise:

$$(∇Rµ + Aᵀ·\Lambda - P·B + R (◣\Omega)ᵀ + R ◣\Omega + \kappa e_1ᵀ) = 0$$

$$(\nabla r_i + Aᵀ \lambda_i - \alpha_i \rho_i - \beta_{i-1}\rho_{i-1} + Rø_i + \kappa \delta_{1i}) = 0$$

Isolate for $\rho_{i-1}$.