Assume matrix $A(\theta)$ is parameterized by some parameters $\theta \mapsto A(\theta)$, and that we only interact with $A$ through matrix-vector products, $A(\theta)v$.

We have derived
$$\nabla_A\mu = \sum_{n=1}^k -\lambda_n r_n^\top -l_n\rho_n^\top$$

and know by definition of the Jacobian of A, $J(A)= J_A(\theta)$

$$dA =  J_A(\theta) d\theta$$

To get the gradient w.r.t. $\theta$:
$$\begin{align*}
\langle\nabla_A\mu, dA \rangle &= \left\langle\sum_{n=1}^k -\lambda_n r_n^\top -l_n\rho_n^\top,  J_A(\theta) d\theta \right\rangle
\\ &= \left\langle\sum_{n=1}^k - J_A(\theta)^\top\lambda_n r_n^\top - J_A(\theta)^\top l_n\rho_n^\top,  d\theta \right\rangle
% \\ &=\left\langle\nabla_\theta \mu ,  d\theta \right\rangle
\end{align*}$$

Now, $ - J_A(\theta)^\top\lambda_n r_n^\top$ is a VJP.

Lets say we have function $$L(A) = u^\top A v$$ 
Also, 
$$\begin{align*}
dL &= u^\top dA v = \text{Tr}(u^\top dA v) = \text{Tr}(vu^\top dA)
\\
&\text{\;\; and}
\\
dL &= \langle\nabla_A L , dA \rangle = \text{Tr}((\nabla_A L)^\top , dA)
\end{align*}$$
So we can pattern-match and infer $\nabla_A L = uv^\top$.

If $A$ is now parameterized by $\theta$, we have $K(\theta) = u^\top A(\theta) v$, and 
$$\nabla_\theta K = J_A(\theta)^\top \nabla_A L = J_A(\theta)^\top uv^\top$$
This expression appears many times. So, we have shown that it can be replaced by the gradient of some function $K$:

$$\begin{align*}
d\mu &= \left\langle\sum_{n=1}^k - J_A(\theta)^\top\lambda_n r_n^\top - J_A(\theta)^\top l_n\rho_n^\top,  d\theta \right\rangle
\\&= \left\langle\sum_{n=1}^k - \nabla(\theta \mapsto \lambda_n^\top A(\theta) r_n) - \nabla(\theta \mapsto \rho_n^\top A(\theta) l_n),  d\theta \right\rangle
\end{align*}$$
# ok other stuff:

If I want gradient only w.r.t. one scalar parameter $\phi$ keeping the others fixed:

$$\begin{align*}
\langle\nabla_A\mu, dA \rangle &= \left\langle\sum_{n=1}^k -\lambda_n r_n^\top -l_n\rho_n^\top, \square d\phi_i \right\rangle
\\ &= \left\langle\sum_{n=1}^k -\lambda_n r_n^\top - l_n\rho_n^\top, \square  \right\rangle d\phi_i
\\ &= \left\langle\sum_{n=1}^k -\lambda_n r_n^\top, \square \right\rangle d\phi_i - \left\langle \sum_{n=1}^k l_n\rho_n^\top, \square  \right\rangle d\phi_i
\\ &= \left\langle\sum_{n=1}^k -\lambda_n, \square r_n \right\rangle d\phi_i - \left\langle \sum_{n=1}^k l_n, \square \rho_n  \right\rangle d\phi_i
\\ &= \nabla\left( \phi_i \mapsto \left\langle\sum_{n=1}^k -\lambda_n, A(\phi_i) r_n \right\rangle\right) d\phi_i - \nabla \left(\phi_i \mapsto \left\langle \sum_{n=1}^k l_n, A(\phi_i) \rho_n  \right\rangle\right) d\phi_i
\\ &= \frac{d}{d\phi_i}\left( \phi_i \mapsto \left\langle\sum_{n=1}^k -\lambda_n, A(\phi_i) r_n \right\rangle\right) d\phi_i - \frac{d}{d\phi_i} \left(\phi_i \mapsto \left\langle \sum_{n=1}^k l_n, A(\phi_i) \rho_n  \right\rangle\right) d\phi_i
\end{align*}$$

How do extend this to all parameters?


\\ &=\left\langle\nabla_\phi \mu ,  d\phi \right\rangle


We can do $v \mapsto Av$, so $v^\top \mapsto Av^\top = vA^\top$... so we can left multiply onto the transposed matrix and right multiply the normal matrix.

$f(v)\mapsto Av$

$df = Adv$

$\mu(f(v))\in \reals$

$$\begin{align*}
d\mu &= \langle \nabla_f, df  \rangle
\\&= \langle \nabla_f, Adv  \rangle
\\&= \langle A^\top\nabla_f, dv  \rangle
\end{align*}$$

The jacobian of $f = (v\mapsto Av)$ is $A$. So, `jvp(v)` is $Av$, while `vjp(w)` is $w^\top A = A^\top w$



Suppose we have scalar function µ, $\mu(f(v))\in \reals$

Then $$VJP: ((\mathcal{M} \to \reals) \times \mathcal{M}) \to (\reals^* \to  \mathcal{TM}^*))$$