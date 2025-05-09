$\nabla A = \sum_{n=1}^k -\lambda_n r_n^\top -l_n\rho_n^\top$

Assume $A$ is parameterized by some parameters $\theta \mapsto A(\theta)$, and that we only interact with $A$ through matrix-vector products, $A(\theta)v$ (and not $w^\top A(\theta)$).

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