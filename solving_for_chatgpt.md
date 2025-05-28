
$\begin{array}{ll}
\text{Adjoint system:} \\
[dl_n] \text{ (for } n \in \{1...k-1\}\text{)} & \nabla l_n + a_n\lambda_n + b_n\lambda_{n+1} - A\rho_n + l_n\sigma_n = 0 \\
[dl_k] & \nabla l_k + a_k\lambda_k - A\rho_k + l_k\sigma_k = 0 \\
[dr_1] & \nabla r_1 - A^\top\lambda_1 + a_1\rho_1 + r_1\omega_1 + \kappa = 0 \\
[dr_n] \text{ (for } n \in \{2...k\}\text{)} & \nabla r_n - A^\top\lambda_n + b_{n-1}\rho_{n-1} + a_n\rho_n + r_n\omega_n = 0 \\
[dres] & \nabla res +\rho_k = 0 \\
[da_n] \text{ (for } n \in \{1...k\}\text{)} & \nabla a_n + l_n^\top\lambda_n + r_n^\top\rho_n = 0 \\
[db_n] \text{ (for } n \in \{1...k-1\}\text{)} & \nabla b_n + l_n^\top \lambda_{n+1} + r_{n+1}^\top \rho_{n} = 0 \\
[dc] & \nabla c - \tilde r^\top\kappa = 0
\\\\
\text{Goal expressions:} & \nabla \tilde r = -c\kappa \\
& \nabla A = \sum_{n=1}^k -\lambda_n r_n^\top -l_n\rho_n^\top \\
\end{array}$

# Solving adjoint system
Can immediately extract $\rho_k = -\nabla res$. 

Define $b_k\lambda_{k+1} = 0$, then equation $[dl_k]$ can be expressed like the others. 

`for n in [k, ... , 2]:` <br>
In equation $[da_n]$, isolate $l_n^\top \lambda_n = - \nabla a_n - r_n^\top\rho_n =: t$

In equation $[dl_n]$, using $LI(n)$, left-multiply $l_n^\top$, and isolate <br> $\sigma_n = -l_n^\top (\nabla l_n + b_n\lambda_{n+1} - A\rho_n) - a_n t$:

In the same equation, now isolate instead <br> 
$\lambda_n = a_n^{-1}(-\nabla l_n - b_n\lambda_{n+1} + A\rho_n - l_n\sigma_n) $

In equation $[db_{n-1}]$, isolate $r_n^\top \rho_{n-1} = -\nabla b_{n-1} - l_{n-1}^\top \lambda_n =: w$

In equation $[dr_n]$, using $RI(n)$, left-multiply $r_n^\top$, and isolate <br>
$\omega_n = -r_n^\top(\nabla r_n - A^\top\lambda_n + a_n\rho_n) - b_{n-1}w$

In the same equation, now isolate instead <br>
$\rho_{n-1} = b_{n-1}^{-1}(-\nabla r_n + A^\top\lambda_n - a_n\rho_n - r_n\omega_n)$
<br>`end`

For the final iteration: <br>
In equation $[da_1]$, isolate $l_1^\top \lambda_1 = - \nabla a_1 - r_1^\top\rho_1 =: t$

In equation $[dl_1]$, using $LI(1)$, left-multiply $l_1^\top$, and isolate <br> $\sigma_1 = -l_1^\top (\nabla l_1 + b_1\lambda_{1+1} - A\rho_1) - a_1 t$:

In the same equation, now isolate instead <br> 
$\lambda_1 = a_1^{-1}(-\nabla l_1 - b_1\lambda_{1+1} + A\rho_1 - l_1\sigma_1) $

Using $[r_1 = c \tilde r]$, rearrange equation $[dc]$ to $\nabla c - c^{-1}r_1^\top \kappa = 0$. Isolating for $r_1^\top \kappa$, we get
<br>
$r_1^\top \kappa = c\nabla c$. Use for substitution in next step:

In equation $[r_1]$, left multiply with $r_1^\top$ and isolate $\omega_1$ as
<br> $\omega_1 = -r_1^\top(\nabla r_1 - A^\top \lambda_1 + a_1 \rho_1) - c\nabla c$

Finally, in the same equation, isolate<br> $\kappa = - \nabla r_1 + A^\top \lambda_1 - a_1\rho_1 - r_1\omega_1$

All unknowns are now determined.

Use $\kappa$ and $\{\rho_i\}_{i=1}^k, \{\lambda_i\}_{i=1}^k$ in equation $[dA]$ and $[d\tilde r]$ to get the gradient of $\mu$ w.r.t. $A$ and $\tilde r$.
Then transform the grad of $\mu$ w.r.t A to the parameter gradient (derived in a different document)