# Constraints:

$L(1): l_1 a_1 = Ar_1$

For $n \in \{2...k\}$:
$L(n): l_n a_n = Ar_n - l_{n-1}b_{n-1}$

For $n \in \{1...(k-1)\}$:
$R(n): r_{n+1} b_n = A^\top l_n - r_n a_n$

$R(k): res = A^\top l_k - r_k a_k$

For $n \in \{1...k\}$:
$LI(n): l_n^\top l_n = 1$

For $n \in \{1...k\}$:
$RI(n): r_n^\top r_n = 1$

$r_1 = c\tilde r$

# Differentiated constraints:

$dL(1): l_1 da_1 + dl_1 a_1 = dAr_1 + Adr_1$

For $n \in \{2...k\}$:
$dL(n): dl_n a_n + l_n da_n = dAr_n + Adr_n - dl_{n-1}b_{n-1} - l_{n-1}db_{n-1}$

For $n \in \{1...(k-1)\}$:
$R(n): dr_{n+1} b_n + r_{n+1} db_n = dA^\top l_n + A^\top dl_n - dr_n a_n - r_n da_n$

$dR(k): dres = dA^\top l_k + A^\top dl_k - dr_k a_k - r_k da_k$

For $n \in \{1...k\}$:
$dLI(n): l_n^\top dl_n = 0$

For $n \in \{1...k\}$:
$dRI(n): r_n^\top dr_n = 0$

$dr_1 = dc\tilde r + cd\tilde r$

# Isolating zero

$dL(1): l_1 da_1 + dl_1 a_1 - dAr_1 - Adr_1 = 0 $

For $n \in \{2...k\}$:
$dL(n): dl_n a_n + l_n da_n - dAr_n - Adr_n + dl_{n-1}b_{n-1} + l_{n-1}db_{n-1} = 0$

For $n \in \{1...(k-1)\}$:
$R(n): dr_{n+1} b_n + r_{n+1} db_n - dA^\top l_n - A^\top dl_n + dr_n a_n + r_n da_n = 0$

$dR(k): dres - dA^\top l_k - A^\top dl_k + dr_k a_k + r_k da_k = 0$

For $n \in \{1...k\}$:
$dLI(n): l_n^\top dl_n = 0$

For $n \in \{1...k\}$:
$dRI(n): r_n^\top dr_n = 0$

$dr_1 - dc\tilde r - cd\tilde r = 0$

# Setting up lagrangian:
In the following, we define $\nabla \circ = \nabla_\circ \mu$ for scalar function $\mu$.

$$\begin{align*}

d\mu &= \langle \nabla c, dc\rangle  
\\ & + \sum_{n=1}^k \langle \nabla l_n, dl_n\rangle  
\\ & + \sum_{n=1}^k \langle \nabla r_n, dr_n\rangle  
\\ & + \langle \nabla res, dres\rangle  
\\ & + \sum_{n=1}^k\langle \nabla a_n, da_n\rangle  
\\ & + \sum_{n=1}^{k-1}\langle \nabla b_n, db_n\rangle  
\\ & \text{Adding Zero constraints and introducing $\lambda, \rho, \sigma, \omega, \kappa$}
\\ & + \langle l_1 da_1 + dl_1 a_1 - dAr_1 - Adr_1, \lambda_1\rangle 
\\ & + \sum_{n=2}^{k} \langle dl_n a_n + l_n da_n - dAr_n - Adr_n + dl_{n-1}b_{n-1} + l_{n-1}db_{n-1}, \lambda_n\rangle 
\\ & + \sum_{n=1}^{k-1}\langle dr_{n+1} b_n + r_{n+1} db_n - dA^\top l_n - A^\top dl_n + dr_n a_n + r_n da_n, \rho_n\rangle  
\\ & + \langle dres - dA^\top l_k - A^\top dl_k + dr_k a_k + r_k da_k, \rho_k\rangle 
\\ & + \sum_{n=1}^k \langle l_n^\top dl_n, \sigma_n\rangle  
\\ & + \sum_{n=1}^k \langle r_n^\top dr_n, \omega_n\rangle  
\\ & + \langle dr_1 - dc\tilde r - cd\tilde r, \kappa\rangle  

\end{align*}$$

# Expanding all inner products and moving negative sign inside:
$$\begin{align*}
d\mu &= \langle \nabla c, dc\rangle  
\\ & + \sum_{n=1}^k \langle \nabla l_n, dl_n\rangle  
\\ & + \sum_{n=1}^k \langle \nabla r_n, dr_n\rangle  
\\ & + \langle \nabla res, dres\rangle  
\\ & + \sum_{n=1}^k\langle \nabla a_n, da_n\rangle  
\\ & + \sum_{n=1}^{k-1}\langle \nabla b_n, db_n\rangle  
\\ & + \langle l_1 da_1, \lambda_1\rangle  + \langle dl_1 a_1, \lambda_1\rangle  + \langle -dAr_1, \lambda_1\rangle  + \langle -Adr_1, \lambda_1\rangle 
\\ & + \sum_{n=2}^{k} \left(\langle dl_n a_n, \lambda_n\rangle  + \langle l_n da_n, \lambda_n\rangle  + \langle -dAr_n, \lambda_n\rangle  + \langle -Adr_n, \lambda_n\rangle  + \langle dl_{n-1}b_{n-1}, \lambda_n\rangle  + \langle l_{n-1}db_{n-1}, \lambda_n\rangle \right)
\\ & + \sum_{n=1}^{k-1}\left(\langle dr_{n+1} b_n, \rho_n\rangle  + \langle r_{n+1} db_n, \rho_n\rangle  + \langle -dA^\top l_n, \rho_n\rangle  + \langle -A^\top dl_n, \rho_n\rangle  + \langle dr_n a_n, \rho_n\rangle  + \langle r_n da_n, \rho_n\rangle  \right)
\\ & + \langle dres, \rho_k\rangle  + \langle -dA^\top l_k, \rho_k\rangle  + \langle -A^\top dl_k, \rho_k\rangle  + \langle dr_k a_k, \rho_k\rangle  + \langle r_k da_k, \rho_k\rangle 
\\ & + \sum_{n=1}^k \langle l_n^\top dl_n, \sigma_n\rangle  
\\ & + \sum_{n=1}^k \langle r_n^\top dr_n, \omega_n\rangle  
\\ & + \langle dr_1, \kappa\rangle  + \langle -dc\tilde r, \kappa\rangle  + \langle -cd\tilde r, \kappa\rangle  
\end{align*}$$

# Rearranging to get differentials on one side:

$$\begin{align*}
d\mu &= \langle \nabla c, dc\rangle  
\\ & + \sum_{n=1}^k \langle \nabla l_n, dl_n\rangle  
\\ & + \sum_{n=1}^k \langle \nabla r_n, dr_n\rangle  
\\ & + \langle \nabla res, dres\rangle  
\\ & + \sum_{n=1}^k\langle \nabla a_n, da_n\rangle  
\\ & + \sum_{n=1}^{k-1}\langle \nabla b_n, db_n\rangle  
\\ & + \langle \lambda_1^\top l_1, da_1\rangle  + \langle a_1 \lambda_1, dl_1\rangle  + \langle -\lambda_1r_1^\top, dA\rangle  + \langle -A^\top\lambda_1, dr_1\rangle 
\\ & + \sum_{n=2}^{k} \left(\langle a_n \lambda_n, dl_n\rangle  + \langle l_n^\top \lambda_n, da_n\rangle  + \langle -\lambda_n r_n^\top, dA\rangle  + \langle - A^\top \lambda_n, dr_n\rangle  + \langle b_{n-1} \lambda_n, dl_{n-1}\rangle  + \langle \lambda_n^\top l_{n-1}, db_{n-1}\rangle \right)
\\ & + \sum_{n=1}^{k-1}\left(\langle b_n \rho_n, dr_{n+1}\rangle  + \langle r_{n+1}\top \rho_n, db_n\rangle  + \langle -l_n\rho_n^\top, dA\rangle  + \langle -A\rho_n, dl_n\rangle  + \langle a_n \rho_n, dr_n\rangle  + \langle r_n^\top \rho_n, da_n\rangle  \right)
\\ & + \langle \rho_k, dres\rangle  + \langle - l_k\rho_k^\top, dA\rangle  + \langle -A\rho_k, dl_k\rangle  + \langle a_k \rho_k, dr_k\rangle  + \langle r_k^\top\rho_k, da_k\rangle 
\\ & + \sum_{n=1}^k \langle l_n\sigma_n, dl_n\rangle  
\\ & + \sum_{n=1}^k \langle r_n\omega_n, dr_n\rangle  
\\ & + \langle \kappa, dr_1\rangle  + \langle -\tilde r^\top\kappa, dc\rangle  + \langle -c\kappa, d\tilde r\rangle  
\end{align*}$$

# Reindexing sums to get differentials without shifted indices:

$$\begin{align*}
d\mu &= \langle \nabla c, dc\rangle  
\\ & + \sum_{n=1}^k \langle \nabla l_n, dl_n\rangle  
\\ & + \sum_{n=1}^k \langle \nabla r_n, dr_n\rangle  
\\ & + \langle \nabla res, dres\rangle  
\\ & + \sum_{n=1}^k\langle \nabla a_n, da_n\rangle  
\\ & + \sum_{n=1}^{k-1}\langle \nabla b_n, db_n\rangle  
\\ & + \sum_{n=1}^{k} \left(\langle a_n \lambda_n, dl_n\rangle  + \langle l_n^\top \lambda_n, da_n\rangle  + \langle -\lambda_n r_n^\top, dA\rangle  + \langle - A^\top \lambda_n, dr_n\rangle  \right) \text{(add 
 "1" to index to absorb previous term)}
\\ & + \sum_{n=1}^{k-1} \left(\langle b_{n} \lambda_{n+1}, dl_{n}\rangle  + \langle \lambda_{n+1}^\top l_{n}, db_{n}\rangle \right) \text{(reindexed)}
\\ & + \sum_{n=2}^{k} \langle b_{n-1} \rho_{n-1}, dr_{n}\rangle  \text{(reindexed)}
\\ & + \sum_{n=1}^{k-1}\left(\langle r_{n+1}\top \rho_n, db_n\rangle  + \langle -l_n\rho_n^\top, dA\rangle  + \langle -A\rho_n, dl_n\rangle  + \langle a_n \rho_n, dr_n\rangle  + \langle r_n^\top \rho_n, da_n\rangle  \right)
\\ & + \langle \rho_k, dres\rangle  + \langle - l_k\rho_k^\top, dA\rangle  + \langle -A\rho_k, dl_k\rangle  + \langle a_k \rho_k, dr_k\rangle  + \langle r_k^\top\rho_k, da_k\rangle 
\\ & + \sum_{n=1}^k \langle l_n\sigma_n, dl_n\rangle  
\\ & + \sum_{n=1}^k \langle r_n\omega_n, dr_n\rangle  
\\ & + \langle \kappa, dr_1\rangle  + \langle -\tilde r^\top\kappa, dc\rangle  + \langle -c\kappa, d\tilde r\rangle  
\end{align*}$$

# Adding all terms with the same differential in the RHS, and setting = 0

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

Use $\kappa$ and ${\rho_i}_{i=1}^k, {\lambda_i}_{i=1}^k$ in equation $[dA]$ and $[d\tilde r]$ to get the gradient of $\mu$ w.r.t. $A$ and $\tilde r$.