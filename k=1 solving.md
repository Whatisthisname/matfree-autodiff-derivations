$l_1 a_1 = Ar_1$

$res = A^\top l_1 - r_1 a_1$

$l_1^\top l_1 = 1$

$r_1^\top r_1 = 1$

$r_1 = c\tilde r$

# diff and set = 0

$l_1 da_1 + dl_1 a_1 - dAr_1 - Adr_1 = 0$

$dres - dA^\top l_1 - A^\top dl_1 + dr_1 a_1 + r_1 da_1 = 0$

$l_1^\top dl_1  = 0$

$r_1^\top dr_1  = 0$

$dr_1 - dc\tilde r - cd\tilde r = 0$

# build lagrangian or something

$$\begin{align*}

d\mu &= \langle \nabla c, dc\rangle  
\\ & + \langle \nabla l_1, dl_1\rangle  
\\ & + \langle \nabla r_1, dr_1\rangle  
\\ & + \langle \nabla res, dres\rangle  
\\ & + \langle \nabla a_1, da_1\rangle

\\ & + \langle l_1 da_1 + dl_1 a_1 - dAr_1 - Adr_1, \lambda_1\rangle 
  
\\ & + \langle dres - dA^\top l_1 - A^\top dl_1 + dr_1 a_1 + r_1 da_1, \rho_1\rangle 

\\ & + \langle l_1^\top dl_1, \sigma_1\rangle  

\\ & + \langle r_1^\top dr_1, \omega_1\rangle  

\\ & + \langle dr_1 - dc\tilde r - cd\tilde r, \kappa\rangle  
\end{align*}$$

# flatten all inner products and regroup by same $d\circ$

$dc: \nabla c - \tilde r ^\top \kappa = 0$

$dl_1: \nabla l_1 + a_1\lambda_1 - A\rho_1 + \sigma_1l_1 = 0$

$dr_1: \nabla r_1 - A^\top\lambda_1 + a_1\rho_1 + \omega_1r_1 + \kappa = 0$

$dres: \nabla res + \rho_1 = 0 $

$da_1: \nabla a_1 + l_1^\top \lambda_1 + r_1^\top\rho_1 = 0$

$d\tilde r: \nabla \tilde r = -c\kappa$

$dA: \nabla A = -\lambda_1r_1^\top - l_1\rho_1^\top$

# solving:

$\rho_1 = -\nabla res$

$t:=l_1^\top\lambda_1 = - \nabla a_1 - r_1^\top \rho_1$

In $dl_1$ left multiply with $l_1^\top:$

$$\begin{align*}
&l_1^\top\nabla l_1 + a_1l_1^\top\lambda_1 - l_1^\top A\rho_1 + \sigma_1l_1^\top l_1
\\ &= l_1^\top\nabla l_1 + a_1t - l_1^\top A\rho_1 + \sigma_1
\\ &= 0
\\ &\implies \sigma_1 = -l_1^\top\nabla l_1 - a_1t + l_1^\top A\rho_1
\end{align*}$$

Use knowledge of $\sigma_1$ in $dl_1$ to get $\lambda_1$:

$$\begin{align*}
&\nabla l_1 + a_1\lambda_1 - A\rho_1 + \sigma_1l_1 = 0
\\&\iff \lambda_1 = a_1^{-1}(- \nabla l_1 + A\rho_1 - \sigma_1l_1)
\end{align*}$$

From $dc$, using $[c \tilde r = r_1]$ infer $r_1^\top\kappa = c \nabla c$.

Now look at $dr_1$ and left multiply $r_1^\top$:
$$\begin{align*}
&r_1^\top \nabla r_1 - r_1^\top A^\top\lambda_1 + a_1r_1^\top \rho_1 + \omega_1r_1^\top r_1 + r_1^\top\kappa
\\&=r_1^\top \nabla r_1 - r_1^\top A^\top\lambda_1 + a_1r_1^\top \rho_1 + \omega_1 + c \nabla c
\\&\implies \omega_1 = -r_1^\top \nabla r_1 + r_1^\top A^\top\lambda_1 - a_1r_1^\top \rho_1 - c \nabla c
\end{align*}$$

In $dr_1$, use knowlege of $\omega_1$ to get $\kappa$.
$$\begin{align*}
&\nabla r_1 - A^\top\lambda_1 + a_1\rho_1 + \omega_1r_1 + \kappa = 0
\\&\iff \kappa = -\nabla r_1 + A^\top\lambda_1 - a_1\rho_1 - \omega_1r_1
\end{align*}$$


solve $\kappa$: