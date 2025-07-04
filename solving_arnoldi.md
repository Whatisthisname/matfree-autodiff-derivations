Arnoldi Iteration constraints.

$A^{n\times n}$, $Q^{n\times k}$, $H^{k\times k}$

$$\begin{align*}
AQ &= QH + re_k^\top
\\
Qe_1 &= vc
\\
◣Q^\top Q &= I
\\
H_{<<} &= 0
\\
Q^\top r &= 0
\end{align*}$$

$Q = [q_1, \dots, q_k]$



$Aq_n = \sum_{i=1}^{n+1}q_ih_{i,n}$

$h_{n+1, n} = ||Aq_n -  \sum_{i=1}^{n}q_ih_{i,n}||$

$Aq_1 = q_1h_{1,1} + q_2h_{2,1}$

### Solution:

$q_1 = v / ||v||$

$Aq_1 = q_1h_{1,1} + q_2h_{2,1} \implies q_1^\top Aq_1 =: h_{1,1} \implies t := Aq_1 - q_1h_{1,1} = q_2h_{2,1} \implies  h_{2,1} := ||t|| \implies q_2 = t/h_{2,1}$

$Aq_2 = q_1h_{1,2} + q_2h_{2,2} + q_3h_{3,2} \implies [q_1, q_2]^\top Aq_2 =: [h_{1,2}, h_{2,2}]^\top \implies t := Aq_2 - [q_1, q_2][h_{1,2}, h_{2,2}]^\top = q_3h_{3,2} \implies  h_{3,2} := ||t|| \implies q_2 = t/h_{3,2}$

$Aq_n = [q_1,\dots, q_n, q_{n+1}][h_{1,n}, \dots, h_{n+1,n}]^\top \implies [q_1,\dots, q_n]^\top Aq_n = [h_{1,n}, \dots, h_{n,n}] \implies Aq_n - [h_{1,n}, \dots, h_{n,n}]$