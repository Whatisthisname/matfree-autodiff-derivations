_nabla.res = [0, nabla.res]

eta = dH @ _e_K - Q.T @ _nabla.res
    = dH_k - Q.T @ _nabla.res

_Q.T @ _nabla.res = Weave(R.T @ nabla.res, 0)

_nabla.H_k = _e_{k-1} * a_k
eta = _e_{k-1} * a_k - Weave(R.T @ nabla.res, 0)

-------------------
lambda_k = _nabla.res + Q @ eta
         = _nabla.res + Q @ (dH_k - Q.T @ _nabla.res)
         = _nabla.res + Q @ dH_k - Q @ Q.T @ _nabla.res
         = _nabla.res + Q @ _e_{k-1} * a_k - Q @ Q.T @ _nabla.res
         = [0, nabla.res] + [0, r_k * a_k] - Q @ Q.T @ _nabla.res

Q @ Q.T = [[L @ L.T, 0], [0, R @ R.T]]
Q @ Q.T @ _nabla.res = [0, R @ R.T @ nabla.res]

         = [0, nabla.res] + [0, r_k * a_k] - [0, R @ R.T @ nabla.res]
         = [0, nabla.res + r_k * a_k - R @ R.T @ nabla.res]

-------------------
