import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(1)
key1, key2 = jax.random.split(key, num=2)
A = jax.random.normal(key1, (3, 3))
x = jax.random.normal(key2, (3,))


def fun(v):
    print(" Calling the function , not the JVP ... ")
    return jnp.dot(v, A @ v)


dfx_ad = jax.jacfwd(fun)(x)  # Calls fun , not JVP


def fun_jvp(primals, tangents):
    print(" Calling the JVP , not the function ... ")
    (v,), (dv,) = primals, tangents
    w = jnp.dot(v, A @ v)
    dw1 = jnp.dot(dv, A @ v)
    dw2 = jnp.dot(v, A @ dv)
    return w, dw1 + dw2


fun = jax.custom_jvp(fun)
fun.defjvp(fun_jvp)
dfx_custom = jax.jacfwd(fun)(x)  # Calls the JVP , not the fun
assert jnp.allclose(dfx_custom, dfx_ad)
