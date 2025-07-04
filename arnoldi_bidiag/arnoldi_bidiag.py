from typing import NamedTuple
from jax import Array
import matfree.decomp
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)


class DecompResult(NamedTuple):
    rs: Array
    ls: Array
    as_: Array
    bs_: Array


def arnoldi_bidiagonalization(
    num_matvecs: int,
    output_size: int,
    custom_vjp: bool = True,
    reortho: str = "full",
):
    num_matvecs *= 2

    def block_matvec(v0, matvec_fun, *params):
        upper, lower = jnp.split(v0, [output_size])
        upper_matvec, vecmat_fun = jax.vjp(lambda v: matvec_fun(v, *params), lower)
        (lower_vecmat,) = vecmat_fun(upper)
        return jnp.concat((upper_matvec, lower_vecmat))

    tridiag = matfree.decomp.tridiag_sym(
        num_matvecs,
        materialize=False,
        custom_vjp=custom_vjp,
        reortho=reortho,
    )

    def wrapped_bidiag(matvec, v0, *params):
        padded_v0 = jnp.pad(v0, (output_size, 0))

        def block_matved_curried(v, *params):
            return block_matvec(v, matvec, *params)

        output = tridiag(block_matved_curried, padded_v0, *params)

        ababab = output.J_small[1]
        as_ = ababab[0::2]
        bs_ = ababab[1::2]

        rlrlrl = output.Q_tall
        rs = rlrlrl[output_size:, 0::2]
        ls = rlrlrl[:output_size, 1::2]

        return DecompResult(rs=rs, ls=ls, as_=as_, bs_=bs_)

    return wrapped_bidiag


# func = arnoldi_bidiagonalization(5, shape=(2, 3))

# func(start_vec, matvec, A)
