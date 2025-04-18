import numpy as np
import dataclasses
import pytest


@dataclasses.dataclass
class BidiagOutput:
    rs: list[list[float]]
    ls: list[list[float]]
    L: int  # for convenience
    B: int  # for convenience
    R: int  # for convenience
    alphas: list[float]
    betas: list[float]
    c: float


def bidiagonalize_jvp(
    primals, tangents, iterations: int
) -> tuple[BidiagOutput, BidiagOutput]:
    A, start_vector = primals
    dA, d_start_vector = tangents
    print("A shape:", A.shape)
    height = A.shape[0]
    width_ = A.shape[1]

    zero_n_vec = np.zeros(width_)
    zero_m_vec = np.zeros(height)
    bs = [0]
    as_ = [0]

    c = 1 / np.linalg.norm(start_vector)
    rs = [zero_n_vec, start_vector * c]
    ls = [zero_m_vec]

    for n in range(1, iterations + 1):
        t = A @ rs[n] - bs[n - 1] * ls[n - 1]
        alpha_k = np.linalg.norm(t)
        # print("alpha_k", alpha_k)
        if np.allclose(alpha_k, 0, atol=1e-7) or np.isnan(alpha_k):
            break
        as_.append(alpha_k)
        ls.append(t / as_[n])

        w = A.T @ ls[n] - as_[n] * rs[n]
        beta_k = np.linalg.norm(w)
        if np.allclose(beta_k, 0, atol=1e-7) or np.isnan(beta_k):
            break

        bs.append(beta_k)
        r_kp1 = w / beta_k
        rs.append(r_kp1)

    # print("as")
    # print(np.array(as_[1:]).round(4))
    # print("bs")
    # print(np.array(bs[1:]).round(4))

    L = np.array(ls[1:]).T
    R = np.array(rs[1:]).T
    if len(as_[1:]) == len(bs[1:]):
        B = np.concatenate((np.diag(as_[1:]), np.zeros((len(as_[1:]), 1))), axis=1)
        for i in range(len(bs[1:])):
            B[i, i + 1] = bs[1:][i]
    else:
        B = np.diag(as_[1:]) + np.diag(bs[1:], k=1)
    # for i in range(len(bs)):
    # B = np.diag(as_[1:]) + np.diag(bs[1:], k=1)

    # print("L shape:", L.shape)
    # print("B shape:", B.shape)
    # print("R shape:", R.shape)
    # print("Bs shape:", B.shape)

    primal_output = BidiagOutput(rs=rs, ls=ls, L=L, B=B, R=R, alphas=as_, betas=bs, c=c)
    # return primal_output, primal_output

    d_as = [0] * len(as_)
    d_bs = [0] * len(bs)
    d_rs = [zero_n_vec.copy() * 0 for _ in range(len(rs))]
    d_rs[1] = (
        d_start_vector
        - (start_vector * (start_vector.T @ d_start_vector))
        / (start_vector @ start_vector)
    ) / np.linalg.norm(start_vector)
    d_ls = [zero_m_vec.copy() * 0 for _ in range(len(ls))]

    # d_rs[1] = d_start_vector, known
    # d_ls[0] = doesn't matter because bs_[0] = 0
    # d_bs[0] = 0

    # In each iteration, assume we already know d_rs[n], d_ls[n-1], d_bs[n-1]
    for n in range(1, len(as_[1:]) + 1):
        d_a_n = ls[n].T @ (dA @ rs[n] + A @ d_rs[n] - d_ls[n - 1] * bs[n - 1])
        d_as[n] = d_a_n
        d_l_n = (
            dA @ rs[n]
            + A @ d_rs[n]
            - bs[n - 1] * d_ls[n - 1]
            - d_as[n] * ls[n]
            - d_bs[n - 1] * ls[n - 1]
        ) / as_[n]
        # there will not be divide by zero here, else the loop would have stopped
        d_ls[n] = d_l_n.copy()

        # There might not be a beta here, so we have to check.
        if n > len(bs[1:]) or np.allclose(bs[n], 0, atol=1e-7) or np.isnan(bs[n]):
            break

        d_b_n = rs[n + 1].T @ (dA.T @ ls[n] + A.T @ d_ls[n] - d_rs[n] * as_[n])
        d_bs[n] = d_b_n
        d_r_np1 = (
            dA.T @ ls[n]
            + A.T @ d_ls[n]
            - d_rs[n] * as_[n]
            - rs[n] * d_as[n]
            - rs[n + 1] * d_bs[n]
        ) / bs[n]
        # there will not be divide by zero here because we checked above.
        d_rs[n + 1] = d_r_np1

    # print("d_as")
    # print(np.array(d_as[1:]).round(4))
    # print("d_bs")
    # print(np.array(d_bs[1:]).round(4))

    d_c = -(start_vector @ d_start_vector) / (
        start_vector @ start_vector * np.linalg.norm(start_vector)
    )

    dL = np.array(d_ls[1:]).T
    dR = np.array(d_rs[1:]).T

    if len(as_[1:]) == len(bs[1:]):
        dB = np.concatenate((np.diag(d_as[1:]), np.zeros((len(as_[1:]), 1))), axis=1)
        for i in range(len(d_bs[1:])):
            dB[i, i + 1] = d_bs[1:][i]
    else:
        dB = np.diag(d_as[1:]) + np.diag(d_bs[1:], k=1)

    tangent_output = BidiagOutput(
        rs=d_rs, ls=d_ls, L=dL, B=dB, R=dR, alphas=d_as, betas=d_bs, c=d_c
    )

    return primal_output, tangent_output


@pytest.mark.parametrize("seed", range(50))
def test_bidiag_jvp(seed):
    np.random.seed(seed)
    n = np.random.randint(2, 6)
    m = np.random.randint(2, 6)
    A = np.random.randn(n, m)
    d_A = np.random.randn(n, m)

    print("Rank:", np.linalg.matrix_rank(A))

    start_vector = 2 * np.eye(m, 1).flatten()
    start_vector = np.random.randn(m)
    d_start_vector = np.random.randn(m)

    result, tangents = bidiagonalize_jvp(
        primals=(A, start_vector),
        tangents=(d_A, d_start_vector),
        iterations=20,
    )

    h = 0.000001
    result_wiggled, _ = bidiagonalize_jvp(
        primals=(A + d_A * h, start_vector + d_start_vector * h),
        tangents=(d_A, d_start_vector),  # doesn't matter
        iterations=20,
    )

    print(f"-- Field: {'c'}".ljust(20), sep="", end="")
    assert np.allclose((result_wiggled.c - result.c) / h, tangents.c, atol=1e-2)
    print(" (OK)")

    for idx in range(1, len(result.rs)):
        for field in ["rs", "alphas", "ls", "betas"]:
            print(f"-- Field: {field}[{idx}]".ljust(20), sep="", end="")
            try:
                aprox = (
                    result_wiggled.__getattribute__(field)[idx]
                    - result.__getattribute__(field)[idx]
                ) / h

                exact = tangents.__getattribute__(field)[idx]
                assert np.allclose(
                    aprox, exact, atol=1e-2
                ), f"\nApprox: {aprox}, \nExact: {exact}"
                print(" (OK)")
            except IndexError:
                print(" (IndexError)")
                continue


@pytest.mark.parametrize("seed", range(50))
def test_bidiag_tall_matrix(seed):
    np.random.seed(seed)
    n = np.random.randint(low=2, high=8 + 1)
    m = np.random.randint(low=2, high=n + 1)
    A = np.random.randn(n, m)  # random tall-or-square matrix
    # if np.random.rand() < 0.4:
    # A[:, 0] = 0

    start_vector = 2 * np.eye(1, m).flatten()
    print("A.shape", A.shape)

    result, _ = bidiagonalize_jvp((A, start_vector), (A, start_vector), iterations=20)

    # print(A)
    print()
    # print(result.L @ result.B @ result.R.T)

    assert np.allclose(result.L @ result.B @ result.R.T, A, atol=1e-3), "A != LBR^T"

    # Inspect reduced iteration count properties:
    r = np.random.randint(1, min(m, n))

    L = result.L[:, :r]
    B = result.B[:r, :r]
    R = result.R[:, :r]

    assert np.allclose(L.T @ L, np.eye(r), atol=1e-5), "L^TL is not identity"
    assert np.allclose(R.T @ R, np.eye(r), atol=1e-5), "R^TR is not identity"
    assert np.allclose(A @ R, L @ B, atol=1e-5), "AR != LB"
    assert np.allclose(
        A.T @ L,
        R @ B.T + np.outer(result.betas[r] * result.rs[r + 1], np.eye(1, r, k=r - 1)),
        atol=1e-5,
    ), "A.T L != R B.T + extra"
    assert np.allclose(L.T @ A @ R, B, atol=1e-5), "L^TAR != B"


@pytest.mark.parametrize("seed", range(50))
def test_bidiag_wide_matrix(seed: int):
    np.random.seed(seed)
    m = np.random.randint(low=2, high=8 + 1)
    n = np.random.randint(low=2, high=m + 1)
    A = np.random.randn(n, m)
    if np.random.rand() < 0.4:
        A[:, 0] = 0
    # print(A.round(4))
    start_vector = np.random.randn(m)

    result, _ = bidiagonalize_jvp((A, start_vector), (A, start_vector), iterations=20)

    assert np.allclose(result.L @ result.B @ result.R.T, A, atol=1e-3), "A != LBR^T"

    r = np.random.randint(1, min(m, n))

    L = result.L[:, :r]
    B = result.B[:r, :r]
    R = result.R[:, :r]

    assert np.allclose(L.T @ L, np.eye(r), atol=1e-5), "L^TL is not identity"
    assert np.allclose(R.T @ R, np.eye(r), atol=1e-5), "R^TR is not identity"
    assert np.allclose(A @ R, L @ B, atol=1e-5), "AR != LB"
    assert np.allclose(
        A.T @ L,
        R @ B.T + np.outer(result.betas[r] * result.rs[r + 1], np.eye(1, r, k=r - 1)),
        atol=1e-5,
    ), "A.T L != R B.T + extra"
