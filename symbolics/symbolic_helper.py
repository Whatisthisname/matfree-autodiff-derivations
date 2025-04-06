from tree_defs import (
    _Sum,
    Dim,
    InnerProduct,
    MatrixLit,
    Shape,
    Var,
    Equation,
    Differential,
    Const,
    niceprint,
    UnitDim,
    MatrixExpr,
)
from tree_simplify import simplify, simplify_equation

A_rows = Dim("A rows")
A_cols = Dim("A cols")

B_rows = Dim("B rows")
B_cols = Dim("B cols")

A = Var(MatrixLit(name="A", shape=Shape((A_rows, A_cols))))
B = Var(MatrixLit(name="B", shape=Shape((B_rows, B_cols))))

L = Var(MatrixLit(name="L", shape=Shape((A_rows, B_rows))))
R = Var(MatrixLit(name="R", shape=Shape((A_cols, B_cols))))

e_1 = Const(MatrixLit(name="e1", shape=Shape((R.cols, UnitDim))))
r_input = Var(MatrixLit(name="r~", shape=Shape((R.rows, UnitDim))))
c = Var(MatrixLit(name="c", shape=Shape((UnitDim, UnitDim))))


dc1 = Equation(A * R, L * B).isolate_zero_RHS().differentiate()
dc2 = Equation(A.T * L, R * B.T).isolate_zero_RHS().differentiate()
dc3 = (
    Equation(L.T * L, Const(MatrixLit(name="1", shape=Shape((B_rows, B_rows)))))
    .isolate_zero_RHS()
    .differentiate()
)
dc4 = (
    Equation(R.T * R, Const(MatrixLit(name="1", shape=Shape((B_cols, B_cols)))))
    .isolate_zero_RHS()
    .differentiate()
)

dc5 = Equation(R * e_1, c * r_input).isolate_zero_RHS().differentiate()

dcs = [dc.lhs for dc in [dc1, dc2, dc3, dc4, dc5]]
lambdas = [
    Var(MatrixLit(name=f"λ{i}", shape=expr.shape())) for i, expr in enumerate(dcs)
]

vars: list[Var] = [R, L, B, c]
grads = [Const(MatrixLit(name=f"∇{var.term.name}µ", shape=var.shape())) for var in vars]
d_mu = Differential(Var(MatrixLit(name="µ", shape=Shape((UnitDim, UnitDim)))))

final_eq = Equation(
    d_mu,
    _Sum(
        [InnerProduct(grad, Differential(var)) for grad, var in zip(grads, vars)]
        + [InnerProduct(lambdas[i], dc) for i, dc in enumerate(dcs)]
    ),
)

niceprint(simplify_equation(final_eq))

"""
clear && mypy symbolic_helper.py --strict
"""
