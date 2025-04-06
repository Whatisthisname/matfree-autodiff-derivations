from tree_defs import (
    Dim,
    MatrixLit,
    Shape,
    Var,
    Equation,
    Differential,
    Const,
    niceprint,
    UnitDim,
)
from tree_simplify import simplify, simplify_equation

X = 5  # Does not matter yet.

A_height = Dim(X)
A_width = Dim(X)

B_height = Dim(X)
B_width = Dim(X)

A = Var(MatrixLit(name="A", shape=Shape((A_height, A_width))))
B = Var(MatrixLit(name="B", shape=Shape((B_height, B_width))))

L = Var(MatrixLit(name="L", shape=Shape((A_height, B_height))))
R = Var(MatrixLit(name="R", shape=Shape((A_width, B_width))))

r_input = Var(MatrixLit(name="r~", shape=Shape((R.rows, UnitDim))))
c = Const(MatrixLit(name="c", shape=Shape((UnitDim, UnitDim))))

niceprint(simplify(Differential(r_input * c)))

eq1 = Equation(A * R, L * B).isolate_zero_RHS()
eq2 = Equation(A.T * L, R * B.T).isolate_zero_RHS()
eq3 = Equation(
    L.T * L, Const(MatrixLit(name="1", shape=Shape((B_height, B_height))))
).isolate_zero_RHS()
eq4 = Equation(
    R.T * R, Const(MatrixLit(name="1", shape=Shape((B_width, B_width))))
).isolate_zero_RHS()


niceprint(simplify_equation(eq1.differentiate()))
niceprint(simplify_equation(eq2.differentiate()))
niceprint(simplify_equation(eq3.differentiate()))
niceprint(simplify_equation(eq4.differentiate()))

"""
clear && mypy symbolic_helper.py --strict
"""
