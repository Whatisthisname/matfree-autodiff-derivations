from collections import defaultdict
from typing import Callable
import typing
from tree_exprs import (
    _Negate,
    _Product,
    _Sum,
    _Transpose,
    Dim,
    InnerProduct,
    Mask,
    MatrixLit,
    Shape,
    Sps,
    Var,
    Equation,
    Differential,
    Const,
    niceprint,
    UnitDim,
    MatrixExpr,
)
from tree_expand import ExpandSettings, expand, expand_equation
from VJP_adjoint_helper_functions import (
    isolate_predicate_in_compact_inner_product_RHS,
    is_differential,
    get_variables_from_expr,
)

# Define dimensions and variables
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

I_L = Const(MatrixLit(name="1", shape=Shape((B_rows, B_rows))))
I_R = Const(MatrixLit(name="1", shape=Shape((B_cols, B_cols))))

zero = Const(MatrixLit(name="0", shape=B.shape()))

inverse_norm_of_r_cubed = Var(
    MatrixLit(name="(||r~||^-3)", shape=Shape((UnitDim, UnitDim)))
)

# Mark input and output variables
input_variables = [A, r_input]
output_variables = [L, B, R, c]

# Define constraints:
constraints = [
    Equation(A * R, L * B),
    Equation(A.T * L, R * B.T),
    Equation(L.T * L, I_L),
    Equation(R.T * R, I_R),
    Equation(R * e_1, r_input * c),
    Equation(Mask(B, Sps.SSUpper) + Mask(B, Sps.SLower), 0 * B),
    # Equation(c - inverse_norm_of_r, 0 * c),
]

# Run the script!

differentiated_constraints = [
    expand_equation(expand_equation(c.isolate_zero_RHS().differentiate()))
    for c in constraints
]
# extra pre-differentiated constraint:
differentiated_constraints.append(
    expand_equation(
        Equation(
            Differential(c)
            - r_input.T * Differential(r_input) * inverse_norm_of_r_cubed,
            0 * c,
        )
    )
)
temp = differentiated_constraints[5]
differentiated_constraints[5] = differentiated_constraints[6]
differentiated_constraints[6] = temp

print()
print("Differentiated constraints:")
for i, dc in enumerate(differentiated_constraints):
    print(f"dc{i+1} = " + dc.str_compact())

dcs = [dc.lhs for dc in differentiated_constraints]

# auto-define the lambda variables.
lambdas = [
    Var(MatrixLit(name=f"λ{i+1}", shape=expr.shape())) for i, expr in enumerate(dcs)
]

grads = [
    Const(MatrixLit(name=f"∇{var.term.name}µ", shape=var.shape()))
    for var in output_variables
]

# mu is a scalar function of the output variables
d_mu = Differential(Var(MatrixLit(name="µ", shape=Shape((UnitDim, UnitDim)))))

d_mu_equation = Equation(
    d_mu,
    _Sum(
        [
            InnerProduct(grad, Differential(var))
            for grad, var in zip(grads, output_variables)
        ]
        + [InnerProduct(lambdas[i], dc) for i, dc in enumerate(dcs)]
    ),
)

# Take all the inner products and expand them
d_mu_equation = expand_equation(d_mu_equation)


# isolate all the inner products

groups = defaultdict(lambda: [])
all_inner_prods = typing.cast(_Sum, d_mu_equation.rhs).exprs
for ip in all_inner_prods:
    ip = typing.cast(InnerProduct, ip)
    isolated_ip = expand(
        isolate_predicate_in_compact_inner_product_RHS(
            typing.cast(InnerProduct, expand(ip)), is_differential
        )
    )
    variable = isolated_ip.right
    groups[variable.str_compact()].append(isolated_ip)

collected_inner_products = _Sum(
    [
        InnerProduct(_Sum([ip.left for ip in inner_prods]), inner_prods[0].right)
        for inner_prods in groups.values()
    ]
)

# Further simplify within the inner products without expanding them.
settings = ExpandSettings(expand_inner_products=False)
simplified = typing.cast(_Sum, expand(collected_inner_products, settings=settings))
# niceprint(simplified)

final_gradient_expressions = []
equations = []
for inner_prod in simplified.exprs:
    inner_prod = typing.cast(InnerProduct, inner_prod)
    if inner_prod.right.str_compact() in [
        Differential(input).str_compact() for input in input_variables
    ]:
        final_gradient_expressions.append(
            Equation(
                inner_prod.left,
                Var(
                    MatrixLit(
                        inner_prod.left.shape(), f"∇{inner_prod.right.expr.term.name}µ"
                    )
                ),
            )
        )
    else:
        equations.append(
            Equation(inner_prod.left, Var(MatrixLit(inner_prod.left.shape(), "0")))
        )

print()
print("Goal expressions:")
[niceprint(eq) for eq in final_gradient_expressions]
print()
print("Adjoint system:")
[niceprint(eq) for eq in equations]
print()

vars = list(set(sum([get_variables_from_expr(eq.lhs) for eq in equations], [])))
vars.sort(key=lambda x: x.str_compact())

for var in vars:
    if var.is_scalar():
        print(f"{var.str_compact().ljust(5)} scalar")
    elif var.is_vector():
        print(f"{var.str_compact().ljust(5)} vector, {var.rows.dim}")
    else:
        print(f"{var.str_compact().ljust(5)} matrix, {var.rows.dim} x {var.cols.dim}")

"""
clear && mypy symbolic_helper.py --strict
"""
