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
    Mask,
)
from tree_expand import ExpandSettings, expand, expand_equation
from VJP_adjoint_helper_functions import (
    isolate_predicate_in_compact_inner_product_RHS,
    is_differential,
)

# Define dimensions and variables
Square_dim = Dim("size")

A = Var(MatrixLit(name="A", shape=Shape((Square_dim, Square_dim))))
Q = Var(MatrixLit(name="Q", shape=Shape((Square_dim, Square_dim))))
R = Var(MatrixLit(name="R", shape=Shape((Square_dim, Square_dim))))
I_R = Const(MatrixLit(name="1", shape=Shape((Square_dim, Square_dim))))
zero = Const(MatrixLit(name="0", shape=Shape((Square_dim, Square_dim))))

# Mark input and output variables
input_variables = [A]
output_variables = [Q, R]

# Define constraints:
constraints = [
    Equation(Mask(Q.T * Q - I_R, Sps.Upper) + Mask(R, Sps.SLower), zero),
    Equation(A, Q * R).flip(),
]

# Run the script!

differentiated_constraints = [
    expand_equation(expand_equation(c.isolate_zero_RHS().differentiate()))
    for c in constraints
]
dcs = [dc.lhs for dc in differentiated_constraints]

for dc in differentiated_constraints:
    niceprint(dc)


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
    isolated_ip = isolate_predicate_in_compact_inner_product_RHS(
        typing.cast(InnerProduct, expand(ip)), is_differential
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

niceprint(simplified)


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


"""
clear && mypy symbolic_helper.py --strict
"""
