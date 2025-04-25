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
    print_variables_in_equations,
)

# Define dimensions and variables
A_rows = Dim("n")
A_cols = Dim("m")
rank_dim = Dim("k")

A = Var(MatrixLit(name="A", shape=Shape((A_rows, A_cols))))
alpha_i = Var(MatrixLit(name="a[i]", shape=Shape((UnitDim, UnitDim))))
beta_prev = Var(MatrixLit(name="b[i-]", shape=Shape((UnitDim, UnitDim))))
beta_i = Var(MatrixLit(name="b[i]", shape=Shape((UnitDim, UnitDim))))

l_i = Var(MatrixLit(name="l[i]", shape=Shape((A_rows, UnitDim))))
l_prev = Var(MatrixLit(name="l[i-]", shape=Shape((A_rows, UnitDim))))
res = Var(MatrixLit(name="res", shape=Shape((A_cols, UnitDim))))
r_1 = Var(MatrixLit(name="r[1]", shape=Shape((A_cols, UnitDim))))
r_i = Var(MatrixLit(name="r[i]", shape=Shape((A_cols, UnitDim))))
r_next = Var(MatrixLit(name="r[i+1]", shape=Shape((A_cols, UnitDim))))

r_input = Var(MatrixLit(name="r~", shape=Shape((A_cols, UnitDim))))
c = Var(MatrixLit(name="c", shape=Shape((UnitDim, UnitDim))))
beta_last = Var(MatrixLit(name="beta", shape=Shape((UnitDim, UnitDim))))
one = Const(MatrixLit(name="1", shape=Shape((UnitDim, UnitDim))))

# Mark input and output variables
input_variables = [A, r_input]
output_variables = [res, c, alpha_i, beta_i, beta_prev, r_i, r_1, r_next, l_prev, l_i]

# Define constraints:
constraints = [
    Equation(l_i * alpha_i, A * r_i - (l_prev * beta_prev)),
    # Equation(r_next * beta_i, A.T * l_i - r_i * alpha_i),
    # Equation(l_i.T * l_i, one),
    # Equation(r_i.T * r_i, one),
    # Equation(r_1, c * r_input),
    # Equation(res, A.T * l_i - r_i * alpha_i),
]

print()
print("Shape of Constraints:")
[print(eq.lhs.shape()) for eq in constraints]

# Run the script!

differentiated_constraints = [
    expand_equation(expand_equation(c.isolate_zero_RHS().differentiate()))
    for c in constraints
]
# extra pre-differentiated constraint:
# differentiated_constraints.append(
#     expand_equation(
#         Equation(
#             Differential(c)
#             - r_input.T * Differential(r_input) * inverse_norm_of_r_cubed,
#             0 * c,
#         )
#     )
# )
# temp = differentiated_constraints[5]
# differentiated_constraints[5] = differentiated_constraints[6]
# differentiated_constraints[6] = temp

print()
print("Differentiated constraints:")
for i, dc in enumerate(differentiated_constraints):
    print(f"dc{i + 1} = " + dc.str_compact())

dcs = [dc.lhs for dc in differentiated_constraints]

# auto-define the lambda variables.
lambdas = [
    Var(MatrixLit(name=f"λ{i + 1}", shape=expr.shape())) for i, expr in enumerate(dcs)
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

print()
print("dµ:")
niceprint(d_mu_equation)

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
    try:
        variable = isolated_ip.right
        groups[variable.str_compact()].append(isolated_ip)
    except:
        pass
    finally:
        pass

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

final_gradient_expressions: list[Equation] = []
equations: list[Equation] = []
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
print("Shape of Adjoint system:")
[print(eq.lhs.shape()) for eq in equations]

print()
print_variables_in_equations(equations + final_gradient_expressions)
"""
clear && mypy symbolic_helper.py --strict
"""
