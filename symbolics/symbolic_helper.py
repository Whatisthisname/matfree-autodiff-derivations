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
    Var,
    Equation,
    Differential,
    Const,
    niceprint,
    UnitDim,
    MatrixExpr,
)
from tree_expand import ExpandSettings, expand, expand_equation

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


def satisfies_predicate(
    expr: MatrixExpr, predicate: Callable[[MatrixExpr], bool]
) -> bool:
    if predicate(expr):
        return True
    if isinstance(expr, Var):
        return False
    if isinstance(expr, Differential):
        return satisfies_predicate(expr.expr, predicate)
    if isinstance(expr, Const):
        return False
    elif isinstance(expr, _Product):
        return satisfies_predicate(expr.left, predicate) or satisfies_predicate(
            expr.right, predicate
        )
    elif isinstance(expr, _Transpose):
        return satisfies_predicate(expr.expr, predicate)
    elif isinstance(expr, _Negate):
        return satisfies_predicate(expr.expr, predicate)
    elif isinstance(expr, InnerProduct):
        return satisfies_predicate(expr.left, predicate) or satisfies_predicate(
            expr.right, predicate
        )
    elif isinstance(expr, _Sum):
        return any(satisfies_predicate(expr, predicate) for expr in expr.exprs)
    else:
        raise NotImplementedError(
            f"satisfies_predicate not implemented for {type(expr)}"
        )


# assumes the target is in the RHS and try to move everything to the LHS
def isolate_predicate_in_compact_inner_product_RHS(
    innerProduct: InnerProduct, predicate: Callable[[MatrixExpr], bool]
) -> InnerProduct:
    if predicate(innerProduct.right):
        return innerProduct
    elif isinstance(transpose := innerProduct.right, _Transpose):
        return isolate_predicate_in_compact_inner_product_RHS(
            InnerProduct(innerProduct.left.T, transpose.expr), predicate
        )

    elif isinstance(negate := innerProduct.right, _Negate):
        return isolate_predicate_in_compact_inner_product_RHS(
            InnerProduct(-innerProduct.left, negate.expr), predicate
        )

    elif isinstance(differential := innerProduct.right, Differential):
        if satisfies_predicate(differential, predicate):
            return InnerProduct(innerProduct.left, differential)
        else:
            raise NotImplementedError("maybe this should be fixed")

    elif isinstance(product := innerProduct.right, _Product):
        if satisfies_predicate(product.left, predicate):
            return isolate_predicate_in_compact_inner_product_RHS(
                InnerProduct(
                    innerProduct.left * product.right.T
                    if not product.left.is_scalar()  # todo: also do this for when inputs to inner prod are (row vector, row_vector * scalar,
                    else product.right.T * innerProduct.left,
                    product.left,
                ),
                predicate,
            )
        elif satisfies_predicate(product.right, predicate):
            return isolate_predicate_in_compact_inner_product_RHS(
                InnerProduct(
                    product.left.T * innerProduct.left,
                    product.right,
                ),
                predicate,
            )
        else:
            raise ValueError(" Not found predicate in product! ")

    elif isinstance(sum := innerProduct.right, _Sum):
        raise ValueError(
            "Call simplify before isolating! Isolating a sum is not supported"
        )
    else:
        niceprint(innerProduct)
        raise ValueError(
            f"Cannot isolate with predicate in {innerProduct.right} of type {type(innerProduct.right)}"
        )


# niceprint(final_eq)

final_eq_simple = expand_equation(final_eq)

# niceprint(final_eq_simple)


def is_differential(x: MatrixExpr) -> bool:
    return isinstance(x, Differential)


# isolate all the inner products
# TODO the transposes are not being isolated correctly

new_eq = Equation(
    final_eq_simple.lhs,
    _Sum(
        [
            isolate_predicate_in_compact_inner_product_RHS(
                typing.cast(InnerProduct, expand(expr)), is_differential
            )
            if isinstance(expr, InnerProduct)
            else expr
            for expr in final_eq_simple.rhs.exprs
        ]
    ),
)

print()
niceprint((new_eq))

groups = defaultdict(lambda: [])
for expr in new_eq.rhs.exprs:
    expr = typing.cast(InnerProduct, expr)
    groups[expr.right.str_compact()].append(expr)

grouped_expr = _Sum(
    [
        InnerProduct(_Sum([ip.left for ip in inner_prods]), inner_prods[0].right)
        for inner_prods in groups.values()
    ]
)

settings = ExpandSettings(expand_inner_products=False)

print()
simplified = typing.cast(_Sum, expand(grouped_expr, settings=settings))
niceprint(simplified)

final_gradient_expressions = []
equations = []
for inner_prod in simplified.exprs:
    inner_prod = typing.cast(InnerProduct, inner_prod)
    if inner_prod.right.str_compact() in ["dA", "dr~"]:
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
