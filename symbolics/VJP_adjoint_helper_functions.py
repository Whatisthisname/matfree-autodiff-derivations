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
    Var,
    Equation,
    Differential,
    Const,
    niceprint,
    UnitDim,
    MatrixExpr,
)
from tree_expand import ExpandSettings, expand, expand_equation


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

    elif isinstance(mask := innerProduct.right, Mask):
        return isolate_predicate_in_compact_inner_product_RHS(
            InnerProduct(Mask(innerProduct.left, mask.sparsity), mask.expr), predicate
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


def get_variables_from_expr(expr: MatrixExpr) -> typing.List[Var]:
    if isinstance(expr, Var):
        return [expr]
    if isinstance(expr, Differential):
        return get_variables_from_expr(expr.expr)
    if isinstance(expr, Const):
        return []
    elif isinstance(expr, _Product):
        return get_variables_from_expr(expr.left) + get_variables_from_expr(expr.right)
    elif isinstance(expr, _Transpose):
        return get_variables_from_expr(expr.expr)
    elif isinstance(expr, _Negate):
        return get_variables_from_expr(expr.expr)
    elif isinstance(expr, InnerProduct):
        return get_variables_from_expr(expr.left) + get_variables_from_expr(expr.right)
    elif isinstance(expr, _Sum):
        return sum([get_variables_from_expr(e) for e in expr.exprs], [])
    elif isinstance(expr, Mask):
        return get_variables_from_expr(expr.expr)
    else:
        raise NotImplementedError(
            f"get_variables_from_expr not implemented for {type(expr)}"
        )


def is_differential(x: MatrixExpr) -> bool:
    return isinstance(x, Differential)


"""
clear && mypy symbolic_helper.py --strict
"""
