from typing_extensions import assert_never
from tree_defs import (
    Dim,
    MatrixLit,
    Shape,
    Var,
    Equation,
    niceprint,
    MatrixExpr,
    Differential,
    _Product,
    Invert,
    _Transpose,
    _Sum,
    Norm,
    _Negate,
    Const,
)


def simplify_equation(eq: Equation) -> Equation:
    return Equation(simplify(eq.lhs), simplify(eq.rhs))


def simplify(expr: MatrixExpr) -> MatrixExpr:
    if isinstance(expr, Var):
        return expr
    elif isinstance(expr, Differential):
        return _simplify_differential(expr)
    elif isinstance(expr, _Product):
        return _simplify_product(expr)
    elif isinstance(expr, Invert):
        return _simplify_invert(expr)
    elif isinstance(expr, _Transpose):
        return _simplify_transpose(expr)
    elif isinstance(expr, _Sum):
        return _simplify_sum(expr)
    elif isinstance(expr, _Negate):
        return _simplify_negate(expr)
    elif isinstance(expr, Norm):
        raise AssertionError("fail")
    elif isinstance(expr, Const):
        return expr
    else:
        assert_never(expr)


def _simplify_differential(expr: Differential) -> MatrixExpr:
    inner = simplify(expr.term)
    if isinstance(inner, _Product):  # If the term is a product, apply the product rule
        return simplify(
            _Product(
                Differential(inner.left),
                inner.right,
            )
            + _Product(
                inner.left,
                Differential(inner.right),
            ),
        )
    elif isinstance(inner, _Sum):
        ok = simplify(
            _Sum(
                [simplify(Differential(expr)) for expr in inner.exprs],
            )
        )
        return ok
    elif isinstance(inner, _Negate):
        return _Negate(Differential(inner.expr))
    elif isinstance(inner, Const):
        return Const(MatrixLit(name="0", shape=inner.shape()))
    else:
        return Differential(inner)


def _simplify_transpose(expr: _Transpose) -> MatrixExpr:
    inner = simplify(expr.expr)
    if isinstance(inner, _Product):
        left = simplify(inner.left)
        right = simplify(inner.right)
        return simplify(_Product(_Transpose(right), _Transpose(left)))
    elif isinstance(inner, _Transpose):
        return inner
    else:
        return _Transpose(simplify(inner))


def _simplify_invert(inv_expr: Invert) -> MatrixExpr:
    inner = inv_expr.expr
    if isinstance(inner, _Product):
        # Apply the inverse of a product rule
        left = simplify(inner.left)
        right = simplify(inner.right)
        return _Product(Invert(right), Invert(left))
    elif isinstance(inv_expr, Invert):  # Inverse of inverse is identity
        return simplify(inv_expr)
    else:
        return Invert(simplify(inner))


def _simplify_sum(term: _Sum) -> MatrixExpr:
    inners = term.exprs
    if len(inners) == 1:
        return simplify(inners[0])
    else:
        # simplify them all
        simplified_terms = [simplify(expr) for expr in inners]

        # Take all the terms that are themselves sums:
        flattened_sums = sum(
            [
                list(expr.exprs) if isinstance(expr, _Sum) else [expr]
                for expr in simplified_terms
            ],
            [],
        )

        nonzeros = [expr for expr in flattened_sums if not expr.is_zero_element()]

        if len(nonzeros) != 0:
            return _Sum([simplify(expr) for expr in list(nonzeros)])

        else:
            return Const(MatrixLit(name="0", shape=term.shape()))


def _simplify_negate(expr: _Negate) -> MatrixExpr:
    inner = simplify(expr.expr)
    if isinstance(inner, _Sum):
        return simplify(_Sum([_Negate(simplify(expr)) for expr in inner.exprs]))
    elif isinstance(inner, _Negate):
        return inner
    elif isinstance(inner, Const):
        if inner.is_zero_element():
            return Const(MatrixLit(name="0", shape=inner.shape()))
        else:
            return _Negate(inner)
    else:
        return _Negate(inner)


def _simplify_product(prod_expr: _Product) -> MatrixExpr:
    left = simplify(prod_expr.left)
    right = simplify(prod_expr.right)

    if left.is_zero_element() or right.is_zero_element():
        return Const(MatrixLit(name="0", shape=left.shape()))

    if left.is_identity_element():
        return right

    if right.is_identity_element():
        return left

    # Swap left and right to move scalar to the left
    if right.is_scalar() and not left.is_scalar():
        temp = left
        left = right
        right = temp

    if isinstance(left, _Sum):
        return _Sum([_Product(simplify(expr), right) for expr in left.exprs])
    elif isinstance(right, _Sum):
        return _Sum([_Product(left, simplify(expr)) for expr in right.exprs])
    elif isinstance(left, Invert):
        return _Product(Invert(simplify(right)), simplify(left.expr))
    elif isinstance(right, Invert):
        return _Product(Invert(simplify(left)), simplify(right.expr))
    else:
        return _Product(left, right)
