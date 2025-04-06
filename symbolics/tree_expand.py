import dataclasses
from typing_extensions import assert_never
from tree_exprs import (
    Dim,
    MatrixLit,
    Shape,
    Var,
    Equation,
    niceprint,
    MatrixExpr,
    Differential,
    _Product,
    Inverse,
    _Transpose,
    _Sum,
    Norm,
    _Negate,
    Const,
    InnerProduct,
)


@dataclasses.dataclass
class ExpandSettings:
    expand_inner_products: bool

    @staticmethod
    def default() -> "ExpandSettings":
        return ExpandSettings(
            expand_inner_products=True,
        )


def expand_equation(
    eq: Equation, settings: ExpandSettings = ExpandSettings.default()
) -> Equation:
    return Equation(expand(eq.lhs, settings), expand(eq.rhs, settings))


def expand(
    expr: MatrixExpr, settings: ExpandSettings = ExpandSettings.default()
) -> MatrixExpr:
    if isinstance(expr, Var):
        return expr
    elif isinstance(expr, Differential):
        return _expand_differential(expr, settings)
    elif isinstance(expr, _Product):
        return _expand_product(expr, settings)
    elif isinstance(expr, Inverse):
        return _expand_inverse(expr, settings)
    elif isinstance(expr, _Transpose):
        return _expand_transpose(expr, settings)
    elif isinstance(expr, _Sum):
        return _expand_sum(expr, settings)
    elif isinstance(expr, _Negate):
        return _expand_negate(expr, settings)
    elif isinstance(expr, Norm):
        raise AssertionError("fail")
    elif isinstance(expr, InnerProduct):
        return _expand_inner_product(expr, settings)
    elif isinstance(expr, Const):
        return expr
    else:
        assert_never(expr)


def _expand_differential(expr: Differential, settings: ExpandSettings) -> MatrixExpr:
    inner = expand(expr.expr, settings)
    if isinstance(inner, _Product):  # If the term is a product, apply the product rule
        return expand(
            _Product(
                Differential(inner.left),
                inner.right,
            )
            + _Product(
                inner.left,
                Differential(inner.right),
            ),
            settings,
        )
    elif isinstance(inner, _Sum):
        ok = expand(
            _Sum(
                [expand(Differential(expr)) for expr in inner.exprs],
            ),
            settings,
        )
        return ok
    elif isinstance(inner, _Negate):
        return _Negate(Differential(inner.expr))
    elif isinstance(inner, _Transpose):
        return _Transpose(Differential(inner.expr))
    elif isinstance(inner, Const):
        return Const(MatrixLit(name="0", shape=inner.shape()))
    else:
        return Differential(inner)


def _expand_inner_product(expr: InnerProduct, settings: ExpandSettings) -> MatrixExpr:
    left = expand(expr.left, settings)
    right = expand(expr.right, settings)
    if settings.expand_inner_products:
        # distribute all left sums
        if isinstance(left, _Sum):
            return expand(
                _Sum([InnerProduct(left_term, right) for left_term in left.exprs]),
                settings,
            )

        # distribute right sums
        if isinstance(right, _Sum):
            return expand(
                _Sum([InnerProduct(left, right_term) for right_term in right.exprs]),
                settings,
            )

    # check if left and right are transposed and then untranspose both
    if isinstance(left, _Transpose) and isinstance(right, _Transpose):
        return expand(InnerProduct(right.expr, left.expr), settings)

    return InnerProduct(left, right)


def _expand_transpose(expr: _Transpose, settings: ExpandSettings) -> MatrixExpr:
    inner = expand(expr.expr, settings)
    if isinstance(inner, _Product):
        left = expand(inner.left, settings)
        right = expand(inner.right, settings)
        return expand(_Product(_Transpose(right), _Transpose(left)), settings)
    elif isinstance(inner, _Transpose):
        return inner.expr
    elif isinstance(inner, _Negate):
        return _Negate(inner.expr.T)
    else:
        return _Transpose(inner)


def _expand_inverse(inv_expr: Inverse, settings: ExpandSettings) -> MatrixExpr:
    inner = inv_expr.expr
    if isinstance(inner, _Product):
        # Apply the inverse of a product rule
        left = expand(inner.left, settings)
        right = expand(inner.right, settings)
        return _Product(Inverse(right), Inverse(left))
    elif isinstance(inv_expr, Inverse):  # Inverse of inverse is identity
        return expand(inv_expr, settings)
    else:
        return Inverse(expand(inner, settings))


def _expand_sum(term: _Sum, settings: ExpandSettings) -> MatrixExpr:
    inners = term.exprs
    if len(inners) == 1:
        return expand(inners[0], settings)
    else:
        # expand them all
        simplified_terms = [expand(expr, settings) for expr in inners]

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
            return _Sum([expand(expr, settings) for expr in list(nonzeros)])

        else:
            return Const(MatrixLit(name="0", shape=term.shape()))


def _expand_negate(expr: _Negate, settings: ExpandSettings) -> MatrixExpr:
    inner = expand(expr.expr, settings)
    if isinstance(inner, _Sum):
        return expand(
            _Sum([_Negate(expand(expr, settings)) for expr in inner.exprs]), settings
        )
    elif isinstance(inner, _Negate):
        return inner
    elif isinstance(inner, Const):
        if inner.is_zero_element():
            return Const(MatrixLit(name="0", shape=inner.shape()))
        else:
            return _Negate(inner)
    else:
        return _Negate(inner)


def _expand_product(prod_expr: _Product, settings: ExpandSettings) -> MatrixExpr:
    left = expand(prod_expr.left, settings)
    right = expand(prod_expr.right, settings)

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
        return _Sum([_Product(expand(expr, settings), right) for expr in left.exprs])
    elif isinstance(right, _Sum):
        return _Sum([_Product(left, expand(expr, settings)) for expr in right.exprs])
    elif isinstance(left, Inverse):
        return _Product(Inverse(expand(right, settings)), expand(left.expr, settings))
    elif isinstance(right, Inverse):
        return _Product(Inverse(expand(left, settings)), expand(right.expr, settings))
    elif isinstance(right, _Negate) and isinstance(left, _Negate):
        return _Product(expand(left.expr, settings), expand(right.expr, settings))
    elif isinstance(right, _Negate):
        return -_Product(expand(left, settings), expand(right.expr, settings))
    elif isinstance(left, _Negate):
        return -_Product(expand(left.expr, settings), expand(right, settings))
    else:
        return _Product(left, right)
