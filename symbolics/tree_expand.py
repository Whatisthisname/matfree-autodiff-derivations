import dataclasses
from typing_extensions import assert_never
from tree_exprs import (
    Dim,
    Mask,
    MatrixLit,
    Shape,
    Sps,
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
    print_tree: bool = False

    @staticmethod
    def default() -> "ExpandSettings":
        return ExpandSettings(
            expand_inner_products=True,
        )


def expand_equation(
    eq: Equation, settings: ExpandSettings = ExpandSettings.default()
) -> Equation:
    return Equation(expand(eq.lhs, settings), expand(eq.rhs, settings))


indentation_level = 0


def expand(
    expr: MatrixExpr, settings: ExpandSettings = ExpandSettings.default()
) -> MatrixExpr:
    global indentation_level
    if settings.print_tree:
        print(
            f"{indentation_level:2}"
            + " | " * (indentation_level - 1)
            + (" L " if indentation_level > 0 else ""),
            end=" ",
        )
        (niceprint(expr),)
    indentation_level += 1
    try:
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
            _expand_inner_product(expr, settings)
        elif isinstance(expr, InnerProduct):
            return _expand_inner_product(expr, settings)
        elif isinstance(expr, Const):
            return expr
        elif isinstance(expr, Mask):
            return _expand_mask(expr, settings)
        else:
            assert_never(expr)
    finally:
        indentation_level -= 1


def _expand_norm(expr: Norm, settings: ExpandSettings) -> MatrixExpr:
    raise NotImplementedError(
        "Norm expansion is not implemented yet. Please use expand_inner_product instead."
    )
    inner = expand(expr.expr, settings)
    if isinstance(prod := inner, _Transpose):
        return expand(InnerProduct(prod.expr, prod.expr), settings)


mask_product_combination = {
    Sps.Lower: {
        Sps.Lower: Sps.Lower,
        Sps.SLower: Sps.SLower,
        Sps.Diag: Sps.Lower,
    },
    Sps.SLower: {
        Sps.Lower: Sps.SLower,
        Sps.SLower: Sps.SLower,
        Sps.Diag: Sps.SLower,
    },
    Sps.Upper: {
        Sps.Upper: Sps.Upper,
        Sps.SUpper: Sps.SUpper,
        Sps.Diag: Sps.Upper,
    },
    Sps.Diag: {
        Sps.Diag: Sps.Diag,
        Sps.Lower: Sps.Lower,
        Sps.SLower: Sps.SLower,
        Sps.Upper: Sps.Upper,
        Sps.SUpper: Sps.SUpper,
    },
}
mask_double_application_list = [
    ({Sps.Dense, Sps.Dense}, Sps.Dense),
    ({Sps.Dense, Sps.Diag}, Sps.Diag),
    ({Sps.Dense, Sps.Upper}, Sps.Upper),
    ({Sps.Dense, Sps.Lower}, Sps.Lower),
    ({Sps.Dense, Sps.SUpper}, Sps.SUpper),
    ({Sps.Dense, Sps.SLower}, Sps.SLower),
    ({Sps.Dense, Sps.SSUpper}, Sps.SSUpper),
    ({Sps.Dense, Sps.Nothing}, Sps.Nothing),
    #
    ({Sps.Diag, Sps.Diag}, Sps.Diag),
    ({Sps.Diag, Sps.Upper}, Sps.Diag),
    ({Sps.Diag, Sps.Lower}, Sps.Diag),
    ({Sps.Diag, Sps.SUpper}, Sps.Nothing),
    ({Sps.Diag, Sps.SLower}, Sps.Nothing),
    ({Sps.Diag, Sps.SSUpper}, Sps.Nothing),
    ({Sps.Diag, Sps.Nothing}, Sps.Nothing),
    ({Sps.Diag, Sps.SSLower}, Sps.Nothing),
    #
    ({Sps.Upper, Sps.Upper}, Sps.Upper),
    ({Sps.Upper, Sps.Lower}, Sps.Diag),
    ({Sps.Upper, Sps.SUpper}, Sps.SUpper),
    ({Sps.Upper, Sps.SLower}, Sps.Nothing),
    ({Sps.Upper, Sps.SSUpper}, Sps.SSUpper),
    ({Sps.Upper, Sps.Nothing}, Sps.Nothing),
    ({Sps.Upper, Sps.SSLower}, Sps.Nothing),
    #
    ({Sps.Lower, Sps.Lower}, Sps.Lower),
    ({Sps.Lower, Sps.SUpper}, Sps.Nothing),
    ({Sps.Lower, Sps.SLower}, Sps.SLower),
    ({Sps.Lower, Sps.SSUpper}, Sps.Nothing),
    ({Sps.Lower, Sps.Nothing}, Sps.Nothing),
    ({Sps.Lower, Sps.SSLower}, Sps.SSLower),
    #
    ({Sps.SUpper, Sps.SUpper}, Sps.SUpper),
    ({Sps.SUpper, Sps.SLower}, Sps.Nothing),
    ({Sps.SUpper, Sps.SSUpper}, Sps.SSUpper),
    ({Sps.SUpper, Sps.Nothing}, Sps.Nothing),
    ({Sps.SUpper, Sps.SSLower}, Sps.Nothing),
    #
    ({Sps.SLower, Sps.SLower}, Sps.SLower),
    ({Sps.SLower, Sps.SSUpper}, Sps.Nothing),
    ({Sps.SLower, Sps.Nothing}, Sps.Nothing),
    ({Sps.SLower, Sps.SSLower}, Sps.SSLower),
    #
    ({Sps.SSUpper, Sps.SSUpper}, Sps.SSUpper),
    ({Sps.SSUpper, Sps.Nothing}, Sps.Nothing),
    ({Sps.SSUpper, Sps.SSLower}, Sps.Nothing),
    #
    ({Sps.Nothing, Sps.Nothing}, Sps.Nothing),
    ({Sps.Nothing, Sps.SSLower}, Sps.SSLower),
    #
    ({Sps.SSLower, Sps.SSLower}, Sps.SSLower),
]
mask_double_application = {frozenset(k): v for k, v in mask_double_application_list}

transposed_mask = {
    Sps.Lower: Sps.Upper,
    Sps.Upper: Sps.Lower,
    #
    Sps.SLower: Sps.SUpper,
    Sps.SUpper: Sps.SLower,
    #
    Sps.Diag: Sps.Diag,
    #
    Sps.SSUpper: Sps.SSLower,
    Sps.SSLower: Sps.SSUpper,
    #
    Sps.Nothing: Sps.Nothing,
    #
    Sps.Dense: Sps.Dense,
}


def _expand_mask(expr: Mask, settings: ExpandSettings) -> MatrixExpr:
    inner = expand(expr.expr, settings)
    if isinstance(inner, Mask):
        my_mask = expr.sparsity
        inner_mask = inner.sparsity
        replacement_mask = mask_double_application[frozenset({my_mask, inner_mask})]
        return expand(Mask(inner.expr, replacement_mask), settings)
    if isinstance(inner, _Negate):
        return _Negate(Mask(inner.expr, expr.sparsity))
    if isinstance(inner, _Transpose):
        return _Transpose(Mask(inner.expr, transposed_mask[expr.sparsity]))
    if isinstance(inner, _Sum):
        return expand(_Sum([Mask(summand, expr.sparsity) for summand in inner.exprs]))
    if isinstance(inner, Const):
        if inner.is_zero_element():
            return inner
        elif inner.is_identity_element() and expr.sparsity in [
            Sps.SLower,
            Sps.SUpper,
            Sps.SSLower,
            Sps.SSUpper,
            Sps.Nothing,
        ]:
            return Const(MatrixLit(name="0", shape=expr.shape()))
        else:
            return Mask(inner, expr.sparsity)
    else:
        return Mask(inner, expr.sparsity)


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
    elif isinstance(inner, Mask):
        return Mask(Differential(inner.expr), inner.sparsity)
    elif isinstance(inner, Const):
        return Const(MatrixLit(name="0", shape=inner.shape()))
    if isinstance(
        inner, InnerProduct
    ):  # If the term is an innerproduct, apply the product rule
        trace_inner = expand(Differential(inner.left.T * inner.right))
        # now take the trace of this:
        return expand(
            Mask(
                trace_inner,
                Sps.Diag,
            ).T
            * Mask(
                trace_inner,
                Sps.Diag,
            )
        )

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
    if inner.is_scalar():
        return inner
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
        return expand(inv_expr.expr, settings)
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
            return inner
        else:
            return _Negate(inner)
    else:
        return _Negate(inner)


def _expand_product(prod_expr: _Product, settings: ExpandSettings) -> MatrixExpr:
    left = expand(prod_expr.left, settings)
    right = expand(prod_expr.right, settings)

    if left.is_zero_element() or right.is_zero_element():
        return Const(MatrixLit(name="0", shape=prod_expr.shape()))

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
