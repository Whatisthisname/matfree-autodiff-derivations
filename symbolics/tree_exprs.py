from abc import abstractmethod
import typing
from enum import StrEnum  # type: ignore[attr-defined]
from typing import Optional, Protocol, Union
from typing_extensions import assert_never, Self


class Displayable(Protocol):
    def str_compact(self) -> str:
        """Return a compact string representation of the object."""
        pass

    def __repr__(self) -> str:
        """Return a detailed string representation of the object."""
        pass


class Dim:
    def __init__(self, dim: Optional[Union[int, str]]):
        self.dim = dim

    def __repr__(self) -> str:
        return f"Dim({self.dim})"


UnitDim = Dim(1)


class Shape(Displayable):
    def __init__(self, shape: typing.Tuple[Dim, Dim]):
        self.shape = shape

    def __repr__(self) -> str:
        return f"Shape({self.shape})"

    def str_compact(self) -> str:
        return f"({self.shape[0].dim}, {self.shape[1].dim})"

    def __eq__(self, other: typing.Any) -> bool:
        if not isinstance(other, Shape):
            return NotImplemented
        return self.shape[0] == other.shape[0] and self.shape[1] == other.shape[1]

    def __getitem__(self, index: int) -> Dim:
        if index < 0 or index >= len(self.shape):
            raise IndexError("Index out of range")
        return self.shape[index]


class MatrixLit(Displayable):
    def __init__(self, shape: Shape, name: str):
        self.shape = shape
        self.name = name

    def __repr__(self) -> str:
        return f"Matrix({self.name}, {self.shape})"

    def str_compact(self) -> str:
        return f"{self.name}"


class MatrixExpr(Displayable, typing.Protocol):
    @abstractmethod
    def shape(self) -> Shape:
        pass

    @property
    def rows(self) -> Dim:
        return self.shape().shape[0]

    @property
    def cols(self) -> Dim:
        return self.shape().shape[1]

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def str_compact(self) -> str:
        pass

    @property
    def T(self) -> "MatrixExpr":
        return _Transpose(self)

    def __neg__(self) -> "MatrixExpr":
        return _Negate(self)

    def __add__(self, other: "MatrixExpr") -> "MatrixExpr":
        return _Sum([self, other])

    def __sub__(self, other: "MatrixExpr") -> "MatrixExpr":
        return _Sum([self, -other])

    def __mul__(self, other: "MatrixExpr") -> "MatrixExpr":
        if isinstance(other, (int)) and other == 0:
            return Const(MatrixLit(name="0", shape=self.shape()))
        return _Product(self, other)

    def __rmul__(self, other: "MatrixExpr") -> "MatrixExpr":
        if isinstance(other, (int)) and other == 0:
            return Const(MatrixLit(name="0", shape=self.shape()))
        return _Product(other, self)

    def __inv__(self) -> "MatrixExpr":
        return Inverse(self)

    def is_zero_element(self) -> bool:
        return isinstance(self, Const) and self.term.name == "0"

    def is_identity_element(self) -> bool:
        return isinstance(self, Const) and self.term.name == "1"

    def is_scalar(self) -> bool:
        return self.shape().shape[0] == UnitDim and self.shape().shape[1] == UnitDim

    def is_vector(self) -> bool:
        return self.shape().shape[1] == UnitDim

    def is_square(self) -> bool:
        return self.shape().shape[0] == self.shape().shape[1]


class Sps(StrEnum):  # type: ignore[misc]
    Dense = "Full"
    Diag = "Diag"
    Upper = "Upper"
    Lower = "Lower"
    SUpper = "Strict Upper"
    SLower = "Strict Lower"
    SSUpper = "I>>"
    Nothing = "Nothing"
    SSLower = "I<<"


class Mask(MatrixExpr):
    def __init__(self, expr: MatrixExpr, sparsity: Sps):
        self.expr = expr
        self.sparsity = sparsity

    def shape(self) -> Shape:
        return self.expr.shape()

    def __repr__(self) -> str:
        return f"Mask({self.expr}, {self.sparsity})"

    @staticmethod
    def _get_symbol(sparsity: Sps) -> str:
        if sparsity == Sps.Dense:
            return f"█"
        elif sparsity == Sps.Diag:
            return f"⟍"
        elif sparsity == Sps.Upper:
            return f"◥"
        elif sparsity == Sps.Lower:
            return f"◣"
        elif sparsity == Sps.SUpper:
            return f"◹"
        elif sparsity == Sps.SLower:
            return f"◺"
        elif sparsity == Sps.SSUpper:
            return f"»"
        elif sparsity == Sps.Nothing:
            return f"☐"
        elif sparsity == Sps.SSLower:
            return f"«"
        else:
            assert_never(sparsity)

    def str_compact(self) -> str:
        if isinstance(self.expr, (_Sum, _Product)):
            return Mask._get_symbol(self.sparsity) + f"({self.expr.str_compact()})"
        else:
            return Mask._get_symbol(self.sparsity) + f"{self.expr.str_compact()}"


class Differential(MatrixExpr):
    def __init__(self, expr: MatrixExpr):
        self.expr = expr

    def shape(self) -> Shape:
        return self.expr.shape()

    def __repr__(self) -> str:
        return f"Differential({self.expr})"

    def str_compact(self) -> str:
        if isinstance(self.expr, (_Sum, _Product)):  # TODO perhaps fix here.
            return f"d({self.expr.str_compact()})"
        else:
            return f"d{self.expr.str_compact()}"


class Const(MatrixExpr):
    def __init__(self, term: MatrixLit):
        self.term = term

    def shape(self) -> Shape:
        return self.term.shape

    def __repr__(self) -> str:
        return f"Constant({self.term})"

    def str_compact(self) -> str:
        return self.term.str_compact()


class Var(MatrixExpr):
    def __init__(self, term: MatrixLit):
        self.term = term

    def shape(self) -> Shape:
        return self.term.shape

    def __repr__(self) -> str:
        return f"Var({self.term})"

    def str_compact(self) -> str:
        return self.term.str_compact()


class _Transpose(MatrixExpr):
    def __init__(self, expr: MatrixExpr):
        self.expr = expr

    def __repr__(self) -> str:
        return f"Transpose({self.expr})"

    def shape(self) -> Shape:
        return Shape((self.expr.shape().shape[1], self.expr.shape().shape[0]))

    def str_compact(self) -> str:
        if isinstance(self.expr, (_Sum, _Product)):  # TODO perhaps fix here.
            return f"({self.expr.str_compact()})ᵀ"
        else:
            return f"{self.expr.str_compact()}ᵀ"


class Inverse(MatrixExpr):
    def __init__(self, expr: MatrixExpr):
        self.expr = expr
        assert expr.is_square(), f"Inverse requires a square matrix, got {expr.shape()}"

    def __repr__(self) -> str:
        return f"Inverse({self.expr})"

    def shape(self) -> Shape:
        return self.expr.shape()

    def str_compact(self) -> str:
        return f"{self.expr.str_compact()}^-1"


class _Sum(MatrixExpr):
    def __init__(self, exprs: typing.Sequence[MatrixExpr]):
        assert len(exprs) > 0, "Sum requires at least one term"
        assert all(
            term.shape() == exprs[0].shape() for term in exprs
        ), f"All terms in a sum must have the same shape, got {[term.shape() for term in exprs]}"
        self.exprs = exprs

    def shape(self) -> Shape:
        return self.exprs[0].shape()

    def __repr__(self) -> str:
        return f"Sum({self.exprs})"

    def str_compact(self) -> str:
        if len(self.exprs) == 1:
            return self.exprs[0].str_compact()
        else:
            total_str = self.exprs[0].str_compact()
            for i, expr in enumerate(self.exprs[1:]):
                total_str += (
                    " + " if not isinstance(expr, _Negate) else " "
                ) + expr.str_compact()

            return "(" + total_str + ")"


class _Negate(MatrixExpr):
    def __init__(self, expr: MatrixExpr):
        self.expr = expr

    def shape(self) -> Shape:
        return self.expr.shape()

    def __repr__(self) -> str:
        return f"Negate({self.expr})"

    def str_compact(self) -> str:
        return f"- {self.expr.str_compact()}"


class _Product(MatrixExpr):
    def __init__(self, left: MatrixExpr, right: MatrixExpr):
        assert (
            left.is_scalar() or right.is_scalar() or left.shape()[1] == right.shape()[0]
        ), f"Product requires shared inner dimension for both sides, got LHS {left.shape()} and RHS {right.shape()}, with left = {left.str_compact()} and right {right.str_compact()}"
        if right.is_scalar():
            self._shape = left.shape()
        elif left.is_scalar():
            self._shape = right.shape()
        else:
            self._shape = Shape((left.shape()[0], right.shape()[1]))
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"Product({self.left}, {self.right})"

    def str_compact(self) -> str:
        return f"{self.left.str_compact()}·{self.right.str_compact()}"

    def shape(self) -> Shape:
        return self._shape


class Norm(MatrixExpr):
    def __init__(self, expr: MatrixExpr):
        assert expr.is_vector(), f"Norm requires a vector, got {expr.shape()}"

        self.expr = expr

    def shape(self) -> Shape:
        return Shape((UnitDim, UnitDim))

    def __repr__(self) -> str:
        return f"Norm({self.expr})"

    def str_compact(self) -> str:
        return f"||{self.expr.str_compact()}||"


class InnerProduct(MatrixExpr):
    def __init__(self, left: MatrixExpr, right: MatrixExpr):
        assert (
            left.shape() == right.shape()
        ), f"Inner product requires same shape for both sides, but got LHS {left.shape()} and RHS {right.shape()} with left = {left.str_compact()} and right = {right.str_compact()}"
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"InnerProduct({self.left}, {self.right})"

    def str_compact(self) -> str:
        return f"〈{self.left.str_compact()} , {self.right.str_compact()}〉"

    def shape(self) -> Shape:
        return Shape((UnitDim, UnitDim))

    def flip(self) -> "InnerProduct":
        """Flip the inner product."""
        return InnerProduct(self.right, self.left)

    def transpose(self) -> "InnerProduct":
        return InnerProduct(self.left.T, self.right.T)


class Equation:
    def __init__(self, lhs: MatrixExpr, rhs: MatrixExpr):
        assert (
            lhs.shape() == rhs.shape()
        ), f"Equation requires same shape for both sides, got LHS {lhs.shape()} and RHS {rhs.shape()}"
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self) -> str:
        return f"Equation({self.lhs}, {self.rhs})"

    def str_compact(self) -> str:
        return f"{self.lhs.str_compact()} = {self.rhs.str_compact()}"

    def __add__(self, other: MatrixExpr) -> "Equation":
        return Equation(_Sum((self.lhs, other)), _Sum((self.rhs, other)))

    __radd__ = __add__

    def __mul__(self, other: MatrixExpr) -> "Equation":
        """Multiply something from the right."""
        return Equation(_Product(self.lhs, other), _Product(self.rhs, other))

    def __rmul__(self, other: MatrixExpr) -> "Equation":
        """Multiply something from the left."""
        return Equation(_Product(other, self.lhs), _Product(other, self.rhs))

    def isolate_zero_RHS(self) -> "Equation":
        """Isolate the zero on the right-hand side."""
        return Equation(
            self.lhs - self.rhs, Const(MatrixLit(name="0", shape=self.lhs.shape()))
        )

    def flip(self) -> "Equation":
        """Flip the equation."""
        return Equation(self.rhs, self.lhs)

    def differentiate(self) -> "Equation":
        """Differentiate the equation."""
        return Equation(Differential(self.lhs), Differential(self.rhs))


def niceprint(expr: Displayable) -> None:
    print(expr.str_compact())
