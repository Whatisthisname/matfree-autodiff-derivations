from abc import abstractmethod
import typing
from typing import Optional, Protocol
from typing_extensions import assert_never, Self


class Displayable(Protocol):
    def str_compact(self) -> str:
        """Return a compact string representation of the object."""
        pass

    def __repr__(self) -> str:
        """Return a detailed string representation of the object."""
        pass


class Dim:
    def __init__(self, dim: Optional[int]):
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

    def __add__(self, other: Self) -> "MatrixExpr":
        return _Sum([self, other])

    def __sub__(self, other: Self) -> "MatrixExpr":
        return _Sum([self, -other])

    def __mul__(self, other: Self) -> "MatrixExpr":
        return _Product(self, other)

    def __rmul__(self, other: Self) -> "MatrixExpr":
        return _Product(other, self)

    def __inv__(self) -> "MatrixExpr":
        return Invert(self)

    def is_zero_element(self) -> bool:
        return isinstance(self, Const) and self.term.name == "0"

    def is_identity_element(self) -> bool:
        return isinstance(self, Const) and self.term.name == "1"

    def is_scalar(self) -> bool:
        return self.shape().shape[0] == UnitDim and self.shape().shape[1] == UnitDim

    def is_vector(self) -> bool:
        return self.shape().shape[1] == UnitDim


class Differential(MatrixExpr):
    def __init__(self, term: MatrixExpr):
        self.term = term

    def shape(self) -> Shape:
        return self.term.shape()

    def __repr__(self) -> str:
        return f"Differential({self.term})"

    def str_compact(self) -> str:
        return f"d{self.term.str_compact()}"


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
        return f"UnitExp({self.term})"

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
        return f"{self.expr.str_compact()}^T"


class Invert(MatrixExpr):
    def __init__(self, expr: MatrixExpr):
        self.expr = expr

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
        ), "All terms in a sum must have the same shape"
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
            left.shape()[1] == UnitDim or left.shape()[1] == right.shape()[0]
        ), f"Product requires shared inner dimension for both sides, got LHS {left.shape()} and RHS {right.shape()}"
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"Product({self.left}, {self.right})"

    def str_compact(self) -> str:
        return f"{self.left.str_compact()} @ {self.right.str_compact()}"

    def shape(self) -> Shape:
        return Shape((self.left.shape().shape[0], self.right.shape().shape[1]))


class Norm(MatrixExpr):
    def __init__(self, expr: MatrixExpr):
        assert expr.is_vector(), f"Norm requires a vector, got {expr.shape()}"


class InnerProduct:
    def __init__(self, left: MatrixExpr, right: MatrixExpr):
        assert (
            left.shape() == right.shape()
        ), "Inner product requires same shape for both sides"
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"<{self.left}, {self.right}>"


class Equation:
    def __init__(self, lhs: MatrixExpr, rhs: MatrixExpr):
        assert lhs.shape() == rhs.shape(), "Equation requires same shape for both sides"
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
