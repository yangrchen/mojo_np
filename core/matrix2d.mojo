from algorithm import vectorize, parallelize
from math import trunc, mod, round
from random import rand
from memory import memset
from tensor import Tensor


struct Matrix2D[dtype: DType = DType.float32](Stringable):
    var dim0: Int
    var dim1: Int
    var _data: DTypePointer[dtype]
    alias simd_width = simdwidthof[dtype]()

    @always_inline
    fn __init__(inout self, *dims: Int):
        self.dim0 = dims[0]
        self.dim1 = dims[1]
        self._data = DTypePointer[dtype].alloc(dims[0] * dims[1])
        rand(self._data, dims[0] * dims[1])

    @always_inline
    fn __copyinit__(inout self, other: Self):
        """Creates a shallow copy (doesn't copy the underlying elements).

        Args:
            other: The `Matrix` to copy.
        """
        self.dim0 = other.dim0
        self.dim1 = other.dim1
        self._data = other._data

    fn _adjust_slice_(self, inout span: Slice, dim: Int):
        if span.start < 0:
            span.start = span.start % dim
        if span.end > dim:
            span.end = dim
        elif span.end < 0:
            span.end = span.end % dim
        if span.start > span.end:
            span.start = 0
            span.end = 0

    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> SIMD[dtype, 1]:
        var row_new = row % self.dim0
        var col_new = col % self.dim1
        return self._data.simd_load[1](row_new * self.dim1 + col_new)

    @always_inline
    fn __getitem__(self, row: Int) -> Self:
        var idx = row % self.dim0
        var src_ptr = self._data.offset(idx * self.dim1)
        var mat_new = Self(1, self.dim1)

        # Vectorize to store multiple elements in the row simul.
        @parameter
        fn set_row[simd_width: Int](idx: Int):
            mat_new._data.simd_store[simd_width](
                idx, src_ptr.simd_load[simd_width](idx)
            )

        vectorize[set_row, self.simd_width](self.dim1)
        return mat_new

        # Old implementation just called the overloaded __getitem__
        # return self.__getitem__(row, slice(0, self.dim1))

    @always_inline
    fn __getitem__(self, row: Int, owned col_slice: Slice) -> Self:
        var start: Int
        var end: Int
        if row < 0:
            start = row % self.dim0
        else:
            start = row
        end = start + 1
        return self.__getitem__(slice(start, end), col_slice)

    @always_inline
    fn __getitem__(self, owned row_slice: Slice, col: Int) -> Self:
        return self.__getitem__(row_slice, slice(col, col + 1))

    @always_inline
    fn __getitem__(self, owned row_slice: Slice, owned col_slice: Slice) -> Self:
        self._adjust_slice_(row_slice, self.dim0)
        self._adjust_slice_(col_slice, self.dim1)

        var src_ptr = self._data
        var mat_new = Self(len(row_slice), len(col_slice))

        # idx_rows and idx correspond to indices in the mat_new array
        @parameter
        fn slice_col(idx_rows: Int):
            src_ptr = self._data.offset(row_slice[idx_rows] * self.dim1 + col_slice[0])

            # SIMD width can differ so "step" in pointer to next element needs to be multiplied by SIMD width
            @parameter
            fn slice_row[simd_width: Int](idx: Int):
                mat_new._data.simd_store[simd_width](
                    idx + idx_rows * len(col_slice),
                    src_ptr.simd_strided_load[simd_width](col_slice.step),
                )
                src_ptr = src_ptr.offset(simd_width * col_slice.step)

            vectorize[slice_row, self.simd_width](len(col_slice))

        parallelize[slice_col](len(row_slice))
        return mat_new

    @always_inline
    fn __setitem__(inout self, row: Int, col: Int, val: Scalar[dtype]):
        self._data.simd_store[1](row * self.dim1 + col, val)

    fn __setitem__(inout self, row: Int, col_slice: Slice):
        pass

    # This printing is probably very slow. Will need to look at addressing this later
    fn __str__(self) -> String:
        var prec = 3
        var rank = 2
        var dim0 = 0
        var dim1 = 0
        var val: Scalar[dtype]
        var str_rep: String = ""
        if self.dim0 == 1:
            rank = 1
            dim0 = 1
            dim1 = self.dim1
        else:
            dim0 = self.dim0
            dim1 = self.dim1

        if dim0 > 0 and dim1 > 0:
            for i in range(dim0):
                if rank > 1:
                    if i == 0:
                        str_rep += "  ["
                    else:
                        str_rep += "\n  "
                str_rep += "["
                for j in range(dim1):
                    if rank == 1:
                        val = self._data.simd_load[1](j)
                    else:
                        val = self[i, j]
                    var s: String = ""
                    var int_str: String = ""
                    if dtype.is_floating_point():
                        int_str = str(trunc(val).cast[DType.int32]())

                        # Change negative integer part to use mod to get the correct fractional part
                        if val < 0:
                            val = -val
                        var float_str = str(mod(val, 1))

                        # Print with 0.3f precision
                        # TODO: Implement better precision implementation
                        s += int_str + "." + float_str[2:prec]
                    elif dtype.is_integral():
                        s += str(val.cast[DType.int32]())
                    if j == 0:
                        str_rep += s
                    else:
                        str_rep += " " + s
                str_rep += "]"
            if rank > 1:
                str_rep += "]"
            if rank > 2:
                str_rep += "]"
            str_rep += "\n"
        str_rep += (
            "  Matrix: "
            + str(self.dim0)
            + " x "
            + str(self.dim1)
            + ", "
            + "DType: "
            + str(dtype)
        )
        return str_rep