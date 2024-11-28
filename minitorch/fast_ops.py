from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function using Numba with the option to always inline."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # When `out` and `in` are stride-aligned, avoid indexing
        if(
            len(out_strides) != len(in_strides)
            or (out_strides != in_strides).any()
            or (out_shape != in_shape).any()
        ):
            for i in prange(len(out)):
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                in_index: Index = np.empty(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                broadcast_index(out_index, out_shape, in_shape, in_index)
                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])
        else:
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])

        # if list(out_shape) == list(in_shape) and list(out_strides) == list(in_strides):
        #     for i in prange(len(out)):
        #         out[i] = fn(in_storage[i])
        #     return
        # else:
        #     for i in prange(len(out)):
        #         out_index: Index = np.empty(
        #             MAX_DIMS, np.int32
        #         )  # changed from int16 to int32 to avoid overflow
        #         in_index: Index = np.empty(
        #             MAX_DIMS, np.int32
        #         )  # changed from int16 to int32 to avoid overflow
        #         to_index(i, out_shape, out_index)
        #         broadcast_index(out_index, out_shape, in_shape, in_index)
        #         o = index_to_position(out_index, out_strides)
        #         j = index_to_position(in_index, in_strides)
        #         out[o] = fn(in_storage[j])

    return njit(_map, parallel=True)  # type: ignore
    # return njit(parallel=True)(_map)

def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 3.1.
        # When `out` and `in` are stride-aligned, avoid indexing
        if (
            len(out_strides) != len(a_strides)
            or len(out_strides) != len(b_strides)
            or (out_strides != a_strides).any()
            or (out_strides != b_strides).any()
            or (out_shape != a_shape).any()
            or (out_shape != b_shape).any()
        ):
            for i in prange(len(out)):
                out_index: Index = np.empty(MAX_DIMS, np.int32)
                a_index: Index = np.empty(MAX_DIMS, np.int32)
                b_index: Index = np.empty(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)
                o = index_to_position(out_index, out_strides)
                broadcast_index(out_index, out_shape, a_shape, a_index)
                j = index_to_position(a_index, a_strides)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                k = index_to_position(b_index, b_strides)
                out[o] = fn(a_storage[j], b_storage[k])
        else:
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])

        # if list(out_shape) == list(a_shape) == list(b_shape) and list(
        #     out_strides
        # ) == list(a_strides) == list(b_strides):
        #     for i in prange(len(out)):
        #         out[i] = fn(a_storage[i], b_storage[i])
        #     return
        # else:
        #     for i in prange(len(out)):
        #         out_index: Index = np.empty(MAX_DIMS, np.int32)
        #         a_index: Index = np.empty(MAX_DIMS, np.int32)
        #         b_index: Index = np.empty(MAX_DIMS, np.int32)
        #         to_index(i, out_shape, out_index)
        #         o = index_to_position(out_index, out_strides)
        #         broadcast_index(out_index, out_shape, a_shape, a_index)
        #         j = index_to_position(a_index, a_strides)
        #         broadcast_index(out_index, out_shape, b_shape, b_index)
        #         k = index_to_position(b_index, b_strides)
        #         out[o] = fn(a_storage[j], b_storage[k])

    return njit(_zip, parallel=True)  # type: ignore
    # return njit(parallel=True)(_zip)


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 3.1.
        for i in prange(len(out)):
            out_index: Index = np.empty(MAX_DIMS, np.int32)
            reduce_size = a_shape[reduce_dim]
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            accum = out[o]
            j = index_to_position(out_index, a_strides)
            step = a_strides[reduce_dim]
            for s in range(reduce_size):
                accum = fn(accum, a_storage[j])
                j += step
            out[o] = accum

        # reduce_size = a_shape[reduce_dim]
        # reduce_stride = a_strides[reduce_dim]

        # for i in prange(len(out)):
        #     out_index: Index = np.empty(MAX_DIMS, np.int32)
        #     to_index(i, out_shape, out_index)
        #     o = index_to_position(out_index, out_strides)  # where to write
        #     j = index_to_position(out_index, a_strides)  # where to read
        #     local = out[o]  # Use local variable for reduction
        #     for _ in range(reduce_size):
        #         local = fn(local, a_storage[j])  # Apply reduction
        #         j += reduce_stride  # Move to next element
        #     out[o] = local

    return njit(_reduce, parallel=True)  # type: ignore
    # return njit(parallel=True)(_reduce)

def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # TODO: Implement for Task 3.2.
    row_stride_a = a_strides[1]
    col_stride_b = b_strides[2]

    for i1 in prange(out_shape[0]):
        for i2 in range(out_shape[1]):
            for i3 in range(out_shape[2]):
                a_inner = i1 * a_batch_stride + i2 * a_strides[1]
                b_inner = i1 * b_batch_stride + i3 * b_strides[1]
                acc = 0.0
                for _ in range(a_shape[2]):
                    acc += a_storage[a_inner] * b_storage[b_inner]
                    a_inner += a_strides[2]
                    b_inner += b_strides[1]
                out_position = i1 * out_strides[0] + i2 * out_strides[1] + i3 * out_strides[2]
                out[out_position] = acc

    # for n in prange(out_shape[0]):  # parallel
    #     for i in range(out_shape[1]):  # rows
    #         for j in range(out_shape[2]):  # cols
    #             o = (
    #                 n * out_strides[0] + i * out_strides[1] + j * out_strides[2]
    #             )  # position
    #             local = 0.0
    #             row_pos_a = n * a_batch_stride + i * row_stride_a
    #             col_pos_b = n * b_batch_stride + j * col_stride_b
    #             for _ in range(a_shape[-1]):
    #                 local += a_storage[row_pos_a] * b_storage[col_pos_b]
    #                 row_pos_a += a_strides[2]
    #                 col_pos_b += b_strides[1]
    #             out[o] = local

    # for n in prange(out_shape[0]): # parallel
    #     for i in range(out_shape[1]): # rows
    #         for j in range(out_shape[2]): # cols
    #             out_index = np.array([n, i, j], np.int32)
    #             o = index_to_position(out_index, out_strides)
    #             for k in range(a_shape[-1]):
    #                 a_index = np.array([n, i, k], np.int32)
    #                 b_index = np.array([n, k, j], np.int32)
    #                 a = index_to_position(a_index, a_strides)
    #                 b = index_to_position(b_index, b_strides)
    #                 out[o] += a_storage[a] * b_storage[b]


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
# tensor_matrix_multiply = njit(parallel=True, fastmatch = True)(_tensor_matrix_multiply)
assert tensor_matrix_multiply is not None
