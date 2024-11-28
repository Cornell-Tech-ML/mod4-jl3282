# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA device.

    Args:
    ----
        fn (Fn): The function to be JIT compiled.
        **kwargs: Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        Fn: The JIT compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for CUDA.

    Args:
    ----
        fn (Callable): The function to be JIT compiled.
        **kwargs: Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        FakeCUDAKernel: The JIT compiled function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on two tensors using CUDA.

        Args:
        ----
            a (Tensor): The first tensor.
            b (Tensor): The second tensor.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

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

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

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
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = fn(in_storage[j])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i < out_size:
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            j = index_to_position(a_index, a_strides)            
            broadcast_index(out_index, out_shape, b_shape, b_index)
            k = index_to_position(b_index, b_strides)
            out[o] = fn(a_storage[j], b_storage[k])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Implementing a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.

    if i < size:
        val = float(a[i])
        cache[pos] = val
        cuda.syncthreads()
    else:
        cache[pos] = 0.0
    
    if i < size:
        for j in [1, 2, 4, 8, 16]:
            if pos % (2 * j) == 0:
                cache[pos] += cache[pos + j]
                cuda.syncthreads()
        if pos == 0:
            out[cuda.blockIdx.x] = cache[0]

jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Perform a practice sum operation on a tensor using CUDA.

    Args:
    ----
        a (Tensor): The input tensor.

    Returns:
    -------
        TensorData: The result of the sum operation.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)  # shared
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x
        cache[pos] = reduce_value

        # pad out to 2^0 if the size is not an even number
        # if its longer than 1024 run redue multuple times to get down to the final value
        # where to read and write?
        # while loop also fine
        # TODO: Implement for Task 3.3.
        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            if out_index[reduce_dim] < a_shape[reduce_dim]:
                in_a = index_to_position(out_index, a_strides)
                cache[pos] = a_storage[in_a]
                cuda.syncthreads()
                x = 0
                while 2**x < BLOCK_DIM:
                    j = 2**x
                    if pos % (2 * j) == 0:
                        cache[pos] = fn(cache[pos], cache[pos + j])
                        cuda.syncthreads()
                    x += 1
            if pos == 0:
                out[o] = cache[0]

        # i = out_index[reduce_dim] * cuda.blockDim.x + cuda.threadIdx.x
        # # cache[pos] = reduce_value
        # to_index(out_pos, out_shape, out_index)

        # if i < a_shape[reduce_dim]:
        #     out_index[reduce_dim] = i
        #     cache[pos] = a_storage[index_to_position(out_index, a_strides)]
        # else:
        #     cache[pos] = reduce_value
        # cuda.syncthreads()

        # if pos == 0:
        #     local = reduce_value
        #     for j in range(cuda.blockDim.x):
        #         if j < a_shape[reduce_dim]:
        #             local = fn(local, cache[j])

        #     out[out_pos] = local

    return jit(_reduce)  # type: ignore
    # return cuda.jit()(_reduce) 


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Implement a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    BLOCK_DIM = 32  # Block size is 32x32 for shared memory usage
    # <Note from the lecture>
    # extra dimension just becomes an extra block, should not leave them empty
    # if the shared memory is too small, it has to move around
    # make sure you implement the sliding window for the shared memory
    # TODO: Implement for Task 3.4.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    if i >= size or j >= size:
        return
    
    a_shared[i, j] = a[i * size + j]
    b_shared[i, j] = b[i * size + j]
    cuda.syncthreads()

    accum = 0.0
    for k in range(size):
        accum += a_shared[i, k] * b_shared[k, j]
    out[i * size + j] = accum
    
    
    # # Calculate the global thread indices
    # # Each thread corresponds to one cell in the output matrix
    # thread_row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # thread_col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # # Allocate shared memory for matrices a and b (shared by threads in the same block)
    # a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    # b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # # Load data into shared memory using loops
    # for row in range(BLOCK_DIM):
    #     for col in range(BLOCK_DIM):
    #         if row < size and col < size:
    #             # Copy elements from global memory to shared memory
    #             a_shared[row, col] = a[row * size + col]
    #             b_shared[row, col] = b[row * size + col]

    # # Synchronize threads
    # cuda.syncthreads()

    # # Perform matrix multiplication
    # # Each thread computes a single element in the output matrix
    # if thread_row < size and thread_col < size:
    #     accum_value = 0
    #     for k in range(size):
    #         # Compute the dot product for the thread's assigned row and column
    #         accum_value += a_shared[thread_row, k] * b_shared[k, thread_col]

    #     # Save result to global memory
    #     pos = thread_row * size + thread_col
    #     out[pos] = accum_value


jit_mm_practice = jit(_mm_practice)
# jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Perform matrix multiplication on two tensors using a practice kernel.

    Args:
    ----
        a (Tensor): The first tensor.
        b (Tensor): The second tensor.

    Returns:
    -------
        TensorData: The result of the matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    # Shared memory allocation for matrix a and b
    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Compute global and local thread indices
    # The final position c[i, j]; global position
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    accum = 0.0
    for k_start in range(0, a_shape[2], BLOCK_DIM):
        k = k_start + pi
        if i < a_shape[1] and k < a_shape[2]:
            a_shared[pi, pj] = a_storage[batch * a_batch_stride + i * a_strides[1] + k * a_strides[2]]
        k = k_start + pj
        if j < b_shape[2] and k < b_shape[1]:
            b_shared[pi, pj] = b_storage[batch * b_batch_stride + k * b_strides[1] + j * b_strides[2]]
        cuda.syncthreads()

        for k in range(BLOCK_DIM):
            if (k_start + k) < a_shape[2]:
                accum += a_shared[pi, k] * b_shared[k, pj]
    if i < out_shape[1] and j < out_shape[2]:
        out[batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]] = accum
    
    
    # # Initialize the accumulator for output
    # accum = 0.0
    # shared_dim = a_shape[2]  # Shared dimension (K dimension)

    # # Loop over shared memory tiles
    # for k in range(0, shared_dim, BLOCK_DIM):
    #     # Load tile from global to shared memory for matrix a
    #     if i < a_shape[1] and (k + pj) < shared_dim:
    #         a_idx = batch * a_batch_stride + i * a_strides[1] + (k + pj) * a_strides[2]
    #         a_shared[pi, pj] = a_storage[a_idx]
    #     else:
    #         a_shared[pi, pj] = 0.0  # if out-of-bounds, set the value to 0.0

    #     # Similarly for matrix b, loads a tile into shared memory
    #     if (k + pi) < shared_dim and j < b_shape[2]:
    #         b_idx = batch * b_batch_stride + (k + pi) * b_strides[1] + j * b_strides[2]
    #         b_shared[pi, pj] = b_storage[b_idx]
    #     else:
    #         b_shared[pi, pj] = 0.0  # if out-of-bounds, set the value to 0.0

    #     cuda.syncthreads()  # Ensure all threads have loaded their tiles

    #     # Compute partial dot product using the shared memory tiles for matrix a and b
    #     if i < out_shape[1] and j < out_shape[2]:
    #         for kk in range(
    #             min(BLOCK_DIM, shared_dim - k)
    #         ):  # only iterate over the valid shared dimension
    #             accum += a_shared[pi, kk] * b_shared[kk, pj]

    #     cuda.syncthreads()  # Ensure all threads have computed their partial dot products

    # # Write result of the matrix multiplication to global memory, only for valid output indices
    # if i < out_shape[1] and j < out_shape[2]:
    #     out_idx = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
    #     out[out_idx] = accum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
