from re import M
from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    new_height = height // kh
    new_width = width // kw

    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    reshaped = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    # Flatten the (kh, kw) into a single dimension
    tiled = reshaped.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width

# TODO: Implement for Task 4.3.

def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    tile_input, new_height, new_width = tile(input, kernel)

    return tile_input.mean(dim=4).contiguous().view(batch, channel, new_height, new_width)

max_reduce = FastOps.reduce(operators.max, -1e9)

def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor

    Args:
    ----
        input: Input tensor
        dim: Dimension along which to find argmax

    Returns:
    -------
        A tensor of the same shape as input with 1s at the positions of maximal values along dim

    """
    max = max_reduce(input, dim)
    return max == input

class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Compute the max along a dimension

        Args:
        ----
            ctx: autodiff context
            input: input tensor
            dim: dimension to compute max

        Returns:
        -------
            The max tensor

        """
        max = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, max)
        return max
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient of the max function

        Args:
        ----
            ctx: autodiff context
            grad_output: gradient of the output

        Returns:
        -------
            Gradient of the loss with respect to the input tensor
            Gradient of the loss with respect to the dimension (always 0.0)

        """
        input, max = ctx.saved_values
        return ((input == max) * grad_output), 0.0

def max(input: Tensor, dim: int) -> Tensor:
    """Compute the max along a dimension

    Args:
    ----
        input: input tensor
        dim: dimension to compute max

    Returns:
    -------
        The max tensor

    """
    return Max.apply(input, input._ensure_tensor(dim))

def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor

    Args:
    ----
        input: Input tensor
        dim: Dimension along which to compute softmax

    Returns:
    -------
        The softmax tensor

    """
    exp = input.exp()
    sum = exp.sum(dim)
    return exp/sum 

def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor

    Args:
    ----
        input: Input tensor
        dim: Dimension along which to compute logsoftmax

    Returns:
    -------
        The logsoftmax tensor

    """
    return softmax(input, dim).log()

def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to the input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel, _, _ = input.shape
    tile_input, new_height, new_width = tile(input, kernel)

    return max(tile_input, 4).contiguous().view(batch, channel, new_height, new_width)

def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off

    Args:
    ----
        input: Input tensor
        p: Probability of dropout
        train: If True, apply dropout; if False, return input

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore:
        return input
    
    random_tensor = rand(input.shape)
    random_drop = random_tensor > p
    return input * random_drop 

