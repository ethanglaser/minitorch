from .fast_ops import FastOps
from .tensor_functions import rand, Function
from . import operators


def tile(input, kernel):
    """
    Reshape an image tensor for 2D pooling

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        (:class:`Tensor`, int, int) : Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.
    """

    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    return (
        input.view(batch, channel, height, width // kw, kw)
        .permute(0, 1, 3, 2, 4)
        .view(batch, channel, width // kw, height // kh, kh * kw)
        .permute(0, 1, 3, 2, 4)
    )


def avgpool2d(input, kernel):
    """
    Tiled average pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    return input.contiguous().view(batch, channel, height * (width // kw), kw).sum(
        3
    ).view(batch, channel, height, width // kw).permute(0, 1, 3, 2).contiguous().view(
        batch, channel, height // kh * width // kw, kh
    ).sum(
        3
    ).view(
        batch, channel, height // kh, width // kw
    ) / (
        kh * kw
    )


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input, dim):
    """
    Compute the argmax as a 1-hot tensor.

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply argmax


    Returns:
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input, dim):
        "Forward of max should be max reduction"
        jitter = rand(input.shape) * 1e-6
        ctx.save_for_backward(input + jitter, dim)
        return max_reduce(input + jitter, dim)

    @staticmethod
    def backward(ctx, grad_output):
        "Backward of max should be argmax (see above)"
        input, dim = ctx.saved_values
        return argmax(input, dim) * grad_output


max = Max.apply


def softmax(input, dim):
    r"""
    Compute the softmax as a tensor.

    .. math::

        z_i = \frac{e^{x_i}}{\sum_i e^{x_i}}

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply softmax

    Returns:
        :class:`Tensor` : softmax tensor
    """
    return input.exp() / input.exp().sum(dim=dim)


def logsoftmax(input, dim):
    r"""
    Compute the log of the softmax as a tensor.

    .. math::

        z_i = x_i - \log \sum_i e^{x_i}

    See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations

    Args:
        input (:class:`Tensor`): input tensor
        dim (int): dimension to apply log-softmax

    Returns:
        :class:`Tensor` : log of softmax tensor
    """
    return input - input.exp().sum(dim=dim).log()


def maxpool2d(input, kernel):
    """
    Tiled max pooling 2D

    Args:
        input (:class:`Tensor`): batch x channel x height x width
        kernel ( pair of ints ): height x width of pooling

    Returns:
        :class:`Tensor` : pooled tensor
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    return max_reduce(
        max_reduce(
            input.contiguous().view(batch, channel, height * (width // kw), kw), 3
        )
        .view(batch, channel, height, width // kw)
        .permute(0, 1, 3, 2)
        .contiguous()
        .view(batch, channel, height // kh * width // kw, kh),
        3,
    ).view(batch, channel, height // kh, width // kw)


def dropout(input, rate, ignore=False):
    """
    Dropout positions based on random noise.

    Args:
        input (:class:`Tensor`): input tensor
        rate (float): probability [0, 1) of dropping out each position
        ignore (bool): skip dropout, i.e. do nothing at all

    Returns:
        :class:`Tensor` : tensor with randoom positions dropped out
    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)
