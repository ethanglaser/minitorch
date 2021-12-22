"""
Collection of the core mathematical operators used throughout the code base.
"""


import math

# ## Task 0.1

# Implementation of a prelude of elementary functions.


def mul(x, y):
    """
    Multiply

    Args:
        x (int): numeric value
        y (int): numeric value

    Returns:
        Product.
    """
    return x * y


def id(x):
    """
    Identity

    Args:
        x (int): numeric value

    Returns:
        Product.
    """
    return x


def add(x, y):
    """
    Add

    Args:
        x (int): numeric value
        y (int): numeric value

    Returns:
        Sum.
    """
    return x + y


def neg(x):
    """
    Negative

    Args:
        x (int): numeric value

    Returns:
        Opposite.
    """
    return -x


def lt(x, y):
    """
    Less than

    Args:
        x (int): numeric value
        y (int): numeric value

    Returns:
        1.0 if x less than y else 0.0
    """
    if x < y:
        return 1.0
    else:
        return 0.0


def eq(x, y):
    """
    Equal to

    Args:
        x (int): numeric value
        y (int): numeric value

    Returns:
        1.0 if x equal to y else 0.0
    """
    if x == y:
        return 1.0
    else:
        return 0.0


def max(x, y):
    """
    Max

    Args:
        x (int): numeric value
        y (int): numeric value

    Returns:
        Max of x and y
    """
    return x if x > y else y


def is_close(x, y):
    """
    Close to

    Args:
        x (int): numeric value
        y (int): numeric value

    Returns:
        1.0 if difference less than 0.01 else 0.0
    """
    if abs(x - y) < 0.01:
        return 1.0
    else:
        return 0.0


def sigmoid(x):
    r"""
    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`

    (See `<https://en.wikipedia.org/wiki/Sigmoid_function>`_ .)

    Calculate as

    :math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}` if x >=0 else :math:`\frac{e^x}{(1.0 + e^{x})}`

    for stability.

    Args:
        x (float): input

    Returns:
        float : sigmoid value
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x):
    """
    :math:`f(x) =` x if x is greater than 0, else 0

    (See `<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>`_ .)

    Args:
        x (float): input

    Returns:
        float : relu value
    """
    if x < 0:
        return 0.0
    else:
        return x


EPS = 1e-6


def log(x):
    ":math:`f(x) = log(x)`"
    if x == 0:
        return math.log(x + EPS)
    else:
        return math.log(x)


def exp(x):
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(x, d):
    r"If :math:`f = log` as above, compute d :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    return (1 / x) * d


def inv(x):
    """
    Inverse

    Args:
        x (int): numeric value

    Returns:
        inverse.
    """
    return 1 / x


def inv_back(x, d):
    r"If :math:`f(x) = 1/x` compute d :math:`d \times f'(x)`"
    # TODO: Implement for Task 0.1.
    return ((-1 / x ** 2)) * d


def relu_back(x, d):
    r"If :math:`f = relu` compute d :math:`d \times f'(x)`"
    if x <= 0:
        return 0
    else:
        return d


# ## Task 0.3

# Small library of elementary higher-order functions for practice.


def map(fn):
    """
    Higher-order map.

    .. image:: figs/Ops/maplist.png


    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """

    def function(input_list):
        return [fn(i) for i in input_list]

    return function


def negList(ls):
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    funcc = map(neg)
    return funcc(ls)


def zipWith(fn):
    """
    Higher-order zipwith (or map2).

    .. image:: figs/Ops/ziplist.png

    See `<https://en.wikipedia.org/wiki/Map_(higher-order_function)>`_

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) on each pair of elements.

    """

    def function(ls1, ls2):
        zipped = zip(ls1, ls2)
        return [fn(z, zz) for z, zz in zipped]

    return function


def addLists(ls1, ls2):
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    funcc = zipWith(add)
    return funcc(ls1, ls2)


def reduce(fn, hello):
    r"""
    Higher-order reduce.

    .. image:: figs/Ops/reducelist.png


    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`
    """

    def function(ls):
        a = hello
        for l in ls:
            a = fn(a, l)
        return a

    return function


def sum(ls):
    "Sum up a list using :func:`reduce` and :func:`add`."
    if len(ls) == 0:
        return 0
    elif len(ls) == 1:
        return ls[0]
    funcc = reduce(add, ls[0])
    return funcc(ls[1:])


def prod(ls):
    "Product of a list using :func:`reduce` and :func:`mul`."
    if len(ls) == 0:
        return 0
    elif len(ls) == 1:
        return ls[0]
    funcc = reduce(mul, ls[0])
    return funcc(ls[1:])
