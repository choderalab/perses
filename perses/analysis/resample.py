import numpy as np
from functools import wraps
from itertools import islice


def samples_correlated(arrays, seed=None):
    """
    Returns an iterator over datasets sampled with replacement from
    the input, each having the same length as the input.

    For each iteration, the set of rows sampled from each input array
    is the same. This is useful when the input arrays have the same
    length and represent correlated measurements.

    Parameters
    ----------
    arrays : list of array
        1-d arrays with consistent length

    seed : int, optional
        Random seed

    Returns
    -------
    iterator of tuple of array
        Resampled arrays, each with same length as the input
    """

    random_state = (
        np.random.mtrand._rand if seed is None else np.random.RandomState(seed)
    )

    n_samples = len(arrays[0])

    for i, array in enumerate(arrays):
        if array.ndim > 1:
            raise ValueError(
                f"Arguments must have dimension 1, but argument {i} has "
                f"dimension {array.ndim}."
            )
        if len(array) != n_samples:
            raise ValueError(
                f"Arguments must have consistent lengths, but the lengths "
                f"of arguments 0 and {i} are inconsistent: "
                f"{n_samples}, {len(array)}."
            )

    while True:
        indices = random_state.randint(0, n_samples, size=(n_samples,))
        yield tuple(array[indices] for array in arrays)


def samples_uncorrelated(arrays, seed=None):
    """
    Returns an iterator over datasets sampled with replacement from
    the input, each having the same length as the input.

    For each iteration, the set of rows sampled from each input array
    is different. This is useful when the input arrays may have
    differing lengths and represent uncorrelated measurements.

    Parameters
    ----------
    arrays : list of array
        1-d arrays with consistent length

    seed : int, optional
        Random seed

    Returns
    -------
    iterator of tuple of array
        Resampled arrays, each with same length as the corresponding input
    """

    random_state = (
        np.random.mtrand._rand if seed is None else np.random.RandomState(seed)
    )

    for i, array in enumerate(arrays):
        if array.ndim > 1:
            raise ValueError(
                f"Arguments must have dimension 1, but argument {i} has "
                f"dimension {array.ndim}."
            )

    while True:
        yield tuple(
            random_state.choice(array, replace=True, size=len(array))
            for array in arrays
        )


def _bootstrap(get_samples, f, n_iters, seed):
    @wraps(f)
    def inner(*arrays, **kwargs):

        args_samples = get_samples(arrays, seed=seed)
        check_args = next(iter(args_samples))
        check_result = f(*check_args, **kwargs)

        if not np.isscalar(check_result):
            raise ValueError(
                f"Function passed to bootstrap should return a scalar, "
                f"but got a {np.ndim(check_result)}-dimensional array"
            )

        samples = np.array(
            [f(*args_sample, **kwargs) for args_sample in islice(args_samples, n_iters)]
        )

        return samples.mean(), samples.std()

    return inner


def bootstrap_correlated(f, n_iters=100, seed=None):
    """
    Transforms a function that computes a sample statistic to a
    function that estimates the corresponding population statistic and
    uncertainty via bootstrap.

    In this version, the positional arguments to the function are
    assumed to be equal-length arrays, with the ith index of the jth
    array representing quantity j measured at observation i; the data
    are assumed correlated across quantities at a fixed observation,
    and uncorrelated across observations of a fixed quantity.

    Parameters
    ----------
    f : callable
        Function of one or more arrays returning a scalar. The
        positional arguments are assumed to be 1-d arrays of
        consistent length to be resampled at each iteration. Keyword
        arguments are passed through but are not resampled.

    n_iter : int
        Number of bootstrap iterations

    seed : int, optional
        Random seed

    Returns
    -------
    callable
        Function with the same input signature as `f`, returning a
        pair of scalars: (estimate, error)

    """

    return _bootstrap(samples_correlated, f, n_iters, seed)


def bootstrap_uncorrelated(f, n_iters=100, seed=None):
    """
    Transforms a function that computes a sample statistic to a
    function that estimates the corresponding population statistic and
    uncertainty via bootstrap.

    In this version, the positional arguments to the function are
    arrays of arbitrary length (not necessarily consistent), with the
    ith index of the jth array representing the ith observation of the
    jth independent experiment; the data are assumed to be
    uncorrelated across both observations and experiments.

    Parameters
    ----------
    f : callable
        Function of one or more arrays returning a scalar. The
        positional arguments are assumed to be 1-d arrays to be
        resampled at each iteration. Keyword arguments are passed
        through but are not resampled.

    n_iter : int
        Number of bootstrap iterations

    seed : int, optional
        Random seed

    Returns
    -------
    callable
        Function with the same input signature as `f`, returning a
        pair of scalars: (estimate, error)

    """

    return _bootstrap(samples_uncorrelated, f, n_iters, seed)
