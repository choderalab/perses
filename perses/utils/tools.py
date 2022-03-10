"Module for useful tools"

import functools
from time import sleep


class ErrorNeverThrown(Exception):
    """This should never be thrown"""

    pass


def retry(max_retry, exceptions=None, delay=0):
    """
    Decorator to retry function max_retry times if a specifed exception is raised.
    Without passing in a an excpetion or a tuple of excpetions, this decorator will
    effectivly do nothing beyond call the decorated function.

    Parameters
    ----------

    max_retry : int
        Number of times to retry function if exception is thrown

    exceptions : Exception or tuple of Exceptions, optional
        exception(s) to catch and keep retrying. If no excpetion is passed in, then the
        function decorated will not retry. This prevents users from accidently retrying
        excpeetions that have no hope of passing eventually. Default None.

    delay : float, optional
        delay in seconds to wait before retrying function after caught exception.
        Default 0 (no delay).

    Examples
    --------

    >>> @retry(5)
    >>> def say_name(name):
    ...     print(name)
    ... say_name("mike")
    mike
    >>>
    >>> @retry(5, exceptions=NameError)
    ... def say_name(name):
    ...     print(name)
    ...     raise NameError
    ...
    >>> say_name("mike")
    mike
    mike
    mike
    mike
    mike
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "perses/utils/tools.py", line 60, in wrapper_repeat
        raise e
      File "perses/utils/tools.py", line 56, in wrapper_repeat
        return func(*args, **kwargs)
      File "<stdin>", line 4, in say_name
    NameError
    >>>

    """

    # User wants all exceptions caught so we will set the exception to an error they will never throw
    if not exceptions:
        exceptions = ErrorNeverThrown

    def decorator_repeat(func):
        @functools.wraps(func)
        def wrapper_repeat(*args, **kwargs):
            attempt = 0
            while attempt < max_retry:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # raise on last attempt
                    if attempt == max_retry - 1:
                        raise e
                attempt += 1
                sleep(delay)

        return wrapper_repeat

    return decorator_repeat
