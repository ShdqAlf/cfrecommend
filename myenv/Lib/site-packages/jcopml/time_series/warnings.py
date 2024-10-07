import functools
import warnings


def ignore_warning(func):
    """
    a decorator to ignore warning
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper
