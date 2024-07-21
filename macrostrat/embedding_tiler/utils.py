from time import time
from macrostrat.utils import get_logger

log = get_logger(__name__)
print(__name__)


def timer(name: str | None = None):
    def decorator(func):
        if name is None:
            timer_name = func.__name__
        else:
            timer_name = name

        def wrapper(*args, **kwargs):
            start = time()
            res = func(*args, **kwargs)
            elapsed = time() - start
            log.info(f"{timer_name}: {elapsed:.2f}s")
            return res

        return wrapper

    return decorator
