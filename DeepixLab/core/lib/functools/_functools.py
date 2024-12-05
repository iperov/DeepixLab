from functools import wraps


def cached_method(func):
    """
    Decorator. Per instance method cache.
    Garbage collected with the instance
    """
    cache_name = f'_{func.__name__}_cache'

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        key = (args, tuple(kwargs.items()))

        if (cache := getattr(self, cache_name, None)) is None:
            setattr(self, cache_name, cache := {})

        if key in cache:
            return cache[key]

        result = cache[key] = func(self, *args, **kwargs)
        return result

    def is_cached(self, *args, **kwargs) -> bool:
        key = (args, tuple(kwargs.items()))
        if (cache := getattr(self, cache_name, None)) is None:
            return False
        return key in cache

    def get_cached_dict(self) -> dict:
        return getattr(self, cache_name, {})

    wrapper.is_cached = is_cached
    wrapper.get_cached_dict = get_cached_dict

    return wrapper
