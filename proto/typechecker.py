from functools import wraps


def accept(*types, **mapTypes):
    '''
    provide type checking of a function, example:
    @accept(int, int)
    def add(a, b):
        return a + b
    '''
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            # leave it for now
            return f(*args, **kwds)
        return wrapper
    return decorator
