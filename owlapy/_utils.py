def MOVE(*args):
    from inspect import currentframe
    f = currentframe()
    f = f.f_back
    mod = f.f_globals['__name__']
    for cls in args:
        cls.__module__ = mod
