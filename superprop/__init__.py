"""
Python property inheritance work-around module

Based on: https://bugs.python.org/file37546/superprop.py
By: Simon Zack
See also: https://bugs.python.org/issue14965

"""

__all__ = ['super_prop']


class SuperProp:
    """
    Super wrapper which allows property setting & deletion. Super can't be subclassed with empty __init__ arguments.
    """

    def __init__(self, super_obj):
        object.__setattr__(self, 'super_obj', super_obj)

    def _find_desc(self, name):
        super_obj = object.__getattribute__(self, 'super_obj')
        if name != '__class__':
            mro = iter(super_obj.__thisclass__.__mro__)
            for cls in mro:
                if cls == super_obj.__thisclass__:
                    break
            for cls in mro:
                if isinstance(cls, type):
                    try:
                        # don't lookup further up the chain
                        return object.__getattribute__(cls, name)
                    except AttributeError:
                        continue
                    except KeyError:
                        return None
        return None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'super_obj'), name)

    def __setattr__(self, name, value):
        super_obj = object.__getattribute__(self, 'super_obj')
        desc = object.__getattribute__(self, '_find_desc')(name)
        if hasattr(desc, '__set__'):
            return desc.__set__(super_obj.__self__, value)
        return setattr(super_obj, name, value)

    def __delattr__(self, name):
        super_obj = object.__getattribute__(self, 'super_obj')
        desc = object.__getattribute__(self, '_find_desc')(name)
        if hasattr(desc, '__delete__'):
            return desc.__delete__(super_obj.__self__)
        return delattr(super_obj, name)


super_prop = SuperProp
