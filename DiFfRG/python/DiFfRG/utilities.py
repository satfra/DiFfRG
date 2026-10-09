import sys
import uuid
import glob
import numpy as np

def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

def is_close(a, b):
   # for strings, just compare them
   if isinstance(a, str) or isinstance(b, str):
       return a == b
   # for numbers, use numpy's isclose
   return np.isclose(a, b)