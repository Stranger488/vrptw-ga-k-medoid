from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()

        output = open("time", "a")
        output.write("{}\n".format(round(te - ts), 2))
        return result
    return wrap