'''
This provides a little wrapper around joblib, giving a default cache directory
which is three levels up from where the module is defined with a directory joblib_cache.
It also adds two convenience functions `is_cacheable` and `is_cached` to check if a
function was wrapped with joblib, and to check if a particular set of keywords and
arguments are already cached.
'''
import joblib
import os, inspect

__all__ = ['cache_path', 'cache_mem', 'func_cache', 'is_cacheable', 'is_cached', 'cache_hash']

print joblib.__version__

basepath, _ = os.path.split(__file__)
print 'basepath = ' , basepath
#basepath =  /chandelierhome/skadir/hybrid_analysis
# When called from a notebook the answer is blank


cache_path = os.path.normpath(os.path.join(basepath, 'joblib_cache'))
print 'cache_path = ', cache_path
#cache_path =  /chandelierhome/skadir/hybrid_analysis/joblib_cache
# When called from a notebook the answer is blank followed by 'joblib_cache'

cache_mem = joblib.Memory(cachedir=cache_path, verbose=0)

func_cache = cache_mem.cache

def is_cacheable(func):
    return hasattr(func, 'get_output_dir')

def is_cached(func, *args, **kwds):
    s = func.get_output_dir(*args, **kwds)
    return os.path.exists(func.get_output_dir(*args, **kwds)[0])

def cache_hash(func, *args, **kwds):
    _, hash = func.get_output_dir(*args, **kwds)
    return hash

if __name__=='__main__':
    @func_cache
    def f(x):
        print 'I am taking the cube'
        return x*x*x
    
    print f(2)
    print f(2)
    print cache_hash(f, 2)
    
