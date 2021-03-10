# -*- coding: utf-8 -*-

import sys
import os
import time

import numpy as np

numpy_integer_types  = [np.dtype(f'int{bits}') for bits in [8, 16, 32, 64]]
                         
numpy_float_types = [np.dtype(f'float{bits}') for bits in [16, 32, 64]] 

numpy_numerical_types  = numpy_integer_types + numpy_float_types
            
def progress_bar(value, endvalue, bar_length=20, endline=""):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()
    if value == endvalue:
        sys.stdout.write(endline)    
    
class TimeIt(object):
    
    base = [0] * 100
    
    def __init__(self, 
                 string_enter, 
                 string_exit_callable, 
                 level_enter=0, 
                 level_exit=0, 
                 verbose=1, 
                 reset_base=True,
                 which_base=0):
        """
        Context manager to compute runtime and print result.
        
        string_enter: str
            String printed before timeit starts.
            
        string_exit_callable: Callable[[float], str]
            String printed once timeit ends. It is callable on the time.
            
        level_enter: int (default: 0)
            The enter string is printed if level is lower than verbose.
            
        level_exit: int (default: 0)
            The exit string is printed if level is lower than verbose.
            
        verbose: int (default: 1)
            Threshold to decide if messages are printed. 
            
        reset_base: bool (default: True)
            If True, the time starts with no lag (base = 0). Else the time starts with a lag determine by the previous 
            calls. This is useful if the context manager is inside a loop and the total time to exit the loop must be 
            printed while every iteration of the loop must print a message.
            
        which_base: 0 <= int <= 100
            If several timeit are nested into each other, they should all have a different `which_base`.
        """
        self.string_enter = string_enter
        self.string_exit_callable = string_exit_callable
        self.level_enter = level_enter
        self.level_exit = level_exit
        self.verbose = verbose
        self.reset_base = reset_base
        self.which_base = which_base
        
    def __enter__(self):
        self.start = time.time()
        if self.level_enter <= self.verbose:
            print(self.string_enter, end="")        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.reset_base:
            TimeIt.base[self.which_base] = 0 
        self.time = (time.time() - self.start) + TimeIt.base[self.which_base]
        TimeIt.base[self.which_base] = self.time
        if self.level_exit <= self.verbose:
            print(self.string_exit_callable(self.time), end="")
        
class RedirectStd:
    
    def __init__(self, where=None):
        """ Redirect the std to the screen, to a file, or discard it.
        
        Argument:
        ---------
        where: str or `sys.stdout` or None (default: None) 
            If str, it is the path to the file where std is written.
            If `sys.stdout`, it is printed on screen.
            If None, std is discarded.
        """
        if where is None: 
            self._out = open(os.devnull, 'w') # do not print
        elif where == sys.stdout:
            self._out = sys.stdout # print on screen
        else:
            self._out = open(where, 'w') # print in file 
            
    def __enter__(self, stdout=None, stderr=None):
        self._old = sys.stdout, sys.stderr
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = self._out, self._out
        
    def __exit__(self, exc_type, exc_value, traceback):
        self._out.flush()
        sys.stdout, sys.stderr = self._old
        
        
class Formatter:
    
    def __init__(self, width=None, precision=None, fixed=False):
        self.width = width
        self.precision = precision
        self.fixed = fixed
        
    def _fmt(self, v):
        if type(v) == str:
            return v
        elif type(v) == int or type(v) == float or type(v).__name__ == "float64":
            fstr = ''
            if self.width is not None:
                fstr += ' ' + str(self.width)
            if self.precision is not None:
                fstr += '.' + str(self.precision)
            if self.fixed:
                fstr += 'f'
            else:
                fstr += 'g'
            return ('{:' + fstr + '}').format(v)
        elif hasattr(v, '__iter__'):
            return '[' + ','.join(self._fmt(vi) for vi in v) + ']'
        else:
            return str(v)
        
    def __call__(self, *args):
        return '\t'.join(self._fmt(a) for a in args)