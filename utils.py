'''
Helper functions for this project.

Author: Gerald Baulig
'''

#Build in
from time import time
from glob import glob, iglob


def myinput(prompt, default=None, cast=None):
    ''' myinput(prompt, default=None, cast=None) -> arg
    Handle an interactive user input.
    Returns a default value if no input is given.
    Casts or parses the input immediately.
    Loops the input prompt until a valid input is given.
    
    Args:
        prompt: The prompt or help text.
        default: The default value if no input is given.
        cast: A cast or parser function.
    '''
    while True:
        arg = input(prompt)
        if arg == '':
            return default
        elif cast is not None:
            try:
                arg = cast(arg)
                if isinstance(arg, ValueError):
                    print(arg)
            except:
                print("Invalid input type. Try again...")
        else:
            return arg
    pass


def ifile(wildcards, sort=False, recursive=True):
    def sglob(wc):
        if sort:
            return sorted(glob(wc, recursive=recursive))
        else:
            return iglob(wc, recursive=recursive)

    if isinstance(wildcards, str):
        for wc in sglob(wildcards):
            yield wc
    elif isinstance(wildcards, list):
        if sort:
            wildcards = sorted(wildcards)
        for wc in wildcards:
            if any(('*?[' in c) for c in wc):
                for c in sglob(wc):
                    yield c
            else:
                yield wc
    else:
        raise TypeError("wildecards must be string or list.")


class Stopwatch():
    def __init__(self):
        self.start = time()
        self.since = time()
        
    def reset(self):
        elapsed = self.elapsed()
        self.start = time()
        return elapsed
    
    def delta(self):
        delta = time() - self.since
        self.since = time()
        return delta
        
    def elapsed(self):
        return time() - self.start
    