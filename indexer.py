#!/usr/bin/env python3

if __name__ == '__main__':
    import os
    from utils import *
    
    index_file = myinput("Name of the index file: ")
    
    with open(index_file, 'w') as fid:
        root = myinput("Set root dir: ",
            cast=lambda x: x if os.path.isdir(x) else ValueError("{} is not a dir!".format(x)),
            default= '.')
        os.chdir(root)
    
        while True:
            wc = myinput("Wildcard for data collection: ")
            label = myinput("Label name: ")
            
            for f in ifile(wc, recursive=False):
                fid.write("{} {}\r\n".format(f, label))
            
            if myinput("Wanna add more? (y/n): ") is not 'y':
                break