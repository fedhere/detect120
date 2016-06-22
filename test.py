from __future__ import print_function 

import time

# status generator
def range_with_status(total):
    """ iterate from 0 to total and show progress in console """
    n=0
    while n<total:
        done = 'n#'*(n+1)
        todo = '-'*(total-n-1)
        s = '<{0}>'.format(done+todo)
        if not todo:
            s+='\n'        
        if n>0:
            s = '\r'+s
        print(s, end='')
        yield n
        n+=1

# example for use of status generator
for i in range_with_status(10):
    time.sleep(0.1)
