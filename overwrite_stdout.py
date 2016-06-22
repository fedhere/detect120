from __future__ import print_function
import sys
import time

for i in range(100):
    print("\rThis i is {0}".format(i),  end="")
    sys.stdout.flush()
    time.sleep(1)
print("")
