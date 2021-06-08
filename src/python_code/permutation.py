#!/usr/bin/env python3

import sys
import numpy as np

try:
    size = int(sys.argv[1])
except:
    print("Error you must give an int as argument")
    print("permutation.py <size> <num>")
    exit()

try:
    num = int(sys.argv[2])
    assert num < size, ""
except:
    print("Error you must give an int as argument lower than size")
    print("permutation.py <size> <num>")
    exit()

aux = [i for i in range(num+1, size)]
# aux.remove(num)

np.save(sys.argv[3], aux)
