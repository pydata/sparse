import sparse
import inspect
import numpy as np
import time
from sparse._utils import assert_eq, random_value_array


#([0, 1],) * 3
#index = ([0, 1],) * 2
index = ([0,1],[0,1])
#index = ([0,1])

print(index)

s = sparse.random((3, 3, 3), density=0.5)
x= s.todense()

r1 = x[index]
r2 = s[index]


print(r1.min(),r1.max(),r1.sum())
print(r2.min(),r2.max(),r2.sum())
print(r2.data)
print(r2.coords)

print(x)
print('\n\n\n')
print(r1)
print('\n\n\n')
print(r2.todense())

#print(assert_eq(x[index], s[index]))

