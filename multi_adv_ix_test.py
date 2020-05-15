import sparse
import inspect
import numpy as np
import time
from sparse._utils import assert_eq, random_value_array

x = sparse.random((20, 20, 20, 20), density=.5)

ii = np.random.randint(0,20,(5000))
index = ([0,1,16],[12,0,1],[12,10,4])

#r1 = x[index]
t1 = time.time()
#r1 = x[ii][:,ii]
#r1 = x[ii,ii,ii]
#r1 = x[ii,97:,:3]
r1 = x[index]
t2 = time.time()
print('te: {}'.format(t2-t1))
print(r1.shape)

print(r1.min(),r1.max(),r1.sum())

d = x.todense()
t1 = time.time()
d = d[index]
t2 = time.time()
print('te2: {}'.format(t2-t1))
print(d.shape)
print(d.min(),d.max(),d.sum())


print(np.where(d))
print(np.where(r1.todense()))

print(assert_eq(d, r1))
