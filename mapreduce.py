import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import pycuda.reduction as reduct
import time
n = 10000
a_gpu = gpuarray.to_gpu(numpy.asarray(range(n), dtype = numpy.int64))

krnl = reduct.ReductionKernel(numpy.int64, neutral="0",
                reduce_expr="a + b", map_expr="x[i] * 2",
                        arguments="long *x")

start1 = time.time()
res1 =  krnl(a_gpu)
stop1 = time.time()
print res1
print ((stop1 - start1)* 1000)

start2 = time.time()

res2 = reduce(lambda x, y: x + y, map(lambda x: x * 2, range(n)))

stop2 = time.time()

print res2

print ((stop2 - start2)*1000)
