import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.reduction as reduct
import pycuda.gpuarray as gpuarray
import numpy
import time
n = 1000

gpuSum = reduct.ReductionKernel(numpy.int32, neutral="0", reduce_expr="a+b",map_expr="1 << x[i]",  arguments="long* x")
a_gpu = gpuarray.to_gpu(numpy.asarray(range(n), dtype = numpy.int64))

t0 = time.time()
krnl = reduct.ReductionKernel(numpy.int64, neutral="0",
                reduce_expr="x[i] + x[i+1]", map_expr="x[i]",
                        arguments="long *x")



t1 = time.time()
#res = krnl(a_gpu).get()
t5 = time.time()

print gpuarray.max(a_gpu).get()

t6 = time.time()


print '%0.6f' % ((t6 - t5)*1000)



print '%0.6f' % ((t1 - t0)*1000)

t2 = time.time()

sum = sum(range(n))
#maxcpu = max(range(n))

t3 = time.time()

print sum


print '%0.6f' % ((t3 - t2)*1000)
