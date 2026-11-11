"""BF16 NT dot products retained in FP32 for TP reduction."""
import functools
import ninetoothed as nt
import ninetoothed.language as ntl
from ntops.kernels.mm import arrangement
def application(input,weight,output,precision):
 accumulator=ntl.zeros(output.shape,dtype=ntl.float32)
 for block in range(input.shape[0]):accumulator+=ntl.dot(input[block],weight[block])
 output=accumulator
def premake():
 tensors=(nt.Tensor(2,dtype=nt.bfloat16,other=0),nt.Tensor(2,dtype=nt.bfloat16,other=0),nt.Tensor(2,dtype=nt.float32),nt.Tensor(0,constexpr=True,value=2))
 return functools.partial(arrangement,block_size_m=32,block_size_n=32,block_size_k=32),application,tensors
