"""NineToothed Qwen RMSNorm with two reference rounding boundaries."""
import functools
import ninetoothed as nt
import ninetoothed.language as ntl
def arrangement(input,weight,output,eps,*,width):
 def rows(tensor):
  x=tensor.tile((1,width));x.dtype=x.dtype.squeeze(0);return x
 affine=weight[None,:].expand((input.shape[0],-1))
 return rows(input),rows(affine),rows(output),eps
def application(input,weight,output,eps):
 values=ntl.cast(input,ntl.float32);variance=ntl.sum(values*values,0)/input.source.shape[-1];inverse_rms=ntl.rsqrt(variance+ntl.cast(eps,ntl.float32));normalized=(values*inverse_rms).to(input.dtype);output=ntl.cast(normalized,ntl.float32)*ntl.cast(weight,ntl.float32)
def premake(width,dtype):
 tensors=(nt.Tensor(shape=(None,width),dtype=dtype,other=0),nt.Tensor(shape=(width,),dtype=dtype,other=0),nt.Tensor(shape=(None,width),dtype=dtype),nt.Tensor(0,dtype=nt.float64))
 return functools.partial(arrangement,width=width),application,tensors
