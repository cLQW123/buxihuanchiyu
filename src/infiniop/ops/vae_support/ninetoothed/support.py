"""NineToothed VAE conversion, padding and weight norm candidates."""
import functools
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Symbol
from ntops.kernels.element_wise import arrangement as elementwise
from ntops.kernels.reduction import arrangement as reduction
def pointwise(input,other,output,operation,lower,upper):
 x=ntl.cast(input,ntl.float32);y=ntl.cast(other,ntl.float32)
 if operation==0:output=input
 elif operation==1:output=x+y
 elif operation==2:output=x*y
 elif operation==3:output=x-y
 elif operation==4:output=x/y
 elif operation==5:output=y-x
 elif operation==6:output=ntl.exp(x)
 elif operation==7:output=ntl.sin(x)
 elif operation==8:output=ntl.cos(x)
 elif operation==9:output=-x
 elif operation==10:output=x*ntl.sigmoid(x)
 elif operation==11:output=x*x
 elif operation==12:output=1.0/x
 elif operation==13:output=ntl.minimum(ntl.maximum(x,lower),upper)
def premake_pointwise(ndim,operation,input_dtype=None,other_dtype=None,output_dtype=None,lower=-float('inf'),upper=float('inf')):
 if not 0<=operation<=13:raise ValueError('Unknown VAE pointwise operation')
 tensors=(nt.Tensor(ndim,dtype=input_dtype),nt.Tensor(ndim,dtype=other_dtype),nt.Tensor(ndim,dtype=output_dtype),nt.Tensor(0,constexpr=True,value=operation),nt.Tensor(0,constexpr=True,value=lower),nt.Tensor(0,constexpr=True,value=upper))
 return functools.partial(elementwise,block_size=512),pointwise,tensors
def padding_arrangement(input,output,*,padding,mode):
 if mode=='constant':padded=input.pad(padding)
 else:
  input._inputs.append([])
  def offsets(indices):
   mapped=[]
   for index,size,(before,after) in zip(indices,input.shape,padding):
    if before==0 and after==0:mapped.append(index);continue
    index=index-before
    if mode=='replicate':mapped.append(ntl.call('minimum',ntl.call('maximum',index,0),size-1))
    else:
     period=2*(size-1);folded=(index+period)%period;mapped.append(ntl.call('minimum',folded,period-folded))
   return (tuple(mapped),)
  padded=nt.Tensor(shape=tuple(size+sum(p) for size,p in zip(input.shape,padding)),dtype=input.dtype,source=input.source,target_dims=input.target_dims,_offsets=offsets,_outputs=[input._inputs[0]])
  input._levels.append([padded])
 return elementwise(padded,output,block_size=512)
def copy_application(input,output):output=input
def premake_padding(ndim,padding,mode,dtype):
 if mode not in ('constant','reflect','replicate') or len(padding)!=ndim:raise ValueError('Unsupported padding')
 return functools.partial(padding_arrangement,padding=padding,mode=mode),copy_application,(nt.Tensor(ndim,dtype=dtype,other=0),nt.Tensor(ndim,dtype=dtype))
def weight_norm_arrangement(input,gain,output,*,dim,block_size):
 expanded=gain.expand(tuple(-1 if i==dim else size for i,size in enumerate(input.shape)))
 return reduction(input,expanded,output,dim=tuple(i for i in range(input.ndim) if i!=dim),block_size=block_size)
def weight_norm_application(input,gain,output):
 squares=ntl.zeros(input.dtype.shape,dtype=ntl.float32)
 for block in range(input.shape[0]):
  x=ntl.cast(input[block],ntl.float32);squares+=x*x
 inverse_norm=ntl.rsqrt(ntl.sum(squares,0))
 for block in range(input.shape[0]):output[block]=ntl.cast(input[block],ntl.float32)*gain[block]*inverse_norm
def premake_weight_norm(ndim,dim=0,dtype=None):
 if not 0<=dim<ndim or ndim<2:raise ValueError('Invalid retained dimension')
 tensors=(nt.Tensor(ndim,dtype=dtype,other=0),nt.Tensor(ndim,dtype=dtype),nt.Tensor(ndim,dtype=dtype))
 return functools.partial(weight_norm_arrangement,dim=dim,block_size=512),weight_norm_application,tensors
