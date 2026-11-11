"""NineToothed Conv1D and phase-decomposed ConvTranspose1D candidate."""
import copy,functools
import ninetoothed as nt
import ninetoothed.language as ntl
from ntops.kernels import mm
def arrangement(input,weight,bias,output,precision,*,stride,pad_left,pad_right,dilation,transpose,kernel_last,kernel_period,block_size_m,block_size_n,block_size_k):
 if transpose:weight=weight[:,:,kernel_last::-kernel_period].permute((1,0,2))
 matrix=functools.partial(mm.arrangement,block_size_m=block_size_m,block_size_n=block_size_n,block_size_k=block_size_k)
 padded=input.pad(((0,0),(0,0),(pad_left,pad_right)))
 columns=padded.tile((1,*weight.shape[1:]),strides=(-1,-1,stride),dilation=(1,1,dilation),floor_mode=True)
 columns=columns.squeeze(1);columns.dtype=columns.dtype.squeeze(0)
 columns=columns.ravel().flatten(end_dim=2).flatten(start_dim=1)
 filters=weight.flatten(start_dim=1).permute((1,0))
 result=output.permute((0,2,1)).flatten(end_dim=2)
 affine=bias[None,:,None].expand((output.shape[0],-1,output.shape[2])).permute((0,2,1)).flatten(end_dim=2)
 _,_,affine,_=matrix(copy.deepcopy(columns),copy.deepcopy(filters),affine,copy.deepcopy(precision))
 columns,filters,result,precision=matrix(columns,filters,result,precision)
 return columns,filters,affine,result,precision
def application(input,weight,bias,output,precision):
 accumulator=ntl.zeros(output.shape,dtype=ntl.float32)
 for block in range(input.shape[0]):accumulator+=ntl.dot(input[block],weight[block])
 output=accumulator+bias
def premake(dtype,stride=1,pad_left=0,pad_right=0,dilation=1,transpose=False,kernel_last=0,kernel_period=1,block_size_m=16,block_size_n=16,block_size_k=32,round_before_bias=False):
 if min(stride,dilation,kernel_period,block_size_m,block_size_n,block_size_k)<=0:raise ValueError('Positive convolution geometry required')
 tensors=(nt.Tensor(3,dtype=dtype,other=0),nt.Tensor(3,dtype=dtype,other=0),nt.Tensor(1,dtype=dtype),nt.Tensor(3,dtype=dtype),nt.Tensor(0,constexpr=True,value=2))
 config=dict(stride=stride,pad_left=pad_left,pad_right=pad_right,dilation=dilation,transpose=transpose,kernel_last=kernel_last,kernel_period=kernel_period,block_size_m=block_size_m,block_size_n=block_size_n,block_size_k=block_size_k)
 selected = application_bias_bf16 if round_before_bias and dtype == nt.bfloat16 else application
 return functools.partial(arrangement,**config),selected,tensors

def application_bias_bf16(input,weight,bias,output,precision):
 accumulator=ntl.zeros(output.shape,dtype=ntl.float32)
 for block in range(input.shape[0]):accumulator+=ntl.dot(input[block],weight[block])
 rounded=ntl.cast(accumulator,ntl.bfloat16)
 output=ntl.cast(rounded,ntl.float32)+bias
