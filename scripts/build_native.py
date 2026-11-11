"""Rebuild changed dispatch units and all InfiniLM sources from this checkout."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig

root = Path(os.environ['SGCUP_ROOT']).resolve()
core = root/'Projects/InfiniCore'
lm = root/'Projects/InfiniLM'
prefix = Path(os.environ['INFINI_ROOT'])
build = root/'build'
objects = build/'objects'
objects.mkdir(parents=True, exist_ok=True)
from ninetoothed._compile_triton import install_hygon_llvm_compatibility
install_hygon_llvm_compatibility()

subprocess.run([sys.executable,str(core/'scripts/build_ntops.py')],check=True)
generated=core/'build/ninetoothed'
ops=('kv_caching','flash_attention','swiglu')
flags=['-O2','-fPIC','-std=c++17','-DENABLE_CPU_API','-DENABLE_HYGON_API',
       '-DENABLE_NINETOOTHED','-DENABLE_CUDNN_API',f'-I{core/"include"}',
       '-I/opt/dtk/include','-I/opt/dtk/cuda/cuda/include']

def compile_source(source, flags):
    obj=objects/('_'.join(source.relative_to(root).parts)+'.o')
    command=['g++',*flags,'-x','c++','-c',str(source),'-o',str(obj)]
    print(json.dumps({'compile':command}),flush=True)
    subprocess.run(command,check=True)
    return obj

sources=[core/f'src/infiniop/ops/{op}/operator.cc' for op in ops]
sources += [generated/f'{op}.c' for op in ops]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    replacements=list(pool.map(lambda source:compile_source(source,flags),sources))
base_objects=sorted((core/'build/.objs/infiniop').rglob('*.o'))
base_objects=[obj for obj in base_objects if not any(f'/ops/{op}/operator.cc.o' in str(obj) for op in ops)]
release=core/'build/linux/x86_64/release'
archives=[release/name for name in ('libinfiniop-cpu.a','libinfiniop-hygon.a','libinfini-utils.a')]
if not base_objects or not all(archive.is_file() for archive in archives):
    raise RuntimeError('The fresh xmake base build is incomplete')
needed=re.findall(r'Shared library: \[(.*?)\]',subprocess.check_output(
    ['readelf','-d',str(release/'libinfiniop.so')],text=True))
linkdirs=[prefix/'lib',Path('/opt/dtk/lib'),Path('/opt/dtk/hip/lib'),Path('/opt/dtk/cuda/cuda/lib64')]
command=['g++','-shared','-Wl,-soname,libinfiniop.so','-Wl,-z,defs',*map(str,base_objects),
         *map(str,replacements),'-Wl,--start-group',*map(str,archives),'-Wl,--end-group',
         *[f'-L{directory}' for directory in linkdirs],*map(str,sorted(generated.glob('*.so'))),
         *[f'-l:{name}' for name in needed],*[f'-Wl,-rpath,{directory}' for directory in (*linkdirs,generated)],
         '-o',str(prefix/'lib/libinfiniop.so')]
print(json.dumps({'link':command}),flush=True)
subprocess.run(command,check=True)

plugin=core/'src/infinicore/ops/multi_head_attention/hygon/mha_ninetoothed_hygon.cc'
command=['g++',*flags,'-shared',str(plugin),'-L'+str(prefix/'lib'),'-linfinicore_cpp_api',
         '-linfiniop','-linfinirt','-Wl,-rpath,'+str(prefix/'lib'),'-o',str(prefix/'lib/mha_ninetoothed.so')]
subprocess.run(command,check=True)

import torch
torch_include=Path(torch.__file__).parent/'include'
includes=[lm/'csrc',lm/'third_party/spdlog/include',lm/'third_party/json/single_include',
          prefix/'include',Path(sysconfig.get_path('include')),torch_include]
flags=['-O2','-fPIC','-std=c++17','-DNDEBUG',*[f'-I{directory}' for directory in includes]]
sources=sorted([*lm.glob('csrc/**/*.cpp'),*lm.glob('csrc/**/*.cc')])
with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.getenv('SGCUP_BUILD_JOBS','32'))) as pool:
    compiled=list(pool.map(lambda source:compile_source(source,flags),sources))
extension=build/'python/_infinilm.so'
extension.parent.mkdir(exist_ok=True)
command=['g++','-shared',*map(str,compiled),'-L'+str(prefix/'lib'),'-linfinicore_cpp_api',
         '-linfiniop','-linfinirt','-linfiniccl','-pthread','-Wl,-rpath,'+str(prefix/'lib'),'-o',str(extension)]
subprocess.run(command,check=True)

import ninetoothed as nt
from infiniop.ops.qwen_rms_norm.ninetoothed.qwen_rms_norm import premake as rms
from infiniop.ops.minimax_h3_rope_cache.ninetoothed.rope_cache import premake as rope
for width in (128,5120):
    output=build/'qwen-rms'
    output.mkdir(exist_ok=True)
    nt.make(*rms(width,nt.bfloat16),caller='cuda',kernel_name=f'sgcup_qwen_rms_v140_{width}',
            output_dir=output,num_warps=4,num_stages=1)
output=build/'h3-rope'
output.mkdir(exist_ok=True)
nt.make(*rope(rope_freq_dim=16,rope_theta=10000.0,position_dtype=nt.float64,dtype=nt.bfloat16),
        caller='cuda',kernel_name='minimax_h3_rope_cache_16_10000_F64_BF16_v65',
        output_dir=output,num_warps=4,num_stages=1)
print('SGCUP_NATIVE_SOURCES_REBUILT',flush=True)
