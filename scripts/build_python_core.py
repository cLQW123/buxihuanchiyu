"""Build InfiniCore's Python binding from the submitted source tree."""
import concurrent.futures
import os
from pathlib import Path
import subprocess
import sysconfig
import torch

root=Path(os.environ['SGCUP_ROOT'])
core=root/'Projects/InfiniCore'
prefix=Path(os.environ['INFINI_ROOT'])
objects=root/'build/python-core-objects'
objects.mkdir(exist_ok=True)
torch_include=Path(torch.__file__).parent/'include'
includes=[core/'include',core/'third_party/spdlog/include',
          core/'src/infinicore/pybind11',torch_include,
          Path(sysconfig.get_path('include'))]
assert (core/'third_party/spdlog/include/spdlog/cfg/env.h').is_file(), \
    'Prepare the pinned InfiniCore spdlog submodule before building'
flags=['-O2','-fPIC','-std=c++17','-DNDEBUG','-DBOOST_STACKTRACE_USE_NOOP',
       *[f'-I{p}' for p in includes]]
def compile_source(source):
    obj=objects/(source.stem+'.o')
    subprocess.run(['g++',*flags,'-c',str(source),'-o',str(obj)],check=True)
    print('CORE_BINDING_COMPILED',source.name,flush=True)
    return obj
sources=sorted((core/'src/infinicore/pybind11').rglob('*.cc'))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    compiled=list(pool.map(compile_source,sources))
output=core/'python/infinicore/lib/_infinicore.so'
output.parent.mkdir(exist_ok=True)
subprocess.run(['g++','-shared',*map(str,compiled),'-L'+str(prefix/'lib'),
                '-linfinicore_cpp_api','-linfiniop','-linfinirt','-linfiniccl',
                '-Wl,-rpath,'+str(prefix/'lib'),'-o',str(output)],check=True)
print('CORE_PYTHON_BINDING_REBUILT',str(output),flush=True)
