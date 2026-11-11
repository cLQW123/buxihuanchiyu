#!/usr/bin/env bash
set -euo pipefail
sgcup_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SGCUP_ROOT="$sgcup_root"
export INFINI_ROOT="${SGCUP_PREFIX:-$sgcup_root/build/install}"
export XMAKE_ROOT=y
sgcup_state=${SGCUP_BUILD_STATE:-$sgcup_root/build}
export XMAKE_GLOBALDIR="$sgcup_state/xmake-global"
export XMAKE_CONFIGDIR="$sgcup_state/xmake-config"
export XMAKE_TMPDIR="$sgcup_state/xmake-tmp"
export NINETOOTHED_HYGON_LLVM_COMPAT=1
export NINETOOTHED_AOT_VARIANT_PROFILE=generic
export INFINICORE_NINETOOTHED_PROFILE=fl2va
export INFINICORE_NINETOOTHED_WORKERS=2
export INFINICORE_NINETOOTHED_NUM_WARPS=4
export INFINICORE_NINETOOTHED_NUM_STAGES=1
export INFINICORE_KV_CACHING_EMB_DIMS=128
export INFINICORE_KV_CACHING_DTYPES=bfloat16
export INFINICORE_NINETOOTHED_OPS=kv_caching,flash_attention,swiglu
export PYTHONPATH="$sgcup_root/Projects/ninetoothed/src:$sgcup_root/Projects/ntops/src:$sgcup_root/Projects/InfiniCore/src:$sgcup_root/Projects/InfiniCore/python:$sgcup_root/Projects/InfiniLM/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$INFINI_ROOT/lib" "$XMAKE_GLOBALDIR" "$XMAKE_CONFIGDIR" "$XMAKE_TMPDIR" "$sgcup_root/build/logs"
python - <<'PY'
from ninetoothed._compile_triton import install_hygon_llvm_compatibility
install_hygon_llvm_compatibility()
import torch
assert torch.cuda.device_count() == 8, 'An exclusive eight-card environment is required'
print('BUILD_DEVICE',torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0),flush=True)
PY
sgcup_xmake=${SGCUP_XMAKE:-xmake}
cd "$sgcup_root/Projects/InfiniCore"
"$sgcup_xmake" global --network=private
"$sgcup_xmake" f -y -m release --hygon-dcu=y --hygon-arch="${SGCUP_ARCH:-gfx936}" \
    --cuda="${HYGON_CUDA_HOME:-/opt/dtk/cuda/cuda}" --ccl=y --aten=y --omp=n --ninetoothed=n
"$sgcup_xmake" build -j "${SGCUP_BUILD_JOBS:-32}"
for target in infinirt infiniop infiniccl infinicore_cpp_api; do
    "$sgcup_xmake" install -y "$target"
done
python "$sgcup_root/scripts/build_native.py"
python "$sgcup_root/scripts/build_python_core.py"
python "$sgcup_root/scripts/validate_aot.py"
source "$sgcup_root/scripts/env.sh"
python "$sgcup_root/scripts/validate_runtime.py"
printf '%s\n' SGCUP_CLEAN_BUILD_COMPLETED
