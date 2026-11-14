#!/usr/bin/env bash
sgcup_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SGCUP_ROOT="$sgcup_root"
export FIRST_FRAME="${FIRST_FRAME:-$sgcup_root/assets/competition-first-frame.png}"
export INFINI_ROOT="${SGCUP_PREFIX:-$sgcup_root/build/install}"
export SGCUP_COMPILER_PYTHONPATH="${SGCUP_COMPILER_PYTHONPATH:-$sgcup_root/build/competition-python}"
# Retain the competition dependency paths while dropping prior model/source
# overlays, including their startup sitecustomize.py and native extensions.
sgcup_dependency_paths=""
IFS=: read -r -a sgcup_inherited_paths <<< "${PYTHONPATH:-}"
for sgcup_path in "${sgcup_inherited_paths[@]}"; do
    case ":${SGCUP_COMPILER_PYTHONPATH:-}:" in
        *":$sgcup_path:"*) continue ;;
    esac
    case "$sgcup_path" in
        ""|*/Projects/InfiniCore|*/Projects/InfiniCore/*|*/Projects/InfiniLM|*/Projects/InfiniLM/*|\
        */Projects/ninetoothed|*/Projects/ninetoothed/*|*/Projects/ntops|*/Projects/ntops/*|\
        */native-python-*|*/sgcup-fixes-*|*/build/python|*/sgcup-*/runtime|\
        "${IFAB_AFS_HOME:-/nonexistent-sgcup-afs}/runtime") continue ;;
    esac
    case ":$sgcup_dependency_paths:" in
        *":$sgcup_path:"*) continue ;;
    esac
    sgcup_dependency_paths="${sgcup_dependency_paths:+$sgcup_dependency_paths:}$sgcup_path"
done
# An explicitly prepared official compiler dependency directory may contain
# Triton when the competition Python environment does not install it globally.
export PYTHONPATH="$sgcup_root/scripts:$sgcup_root/build/python:$sgcup_root/Projects/ninetoothed/src:$sgcup_root/Projects/ntops/src:$sgcup_root/Projects/InfiniCore/src:$sgcup_root/Projects/InfiniCore/python:$sgcup_root/Projects/InfiniLM/python${SGCUP_COMPILER_PYTHONPATH:+:$SGCUP_COMPILER_PYTHONPATH}${sgcup_dependency_paths:+:$sgcup_dependency_paths}"
export LD_LIBRARY_PATH="$INFINI_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NINETOOTHED_HYGON_LLVM_COMPAT=1
export INFINILM_HYGON_NINETOOTHED_FLASH=1
export SGCUP_TP_FP32=0
unset SGCUP_TP_MATMUL_KERNEL_DIR
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TRITON_CACHE_DIR="${SGCUP_CACHE_DIR:-$sgcup_root/build/cache/triton}"
export NINETOOTHED_CACHE_DIR="${NINETOOTHED_CACHE_DIR:-$sgcup_root/build/cache/ninetoothed}"
export SGCUP_H3_ROPE_KERNEL_DIR="$sgcup_root/build/h3-rope"
export SGCUP_QWEN_RMS_KERNEL_DIR="$sgcup_root/build/qwen-rms"
mkdir -p "$TRITON_CACHE_DIR" "$NINETOOTHED_CACHE_DIR"
unset sgcup_path sgcup_inherited_paths sgcup_dependency_paths
