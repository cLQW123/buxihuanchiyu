"""Run the Triton AOT compiler with opt-in Hygon LLVM text compatibility.

Some DTK backends emit GEP no-wrap flags that their assembly-side LLVM parser
cannot read. Dropping these optimization promises preserves defined address
calculations. This adapter only affects NineToothed's compiler subprocess.
"""

import functools
import os
import re
import runpy


def strip_gep_wrap_flags(source):
    return re.sub(
        r"(\bgetelementptr(?:\s+inbounds)?)(?:\s+(?:nuw|nusw))+\s+",
        r"\1 ",
        source,
    )


def install_hygon_llvm_compatibility():
    from triton.backends.amd.compiler import HIPBackend

    original_amdgcn = HIPBackend.make_amdgcn
    original_hash = HIPBackend.hash
    original_parse_options = HIPBackend.parse_options

    @functools.wraps(original_parse_options)
    def parse_options(self, options):
        options = dict(options)
        matrix_size = os.getenv("NINETOOTHED_HIP_MATRIX_INSTR_NONKDIM")
        if matrix_size:
            matrix_size = int(matrix_size)
            if matrix_size not in (16, 32):
                raise ValueError("HIP matrix instruction size must be 16 or 32")
            options["matrix_instr_nonkdim"] = matrix_size
        return original_parse_options(self, options)

    HIPBackend.parse_options = parse_options

    @functools.wraps(original_amdgcn)
    def make_amdgcn(source, metadata, options):
        return original_amdgcn(strip_gep_wrap_flags(source), metadata, options)

    @functools.wraps(original_hash)
    def backend_hash(self):
        return f"{original_hash(self)}-ninetoothed-hygon-gep-v1"

    HIPBackend.make_amdgcn = staticmethod(make_amdgcn)
    HIPBackend.hash = backend_hash


if __name__ == "__main__":
    if os.getenv("NINETOOTHED_HYGON_LLVM_COMPAT") == "1":
        install_hygon_llvm_compatibility()
    runpy.run_module("triton.tools.compile", run_name="__main__", alter_sys=True)
