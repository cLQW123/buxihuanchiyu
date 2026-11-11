import os
import pathlib

import ninetoothed.aot as aot


def test_wrap_hip_source_keeps_hip_headers_and_device_state_per_device():
    kernel_name = "test_kernel_hash"
    source = f"""#include <hip/hip_runtime.h>
#include <stdint.h>

#define HSACO_NAME test_kernel_hsaco
hipModule_t {kernel_name}_mod = nullptr;
hipFunction_t {kernel_name}_func = nullptr;
static unsigned char HSACO_NAME[] = {{0}};

void unload_{kernel_name}() {{
    if ({kernel_name}_mod != nullptr) hipModuleUnload({kernel_name}_mod);
}}

void load_{kernel_name}() {{
    hipModuleLoadData(&{kernel_name}_mod, HSACO_NAME);
    hipModuleGetFunction(&{kernel_name}_func, {kernel_name}_mod, "{kernel_name}");
}}

hipError_t {kernel_name}(hipStream_t stream, int *value) {{
    if ({kernel_name}_func == nullptr) load_{kernel_name}();
    return hipModuleLaunchKernel(
        {kernel_name}_func, 1, 1, 1, 64, 1, 1, 0, stream, nullptr, nullptr
    );
}}
"""
    launch = (
        "extern \"C\" NineToothedResult launch_test(NineToothedStream stream) {\n"
        "    int device_id = 0;\n"
        "    hipError_t device_status = hipGetDevice(&device_id);\n"
        "    if (device_status != hipSuccess) "
        "return static_cast<NineToothedResult>(device_status);\n"
        f"    return kernels_test[device_id].{kernel_name}(stream);\n"
        "}"
    )

    wrapped = aot._wrap_hip_source(source, kernel_name, "test", "int", launch)
    struct_start = wrapped.index(f"namespace {kernel_name} {{")

    assert wrapped.index("#include <hip/hip_runtime.h>") < struct_start
    assert wrapped.index("static unsigned char HSACO_NAME[]") < struct_start
    assert wrapped.index(f"hipModule_t {kernel_name}_mod = nullptr;") > struct_start
    assert wrapped.index(f"hipFunction_t {kernel_name}_func = nullptr;") > struct_start
    assert "ThreadSafeUnorderedMap<int" in wrapped
    assert "hipGetDevice(&device_id)" in wrapped


def test_compile_keeps_triton_hip_cpp_launcher(monkeypatch):
    def fake_triton_compile(command, check):
        output_path = pathlib.Path(command[command.index("--out-path") + 1])
        pathlib.Path(f"{output_path}.a1b2c3_0.cpp").write_text("hip source")
        pathlib.Path(f"{output_path}.a1b2c3_0.h").write_text("hip header")

    monkeypatch.setattr(aot.subprocess, "run", fake_triton_compile)
    signature_hash, output_contents = aot._compile(
        pathlib.Path("kernel.py"), "swiglu", "*fp16, *fp16", "1, 1, 1", 4, 3
    )

    assert signature_hash == "a1b2c3_0"
    assert output_contents["swiglu.a1b2c3_0.cpp"] == "hip source"
    assert output_contents["swiglu.a1b2c3_0.h"] == "hip header"


def test_compile_keeps_triton_cuda_c_launcher(monkeypatch):
    def fake_triton_compile(command, check):
        output_path = pathlib.Path(command[command.index("--out-path") + 1])
        pathlib.Path(f"{output_path}.d4e5f6_0.c").write_text("cuda source")
        pathlib.Path(f"{output_path}.d4e5f6_0.h").write_text("cuda header")

    monkeypatch.setattr(aot.subprocess, "run", fake_triton_compile)
    signature_hash, output_contents = aot._compile(
        pathlib.Path("kernel.py"), "swiglu", "*fp16, *fp16", "1, 1, 1", 4, 3
    )

    assert signature_hash == "d4e5f6_0"
    assert output_contents["swiglu.d4e5f6_0.c"] == "cuda source"
    assert output_contents["swiglu.d4e5f6_0.h"] == "cuda header"


def test_compile_library_uses_hipcc_for_hip_backend(monkeypatch, tmp_path):
    (tmp_path / "swiglu.backend").write_text("hip\n")
    (tmp_path / "swiglu.variant.cpp").write_text("hip source")
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(aot.subprocess, "run", fake_run)
    aot._compile_library("swiglu", tmp_path)

    expected_compiler = os.getenv("NINETOOTHED_HIPCC") or os.getenv("HIPCC") or "hipcc"
    assert captured["command"][0] == expected_compiler
    assert "-lcuda" not in captured["command"]
    assert captured["command"][-1] == tmp_path / "swiglu.variant.cpp"
    assert captured["check"] is True


def test_compile_library_keeps_nvcc_for_cuda_backend(monkeypatch, tmp_path):
    (tmp_path / "swiglu.backend").write_text("cuda\n")
    (tmp_path / "swiglu.variant.cpp").write_text("cuda source")
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(aot.subprocess, "run", fake_run)
    aot._compile_library("swiglu", tmp_path)

    assert captured["command"][0] == "nvcc"
    assert "-lcuda" in captured["command"]
    assert captured["check"] is True
