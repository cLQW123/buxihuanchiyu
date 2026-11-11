# 嵩山杯 Hygon/HIP AOT 适配

此源码副本基于 NineToothed 0.26.0 官方发布源码，保留原 Apache-2.0 `LICENSE`。

Triton HIP AOT 输出 `.cpp` launcher 和 `.h` 声明；NineToothed 0.26.0 的 AOT 包装层原先只查找 `.c`，并以 CUDA context 和 `nvcc` 编译 launcher。本适配保留 HIP 生成的设备代码，通过 `hipGetDevice` 按设备保存模块状态，并使用 `hipcc` 编译 host launcher。可通过 `NINETOOTHED_HIPCC` 或 `HIPCC` 指定编译器路径；未设置时使用 `hipcc`。

从 `Projects/InfiniCore` 目录运行 `python scripts/build_ntops.py` 时，脚本优先加载 `Projects/ninetoothed/src` 和 `Projects/ntops/src` 的本地源码。这样生成的算子与随作品提交的后端补丁保持一致。

已完成本地 Python 语法检查，以及 HIP launcher 包装和 Triton `.cpp/.h` 产物映射的单元检查。尚未在 Hygon 设备上完成编译、算子运行或 FL2VA 验证；正式验证仍须在统一环境运行 124 帧、50 步任务，并检查视频规格与比较脚本结果。
