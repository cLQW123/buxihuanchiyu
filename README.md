# MiniMax H3 FL2VA 八卡适配

本工程以比赛指定 InfiniCore `minimax-h3`、InfiniLM `minimax_h3` 代码基础及 NineToothed 0.26.0 构建首帧视频生成路径。具体公开提交在 `source-lock.json` 中固定。权重和比赛数据由环境提供，不包含在仓库内。

本次新增独立venv、73个固定wheel的哈希清单、可配置的DTK／GCC路径和独立九齿缓存。完整新流程见`reproduce/干净环境复现.md`，依赖边界见`dependencies/离线依赖说明.md`。这些新增部署代码当前仍在干净环境验证，不能把旧版生成结果作为本版已验证的证明。

正确性状态需区分：此前完整八卡FL2VA输出与精度兼容参考的SSIM为92.148%，规格一致。当前已找到并实际测试官方Diffusers提交`d5baa4fb548294f47dbca49890abd4b291204c60`（仍为文档指定的`0.40.0.dev0`），原版`--legacy`入口已完成规定生成，后续以该未经自行修改的参考验证新部署版本。最终SSIM、连续运行及干净构建结果以对应实测日志为准。

## 环境

使用赛事原生独占 8 卡 Hygon gfx936 环境、DTK/HIP 6.3、Python 3.11、PyTorch 2.9、Triton 3.5.1。Diffusers、Transformers、NTops 及离线头文件来自比赛允许的依赖。运行前激活比赛 Python 环境并 `source /opt/dtk/env.sh`。保留平台网络配置；生成过程设置 Hugging Face 离线模式。

Triton 必须使用赛事厂商提供的版本。构建前用 `prepare_compiler_dependency.py --triton-source` 指定官方 `triton/` 包目录。脚本只读取版本，在本工程 `build/competition-python/` 中创建指向官方 `triton/`、伴随 `triton.libs/` 和匹配 `triton-3.5.1*.dist-info/` 的链接，保留厂商动态库与版本元数据，不修改源码；`env.sh` 默认使用这个独立依赖入口。即使官方包原来存放在旧任务 `runtime/` 下，也不会加入其父目录的启动补丁或模型源码。使用其他独立官方依赖目录时可显式设置 `SGCUP_COMPILER_PYTHONPATH`。

NTops 只提供九齿算子的张量排布表达，计算经提交的 NineToothed 编译链产生；未手写 Triton 或 CUDA 设备内核。Torch 用于权重加载、存储、视图和媒体辅助组件，神经 VAE 的数值算子以严格 dispatch 模式执行，不支持的数值运算直接报错。

`scripts/env.sh` 固定 `INFINILM_HYGON_NINETOOTHED_FLASH=1`、`SGCUP_TP_FP32=0`。Attention 使用本工程九齿实现及 `mha_ninetoothed.so` 注册插件；张量并行线性层沿用原 InfiniCore GEMM 后接 allreduce，未启用实验性 FP32 累加桥接。入口记录有效开关、Python 来源和本次构建的基础库、插件与 AOT 库 SHA256，缺失产物或外部路径会立即报错。环境脚本移除旧工程的源码、原生扩展及启动补丁路径，保留比赛依赖路径；无需继承之前任务的模型环境。

## 构建

将公开提交检出到 `Projects/InfiniCore`、`Projects/InfiniLM`、`Projects/ninetoothed`，NTops 离线源码置于 `Projects/ntops`。InfiniCore 需要固定版 spdlog 头文件；InfiniLM 需要 spdlog、nlohmann/json 头文件。可在有网的准备阶段初始化上游仓库固定的子模块，随后离线构建。

```bash
source /path/to/competition/venv/bin/activate
source /opt/dtk/env.sh
python scripts/prepare_compiler_dependency.py --triton-source /path/to/competition/official-triton/triton
source scripts/env.sh
SGCUP_XMAKE=/path/to/xmake bash scripts/build.sh
source scripts/env.sh
```

脚本在本工程 `build/` 内建立独立安装目录，重建 InfiniCore 基础库、九齿 AOT 算子、InfiniLM 与 InfiniCore Python 扩展。禁止使用之前任务遗留的模型扩展或计算库替代本工程产物。

## 运行与检查

```bash
source scripts/env.sh
bash scripts/run.sh fl2va \
  --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" \
  --text 'A dog falls asleep' --gpus 0,1,2,3,4,5,6,7 \
  --num-frames 124 --steps 50 --seed 42 --output outputs/generated.mp4
python Projects/InfiniLM/examples/minimax_h3_compare.py \
  --tasks fl2va --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" \
  --reference-fl2va-video "$REFERENCE_VIDEO" \
  --infinilm-fl2va-video outputs/generated.mp4 --output-dir outputs/comparison
python scripts/continuous.py --model-path "$MODEL_PATH" \
  --first-frame "$FIRST_FRAME" --output-dir outputs/continuous --seeds 42,43
```

50 个指定调度点包含终止零点，按未修改的官方调度语义执行 49 次模型更新。每个用例重新执行规定模型计算，保存实际输出、完整日志、退出码、耗时及哈希。连续测试失败会立即返回非零并保留失败记录。

## 修改范围

InfiniLM 包含 H3 原生推理与多卡同步、Qwen RMSNorm 桥接、严格神经 VAE、随机数与采样、归一化、调度器、位置布局及初始化缓冲区。InfiniCore 包含新增九齿卷积、归一化、矩阵乘法、Attention、KV/SwiGLU 调度、连续 RoPE 与 VAE 辅助算子。NineToothed 包含 HIP AOT launcher、平台编译兼容和标量指针的广播修复。

图像读取和缩放、分词及音视频文件封装沿用原许可组件；神经网络视频解码使用九齿算子。原模型权重、统一样例、官方入口和比较脚本保持不变。数值偏差、失败记录、公开源码版本和最终相似度须以报告及实测证据为准。
