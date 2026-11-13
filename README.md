# MiniMax H3 FL2VA 八卡适配

2026年11月14日在赛事f12r2n14独占八卡Hygon gfx936节点完成全新隔离环境验证。源码重建、330项数值与运行时检查、原版官方参考生成、seed42/43两例连续完整生成、官方SSIM比较及严格音视频规格检查全部通过，所有阶段退出0。最终SSIM为**92.371%**，两例均为768×928、124帧、24 FPS、H.264/yuv420p及32 kHz双声道AAC。

实际完整运行的集成提交为`b166ade6736dc0a514c845a47e1b02630a625d00`；本次发布后续仅更新说明、验证证据和依赖资产元数据，参与构建和推理的代码保持一致。组件分支及完整commit由`source-lock.json`固定。最终日志、配置、退出码、输出示例和依赖包见GitHub Release `offline-runtime-20261114`。最终评测由赛事方统一执行，本仓库列出实测技术结果。

## 准备源码和离线依赖

在联网准备机器执行：

```bash
git clone https://github.com/cLQW123/buxihuanchiyu.git
cd buxihuanchiyu
bash scripts/prepare_sources.sh
python scripts/fetch_dependencies.py
```

`prepare_sources.sh`检出固定的InfiniCore、InfiniLM、NineToothed、NTops及必需头文件子模块，核对源码清单并写入`SOURCE_PROVENANCE.json`。下载脚本获取73个固定wheel、完整厂商Triton包及本次实测xmake工具，按`dependencies/artifacts.json`逐项校验SHA256。保留各依赖中的许可证。也可通过`--archive-dir /path/to/assets --offline`校验已有资产。

将完整源码目录和`inputs/`一起移入比赛离线节点。统一基础环境提供Linux x86_64、CPython 3.11、厂商Torch 2.9.0+das.opt1.dtk2604、DTK 26.04、GCC 11.4.0和驱动；其他Python依赖以及xmake随资产提供。固定模型权重和统一首帧由赛事路径读取，不随工程重复打包。

## 创建独立环境

在已分配的独占八卡节点使用比赛Python执行。下列基础环境路径需替换为当前平台实际路径：

```bash
/path/to/competition/python3.11 scripts/setup_offline.py \
  --wheelhouse "$PWD/inputs/wheelhouse" \
  --vendor-torch-site /path/to/vendor/site-packages \
  --vendor-triton-site "$PWD/inputs/vendor-triton" \
  --dtk-root /path/to/dtk \
  --gcc-root /path/to/gcc \
  --xmake-root "$PWD/inputs/vendor-xmake"
```

脚本建立`build/venv`，不继承系统site-packages，只显式引入统一环境的厂商Torch和随包Triton。安装前检查全部wheel哈希和依赖闭包，离线安装后执行`pip check`。新源码目录使用独立的原生安装、九齿、Triton、xmake和临时文件目录。实际Python为3.11.15，厂商Triton为3.5.1+das.opt1.dtk2604.torch290，xmake为3.1.2。

## 完整构建与验证

```bash
export MODEL_PATH=/path/to/MiniMax-H3/diffusers
export FIRST_FRAME=/path/to/competition-first-frame.png
unset REFERENCE_VIDEO REFERENCE_PROVENANCE
bash reproduce/validate_offline.sh
```

入口从源码重建InfiniCore基础库、九齿AOT算子、Attention注册插件及InfiniLM/InfiniCore Python扩展，完成6项RoPE、4项Qwen RMSNorm、88项八卡原生分发和232项辅助计算对照。随后在独立子进程中运行未修改的官方`--legacy`入口，新生成参考视频，再连续生成seed42、43两例，运行原版比较脚本和两例严格媒体检查。参数固定为`A dog falls asleep`、8卡、124帧、50个调度点及默认尺寸；50个点含终止零点，实际每例执行49次H3更新。每例重新执行模型计算。

如赛方另有固定参考，可同时设置`REFERENCE_VIDEO`与`REFERENCE_PROVENANCE`；来源记录必须明确区分赛方标准、原版官方执行和兼容参考。所有阶段日志、PID、退出码、计时、参数和来源证明写入`outputs/offline-validation/`，成功后生成`validation.json`。

## 实现与范围

InfiniLM负责Qwen3-VL条件编码、H3模型、权重加载及八卡调度，InfiniCore提供原有GEMM、运行时和通信。新增或重写的Attention、KV缓存、SwiGLU、RoPE、Qwen RMSNorm、卷积、归一化、神经VAE、随机采样、浮点坐标与调度算术均由NineToothed编译链生成。严格VAE分发对不支持的数值计算直接报错。整数行索引、存储视图、文件读取、分词和媒体封装使用基础辅助组件。

`scripts/env.sh`固定`INFINILM_HYGON_NINETOOTHED_FLASH=1`和`SGCUP_TP_FP32=0`，原生库与AOT来源在运行前检查。Attention使用本工程九齿实现；张量并行线性层沿用InfiniCore GEMM及allreduce。NTops仅提供九齿排布表达。原始权重、统一输入、官方入口和比较脚本保持不变。

Diffusers固定官方`0.40.0.dev0`提交`d5baa4fb548294f47dbca49890abd4b291204c60`，其1022个Python文件与公开源码一致，包含VAE精度衔接的官方修复。最终参考没有自行精度补丁。

本轮程序端到端计时：seed42为1456.045秒，seed43为1277.743秒，官方参考为595.849秒；当前尚无端到端加速。第一次包括冷编译开销，两例进程墙钟时间另有记录。实现覆盖固定Base FL2VA与首帧锚点；模型实际12 taps滤波器及24/48 taps对照通过，额外128 taps探索未达到误差门槛。完整实现边界与限制见`evidence/NUMERICAL_BOUNDARY_AUDIT_20261114.md`，复现细节见`reproduce/干净环境复现.md`。
