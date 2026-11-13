# 最终固定 FL2VA 路径的实现边界

2026年11月14日在新目录`ssb-cjh-isolated-20261114-04`完成全部离线流程，计算源码集成提交为`b166ade6736dc0a514c845a47e1b02630a625d00`。构建、原版参考生成、seed42/43连续生成、原版比较和两例媒体检查均退出0。最终SSIM为92.371%，规格一致。

## 源码和运行路径

InfiniCore为`4a89f7f0c20b24e3edf44fc249156341fcf0a1b3`，InfiniLM为`71a34004bf93ed11a6fc3ec060b6543021dbe7aa`，NineToothed为`db28777b0e4cba1e76db9c47579879bbf4021c61`；NTops为未修改的官方`82cba6cf2200919b5d12c86adbea60cbfb516040`。全部修改见对应公开分支及patches。构建开始前逐文件核对6208个源码文件，结束后再次核对，未发现源码变化。

| 环节 | 数值路径与边界 | 实测证据 |
| --- | --- | --- |
| Qwen3-VL及H3原生模型 | InfiniLM模型、InfiniCore原有GEMM和通信；新Attention、KV缓存、SwiGLU、连续RoPE和Qwen RMSNorm走九齿 | 当前源码重建的原生模块、库和AOT SHA256；八卡88项原生分发、RoPE 6项、RMSNorm 4项；完整生成每例H3 49次 |
| 神经视频编码 | `NineToothedVAEEncoderMode`拦截卷积、归一化、激活、算术和浮点坐标，未知数值运算报错 | 两例`AutoencoderKLMiniMaxH3.encode`完成；卷积680次、GroupNorm 500次，完整输入928×768 |
| 神经视频解码 | `NineToothedVAEVideoMode`拦截线性、Attention、归一化及继承的编码模式算术 | 两例完整decode完成；输出各124帧，严格媒体字段与新参考一致 |
| 神经音频解码 | `NineToothedVAEAudioMode`的九齿Conv1D、转置卷积、深度卷积、weight norm及算术 | 两例完整audio decode完成；32 kHz双声道AAC，各解码165888采样点 |
| 像素、潜变量归一化及后处理 | 模型归一化和反归一化由九齿pointwise实现；媒体文件读写和封装为辅助组件 | 辅助对照中的实际画布像素测试；完整生成boundary-calls及vae-ops记录 |
| posterior与初始噪声 | 九齿MT19937流和正态变换，读写CPU生成器状态仅作为状态字节组织 | 三种浮点dtype和规定视频潜变量规模的对照；生成器状态逐字节一致；seed42/43独立完整生成 |
| 空间、时间及音频位置 | 浮点位置由九齿生成，整数行索引和模态标签为布局元数据；VAE浮点arange不再走Torch数值构造 | 规定31帧潜变量、58×48布局及其他规模对照；36项浮点范围逐值一致 |
| VAE初始化常量 | 九齿旋转频率与Kaiser-sinc滤波器表达；宿主标量用于静态配置 | 实际12 taps及24/48 taps、48/96维旋转频率对照 |
| 迭代调度 | 九齿生成50点sigma网格、scale_noise及Euler更新；终止零点没有模型调用 | FP32/FP16/BF16各49次更新对照；每例H3 49次，音视频scheduler分别49次，共98次 |

232项辅助对照与98项原生/AOT对照全部通过。该数量表示具体受测用例，不代表任意形状、dtype、任务或未来模型配置均已证明。

## 基础组件和执行来源

权重读取、分词、图像读写与缩放、存储视图、宿主调度、媒体封装和官方评测使用声明的基础组件。严格神经VAE中的浮点运算通过九齿表达执行；整数行索引、常数存储初始化和复制不承担模型数值核。NTops仅提供九齿排布表达，未直接调用其Torch计算实现。

正式入口固定`INFINILM_HYGON_NINETOOTHED_FLASH=1`和`SGCUP_TP_FP32=0`。Attention使用本工程九齿AOT及注册插件；张量并行线性层沿用InfiniCore GEMM和allreduce。厂商FlashAttention wheel保留既有动态链接依赖，正式Attention路径由上述开关和插件选择。九齿经其Triton编译链生成HIP目标代码，新增模型算子没有手写Triton/CUDA旁路或内部调用第三方现成模型计算内核。

`run.json`记录实际argv、八卡数量、源码模块路径和本次原生/AOT库哈希。所有模型扩展及AOT库来自本次新构建目录；独立venv不继承系统site-packages，明确链接统一环境厂商Torch，随包Triton和全部离线wheel有独立来源及哈希。缓存仅包含本工程编译产物，不保存或复用历史模型输出。

## 参考及限制

新参考直接执行原版官方`minimax_h3.py --legacy`；入口SHA256为`ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4`，原版比较脚本SHA256为`e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590`。Diffusers固定官方提交`d5baa4fb548294f47dbca49890abd4b291204c60`，1022个Python文件逐字节核对；参考没有自行修改的精度补丁或参赛数值适配器。

本次范围是固定Base FL2VA、统一首帧、默认画布、124帧及50个调度点。额外128 taps滤波器探索未达到误差门槛；固定模型实际使用12 taps，其对照通过。自定义配置和其他任务不属于本次验证结论。完整流程记录技术检查结果，赛方最终验收与成绩尚不在本工程记录内。

seed42程序端到端1456.045秒，seed43为1277.743秒；同配置参考为595.849秒，当前尚未取得端到端加速。第一次包括冷编译开销。实际配置、日志、逐阶段退出码、视频和来源记录随最终Release资产提供。
