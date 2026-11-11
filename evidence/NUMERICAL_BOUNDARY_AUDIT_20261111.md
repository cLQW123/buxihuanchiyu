# 正式路径数值边界审计（进行中）

2026-11-11。此记录列出尚未闭环的计算，不能作为合规证明。

| 环节 | 当前证据 | 仍需执行的验证 |
|---|---|---|
| 条件编码与H3原生引擎 | v171完整124帧/50配置步实际运行，官方比较92.955% | 最终VAE/辅助计算版本重新生成；核对基础InfiniCore与新增九齿调用边界 |
| 神经视频编码器 | v168 64×64小样通过；规定分辨率/图像内部误差门槛失败 | 保留并诊断误差；最终端到端相似度验证 |
| 神经视频解码器 | v194小样通过；v207/v215 124帧编译断言 | v223记录具体形状后独立复现和回归，重新验证124帧 |
| 神经音频解码器 | v211实际207潜变量帧、165600采样点通过 | 接入最终完整生成并记录实际调用 |
| 输入像素归一化 | encoders.py的encode_visual_condition使用转换、除法、减法 | 分清允许的图像IO与模型归一化，后者使用九齿并对照数值 |
| posterior采样 | encode_visual_condition在vae.encode返回之后调用posterior.sample | 审计随机数来源及mean/std组合，保持seed和舍入顺序；不以VAE方法wrapper覆盖声明其已迁移 |
| 条件潜变量归一化 | encode_visual_condition在CPU上计算(latents-mean)/std；含float16往返 | 按相同舍入语义迁移/验证，避免把宿主数值运算归入调度例外 |
| 迭代调度器更新 | 尚未收齐当前安装版本的scheduler/denoise源文件 | 记录实际数值op，保留50步配置与官方49次H3调用，不缓存历史结果 |
| 输出潜变量反归一化 | decoders.py在vae.decode调用前计算latent*std+mean | 九齿迁移及数值验证；神经VAE方法wrapper尚未涵盖 |
| 输出像素反归一化 | decoders.py在vae.decode返回后进行float转换、乘加、clamp | 确认模型与媒体封装边界并验证九齿表达 |
| 空间/时间位置网格 | before_denoise.py存在NumPy/CPU float64网格、cumsum及算术 | 保持参考浮点构造语义，区分索引组织与模型位置数值计算，逐项记录 |

来源为本地保留的固定Diffusers源码目录`runtime/audit-sources/diffusers-pinned-v19/diffusers/modular_pipelines/minimax_h3/`，以及实际wrapper `runtime/evidence-20261110-v216/runtime/run_strict_vae_official_v212.py`。最终审计还需与实际节点import来源及执行计数比对。

v212只包裹video encode/decode和audio decode，不包裹这些方法前后的归一化、posterior采样或scheduler。因此即使神经VAE全部通过，也不能直接宣称整条推理路径满足实现边界。
