# 固定版本源码与离线复现

当前经过完整验证的流程见`干净环境复现.md`和仓库README。主入口为`reproduce/validate_offline.sh`：独立venv与缓存、完整源码重建、新生成原版官方参考、连续seed42/43生成、原版SSIM和严格音视频检查。

联网阶段先执行`bash scripts/prepare_sources.sh`和`python scripts/fetch_dependencies.py`，把完整源码和inputs移入离线节点后，使用`setup_offline.py`配置基础环境及inputs内的Triton/xmake，设置MODEL_PATH和FIRST_FRAME，再执行主入口。

如需导出确切公开commit的源码包，可运行：

```bash
python reproduce/prepare_published.py \
  --commit FULL_40_CHARACTER_MAIN_HASH \
  --checkout ../isolated-published-source \
  --archive ../published-source.tar.gz
```

导出包保留逐文件SOURCE_PROVENANCE，不包含旧模型构建库、生成结果、模型权重或统一首帧。离线依赖仍由固定Release资产提供。解压后按同一独立环境流程构建，不复用旧编译缓存或模型扩展。

`run_offline.sh`保留为旧版单例验证入口；本版完整隔离验证与提交复现使用上述主入口。
