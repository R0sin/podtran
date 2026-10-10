# 发布 podtran

正式版本由维护者准备，推送 `vX.Y.Z` 标签后由 GitHub Actions 验证、构建并自动公开 GitHub Release。仅支持正式版本标签，从接入此流程后的下一个版本开始，不补发历史版本。

## 准备与发布

1. 在 `CHANGELOG.md` 顶部添加 `## [X.Y.Z] - YYYY-MM-DD` 章节，用中文记录用户可感知的变化。
2. 更新 `src/podtran/__init__.py` 的 `__version__`，并运行 `uv lock` 检查锁文件与项目依赖一致。
3. 运行 `uv run ruff check src tests scripts` 和 `uv run pytest -q`，审阅并提交本次发布内容，合入 `main`。
4. 在确认的发布提交上创建标签，然后推送分支和该标签。例如下列版本号仅为示例，请替换为实际版本：

```powershell
git tag -a v0.4.1 -m "Release v0.4.1"
git push origin main
git push origin v0.4.1
```

工作流自身必须已经包含在标签对应的提交中。不要移动或覆盖已有版本标签。仓库中的流程不依赖维护者本机的 `.codex` 技能。

## 自动检查与产物

- 标签必须为 `vX.Y.Z`，与包版本一致，并存在唯一且非空的对应 changelog 章节。
- Windows / Python 3.11，以及 Linux / Python 3.10、3.11、3.12 运行单元测试；Linux / Python 3.11 运行 lint。
- Windows、Linux 各自用 `uv build --no-sources` 构建 sdist，再从 sdist 构建 wheel；在仓库外的干净环境安装 wheel，验证包版本和 CLI 入口。
- CI 只安装轻量依赖，不调用在线服务、不下载模型。这验证打包和 Python 代码兼容性，不代表完整 WhisperX、Qwen、PyTorch 环境或 GPU 推理已经验证。
- 发布 Linux 构建出的 wheel 和 sdist，说明取自该版本的中文 changelog。发布 job 是唯一具有仓库写权限的 job。

正常分支和 PR 的 CI 也执行这些检查，以便在打标签前发现问题。

GitHub CLI 上传附件时先创建草稿，上传完成后自动公开，不需要手动审核。Release 附件是版本归档和可选安装来源；README 继续推荐 Git 默认分支安装及 `uv tool upgrade podtran`，不要求现有用户换渠道。默认分支可能包含尚未正式发布的提交。

## 失败与重跑

- 临时网络或 runner 故障：在 Actions 中重跑失败的 job。
- 上传中断留下草稿：在 Releases 删除该草稿，**保留 Git 标签**，然后重跑发布 job。流程不会自动覆盖已有 Release。
- 代码、版本或 changelog 校验失败：修复并提交，使用新的版本号和标签。重跑旧标签仍会使用旧提交，不会包含修复。
- 已公开的 Release：重复创建会失败，不替换附件、不移动标签；代码修复发布新版本。

本流程不发布 PyPI、不制作含模型依赖的独立可执行文件，也不更改用户配置或任务产物。
