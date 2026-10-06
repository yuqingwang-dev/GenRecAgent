# 本地分享包与演示回放

当前只准备本地交付，不自动发布到 GitHub 或其他平台。

## 生成

确认源码变更已经审核并提交，且已有 `outputs/project_delivery/demo/` 的三种 Fixed 演示与正式实验记录，再运行：

```powershell
conda activate genrecagent
$env:PYTHONNOUSERSITE = "1"
python scripts/package_project.py
```

程序要求干净的工作树，输出到新的 `outputs/share/` 时间戳目录，不覆盖旧包，也不启动训练或下载。

## 交付文件

- `GenRecAgent_review_bundle.zip`：轻量源码、文档、结果摘要和离线回放。
- `demo_replay.html`：双击即可查看三种场景、逐步回放和最终商品；不需要 GPU、网络或服务器。
- `share_manifest.json`：源码提交号、包内文件哈希、脱敏清单和审查边界。
- `package_receipt.json`：压缩包 SHA256 与大小。

解压后先看 `START_HERE.md`。简历草稿在 `docs/resume_project_zh.md`，正式结果表在 `evidence/results.md`。

HTML 是已保存真实工具执行结果的动画回放，不是实时模型推理，不是 MP4 录屏，也不是新的实验。回放的 Fixed 成功演示不能替代正式模型对照；SFT 仍是 4/12，只有一次填满可行推荐。价格是离线 metadata，不是实时价格。

## 本次验证范围

Python 侧验证凭据扫描、路径脱敏和 script 注入防护；生成 ZIP 后逐项核对文件哈希与排除目录。页面以独立 Edge headless 配置做过初始渲染检查，不操作用户已有浏览器窗口。交互逻辑可用本机已有 Node 做开发测试：

```powershell
node scripts/check_replay.cjs outputs/project_share/demo_replay.html
```

测试执行页面原有 JavaScript，验证场景切换、下一步、播放、暂停、重置以及结束前不展示最终商品。该测试使用轻量模拟 DOM，不冒充真实鼠标操作或自动浏览器交互。查看实际页面时直接双击生成的 HTML。

## 哪些不在包里

不会收集模型权重、完整数据、缓存、训练日志、原始用户历史、Git 历史或真实环境配置。可执行源码保持原始字节；只在包内文档副本中将本机路径、私有服务器地址与账号路径替换为占位符。原始项目文件不变。

静态凭据扫描只报告文件、行号和问题类别，不输出疑似凭据。检测到非空密钥配置、常见令牌、私钥或被追踪的大模型/原始数据时拒绝出包。

这个检查只针对本次本地源码与结果摘要快照，不认证完整 Git 历史已经安全，也不替代第三方数据、模型与素材分发条款审查。将来公开仓库、上传模型或分享更多数据，需要单独确认范围和审查；当前没有执行任何外部发布。
