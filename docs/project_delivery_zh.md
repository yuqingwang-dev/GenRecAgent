# GenRecAgent：运行与项目讲解

更新日期：2026-10-12。当前已打通真实数据、学习型 RQ-VAE、Transformer、工具执行、真实 Qwen LoRA SFT、评测和结果回传，适合研究型求职展示。工程闭环完整不等于模型效果领先或生产可用。默认仍是 Fixed；Dense 是主要召回参考，双路预设沿用已冻结的旧 K-Means Transformer 和融合配置，新学习型编码模型作为独立实验保留。

## 先怎么运行

在本机 PowerShell 中执行：

```powershell
cd F:\AIProjects\GenRecAgent
conda activate genrecagent
$env:PYTHONNOUSERSITE = "1"
$env:OMP_NUM_THREADS = "4"
$env:OPENBLAS_NUM_THREADS = "4"
python scripts/run_project.py check
python scripts/run_project.py demo
python scripts/run_project.py report
```

三个命令分别检查资源、运行真实目录演示、汇总已有实验。不会下载模型、启动训练或覆盖旧结果。不传输出目录时，程序会自动使用新的时间戳目录。演示需要 CUDA；报告汇总不需要 CUDA，但需要本地已有的实验记录和 adapter。

已经运行过的交付文件：

- `outputs/final_delivery_20261012/demo/demo_summary.json`：本机本次重新执行的三个 Fixed 场景。
- 同目录的 `hybrid.json / cold.json / impossible.json`：完整工具轨迹；对应 `.log` 保留运行记录。
- `outputs/final_delivery_20261012/report/results.md`：包括历史对照、新 M2/M3 和原始 v3 SFT 的统一表。
- 同目录的 `metrics_summary.json`：原始精度指标和来源文件哈希，方便核对。
- `outputs/tool_sft_recovery_v3_server/`：最新原始 v3 adapter、完整验证及配对测试。
- `outputs/m3_transformer_rqvae_stable_full/`：新学习型编码对应的独立 Transformer，不能与旧编码表混用。
- `outputs/final_delivery_20261012/share/`：审阅 ZIP、离线回放和文件指纹。

旧输出和 v2 adapter 均保留。统一报告在 CPU 上核对权重/输入/源码指纹，重算全部 16,302 次新 M3 候选排名，并核验 v3 的训练选择与 288 次完整执行；产物不齐或配对错误会拒绝报告。

代码仓库不包含大数据、基座模型和 adapter。将来换电脑或只克隆 Git 仓库，应先运行 `check`，按缺失清单补回对应资源；不要把缺少资源误认为模型效果差。已有数据和模型的详细目录见 README 与 SFT 报告。

## 演示时讲什么

默认演示使用真实商品目录，不是 `run_demo.py` 的早期 mock。

| 场景 | 本次运行结果 | 展示重点 |
| --- | --- | --- |
| hybrid | 7 步，5 件商品，静态价格均不超过 $20 | 实际调用 Dense 和训练后的 Transformer，再过滤、排序、校验 |
| cold | 6 步，5 件商品 | 无历史时使用训练期热门度，不伪造个性化依据 |
| impossible | 2 步，澄清，0 件商品 | $0 预算下没有足够合格商品，不放宽约束、不伪造成功 |

可以先展示商品与动作序列，再打开完整 JSON 看每一步的状态、参数、观测与校验。价格来自离线 metadata，不是实时商城报价。三个预设演示通过不等于新增正式 benchmark。

如果要展示训练后的模型分支：

```powershell
python scripts/run_project.py check --policy compact_sft --verify-hashes
python scripts/run_project.py demo --policy compact_sft --scenario hybrid
```

`compact_sft` 加载已回传的原始 v3 adapter，没有隐藏 Fixed 接管；旧 `sft` 仍是 v2。不启动训练或下载。最终本机原始 SFT 复测在 Transformers 模型加载阶段失败，日志保留于 `outputs/final_delivery_20261012/compact_demo/hybrid.log`；不是模型生成失败，也不宣称本机 SFT 已通过。该分支使用稳定 lab95 环境验证，具体最终范围见验收记录。

最新 72 个限定合成测试：Fixed 72/72、同输入 Compact Prompt 0/72、原始 Compact SFT 72/72，后者填满 36/36 可行推荐、正确澄清 36/36 无结果目标。完整验证为 70/72，仍有两次预算失败。输入提供规范化意图、约束和静态资格数量，Prompt 又是未专门优化的弱对照，不能宣称普遍自主规划或相对强 Prompt 的全面优势。历史 v2 的 12 题/SFT 4/12 单独保留，不与 v3 拼成提升百分比。

## 一分钟项目讲解

> 我做的是电商推荐研究型 Agent，在 Amazon Reviews 2023 的 112,590 件商品和 8,151 个用户上建立热门度与 Dense 基线，实际训练 RQ-VAE 编码和 Transformer 生成式召回，并完成全量离线评测。
>
> Agent 用真实工具读取历史、召回、过滤、排序、验证，保留预算、历史排除和有界恢复。在执行过的合成任务上选取 1,080 条训练动作，完成两轮 Qwen2.5-1.5B LoRA SFT，按验证执行选模型，再做同题策略对照。
>
> 工程和效果分开报告。学习型 RQ-VAE 通过最低码本门槛，但新生成式 Recall@20 只有 0.002699，低于 Dense 的 0.036560。原始 SFT 在限定合成测试上通过 72/72，完整验证仍有两次失败。因此保留可靠默认路线，并展示模型失败、训练记录和可复核轨迹，不夸大为线上收益。

## 简历可以写哪些事实

- 基于 Amazon Reviews 2023 实现离线推荐数据处理与评测，完成 112,590 商品目录上的 Popularity、Dense 和 Semantic ID 生成式召回对照。
- 实现 catalog-valid 约束解码、碰撞候选展开、实际 item-level 双路融合，以及具备价格约束、结果校验和有界恢复的多步工具环境。
- 构造并校验合成工具任务动作数据，在 RTX 3090 上完成 Qwen2.5-1.5B 的 LoRA SFT；保存模型、数据、代码哈希与训练日志，并通过同题评测和消融分析其可靠性限制。

可以写“学习型 RQ-VAE 通过预设最低门槛”，不能写“全面优于聚类或实现高质量推荐”。不要写“生成式召回超过 Dense”“Agent 普遍成功率 100%”“提升线上点击率”。1,080 条选取动作只有 1,035 条互异，不是独立用户轨迹；合成任务不是线上反馈。BGE、Qwen、PEFT、PCA 和 scikit-learn 是外部组件。

## 面试时最容易追问的点

1. 为什么生成式召回弱？实际测试与验证诊断指向代码预测能力不足；有效编码约束和碰撞展开解决的是合法性，不保证命中正确商品。
2. 为什么测试 72/72 不等于可靠 Agent？验证为 70/72，且测试提供规范化状态，不覆盖广泛表达和未知目录探索。
3. 为什么默认 Fixed？它是低成本的执行参考，不能把回放当作原始 SFT 实时推理或推荐效果证明。
4. 评测边界是什么？训练期前缀/热门度、验证选模型、v3 用户/模板/价格隔离；旧检索测试已观察，不声称新的独立 holdout。
5. 剩余限制是什么？生成式相关性弱、价格静态、超时不是硬中断、Windows 模型加载不稳定，均保留证据而非宣称解决。

现在优先熟悉演示轨迹、指标定义和这些边界。后续再考虑独立表达任务、学习策略质量、生成式推荐效果和公开发布；当前没有自动发布代码、数据或模型到外部平台。
