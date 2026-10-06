# GenRecAgent：运行与项目讲解

当前版本已经打通数据、召回、工具执行、真实 SFT、对照评测和结果回传。它适合用作可运行、可解释的研究型求职项目，但不是生产推荐系统。默认用固定策略演示，Prompt 和 SFT 保留为独立实验分支。

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

- `outputs/project_delivery/demo/demo_summary.json`：三种场景的商品标题、静态价格、步骤和判定。
- 同目录的 `hybrid.json / cold.json / impossible.json`：完整工具轨迹；对应 `.log` 保留运行记录。
- `outputs/project_delivery/report/results.md`：统一结果表。
- 同目录的 `metrics_summary.json`：原始精度指标和来源文件哈希，方便核对。
- `outputs/tool_sft_lab95_server/`：lab95 已回传的真实 adapter、训练记录和同题评测。

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
python scripts/run_project.py check --policy sft --verify-hashes
python scripts/run_project.py demo --policy sft --scenario hybrid
```

这只是加载 lab95 已训练的 adapter 做本地推理，不在本机训练。原始模型出错时不会偷偷替换为固定策略并算成模型成功。完整小样本对照仍是 Fixed 12/12、Prompt 0/12、SFT 4/12；SFT 的四次通过里仅一次实际填满推荐，三次为正确澄清。

## 一分钟项目讲解

> 我做的是一个面向电商推荐的研究型 Agent。项目使用 Amazon Reviews 2023 的 All_Beauty 数据，保留 112,590 件商品和 8,151 个符合划分要求的用户，先建立热门度和 Dense 召回基线，再接入 Semantic ID 和训练后的 Transformer 生成式召回。
>
> Agent 使用实际工具读取历史、召回、过滤、排序和验证。价格约束、历史排除和数量要求由环境校验，错误会写入观测，执行有明确的步数限制。随后，我使用经过真实工具执行验证的合成任务构造 608 条训练动作，在 lab95 完成一轮 Qwen LoRA SFT，并做了同题对照和恢复、结束校验消融。
>
> 结果并非所有模型都更强。Dense 的测试 Recall@20 为 0.03656，生成式 Transformer 只有 0.002454；原始 SFT 策略也只完成 4/12 个合成任务。所以演示版保留固定策略作为默认路径，完整记录模型失败和训练证据，而不把模型接口接通当成效果提升。

## 简历可以写哪些事实

- 基于 Amazon Reviews 2023 实现离线推荐数据处理与评测，完成 112,590 商品目录上的 Popularity、Dense 和 Semantic ID 生成式召回对照。
- 实现 catalog-valid 约束解码、碰撞候选展开、实际 item-level 双路融合，以及具备价格约束、结果校验和有界恢复的多步工具环境。
- 构造并校验合成工具任务动作数据，在 RTX 3090 上完成 Qwen2.5-1.5B 的 LoRA SFT；保存模型、数据、代码哈希与训练日志，并通过同题评测和消融分析其可靠性限制。

不要写“修复 RQ-VAE 并达到高质量”“生成式召回超过 Dense”“SFT Agent 成功率 100%”“提升线上点击率”。不要把 608 条动作写成 608 条独立用户轨迹，也不要把合成任务写成真实线上用户反馈。BGE、Qwen 和 scikit-learn 是外部组件，不属于原创模型算法。

## 面试时最容易追问的点

1. 为什么生成式召回弱？实际测试与验证诊断指向代码预测能力不足；有效编码约束和碰撞展开解决的是合法性，不保证命中正确商品。
2. 为什么低 SFT loss 不等于可靠 Agent？一步 token 监督与多步执行是不同目标；本次 SFT 仍有重复动作、未完成前置召回、错误澄清等问题，正式结果保留在完整轨迹里。
3. 固定策略 12/12 能说明什么？只说明这组有限合成任务上的执行参考有效，不代表模型自主规划或广泛自然语言理解。
4. 怎样避免泄漏和不公平比较？训练使用训练期历史前缀，热门度来自训练交互；SFT 用户、价格和完整目标组合分组隔离；各策略使用同一批任务、同一工具与 grader。语言模板仍复用，静态目录和按用户时间划分也有局限。
5. 为什么没有继续加训？当前快速交付版先保留冻结结果和可运行流程。后续优化应在独立验证任务上进行，再使用新的 held-out 评测，不根据这 12 个测试任务反复调到通过。

现在优先熟悉演示轨迹、指标定义和这些边界。后续再考虑独立表达任务、学习策略质量、生成式推荐效果和公开发布；当前没有自动发布代码、数据或模型到外部平台。
