# 训练流程:预训练模型 → SFT → Model Merging → RL(DPO) → Agent 能力训练

这是 `fin_risk` 项目的第二条主线:在"Prompt 工程 + RAG"原型（根目录 README）之上,完整跑一遍现代 LLM 对齐流程的五个阶段,全部紧扣同一个业务——生成结构化、有证据引用、不编造缺失数据的金融风险报告,并让模型学会自己调用 `risk_scoring`/`rag` 这些真实业务函数做多步编排。

```
Qwen2.5-1.5B (base, 未微调)
    │  Stage 1: LoRA SFT ──教会它按 v4 五段式结构写报告
    ▼
stage1_sft (LoRA adapter) ──merge_and_unload──▶ stage1_sft_merged (满血权重)
    │  Stage 2: mergekit 把 stage1_sft_merged 和原始 base 按 linear/TIES/SLERP 合并
    ▼                                            ──防止 SFT 在小数据集上过拟合/遗忘通用能力
stage2_merged (满血权重)
    │  Stage 3: DPO ──偏好对来自规则验证器(report_verifier)打分的 rejection sampling
    ▼
stage3_dpo (LoRA) ──merge──▶ stage3_dpo_merged (满血权重)
    │  Stage 4: LoRA SFT on ReAct 工具调用轨迹 ──教会模型自己决定何时调用哪个工具
    ▼
stage4_agent (LoRA) ──merge──▶ stage4_agent_merged (满血权重, 最终产物)
    │
    ▼
agent_runtime.py ──真正跑通:模型输出 Action → 执行 fin_risk.agent.tools 里的真实函数 → 回填 Observation → 循环 → Final 报告
```

每一步"为什么这么设计"的详细论证写在对应脚本的模块 docstring 里（如 `stage3_dpo/train_dpo.py` 顶部解释了为什么用 DPO 不用 PPO），这里只给"怎么跑"和"预期数字"。

## 算力:租一张卡

本地只有 CPU 跑不动,推荐按小时租云 GPU。选 **单卡 RTX 3090/4090(24GB 显存)**,选一个预装 `PyTorch 2.x + CUDA 12.x` 的镜像,足够跑通全部四个训练阶段。

不在中国大陆使用(比如没有中国大陆手机号/支付宝微信支付)的话,[AutoDL](https://www.autodl.com/) 注册和充值会比较麻烦,更适合用国际信用卡直接付款的平台:

- **[RunPod](https://www.runpod.io/)** ——邮箱+信用卡注册,几分钟搞定,不需要实名/中国手机号。Community Cloud 的 RTX 4090 大概 $0.3-0.5/小时,有官方 PyTorch 镜像,支持 Jupyter 或 SSH 连接,部署简单,是目前海外用户最常用的平台之一。
- **[Vast.ai](https://vast.ai/)** ——全球算力市场(个人卖家挂机器出租),价格通常是几个平台里最便宜的,但机器质量/网络参差不齐,需要自己挑评分高、位置离你近的实例。适合对价格敏感、愿意自己排查问题的情况。
- 如果你人在中国大陆或者有支付宝/微信支付和中国大陆手机号,[AutoDL](https://www.autodl.com/) 依然是不错的选择:界面全中文、有大量社区教程、按小时计费很便宜。

用完记得关机/释放实例,按量计费,不用了就停,避免空跑扣费。

预计总耗时(1.5B 模型,数据规模见下文"数据集规模"一节):SFT 数十分钟,Merge 几分钟,DPO 数十分钟(取决于采样 K 值),Agent SFT 数十分钟,合计大概 2-4 小时算力,单卡 3090/4090 大概几块到十几块钱人民币。

## 环境准备(在租的实例上)

```bash
git clone <your-repo-url> fin_risk && cd fin_risk
pip install -r requirements.txt
pip install -r training/requirements-train.txt

# 如果从国内访问 HuggingFace 慢/连不上,配置镜像:
export HF_ENDPOINT=https://hf-mirror.com

export PYTHONPATH=src   # 本项目的包在 src/ 下,不是通过 pip install -e . 安装的
```

## 数据生成(CPU 即可,本地或云端都行,不花钱)

```bash
python -m training.data_gen.build_sft_dataset --n-per-company 40 --out training/datasets/sft.jsonl
python -m training.data_gen.build_agent_trajectories --n-per-company 15 --out training/datasets/agent_sft.jsonl
```

`--n-per-company` 是每家公司生成的（真实公司financials + 若干扰动后的counterfactual财务快照）样本数——只有 3 家真实公司,靠这个把数据集从 3 条扩到几十~上百条,细节和取舍写在 `training/data_gen/synthetic_companies.py` 的模块 docstring 里。DPO 偏好对（`build_preference_dataset.py`）要等 Stage 2 有 checkpoint 之后才能跑,见下文。

## Stage 1: SFT

```bash
python -m training.stage1_sft.train_sft --config training/configs/sft.yaml
```
产出:`training/outputs/stage1_sft`(LoRA adapter)。

## Stage 2: Model Merging

```bash
python -m training.stage2_merge.run_merge --config training/configs/merge.yaml
python -m training.stage2_merge.eval_general_capability \
    --sft-merged-model training/outputs/stage1_sft_merged \
    --merged-model training/outputs/stage2_merged
```
产出:`training/outputs/stage2_merged`(满血权重)+ `training/outputs/stage2_merge_eval.json`(base / SFT / merged 三者的"通用能力 loss vs 领域 verifier 分数"对比表——**这是"Model Merging 解决了过拟合"这条简历表述的证据**)。`configs/merge.yaml` 里默认用 `linear`(基础模型/SFT模型各 50% 权重平均),`ties`/`slerp` 的备选配置也写在同一个文件里,值得都跑一遍对比。

## Stage 3: RL Pipeline(DPO)

```bash
python -m training.data_gen.build_preference_dataset \
    --model-path training/outputs/stage2_merged --k 4 --out training/datasets/dpo_pairs.jsonl
python -m training.stage3_dpo.train_dpo --config training/configs/dpo.yaml
```
第一条命令对 Stage 2 模型在同一 prompt 上采样 K 个回答,用 `report_verifier`(格式合规/证据真实性/免责声明等规则打分)选最高分/最低分组成偏好对——细节和为什么不训单独的奖励模型网络,见 `build_preference_dataset.py`/`train_dpo.py` 顶部。产出:`training/outputs/stage3_dpo_merged`。

## Stage 4: Agent 能力训练

```bash
python -m training.stage4_agent.train_agent_sft --config training/configs/agent_sft.yaml

# 跑通真实的多步工具调用(会实际执行 fin_risk.agent.tools 里的函数):
python -m training.stage4_agent.agent_runtime --model-path training/outputs/stage4_agent_merged --company-id 600585
```
产出:`training/outputs/stage4_agent_merged`——**这是整个 pipeline 最终交付的模型**。`agent_runtime.py` 打印出的完整 Thought/Action/Observation/Final 轨迹,就是"模型自己决定调用哪个工具、拿到真实返回值、再继续推理"的直接证据,建议把这段输出截图/保存下来。

## 最终评测

```bash
python -m training.eval.run_eval --out training/outputs/final_eval.json
```
对 base/stage1_sft/stage2_merged/stage3_dpo/stage4_agent 五个 checkpoint 跑同一批 held-out 用例,输出对比表:通用能力 loss、领域 verifier 分数、(仅 stage4)任务完成率与工具调用后的报告质量。**这张表是简历里"量化结果"部分最直接的素材**。

## 数据集规模与已知局限(诚实写明,别在简历里假装是工业级规模)

- 只有 3 家真实公司,SFT/Agent 数据集靠财务数值扰动(`synthetic_companies.py`)扩充到几十~上百条,RAG 证据始终是真实公告/新闻,不编造文本——但样本多样性终究有限,不是几万条量级的数据集。
- RL 阶段用 DPO 而非完整 PPO,理由和取舍见 `stage3_dpo/train_dpo.py` docstring。
- 通用能力评测用仓库内置的一小组手写 prompt（`training/eval/general_probe.jsonl`),不是标准 benchmark(如 C-Eval),零外部依赖但严谨性有限。
- Held-out 评测目前只覆盖 3 家真实公司在不同 `--as-of-date` 下的表现,没有覆盖训练时用了不同随机种子的合成财务快照——这是一个诚实的后续改进方向,而不是被隐藏的缺陷。

把这些局限性写进简历项目描述里(比如"在数据规模受限的场景下完整实现了..."),比夸大规模更可信,也更能体现工程判断力。

## 简历怎么写(建议框架)

按阶段对应一条 bullet,每条都挂一个具体产出/数字,而不是"训练了一个模型":

- **SFT**:基于 Qwen2.5-1.5B 预训练模型,用规则化"教师"(确定性风险评分+报告渲染逻辑)蒸馏生成 SFT 数据,微调出能按五段式结构、附证据引用生成金融风险报告的模型。
- **Model Merging**:用 mergekit 将 SFT 模型与原始 base 模型按 linear/TIES 合并,通用能力 loss 从 X 降到 Y(相比纯 SFT 模型),领域任务分数仅从 A 降到 B——量化展示了合并对"防止过拟合/保留通用能力"的效果。
- **RL(DPO)**:设计基于规则验证器(格式合规、证据真实性、缺失数据诚实披露)的自动化偏好对构造流程,替代人工标注,用 DPO 做偏好优化,领域 verifier 分数从 X 提升到 Y。
- **Agent 能力训练**:将业务函数封装为带 JSON Schema 的工具,构造 ReAct 风格专家示范轨迹做行为克隆,训练模型自主完成"查询风险指标→检索证据→生成报告"的多步工具编排,任务完成率 X%,替代了原有硬编码的检索决策逻辑。

把 `training/outputs/*_eval.json` 里跑出来的真实数字填进 X/Y——这些数字应该是你自己跑出来的,不是我编的占位符。
