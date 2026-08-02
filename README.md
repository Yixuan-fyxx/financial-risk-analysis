# Fin Risk — 金融风险分析系统(Prompt 工程 + RAG 检索增强原型)

一个financial risk分析原型系统,围绕两条主线构建:

1. **Prompt 模板的设计与迭代**:通过优化角色设定、上下文组织、输出格式,提升模型对风险指标解释的准确性,把专业金融分析结果转化为非专业用户能看懂的报告。详见 [`docs/prompt_versions.md`](docs/prompt_versions.md)。
2. **RAG 检索增强**:结合真实的企业公告、新闻资讯等外部公开信息,对风险评分结果进行检索增强,为生成报告的 LLM 提供业务上下文,提升分析结果的时效性、完整性与可解释性。

演示数据用的是 3 家**真实上市/曾上市公司**(而非虚构数据),风险特征分别为低/中/高,数据经 WebSearch/WebFetch 检索、逐条附来源引用后手工整理:

| 公司 | 代码 | 风险特征 | 说明 |
|---|---|---|---|
| 美的集团 | 000333.SZ | 财务稳健 | 低杠杆、盈利增长、评级上调 |
| 海螺水泥 | 600585.SH | 营收连续下滑,但利润率与杠杆仍健康 | 水泥行业价格战/去产能背景下的真实中间态案例 |
| 中国恒大集团 | 3333.HK(已退市) | 严重信用风险 | 资不抵债、审计无法表示意见、香港法院清盘令、强制退市全过程 |

每家公司的财务数据、公告、新闻均在 `data/` 目录下的 JSON 文件里标注了 `source`/`url` 字段,可溯源核实;数据完整性上的缺口(比如某些公开摘要没有披露的科目)也如实标注为缺失,而不是编造数字填充——见下文"数据来源与已知缺口"。

## 系统架构

```
data/companies/*.json          真实财务数据(含逐字段 source 引用)
data/announcements/*.json      真实企业公告(RAG 语料)
data/news/*.json                真实新闻资讯(RAG 语料)
        │
        ▼
risk_scoring/indicators.py     7 个风险指标,缺字段则标记不可用,不假装计算
risk_scoring/scorer.py         按可用指标权重归一化,给出 0-100 风险分与等级
        │
        ▼
rag/retriever.py               TF-IDF + 时效性加权检索(jieba 分词,零外部依赖服务)
        │
        ▼
prompts/templates.py           v1→v4 四版 Prompt 模板(角色/上下文/输出格式递进优化)
        │
        ▼
llm/client.py                  MockLLMClient(离线免费) / AnthropicClient(真实调用)
        │
        ▼
pipeline.py                    串联以上所有环节,产出 ReportResult
examples/run_demo.py           命令行入口
```

### 风险评分引擎:容错优先于假装完整

真实的公开披露信息很少覆盖教科书式比率所需的全部科目(比如年报摘要往往只给资产负债率这个比值,不给流动资产/流动负债明细)。所以 `indicators.py` 里每个指标的计算函数在缺字段时返回 `None`,而不是当成 0 处理;`scorer.py` 汇总时只对**可用**指标的权重做归一化,并把 `data_coverage`(数据覆盖率)显式返回,报告里也会告知用户"多少比例的评估基于可得数据"。这个设计在用真实数据接入美的/海螺/恒大后被反复验证——例如恒大 2021 年报因财务净资产为负,会让 ROE = 净利润/净资产 变成一个误导性的正数(亏损 ÷ 负资产 = 正数),`indicators.py` 里专门加了保护(净资产 ≤ 0 时该指标标记为不可用),而不是让这个数学假象拉低风险分。

### RAG 检索:相关性 + 时效性

`rag/retriever.py` 是一个不依赖外部检索服务的轻量 TF-IDF 实现(jieba 分词 + numpy 余弦相似度),排序时按 `(1-recency_boost) * 相关度 + recency_boost * 时效权重` 加权,而不是纯语义相关度排序——因为金融风险叙事对时效性很敏感,半年前的评级下调远不如上周的公告重要。检索结果连同相似度、时效权重、综合分一起返回,报告里引用的每条 `[证据n]` 都能追溯到具体公告/新闻。

## 环境搭建

项目自带专属虚拟环境 `riskbot/`(已创建并安装好依赖,无需重新 `pip install`)。如果需要在别处重建:

```bash
python -m venv riskbot
riskbot/Scripts/pip install -r requirements.txt   # Windows
# riskbot/bin/pip install -r requirements.txt     # macOS/Linux
```

默认的 `--llm mock` 模式完全离线、免费,不需要任何 API Key。如果要用 `--llm anthropic` 做真实调用:

```bash
cp .env.example .env   # 填入 ANTHROPIC_API_KEY
```

**注意**:`ANTHROPIC_API_KEY`(console.anthropic.com 申请)是独立于 claude.ai / Claude Code 订阅的按量计费账户,即使有 Pro/Max 订阅,用 API Key 调用也会单独计费。

## 快速开始

```bash
# 列出所有可用公司
riskbot/Scripts/python examples/run_demo.py --list

# 生成某家公司的风险报告(默认离线 mock,免费)
riskbot/Scripts/python examples/run_demo.py --company 000333
riskbot/Scripts/python examples/run_demo.py --company 600585
riskbot/Scripts/python examples/run_demo.py --company 3333HK

# 对比不同 Prompt 版本的效果(见 docs/prompt_versions.md)
riskbot/Scripts/python examples/run_demo.py --company 600585 --prompt-version v1

# 用真实 Claude API 生成报告(会产生 API 费用,需先配置 .env)
riskbot/Scripts/python examples/run_demo.py --company 600585 --llm anthropic

# 跑单元测试
riskbot/Scripts/python -m pytest -q
```

## 数据来源与已知缺口

- **美的集团**:2024/2025 年年度报告摘要(深交所巨潮资讯网),公告/新闻另附来源链接。年报摘要未披露流动资产/流动负债/存货等明细,总负债为"总资产 × 官方披露资产负债率"反推得到(非直接披露数字,已在 `data_note` 中注明)。
- **海螺水泥**:2025 年年报及三季报(财经媒体对上交所披露信息的转述整理)。同样缺流动资产/流动负债/存货/短期借款明细;用于计算营收增长率的上年同期营收是根据年报披露的同比增速反推,非直接数字。
- **中国恒大**:2021 年年报(2023-07-17 延迟披露,审计机构出具"无法表示意见"),这是公司进入司法清算前最后一份可获得的完整年报;此后公司已停牌、清盘、退市,未再发布经审计的完整财报,因此本系统只用这一期数据,营收增长率因缺少可比期而标记为不可用。货币资金字段因多个信源数据互相矛盾未采用。

所有报告都会附带免责声明:仅作分析参考,不构成投资建议。

## 项目结构

```
src/fin_risk/
  config.py                路径与常量配置
  data/loader.py           财务数据 & RAG 语料加载
  risk_scoring/            指标计算 + 综合评分
  rag/retriever.py          TF-IDF + 时效性检索
  prompts/templates.py      v1-v4 Prompt 模板与迭代记录
  llm/client.py             Mock / Anthropic 客户端
  agent/tools.py            把业务函数封装成带 JSON Schema 的工具(供下面的训练流程使用)
  pipeline.py               端到端编排
examples/run_demo.py       CLI 演示入口
tests/                     44 个单元测试,覆盖指标计算边界情况、检索排序、模板结构、端到端报告生成、工具封装
docs/prompt_versions.md    Prompt 迭代详细记录与真实样例输出
data/                      真实公司财务数据 + 公告 + 新闻语料
training/                  第二条主线:预训练模型→SFT→Model Merging→RL(DPO)→Agent能力训练全流程,见下文
```

## 训练流程:预训练模型 → SFT → Model Merging → RL → Agent 能力训练

在上面的 Prompt/RAG 原型之上,`training/` 目录完整实现了一遍现代 LLM 对齐流程:用 Qwen2.5-1.5B 预训练 base 模型做 LoRA SFT 学会生成本项目的结构化风险报告,用 mergekit 把 SFT 模型与原始 base 合并以防止过拟合/保留通用能力,用规则验证器构造偏好对做 DPO 偏好优化,最后训练模型学会自主调用 `risk_scoring`/`rag` 等真实业务函数完成多步任务编排(替代原本硬编码的检索逻辑)。全部数据生成、训练脚本、评测工具已经就绪;实际训练需要 GPU,详见 [`training/README.md`](training/README.md)(含云 GPU 租用指南、逐阶段命令、预期产出)。
