# 第四阶段：使用与工程 —— 详解

> 读完这篇，你会从"会用 ChatGPT 聊天"变成"能用 API 搭出自己的 AI 应用"。
> 预计阅读时间：50-70 分钟。

---

## 目录

1. [概览：模型训好了，然后呢？](#概览模型训好了然后呢)
2. [4.1 API 调用：把 AI 嵌入你的程序](#41-api-调用把-ai-嵌入你的程序)
   - [2026 年主流 API 提供商](#2026-年主流-api-提供商)
   - [核心参数详解](#核心参数详解)
   - [流式输出（Streaming）](#流式输出streaming)
   - [Token 计费](#token-计费)
   - [动手：命令行 AI 翻译工具](#动手命令行-ai-翻译工具)
3. [4.2 Prompt Engineering（提示工程）](#42-prompt-engineering提示工程)
   - [Zero-shot vs Few-shot](#zero-shot-vs-few-shot)
   - [Chain-of-Thought（思维链）](#chain-of-thought思维链)
   - [结构化输出](#结构化输出)
   - [System Prompt 设计](#system-prompt-设计)
   - [Prompt Engineering 的边界](#prompt-engineering-的边界)
4. [4.3 RAG（检索增强生成）](#43-rag检索增强生成)
   - [RAG 全景架构](#rag-全景架构)
   - [分块策略（Chunking）](#分块策略chunking)
   - [嵌入与向量数据库](#嵌入与向量数据库)
   - [检索质量优化](#检索质量优化)
   - [动手：本地文档问答系统](#动手本地文档问答系统)
5. [4.4 本地部署](#44-本地部署)
   - [推理框架对比](#推理框架对比)
   - [量化格式选型](#量化格式选型)
   - [显卡选型指南](#显卡选型指南)
   - [动手：Ollama 跑本地模型](#动手ollama-跑本地模型)
6. [本阶段小结](#本阶段小结)

---

## 概览：模型训好了，然后呢？

前三阶段我们把一个 LLM 从裸架构一路训成了能对话的 AI 助手。现在要回答的问题是：**我怎么用它？**

答案分三条路：

| 使用方式 | 门槛 | 灵活度 | 成本 | 适合场景 |
|---------|------|--------|------|---------|
| 网页/App 聊天 | 零门槛 | 低 | 免费到 $20/月 | 日常使用、写作辅助 |
| API 调用 | 会写代码 | 高 | 按量付费 | 自动化、产品集成 |
| 本地部署 | 有显卡 | 最高 | 硬件成本 | 隐私敏感、离线需求 |

这一阶段重点讲后两种——**API 调用**和**本地部署**，以及"怎么让模型回答得更好"的通用技术：Prompt Engineering 和 RAG。

---

## 4.1 API 调用：把 AI 嵌入你的程序

### 2026 年主流 API 提供商

| 提供商 | 代表模型 | 中文能力 | 价格（百万 token） | 特点 |
|--------|---------|---------|-------------------|------|
| DeepSeek | DeepSeek-V4 Pro | 极好（中文母语） | $0.44/输入, $0.87/输出 | 性价比之王，开源，价格仅为 Claude 的 1/50 |
| 阿里通义千问 | Qwen3 | 极好 | ~$0.50/输入, ~$2/输出 | 中文理解最细腻 |
| 智谱 | GLM-5 | 极好 | ~$0.50/输入, ~$2/输出 | 国产自主，学术背景硬 |
| Moonshot | Kimi K3 | 极好 | $3/输入, $15/输出 | 超长上下文（512K+），长文本性价比最高 |
| OpenAI | GPT-5.x | 优秀 | ~$5/输入, ~$15/输出 | 生态最成熟，Function Calling 强 |
| Anthropic | Claude Fable 5 | 优秀 | $10/输入, $50/输出 | 长上下文（200K+），安全性最好，价格最高 |

> 数据来源：各平台 2026 年 7 月官方定价。**DeepSeek 和 Claude 的输出价格差约 57 倍**——同样的任务，选对供应商成本天差地别。选 API 不只是比价格——**长文本用 Claude/Kimi，中文优先 Qwen/DeepSeek，生态兼容选 OpenAI**。价格下降是大趋势，但各家的降幅不同步，建议关注官方定价页的最新数据。

所有主流 API 都兼容 OpenAI 的请求格式。换一家提供商，改个 `base_url` 和 `api_key` 就行，代码不用动：

```python
# 通用范式（OpenAI 兼容 SDK）
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1"  # 换成任何兼容的地址
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
    max_tokens=1024,
)

print(response.choices[0].message.content)
```

### 核心参数详解

调用 API 不是简单地"发一句话，收一句话"。几个关键参数直接影响输出的质量和风格：

#### temperature（温度，0~2，默认 1）

温度控制输出的**随机性**。把它理解成"创造力旋钮"：

```
temperature = 0   →  每次都选概率最高的词 → 结果完全一致（适合翻译、分类、代码生成）
temperature = 0.7 →  有一定随机性 → 每次输出略有不同（适合对话、写作）
temperature = 1.5 →  概率分布被"拉平" → 模型胡言乱语概率大增
```

底层原理很简单：输出层算出每个词的概率后，在应用 Softmax 之前把分数除以 temperature。temperature < 1 让高分更高（"赢家通吃"），temperature > 1 让分数差距变小（"大家都有机会"）。

所以 **temperature = 0 不是绝对确定**——有浮点精度的原因，极端情况下可能仍有微小波动。需要绝对确定时，有些 API 支持设置 `seed` 参数。

#### top_p（核采样，0~1，默认 1）

另一种控制随机性的方式。模型把所有候选词按概率从高到低排序，只从"累积概率达到 top_p"的那一小撮词里挑。top_p=0.1 意味着只考虑概率最高的几个词（非常保守），top_p=0.9 意味着考虑大多数词。

```
top_p = 0.1 → 只从"最高概率、加起来占 10%"的那几个词中选 → 极保守
top_p = 0.9 → 从"加起来占 90%"的词中选 → 宽松
top_p = 1.0 → 所有词都有机会 → 不限制
```

实践中 **temperature 和 top_p 通常只调一个**，不要两个同时大改。

#### max_tokens（最大输出长度）

注意：这个限制的是**输出 token 数量**，不是输入+输出的总和。设太少话没说完就会截断，设太多会增加成本（输出 token 通常比输入贵）。

#### stop sequences（停止符）

告诉模型"遇到这个就停"。比如想生成一个 JSON 最后不要有多余文字：

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    stop=["\n\n", "###"],  # 遇到空行或 ### 就停
)
```

#### system prompt vs user prompt

```
┌─────────────────────────────┐
│  system prompt（系统提示）    │  ← 设定角色、规则、输出格式（一次性设定）
├─────────────────────────────┤
│  user: "帮我写一首诗"        │  ← 具体问题
│  assistant: "明月几时有..."  │  ← 模型回答
│  user: "换成五言绝句"        │  ← 追问
│  assistant: "窗前明月光..."  │
└─────────────────────────────┘
```

System prompt 是"一次性设定"，放在对话最前面，模型对它的遵从度很高。User prompt 是具体的问题和追问。

### 流式输出（Streaming）

体验过 ChatGPT "一个字一个字往外蹦"的效果吗？那就是流式输出（Streaming）。

非流式：把所有 token 一次性生成完 → 返回完整回答。用户要等几秒甚至几十秒。
流式：生成一个 token → 立刻返回 → 再生成下一个 → 立刻返回。用户看到的是"边想边写"。

代码只需要加一个参数：

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "给我讲个故事"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

内部原理和第三阶段讲的"自回归生成 + KV Cache"完全一致——每生成一个 token 就立即输出，然后把新的 KV 缓存拼上，继续生成下一个。

### Token 计费

API 按 token 收费，不是按字数。**中文一个字约 2 个 token，英文一个单词约 1.3 个 token。** 这意味着同样的信息量，中文 API 成本是英文的 1.5-2 倍。

粗略估算公式：

```
成本 = (输入Token数 × 输入单价 + 输出Token数 × 输出单价)
```

输入比输出便宜，因为输入只需要前向传播一次（所有 token 可以并行处理），输出是逐个 token 串行生成的。

举个例子：用 DeepSeek 翻译一篇 5000 字的英文文章：
- 输入 ~7K token（5000 英文词 × 1.3）+ 系统提示
- 输出 ~10K token（5000 中文字 × 2）
- 成本 ≈ 7K × $0.44/M + 10K × $0.87/M ≈ $0.003 + $0.009 = **不到 1.5 美分**

日常使用花不了多少钱。大规模商用才需要在意 token 优化。

### 动手：命令行 AI 翻译工具

20 行代码写一个中英互译命令行工具：

```python
#!/usr/bin/env python3
"""ai_translate.py — 命令行翻译工具"""
import sys
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1",
)

text = sys.argv[1] if len(sys.argv) > 1 else input("请输入要翻译的内容：")

response = client.chat.completions.create(
    model="deepseek-chat",
    temperature=0.1,  # 低温度确保翻译稳定
    messages=[
        {"role": "system", "content": "你是专业翻译。中文→英文时输出英文，英文→中文时输出中文。只输出翻译结果，不加任何解释。"},
        {"role": "user", "content": text},
    ],
)

print(response.choices[0].message.content)
```

用法：`python ai_translate.py "今天天气真好"` → 输出英文翻译。

---

## 4.2 Prompt Engineering（提示工程）

Prompt Engineering 的本质是：**模型已经有这个能力，你要做的是通过措辞把它"激活"出来。** 它不是"教模型新知识"，而是"告诉模型用哪种模式回复"。

### Zero-shot vs Few-shot

**Zero-shot**：直接问，不给例子。

```
判断以下句子的情感（正面/负面）：
"这家店的服务态度太差了"
```

模型有时候对，有时候错——它可能不知道你要什么格式。

**Few-shot**：先给几个例子，再问。

```
判断以下句子的情感（正面/负面）：

示例：
"我非常喜欢这个产品" → 正面
"等了两个小时还没上菜" → 负面
"质量不错，价格也合理" → 正面

现在判断：
"这家店的服务态度太差了" → 
```

模型看到你给的例子，立刻懂了——哦，只需要输出一个词。Few-shot 就是"打样"，告诉模型你期望的格式和风格。2-3 个例子通常就够了。

### Chain-of-Thought（思维链）

核心技巧就一句话：**"让我们一步一步思考。"**

对比一下效果。直接问：

```
问：小明有 5 个苹果，他给了小红 2 个，又买了 3 个，然后吃掉了 1 个，
最后把剩下的苹果平均分给了 2 个朋友。每个朋友得到几个？

模型直接回答：2.5  →  错了
```

加上 CoT：

```
问：小明有 5 个苹果，他给了小红 2 个，又买了 3 个，然后吃掉了 1 个，
最后把剩下的苹果平均分给了 2 个朋友。每个朋友得到几个？

让我们一步一步思考：
```

模型会输出：
```
第 1 步：小明一开始有 5 个苹果
第 2 步：给了小红 2 个，剩余 5-2=3 个
第 3 步：又买了 3 个，剩余 3+3=6 个
第 4 步：吃掉 1 个，剩余 6-1=5 个
第 5 步：平均分给 2 个朋友，5÷2=2.5 个
答案：每人 2.5 个
```

为什么有效？自回归生成是逐 token 产生的——前面的 token 是后面 token 的"上下文"。如果模型直接跳到"2.5"，中间没有推理步骤，公式计算容易出错。但把它拆成"第 1 步……第 2 步……"，每一步都是简单计算，错误率大幅下降。

**CoT 的变体：**

- **Zero-shot CoT**：不需要给例子，只在 prompt 末尾加一句"让我们一步一步思考"
- **Few-shot CoT**：给几个带推理步骤的例子（在示例里展示"正确回答 + 推理过程"）
- **Auto-CoT**：让模型自己生成中间步骤（"你是怎么想的？写出来"）
- **Self-Consistency**：让模型推理 5 次，取最频繁出现的答案（用 temperature > 0 多次采样再投票）

**什么时候 CoT 有用？** 数学推理、逻辑题、多步规划——需要多步推导的任务。什么时候没用？简单的信息查询（"法国的首都是哪"）——不需要推理，CoT 只是浪费 token。

### 结构化输出

大多数业务场景需要的是**结构化数据**，不是一段散文。两种做法：

**做法一：Prompt 里要求 JSON**

```
把以下会议记录整理成 JSON，包含以下字段：日期、议题、参与人、决议。
只输出 JSON，不要任何解释。

会议记录：2026年7月20日，技术部周会，参加人有张三李四王五，讨论了Q3版本排期，
决定将发布日期从8月1日推迟到8月15日。
```

模型输出：
```json
{
  "日期": "2026-07-20",
  "议题": "Q3版本排期",
  "参与人": ["张三", "李四", "王五"],
  "决议": "发布日期从8月1日推迟到8月15日"
}
```

90% 的情况这就够了。但偶尔模型会多输出一个"好的，这是 JSON："前缀，或漏一个闭合括号。

**做法二：Function Calling / Tool Use（更可靠）**

告诉模型"你可以调用以下函数"，模型会输出结构化的函数名+参数。API 层面保证格式正确：

```python
tools = [{
    "type": "function",
    "function": {
        "name": "extract_meeting_info",
        "description": "提取会议关键信息",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "topic": {"type": "string"},
                "participants": {"type": "array", "items": {"type": "string"}},
                "decision": {"type": "string"},
            },
            "required": ["date", "topic", "participants", "decision"],
        },
    },
}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "会议记录：..."}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_meeting_info"}},
)

# response.choices[0].message.tool_calls[0].function.arguments
# → '{"date":"2026-07-20","topic":"Q3版本排期",...}'  格式100%合法
```

Function Calling 比 prompt 约束更可靠，因为它在推理时对不符合 schema 的 token 直接截断重试。需要确定性的生产场景用 Function Calling，快速原型用 prompt 约束就够了。

### System Prompt 设计

优秀的 system prompt 分四层：

```
第 1 层：角色定位（"你是谁"）
第 2 层：能力边界（"你能做什么/不能做什么"）
第 3 层：输出格式（"怎么回复"）
第 4 层：示例（"好的回复长这样"）
```

实例——一个"代码审查助手"的 system prompt：

```
你是一位资深代码审查专家。你的任务是：
1. 检查代码中的 bug、安全漏洞、性能问题
2. 给出具体的修改建议和理由
3. 对好的代码片段也要指出值得学习的地方

回复格式：每个问题用 ### 标出，包含「问题类型」「所在行」「具体描述」「修改建议」四个字段。

不要做的事：
- 不审查代码风格（缩进、命名风格等）
- 不确定的问题标注"可能"，不要假装确定
- 发现严重的 SQL 注入 / 任意代码执行漏洞，在开头用 [严重] 标出
```

### Prompt Engineering 的边界

有几种情况 **prompt engineering 救不了**：

| 情况 | 为什么无效 | 正确做法 |
|------|-----------|---------|
| 模型知识截止后的事实 | 训练数据里没有，prompt 编不出来 | 用 RAG 喂外部知识 |
| 复杂的多步推理 | 单次前向传播的计算量有限 | 拆成多轮对话 + CoT |
| 长文档精确对比 | 上下文太长模型注意力衰减 | RAG 检索出相关段落再对比 |
| 需要外部工具 | 模型不能调 API、读网页、运行代码 | Function Calling + Agent |
| 模型本身的幻觉倾向 | 底层机制决定的 | 用低温度 + 事实核查流程 |

一句话总结 prompt engineering 的哲学：**能通过调 temperature 和 system prompt 解决的问题就别上 RAG，能通过 RAG 和 few-shot 解决的问题就别微调。**

---

## 4.3 RAG（检索增强生成）

### RAG 全景架构

LLM 的知识冻结在训练截止日期。问它"今天天气怎么样"它会编一个。RAG 的思路：**先检索相关文档，把文档当参考资料一起喂给模型。**

![RAG 架构](/images/rag_architecture.png)

整体流程分五个阶段：

```
文档     →  分块     →  嵌入     →  存入向量数据库
用户问题  →  嵌入     →  检索相似块  →  拼入 prompt  →  LLM 生成回答
```

简单说就是：先把你的文档库"预处理"成可检索的形式，用户提问时搜出最相关的片段，塞进 prompt 里让模型基于这些片段回答。

> **2026 年更新：RAG 没有被长上下文取代。** 2025-2026 年多篇研究对比了 RAG 和直接将全文塞入长上下文模型的效果。结论是：RAG 在"大海捞针"类精确检索任务上保持 82.58% 的胜率，成本仅为长上下文方案的 ~1/100（在处理大型文档库时）。实际最佳策略是**混合使用**——RAG 定位相关段落 + 长上下文模型处理完整段落。
>
> **RAG 的边界：什么时候不如精确搜索？** RAG 擅长的是"语义模糊匹配"——用户问"怎么实现登录"能命中 `handleAuth()` 的文档。但在**精确符号查找**场景（找 `calculateTax` 这个函数名在哪），Grep 100% 命中、零索引成本、即时出结果，RAG 反而可能返回语义相近但完全无关的结果。Claude Code 等 AI 编程工具 2026 年已舍弃 RAG，改用 Grep + Glob + Read 的精确工具链。**代码搜索要精确，文档问答要语义**——两种场景，两种策略。
>
> **附：Claude Code 的 Grep 工具到底是什么？** 名字叫 Grep，但底层跑的不是 1974 年的 Unix `grep`，而是 **ripgrep (rg)**——一个用 Rust 重写的现代搜索引擎。比传统 grep 快 5-10 倍，原生支持正则、`.gitignore` 自动忽略、多线程并行。Claude Code 把它封装为内置工具（不是 Bash 调用），每次搜索都是实时 `rg` 执行，**不做索引、不做嵌入、不建向量库**——官方称之为"Search, Don't Index"策略。结果精确到行号和字符位置，默认上限 250 条匹配防止 token 溢出。Grep（ripgrep）、Glob（文件路径匹配）、Read（按行读取）三者组合，构成了 AI 编程助手的精确检索链。

### 分块策略（Chunking）

不能把一整本书塞进 prompt（也塞不下），需要切成小块。

**固定大小分块**：最简单的做法——每 N 个 token 切一刀，块之间设 overlap（重叠）。overlap 的作用是防止关键信息恰好卡在切缝上被切断。

```
文档：  今天天气真好我们去公园玩吧
分块1： 今天天气真好我们  (10 字)
分块2： 好我们去公园玩吧  (10 字, overlap 2)
```

**语义分块**：不按字数切，而是找"自然断点"——段落结尾、句号、Markdown 标题。这需要一个小模型来判断两个句子是否"还在说同一件事"。

**父子分块**：把文档切成小块（子块）做检索（检索精度高），检索到之后把该小块所属的大块（父块）一起返回给 LLM（上下文更完整）。

实际建议：**先试试固定大小 512 token + overlap 64 token**，90% 的场景够用了。有特殊需求再上语义分块。

### 嵌入与向量数据库

切好的文本块不能直接检索——需要转成**向量**。

嵌入（Embedding）就是把一段文本映射成一个高维向量（比如 1536 维的浮点数列表）。语义相似的文本 → 向量距离近。第二阶段讲过，Attention 里就是用点积算两个向量的相似度，这里同一回事——检索时把用户问题也转成向量，然后搜出距离最近的 K 个文档块。

2026 年常用嵌入模型：

| 模型 | 维度 | 中文质量 | 免费 | 特点 |
|------|------|---------|------|------|
| text-embedding-3-large (OpenAI) | 256-3072 | 好 | 否 | 可变维度，性价比高 |
| BGE-M3 (BAAI) | 1024 | 极好 | 是 | 多语言，支持稠密+稀疏混合检索 |
| GTE-Qwen2 (阿里) | 1536-4096 | 极好 | 是 | 中文最佳选择之一 |
| Jina embeddings v3 | 1024 | 好 | 有限免费 | 支持长文本（8K token） |

向量数据库负责高效存储和检索这些向量：Chroma（轻量，Python 一行启动）、Milvus（生产级，支持十亿级向量）、Qdrant（性能优秀，Rust 实现）、Pinecone（全托管，零运维）。

### 检索质量优化

基础 RAG 搭好之后，通常有 30-40% 的情况检索出来的结果不太相关。几个优化维度：

**1. 混合检索（Hybrid Search）**：同时用向量检索（语义相似）+ 关键词检索（BM25，精确匹配），把两边的结果合并。靠语义搜不到的专有名词（比如产品型号"XJ-402B"），关键词能精确命中。

**2. Reranking（重排序）**：先用简单方法搜出 Top 50，再用一个更精准但更慢的排序模型（如 BGE-Reranker）重新给这 50 个排序，取 Top 5。大幅提升精准度。

**3. HyDE（假设文档嵌入）**：让 LLM 先"假想"一份完美回答的文档，把这个假想文档做嵌入去检索。因为假想文档的"语义密度"比问题本身高得多，检索命中率显著提升。

**4. 多路召回**：从不同角度检索（语义、关键词、问题重写后的版本），合并去重。

组合拳：HyDE + 混合检索 + Reranking = 大多数场景下检索质量提升 50% 以上。但这些步骤每多一步就多花一些时间和 token，在小文档库（几千条）上没必要，直接上基础 RAG 就够了。

### 动手：本地文档问答系统

用 Chroma + BGE-M3，~50 行代码搭一个能问自己文档的问答系统：

```python
import chromadb
from openai import OpenAI

# 初始化
client = OpenAI(api_key="...", base_url="https://api.deepseek.com/v1")
chroma = chromadb.PersistentClient(path="./my_docs_db")
collection = chroma.get_or_create_collection("my_documents")

# ===== 离线：索引文档 =====
def index_documents(file_paths):
    for fp in file_paths:
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        # 简单固定分块：每 500 字一块，重叠 50 字
        chunks = [text[i:i+500] for i in range(0, len(text), 450)]
        for j, chunk in enumerate(chunks):
            embedding = client.embeddings.create(
                model="text-embedding-3-small", input=chunk
            ).data[0].embedding
            collection.add(
                ids=[f"{fp}_{j}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": fp}],
            )

# ===== 在线：回答问题 =====
def ask(question, top_k=5):
    q_emb = client.embeddings.create(
        model="text-embedding-3-small", input=question
    ).data[0].embedding
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    context = "\n---\n".join(results["documents"][0])

    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature=0.3,
        messages=[
            {"role": "system", "content": f"根据以下参考资料回答问题。如果资料中没有答案，如实说不知道。\n\n参考资料：\n{context}"},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content

# 使用
# index_documents(["公司制度.pdf.txt", "产品手册.md"])
# print(ask("年假可以累积到下一年吗？"))
```

---

## 4.4 本地部署

### 推理框架对比

想在自己电脑上跑模型，需要一个"推理引擎"。2026 年主流选择：

| 框架 | 上手难度 | 性能 | 量化支持 | 适合场景 |
|------|---------|------|---------|---------|
| Ollama | 极低（一行命令） | 中 | GGUF | 个人使用、快速体验 |
| llama.cpp | 低 | 中高 | GGUF（最好） | CPU 推理、边缘设备 |
| vLLM | 中 | 极高 | GPTQ/AWQ | 生产级 API 服务 |
| SGLang | 中 | 极高 | GPTQ/AWQ | 批量推理、结构化输出 |
| LLaMA-Factory | 低 | — | 多种 | 微调+推理一体 |

**Ollama** 最推荐入门——`ollama run qwen3:8b` 就能在本地跑起来，自动下载模型、自动配置量化、提供 OpenAI 兼容 API。

**llama.cpp** 是底层引擎，Ollama 底层用的就是它。CPU 推理能力极强——没有显卡也能在 64GB 内存的 MacBook 上跑 70B 模型（当然速度慢）。

**vLLM** 是生产环境的标配。它的核心优化是 PagedAttention——把 KV Cache 像操作系统管理内存分页一样管理，大幅减少显存碎片。吞吐量比 Ollama 高 10-20 倍。搭 API 服务必选。

### 量化格式选型

第三阶段讲过量化的原理（计算机底层概念速查里也有），这里只说"下载模型时选哪种格式"：

| 格式 | 适用框架 | 压缩比 | 质量 | 适用场景 |
|------|---------|--------|------|---------|
| GGUF (Q4_K_M) | Ollama / llama.cpp | ~4× | 好 | CPU 推理、个人使用 |
| GPTQ (INT4) | vLLM / SGLang | ~4× | 好 | GPU 推理、API 服务 |
| AWQ (INT4) | vLLM / SGLang | ~4× | 略好于 GPTQ | GPU 推理（新模型首选） |
| EXL2 | ExLlamaV2 | 可变 | 最好 | 高端 GPU、要求最高质量 |

GGUF 文件名里的 `Q4_K_M` 代表"4-bit 量化，K-quant 算法，中等平衡"。K-quant 是针对不同层用不同精度的技术——重要层多给点精度，次要层多压一点。M 代表中等档。一般选 Q4_K_M（速度与质量的甜点），显存足够就 Q5_K_M（质量优先），不够就 Q3_K_M（空间优先）。

### 显卡选型指南

一块显卡能跑多大模型？核心公式：

```
所需显存 ≈ 参数量 × 每参数占用字节 + 上下文开销（~1-2 GB）
```

| 显卡 | 显存 | 能跑什么（4-bit 量化） | 速度体验 |
|------|------|---------------------|---------|
| RTX 3060 12GB | 12 GB | 7B-14B | 流畅（~30 token/s） |
| RTX 4090 24GB | 24 GB | 7B-34B | 极快（~80 token/s） |
| RTX 5090 32GB | 32 GB | 7B-70B | 极快 |
| 2×RTX 4090 48GB | 48 GB | 70B | 流畅 |
| A100 80GB | 80 GB | 70B (BF16) / 180B (4-bit) | 极快（推理浪费） |
| Mac M2 Ultra 192GB | 192 GB 统一内存 | 70B (4-bit) / 180B (4-bit) | 尚可（~10 token/s） |

关键洞察：
- **4-bit 量化后，参数量(B) ≈ 所需显存(GB)的一半**。比如 70B 模型约需 35GB。
- 消费级单卡（24GB）够跑 34B 以下的 4-bit 模型，体验很好。
- 想跑 70B 以上要么双卡、要么用 Mac 统一内存、要么租云 GPU。
- 苹果 M 系列芯片的统一内存（CPU 和 GPU 共用）在跑大模型上有独特优势——192GB 能装下 180B 模型。速度不快但能跑，这是所有消费级方案里唯一能单机跑千亿级模型的选择。

### 动手：Ollama 跑本地模型

3 分钟用上本地 AI：

```bash
# 1. 安装 Ollama（官网下载或一行命令）
curl -fsSL https://ollama.com/install.sh | sh   # macOS/Linux
# Windows: 去 ollama.com 下载安装包

# 2. 下载并运行模型（自动下载 + 自动量化 + 启动服务）
ollama run qwen3:8b

# 3. 在终端里直接聊天了！
>>> 给我写一个 Python 快速排序

# 4. 另开一个终端，用代码调用（Ollama 自动提供 OpenAI 兼容 API）
```

```python
# 调用本地 Ollama API — 和调用云端 API 完全一样的代码
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

Ollama 也支持自定义 system prompt 和加载 GGUF 自定义模型文件。想微调后在本地跑的，LLaMA-Factory 微调导出 GGUF → Ollama 加载，一条龙。

---

## 本阶段小结

第四阶段的核心只有一个：**把模型当成基础设施来用，而不是玩具。**

- **API 调用**：改个 `base_url` 就能切换提供商，掌握 temperature / top_p / streaming / function calling 就能覆盖 90% 的业务需求。
- **Prompt Engineering**：不是魔法，是说清楚"角色、边界、格式、示例"四件事。CoT 的核心价值在于强制模型"显式推理"。
- **RAG**：让模型能回答训练数据之外的问题。分块→嵌入→检索→生成，记住基础架子，优化按需叠加。
- **本地部署**：Ollama 一键启动，GGUF 4-bit 量化是甜点，vLLM 是生产标配。

前三阶段讲的是"怎么造模型"，这一阶段讲的是"怎么用模型"。下一阶段讲的是"怎么让模型不只是聊天——能自主规划、能看懂图片、能在评估中证明自己没那么菜"。

---

## 参考文献与图片来源

1. OpenAI API Documentation — Chat Completions, 2026. https://platform.openai.com/docs/api-reference/chat
2. Anthropic Claude API Documentation, 2026. https://docs.anthropic.com/en/api
3. DeepSeek API Documentation, 2026. https://platform.deepseek.com/api-docs
4. J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", NeurIPS 2022. https://arxiv.org/abs/2201.11903
5. X. Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in Language Models", ICLR 2023. https://arxiv.org/abs/2203.11171
6. P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020. https://arxiv.org/abs/2005.11401
7. Yunfan Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey", 2024. https://arxiv.org/abs/2312.10997
8. BGE-M3 Embedding Model, BAAI, 2024. https://huggingface.co/BAAI/bge-m3
9. Chroma Vector Database, 2026. https://www.trychroma.com/
10. Ollama — Get up and running with large language models, 2026. https://ollama.com/
11. vLLM — Easy, Fast, and Cheap LLM Serving for Everyone, 2026. https://github.com/vllm-project/vllm
