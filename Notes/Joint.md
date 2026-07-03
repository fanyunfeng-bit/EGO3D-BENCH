# 联合方向（Joint：既加速又提升）方案库

> 本文件长期记录"同一机制同时实现 token 压缩 + 空间推理提升"的方案。每个方案含：状态 / 它是什么（与 C、R 的关系）/ 联合独有的 Challenge / Motivation / 详尽方案（算法·框架图·伪代码·实验）/ 计算成本 / 新颖性 / 开放问题。
> 关联：压缩方案 [Visual-Compression.md]、推理方案 [Spatial-Reasoning.md]。
> 共享原语：**pose-free 跨视角对应**。**调研裁定：「一份对应、两处用」（同一套跨视角对应同时驱动关系图推理与 token 去重）的组合论文目前未见——这是最稳的差异化叙事。**
> 状态图例：🟢已验证 / 🟡进行中 / ⚪构思 / 🔴存疑。最近更新 2026-06-04。
> **⚠️ 2026-06-18 时效**:本文(U1 = "压缩即提精度、一份对应两处用")写于实证之前。截至 2026-06-18 的实测(见 [Visual-Compression.md] §实测发现 D–J + [CVSP-Story.md] 实验现状):**多视角压缩并不可靠地提精度,query-agnostic 的 informed 选择(显著/多样/leverage/锚点)均未稳定赢过随机**。U1 的"压缩↑精度"前提**待重估**;本文作历史构思保留。

---

## U1：任务感知跨视角认知速写（Task-aware Cross-View Cognitive Sketch）⚪ ★课题主线
一句话：**同一个 query 感知的跨视角物体结构，既是"保留哪些 token"的压缩掩码，又是 MLLM 推理的脚手架**——一次计算，两处用。

### 它是 C+R 的"整合"吗？（回答你的提问）
**不是"先跑 C1 再跑 R1 然后拼接"，而是两者坍缩到同一个结构上。** 精确地说：
- U1 = **C1-B（query 感知跨视角选择）与 R1（跨视角关系图）共用同一原语 + 同一 query 条件，落到同一个产物**。
- 这个产物（认知速写 = 带跨视角对应的物体图）**同时定义**：① 哪些视觉 token 存活（= R1 图中节点对应的最佳视角 token，其余删 → 压缩）；② 模型推理所依据的结构（= 关系图本身 → 推理）。
- 即：**"建推理脚手架的那一次跨视角抽取，顺带就产出了压缩掩码"**。R1 是它的"只建图不删 token"投影，C1 是它的"只删 token 不显式建图"投影，U1 把两者合一。

### 联合独有的 Challenge（不同于"加速的 challenge + 推理的 challenge"之和）
这是你问的关键——以下是 joint 才有、单独 C 或 R 不会遇到的难点：
1. **充分性–最小性张力（核心）**：压缩要 token 最少，推理要结构足够覆盖**所有**被 query 隐含的推理步（有些不在浅层 query 解析里）。被 C 当"冗余"删掉的 token，可能正是某个下游推理步需要的。纯 C（除精度外无推理依赖被删 token）和纯 R（全保留）都不面对这个**耦合约束**。→ 需"**推理感知的保留**"（保留建立参照系、关系的 token，不只被指物体）。
2. **误差复合、无回退（严重）**：纯 R 中坏对应只是图里多一条噪声边，模型**仍有全部原始 token 可回退**；纯 C 中坏去重删个 token，但没有推理结构依赖它。**U1 中一次对应错误同时**：删错 token + 污染脚手架 + 抹掉模型本可回退的原始 token → **误差复合且不可恢复**。鲁棒性门槛更高。
3. **粒度绑定**：推理用**物体级符号节点**，压缩在**patch 级 token** 操作。U1 必须保证"图节点 ↔ 存活 patch token"一致绑定，否则图说"物体3"但物体3的 token 已被删。纯 C/纯 R 无此绑定问题。
4. **query 条件双重职责且会冲突**：query 同时决定"留什么"（预算）与"推理什么"（脚手架）。一次 query 解析错误**同时**伤两者（删了需要的 token + 错搭脚手架），而纯 R 中解析错只错脚手架、token 还在。
5. **联合评测的归因难题**：U1 掉点时，是压缩删了信息、还是脚手架误导？必须用 R1/C1 两个投影作消融拆解——**这正是把 R1、C1 保留为独立 baseline 的方法学必要性**。
6. **延迟账非单调**：U1 既加建图开销（像 R1 的辅助前向），又省 LLM 输入 token（像 C1），还可能加推理输出（像 Ego3D-VLM +31%）。**净延迟可正可负，必须实测**——"联合是否真的更快"是个纯 C/纯 R 都不必回答的问题。

### Motivation
构建"任务相关跨视角空间结构"这一动作本身就标定了哪些 token 重要（被指物体、最佳视角）、哪些冗余（重复视角、无关背景）→ 建推理脚手架与产出压缩掩码是**同一件事**。结果：token↓（加速）+ 任务相关信号更密、distractor 更少（推理↑、跨视角幻觉↓）。

### 方案
**算法（training-free）**
1. query → 被指物体 + query 类型。
2. pose-free 跨视角对应（共享原语）→ 同物体簇 + 全局 id。
3. **query 感知的联合选择**（一次决定双产物）：被指物体按 query 预算保最佳/多视角 token（C 侧"留什么"）；同时这些物体+关系构成认知速写图（R 侧"脚手架"）。**推理感知保留**：额外保参照系/关系所需 token（应对充分性张力）。
4. 删冗余视角与 distractor token；**绑定**图节点 ↔ 存活 token。
5. {裁后 token + 认知速写图} 喂 MLLM。

**框架图**
```
query ─┐
       ├─► 跨视角对应(共享原语) ─► query感知联合选择 ─┬─► 压缩掩码(存活token)  ┐
views ─┘                                            └─► 认知速写图(脚手架)    ├─► MLLM ─► answer
                                          (节点↔token绑定 + 推理感知保留)     ┘
```

**伪代码**
```python
def U1(views, query):
    objs, qtype = parse_query(query)
    clusters = cross_view_match(region_embeddings(views))   # 共享原语
    sketch, kept = Graph(), []
    for c in clusters:
        if not verify_identity(c): c = split(c)
        node = sketch.add_object(c)                          # R: 脚手架节点
        budget = view_budget(obj(c), qtype, referred=obj(c) in objs)
        toks = topk_diverse_views(c, budget, score=quality)  # C: 存活token (与node同源)
        bind(node, toks)                                     # 粒度绑定
        kept += toks
    sketch.add_relations(); sketch.add_camera_order()
    kept += reasoning_aware_keep(views, clusters, qtype)     # 充分性: 参照系/关系token
    return MLLM(prompt(query, serialize(sketch)), kept)
```

**实验设计（TODO）**：基准 Ego3D-Bench + All-Angles。**关键消融（应对挑战5）**：U1 vs R1(只图) vs C1(只删) vs 原始——拆解"压缩贡献 / 脚手架贡献 / 联合增益"。指标：Acc + IC + 压缩率 + **端到端 FLOPs/延迟/显存**（验证挑战6"是否真省时"）+ counting。鲁棒性实验（挑战2）：注入对应错误，比较 U1 vs R1 的崩溃幅度。

### 计算成本
建图/对应开销（同 R1，辅助模型一次性）− LLM 输入 token 减少（同 C1）+ 可能的推理输出增长。**净值需实测**（见挑战6）。预期：token 重的长多视角输入下净省时，短输入下可能净增。

### 新颖性
**最稳：「一份对应、两处用」的统一框架调研未见。** 与 Coarse Correspondences(2408.00754，只叠ID标、不建图、不压缩)、Ego3D-VLM(2509.06266，附加坐标map、不删token、post-train)、Cog3DMap(2603.23023，去冗余提精度但要训练+为视频+几何模型)、Geo3DPruner(只压缩不推理) 全部区分：U1 是 **training-free + pose-free + 压缩掩码即推理脚手架**。逐条反驳 Prune2Drive/Geo3DPruner/Cog3DMap 的缺陷（见各方案表）。

### 开放问题 / TODO
- 充分性张力的量化：如何定义"推理充分但 token 最小"的保留准则（可学一个轻量预算预测器作为极小训练版）。
- 误差复合的缓解：低置信对应处**保守不删**（保留原始 token 作回退）→ 用置信度调节"删/留"。
- 节点↔token 绑定的实现（mask→token 索引映射）。
- 与 R2 身份绑定的协同（U1 可内置 R2 的去重复计数标签）。

---

## 引用速查
Coarse Correspondences 2408.00754 · Ego3D-VLM 2509.06266 · Cog3DMap 2603.23023 · Geo3DPruner 2604.18260 · Graph-of-Mark 2603.06663 · All-Angles 2504.15280 · Ego3D-Bench 2509.06266 · VSI-Bench 2412.14171
