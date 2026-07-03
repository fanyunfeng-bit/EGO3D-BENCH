# GeoScaffold — query-aware 多视角 token 压缩（故事 / 方案工作版，2026-06-18）

> 本文件记录从 CVSP（query-agnostic）转向 **query-aware 分阶段压缩** 的整合方案。
> 关联：[Visual-Compression.md]（§实测发现 / §思路1 CVSP 历史草案）、[CVSP-Story.md]（前序 motivation）。
> 工作名 **GeoScaffold**（亦可叫 CVSP-2）。状态：motivation + 方案骨架已收敛，待实现 + 验证。
> 一句话：**先在 LLM 前搭一个 query 无关、可缓存的"全局跨视角 3D 骨架"；再在 LLM 内用去偏的 query 相关性做"由粗到精"的跨视角精修。每个阶段都跨视角、绝不逐视角。**

---

## 0. 为什么转向 query-aware（承接实测）
query-agnostic 路线（saliency / diversity / FPS / leverage / 纯 anchor）在 keep10/25 全平 random、keep5 仅几何 anchor 在 VSI 微赢（见 [Visual-Compression.md] §B–I）。根因：可争夺视觉盘子 ~9–14pp、高冗余 + 重语言先验。**没被开采的信号 = query 相关性 × 几何/anchor 承重度**：`importance = f(query相关度, 几何承重)`。本方案据此设计。

约束（用户拍板）：**training-free · 无额外模型**（不用检测器/深度/VGGT/CLIP；信号只取 MLLM 自身的 ViT 特征 / 文本·视觉 token 嵌入 / 注意力）· 跨 VSI + Ego3D 普适。

---

## 1. 三篇关键近邻（已核实，2026-06-18 精读）

### Nüwa（2602.02951，单图，training-free）—— anchor 保护的样板
- **诊断**：剪枝破坏"由 token 位置嵌入交互构成的**全局空间参照系**" → grounding 任务崩。位置策略：PERC（PE 压成小区间，丢全局框架，差）/ PESP（稀疏保留原 PE，残缺框架）/ **RPME**（保留相对距离 + 线性重映射铺满原程，+5.6–13.4%，最佳）。
- **Stage 1（编码器后）**：separation（N×N 切 M×M 网格，均匀覆盖、不留空洞）→ alignment（显著分 `S(t_i)=α_cls,i·‖k_i‖₂` = **CLS注意力 × key范数**，每区域 top-k = 基准token）→ aggregation（角色分：**pillar = key范数 top四分位 = ViT register token，原样冻结 W=δ**；collector = 其余基准，按 `A_ij·P_ij`= 语义相似 × 空间邻近 聚合邻居）。
- **Stage 2（LLM 中层）**：均值池化文本 query `q̄`，`R_i=cos(proj(v'_i),q̄)`，留 top-K。
- **"spatial anchor" 是宽泛词** = 保下来的网格结构 + 受保护 pillar；**pillar = 保护级最高的子集**（高范数 register）。
- 指标：88.9% 剪枝、VQA 留 95%、VG 7%→47%、TFLOPs −89%、prefill −62%。
- 注：Nüwa 的 anchor（register/网格，保特征完整）≠ 我们的 anchor（几何地标，保 3D 参照系）；可两者都要。`CLS-attn × key-norm`、register 思路是我们现有 anchor 分没有、可补的免费信号。

### VisionTrim（2601.22674，单图+视频，training-free，Qwen2.5-VL-7B ~1/3 token、~0.1% 损失）
- **DVTS**：ViT CLS 注意力(倒数第二层)全局重要性 + **LTAM**（局部 token 亲和=特征相似×空间邻近，双核）+ **方差自适应加权** `α=σ²_l/(σ²_g+σ²_l)`。
- **TGVC**：**用 CLIP text encoder**（= 额外模型，违我们约束 → 改用 MLLM 自有文本嵌入 + 修模态gap）。`S_t2v=softmax(TV_rᵀ/√d)`，**被删 token 按 text 加权平均合并回**簇中心，`V_final=[V_dom; V_com]`。
- **两阶段**：编码器(pre-LLM) + LLM 层间(用**首个生成 token**的注意力)。

### AdaTP（2505.20100，视频，training-free）—— 坐实"视角位置偏置"+ 去偏
- **全局偏置**：top10% 注意力 **86.8% 落在 32 帧的最后 4 帧**（layer 1）→ 多视角 query 注意力系统性偏向靠后视角。**全局去偏**：用 text-visual 相似度找"显著片段"再分预算（而非信原始有偏注意力）。
- **局部偏置**：注意力固定堆某空间位（5.77×均值）。**局部去偏**：每空间位每段只留一代表（跨帧去重）。
- 信号：LLM 自注意力 text→visual，层 2..N−12；分段相邻余弦 ≥0.95。27.3% FLOPs 无损。

### DyToK（2512.06866，视频，training-free）—— 分阶段 + 逐帧预算
- 帧预算 `w_f=mean_{深层} softmax(Q_l K_lᵀ/√D)`（**最后一个 query token → visual**，**深层 20–23/24** 比浅层准），`a_f=⌊ŵ_f·T⌋`。
- ⚠️ 用了 14× 小的同族辅助模型算 prior（= 额外模型 → 改用目标模型 partial forward）。
- 分阶段：先定逐帧配额 → 喂任意下游压缩器（VisionZip 编码端 / FastV LLM 端）。无 anchor/去重。
- 经验：**分预算用深层最准、视角内砍 token 可早层**。

> 共性：三篇都不是同步多视角（Nüwa 单图、AdaTP/DyToK 时序视频）→ **多视角 3D 是空位**。

---

## 2. 统一框架 = 由粗到精的 query 条件 coreset（总纲）
> 用 query 无关、可缓存的 **全局跨视角 3D 骨架** 表示场景结构；再用一次 **query 条件的跨视角精修** 决定"每块该多细"。
> `min tokens  s.t.  场景3D结构被覆盖(stage1, query无关) + query条件信息被保留(stage2)`；预算与 λ 自适应。

### Stage 1 ｜ LLM 前（query 无关、可缓存）—— 全局联合 coreset，**处处跨视角**
对**所有视角 token 汇成的一个池**做选择（不是逐视角）：
- **跨视角冗余折扣**：高跨视角支持度（同一 3D 内容被 ≥2 视角看到）→ 留一代表，其余池化成摘要。
- **跨视角 3D 锚点（核心新颖性）**：`跨视角支持度 × 几何独特性(Lowe margin)` → 多视角共同确认 + 独特 → 撑 3D 参照系，强保护。
- **全局均匀覆盖**：覆盖项在**联合池**上算 → 低重叠视角也有代表，不被饿死。
- **局部空间连续性**：视角内邻域亲和力，仅作**平滑正则**；选择/预算仍全局。
- 产出 = **多分辨率集** {锚点(全分辨率) + 跨视角簇摘要}；**成员特征入零算力 side-cache**（供 Stage 2 重注入）。
- **dedup↔coverage 行为随重叠度涌现**（VSI 偏去重、Ego3D 偏覆盖/锚点）= 自适应率，不是两套 pipeline。

### Stage 2 ｜ LLM 内 早层(~2–4)（query 条件、由粗到精）—— **三处都跨视角**
1. **去偏 query 相关性**：`cos(视觉隐状态, 自有均值query嵌入)`（修模态gap：逐模态去均值+L2+去sink）÷ **逐视角位置基线**、去 sink → 真·相关性、**视角间可比**。
2. **跨视角预算分配**：在去偏相关性上对**视角做全局 softmax** → 每视角不同预算 `k_v`（视角彼此竞争），含**每视角地板**（不盲任何相机），总预算 K 自适应（随 relevance 熵 + 重叠度）。
3. **去选 + 再稠密化**：删 query 无关的骨架/摘要；对 query 强相关的**跨视角簇摘要从 cache 重注入成员** → 该细的地方才细。
4. **位置(RPME)**：精修后集合重映射铺满原程；重注入成员恢复原始位置。

### 跨视角保证（自检表）
| | 逐视角(✗ 已避免) | 跨视角(✓ 我们做的) |
|---|---|---|
| **Stage 1** | 不逐视角独立选 | 全局联合 coreset：跨视角支持 / 锚点 / 覆盖 |
| **Stage 2** | 不每视角各自定率 | 视角间竞争分预算 + 重注入跨视角簇 + 跨视角去偏归一 |

---

## 3. 冲突检查 + 解法（4 个，已解）
1. **去偏 × 逐视角不同预算（顺序冲突）**：必须**先去偏、再在去偏相关性上分视角预算**；否则放大偏置。
2. **自适应率 × stage1 必须跨视角（表面冲突）**：stage1 重构为**对所有视角的 GLOBAL 联合 coreset** → 高重叠表现为去重、低重叠表现为跨视角覆盖/锚点，**同一目标随重叠涌现**。
3. **再稠密化 × 位置/机械（机械冲突，最该讲清）**：stage1 不硬删→池化成摘要进 LLM + 成员特征留 side-cache；**LLM 早层一次性决定**（删无关摘要 + 重注入相关摘要的成员）→ RPME 重映射 → 跑剩余层。**早层决定一次、之后静态**，不是边解码边插。
4. **stage1 去冗余 × stage2 也删冗余（非冲突）**：stage1 删**外观/特征冗余**(query无关, de-duplicate)；stage2 删**任务冗余**(对当前 query 无关, de-select)。分工干净。
> W5(自有嵌入)与 W4(去偏)互补不冲突：前者给信号、后者清洗。

---

## 4. Abstract（顶会风）
Multi-view VLMs for 3D spatial reasoning explode in visual tokens as views grow, making prefill the dominant cost. We show single-image token compressors break here: multi-view tokens are massively redundant (random 75–90% pruning rarely hurts, often helps), informed single-image pruning therefore **fails to beat random** because its saliency and diversity axes are computed *per view*, and the obvious query-aware fix—text-to-visual attention—is **position-biased across views**, over-selecting later views. We present **GeoScaffold**, a training-free, no-extra-model compressor cast as a **coarse-to-fine, query-conditional coreset**, in which *every* stage operates **across views, never per view**. A **query-agnostic, cacheable stage** builds a global cross-view *geometric scaffold*: it discounts cross-view redundancy, protects **cross-view-consistent, geometrically-unique 3D anchors** that hold the global reference frame, and guarantees uniform scene coverage over the joint token pool—pruned tokens are pooled into cross-view *summaries* rather than discarded. A **query-conditional in-LLM stage** then, on **debiased** question relevance, (i) allocates *per-view token budgets by inter-view competition*, (ii) de-selects task-irrelevant scaffold tokens, and (iii) **re-densifies** only the cross-view regions the question needs by expanding their summaries—spending resolution where the query demands it; relative position embeddings are remapped to preserve spatial integrity. The total rate adapts to relevance entropy and cross-view overlap, so one mechanism spans high-overlap indoor video and low-overlap surround driving. On VSI-Bench and Ego3D-Bench across InternVL3-8B and Qwen2.5-VL-7B, GeoScaffold matches or exceeds full-token accuracy at <15% of tokens (−X% prefill FLOPs, −Y% KV) and, by removing cross-view distractors, **improves** spatial-reasoning accuracy—the first informed multi-view compressor to consistently beat random.

---

## 5. 仍开放 / 下一步 TODO
- **最该先抠**：③ 再稠密化的精确数据流（side-cache 存什么、早层在哪决定、重注入后 RoPE/RPME 怎么对位、KV 如何处理）——新颖性最高、最易被审稿/实现挑。
- 便宜诊断（验 point 1）：在 InternVL3(1D-RoPE)/Qwen2.5-VL(M-RoPE) 上画"每视角拿到的 text→visual 注意力质量分布"，确认靠后视角偏置形状。
- 信号决策：Stage1 anchor 是否加 `CLS-attn × key-norm` + register(高范数) 信号；跨视角支持度具体算法（哪层特征、互最近邻 vs 软聚合、阈值）。
- Stage2 相关性：中层 vs 早层、自有文本嵌入修模态gap 的具体做法、去偏基线怎么估（D2Pruner 内容无关基线 / PoRe 拟合）。
- 自适应率/预算的具体函数（熵、重叠度 → K、k_v）；每视角地板大小。
- 加速账：含估计开销的端到端 FLOPs/KV/prefill；主战场放大 N（多帧/高分辨率）；wall-time 不承诺（decode 主导）。
- eval：打视觉扛大头任务（物体距离/counting）；极端预算下含真 random baseline + 逐类别拆解。
- 消融：逐视角 vs 全局（两阶段各一）；anchor / 去偏 / 再稠密化 / 自适应率 逐项开关；vs VisionTrim(改自有嵌入)/Nüwa/AdaTP/DyToK/CDPruner。
- 并行：38B/API 验证"可争夺盘子是否随规模变大"。

## 引用速查
Nüwa 2602.02951 · VisionTrim 2601.22674 · AdaTP 2505.20100 · DyToK 2512.06866 · VisPruner/FasterVLM 2412.01818 · CDPruner 2506.10967 · D2Pruner 2512.19443 · Attention-Debiasing/PoRe 2508.17807 · VScan 2505.22654 · SparseVLM 2410.04417 · PruneVid 2412.16117 · VSI-Bench 2412.14171 · Ego3D-Bench 2509.06266
