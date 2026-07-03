# 空间推理提升（Spatial Reasoning）方案库

> 本文件长期记录"多视角 MLLM 空间推理提升"方向下的所有方案，每个方案含：状态 / Challenge / Motivation / 详尽方案（算法·框架图·伪代码·实验）/ 计算成本 / 新颖性与区分 / 开放问题。目标基准：**Ego3D-Bench**（2509.06266，同步多相机多图）、**All-Angles**（2504.15280，多机位静态多图）。设计原则：只用跨视角空间结构、**不假设时序**、pose-free、training-free 优先。
> 关联：压缩方案见 [Visual-Compression.md]，联合方案见 [Joint.md]；背景调研见 [论文笔记.md]。
> 共享原语：**pose-free 跨视角对应**（视觉特征匹配 → 同物体跨视角全局 id + 最佳视角代表）。R1/R2 与 C 系列共用此原语。
> 状态图例：🟢已验证 / 🟡进行中 / ⚪构思 / 🔴存疑。最近更新 2026-06-04。

---

## R1：跨视角「局部→全局」结构化推理（Cross-View Graph-of-Mark）⚪
一句话：逐视角抽局部空间关系 → 跨视角物体对应 → 把单视角观测不到的**跨视角关系**补出来 → MLLM 作空间推理。核心是把"局部好/全局差"重铸为 **co-visibility gap（共现缺口——视角现象，而非 VSI-Bench 的距离现象）**。

> **本方案已于 2026-06-05 拆为两篇独立工作，各自完整展开成独立文件，本处仅作索引/对应：**
> - **方案一 · X-GoM（符号组合）** → [R1-Composition-XGoM.md]
>   哲学 = **符号外包**：用 pose-free 的"关系搬运 τ + 三轴符号代数 + 多路径自一致"把缺口关系*算出来*，MLLM 只读结果；**结构性降 IC**。headline 最锐利、最易过审。
> - **方案二 · ACVR（渐进视觉脚手架）** → [R1-Progressive-ACVR.md]
>   哲学 = **视觉脚手架**：query 引导按需稀疏构建、把标记/关系*画回原图*，MLLM 在富像素上自推理；卖点 = 精度（不丢像素）+ 效率（按需稀疏，唯一碰压缩主线）+ 可验证视觉推理链。
>
> **两篇共性**：共享原语 pose-free 跨视角对应；training-free / pose-free / 不假设时序；基准 **All-Angles（主，盯 IC）+ VSI-Bench（次，当无序多视角、丢帧序、仅 Acc 广度）**；Ego3D 暂 park（标定低重叠几何性弱适配，留给 C/U 系列）。
> **头号近邻**：Coarse Corr 2408.00754（training-free/pose-free 跨视角对应+叠 id，但**无关系/无组合/靠视频 tracking**，未测目标基准）——两篇新颖性都需对它逐句切割。
> **背景**：旧的粗糙 4 步方案已被上述两文件取代；ViKey 标记经验、竞品精读详见各文件与 [论文笔记.md]。

---

## R2：视角 distractor 因果归因 + 跨视角冗余/干扰消解（推理增强，不强调提速）⚪
一句话：先证明"视角/token 过多→空间推理变差"，再归因"是哪些视角/token/信息导致变差"，最后针对性设计跨视角冗余/干扰消解方法（允许 attention 类、允许加模块、不以提速为目标）。

### Challenge / 现有缺陷
- "少而精"证据全在**帧/时序**层面（Less-is-More 2508.03337、Struct2D 2506.04220）；**"视角数/视角 distractor → 空间推理变差"无人隔离**；"压缩→提精度"从未当多视角一等目标；FastV 式注意力剪枝信号被 VisPruner(2412.01818) 证不可靠。

### Motivation
多视角下，无关/冗余视角分散注意力、当 distractor，且跨视角同物体重复编码会引起**误绑定/重复计数**。若先把"**为何**变差"查清，再针对病因消解跨视角冗余/干扰，就能把剪枝/合并**重定义为推理增强**（而非提速）。

### 方案（三阶段研究程序）
**Stage 0 — 存在性实验（distractor 是否真伤推理）**
- 构造受控视角集：对每题取"最小充分视角集 S*"（含被问物体的视角）；造变体 V_k = S* + k 个 distractor 视角（无关/冗余）。
- 控制变量：① 固定 token 预算（下采样）vs 增长 → 分离"视角数"与"token 数"；② 关键视角位置（首/中/尾）→ 测"视角版 lost-in-the-middle"；③ 加近重复视角 → 测重复对计数的影响。
- 指标：Acc、IC、counting error。假设：k↑→Acc↓/IC↑；中间位最差；重复→计数虚高。

**Stage 1 — 病因归因实验（哪些视角/token/信息导致变差）**
候选机制 + 检验手段（允许 attention 分析）：
- (Ca) **注意力稀释**：随 distractor 增多，answer/末位 token 对关键物体 token 的注意力质量下降 → 注意力归因(attention attribution/rollout)度量"关键 vs distractor"注意力占比。
- (Cb) **跨视角误绑定**：模型把视角 i 的物体 A 与视角 j 的 B 绑一起（看相像）→ 受控注入/移除"跨视角相似干扰物"，看错误相关性 + 是否注意到错误视角实例。
- (Cc) **重复过权/过计数**：同物体在 N 视角→N×表征→偏置计数与显著性 → 计数误差 vs 重复视角数。
- (Cd) **位置偏置**：来自 Stage 0 位置实验。
- (Ce) **证据冲突**：各视角局部线索轻微不一致、模型无法调和 → 一致 vs 故意冲突视角集对比。
- 工具：注意力归因、**留一法 token/视角消融**（移除看答案是否翻转→重要性）、隐状态线性探针（"模型是否知道 X 跨视角是同一个"）、logit-lens。
- 产出：**哪些视角/token/信息**致害的排序归因（如"跨视角重复 token 抢走 X% 注意力、致 Y% 错误"）。

**Stage 2 — 针对病因的方法（跨视角冗余/干扰消解；可加结构；不以提速为目标）**
- 若 (Cb) 误绑定主导 → **跨视角身份绑定**：用共享原语建对应，给同物体 token 打**共享身份嵌入/共享位置标签**，使注意力共享而非重复 → 防误绑定+防重复计数（加一个 identity-binding 模块）。
- 若 (Ca)/(Cc) 稀释/过权主导 → **注意力再分配**：把跨视角重复 token 合并为**带计数标签的单 token**（"该物体出现于 k 个视角"），和/或对 query 相关视角做**经病因验证的注意力引导**（不是 naive FastV）。
- 若 (Cd) 位置 → 关键视角前置/位置去偏。
- 若 (Ce) 冲突 → 加**跨视角一致性消解**：比较不同视角子集的预测、抑制不一致注意力（self-consistency 作正则）。
- **与 C1 区别**：R2 也合并/重排 token，但**目标是推理精度（消 distractor/干扰），用 Acc+IC 度量，不为提速**；可保留全部信息只**重绑定/重加权**，或加轻量"跨视角冲突消解"模块。本质是"**压缩即降噪**用于推理"。

**伪代码（Stage2 身份绑定版）**
```python
def R2_bind(views, query):
    gid = cross_view_match(region_embeddings(views))   # 共享原语
    tokens = visual_tokens(views)
    tokens = tag_shared_identity(tokens, gid)          # 同物体跨视角共享身份/位置标签
    tokens = merge_duplicates_with_count(tokens, gid)  # 重复→单token + "出现k视角"计数标签
    return MLLM(prompt(query), tokens)                 # 注意力共享, 不重复计数, 信息不删为提速
```

**实验设计（TODO）**：Stage0/1 在 All-Angles + Ego3D-Bench 上做受控集；Stage2 对比 FastV/SparseVLM(naive 注意力)、不绑定基线；指标 Acc + IC + counting，附注意力占比可视化。

### 计算成本
保留信息为主、不为提速，故 token 数可不降；额外开销=对应原语（同 R1，辅助模型一次性）+ 身份标签/合并（廉价）+ 可选一致性消解（多一次/几次前向）。**定位是"准"而非"快"**。

### 新颖性 & 区分
- 首个把"视角 distractor 对空间推理的因果"隔离 + 归因 + 针对性消解。与通用 token 剪枝（FastV/SparseVLM/DART 2502.11494）区分：它们为提速、用通用相似度/注意力；R2 为推理精度、用**病因驱动 + 跨视角身份**。与 C1 区分：目标与度量不同（精度 vs 压缩率）。

### 开放问题 / TODO
- Stage1 归因方法学要严谨（注意力≠因果，需配留一法消融交叉验证）。
- 病因可能多因并存 → 方法需自适应组合。

---

## R3：真实视角子集的空间自一致（test-time，可选）🔴
- 要点：**不生成新视角**（ViSA 2512.05809 证生成式想象视角在 MMSI-Bench 失效、verifier≈随机），而是对**真实视角的不同子集**分别推理、按几何一致性聚合/投票。VSI-Bench 说"整段 self-consistency 无效"，但"**视角维度** self-consistency"是干净反直觉假设。
- 风险：verifier/聚合准则难设。**定位**：作 R1 的增强模块或小实验，不作主线。
- TODO：设计帧锚定可验证 micro-claim 作聚合准则（借 ViSA 思路）。

---

## 引用速查
All-Angles 2504.15280 · Ego3D-Bench/VLM 2509.06266 · VSI-Bench 2412.14171 · Set-of-Mark 2310.11441 · Graph-of-Mark 2603.06663 · Scaffold 2402.12058 · **Coarse Correspondences 2408.00754** · V²-SAM 2511.20886 · VICP 2508.21222 · MMVM/CoLVA 2501.04670 · HATCH/XVR 2602.08735 · FastV 2403.06764 · VisPruner 2412.01818 · DART 2502.11494 · ViSA 2512.05809 · Less-is-More 2508.03337 · Struct2D 2506.04220
