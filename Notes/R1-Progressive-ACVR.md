# R1′ · ACVR：query 渐进式跨视角推理（Active Cross-View Reasoning，视觉脚手架版）

> 状态 ⚪构思 ｜ 工作名 **ACVR / "Walk-the-Marks"** ｜ 最近更新 2026-06-05
> **对应**：本文件是 [Spatial-Reasoning.md] 中 **R1** 的"方案二"完整展开（渐进按需·效率路线）。姊妹方案见 [R1-Composition-XGoM.md]（一次性完整构建·正确性路线）。共享"对应原语"与 GoM 式"画关系→MLLM 答"接口，**分界在构建策略**（见哲学定位），作为两篇独立工作；背景见 [论文笔记.md]。
> **一句话**：不预建全图，而是从 query 物体出发，在"潜在跨视角场景图"上做 **query 引导的最佳优先扩展**——按需感知/桥接/(可选)组合/验证，**连通即停**；把逐步构建的标记/关系**画回原图**（GoM 式），**MLLM 在富像素上自己推理**。构建顺序本身 = 一条**可验证的视觉 local→global 推理链**。
> **与姊妹 X-GoM 的分界（2026-06-05 重定）**：两者都"画关系上图→MLLM 答"（接口同源、沿用 GoM），分界在**构建策略**——**ACVR = 按 query 渐进/按需构建，重点是*效率*（少触达、又快又准、可验证推理链）**；**X-GoM = 一次性建好完整跨视角图，重点是组合演算的*正确性*（τ/代数/降 IC 机理）**。（旧的"视觉脚手架 vs 符号外包"分界在"两者 MLLM 都始终作答"后已不成立，作废。）
> **设计原则**：training-free、pose-free、不假设时序；**保留像素（不丢视觉信息）**、稀疏聚焦（避 GoM 稠密图反伤）、按需（省辅助前向）。

## Challenge / 现有缺陷
- **GoM 自证"稠密图反伤"**：>10 物 / >16 关系反而掉点 → 把全场景图一股脑喂 MLLM 是错的。
- **只喂文本子图会丢视觉信息**：GoM 实测视觉 SG 比文本 SG 高 +10% → 纯文本子图精度崩（这是采纳"画回原图"的直接依据）。
- **MLLM 跨视角缝合本就差**：直接喂多图它对应/位姿/IC 都失败（All-Angles 实测）。
- 现有 agentic 视觉搜索（V*/SEAL）是**单图区域放大**；工具诱导（VisProg/ViperGPT）是**通用程序**——都不做跨视角结构构建。

## Motivation
把"逐步构建局部→全局"做成**真正的按需搜索**：从 query 物体出发，只构建/感知 query 相关的稀疏路径，**画回原图**让 MLLM 在不丢像素的前提下沿干净脚手架推理。既避开稠密图噪声（↑精度），又只触达相关视角/物体（↓辅助开销），且构建轨迹**人可见、可验证**（胜过文本 CoT 的空间幻觉）。

## 方法（α：符号策略 + MLLM 在标注原图上作答）

**状态**：部分图 G_t = {query 提到的物体，定位在其出现视角} + 前沿 frontier。

**扩展动作（按需调用感知/对应/组合算子）**：
- **Localize**：对某未处理视角跑 GoM 感知（按需，不全量）。
- **Bridge**：找共享锚点把当前连通块接到另一视角/块（按需对应）。
- **Compose（可选）**：沿刚连通路径做 搬运+组合（复用 X-GoM 算子）导出候选跨视角关系——*作为标注来源，非必需*。
- **Verify**：为已连通对再找一条锚点路径 → 多路径一致 → 升/降该标注置信。

**策略（training-free 启发式，best-first / A\*）**：优先扩展 (a) 落在 query 两物体路径上、(b) 对应/τ 置信高、(c) 更短（少跳）、(d) 最降低答案不确定度（信息增益）的 frontier。
**停止**：query 关系足够置信判定 / frontier 耗尽 / 触达预算到顶。

**★ 输出（关键，采纳用户修正）**：把逐步构建的 全局一致 id 标记 + 局部关系 +（可选）跨视角连边 **画在原图上**（GoM 式视觉提示，ViKey 标记规范）；**保留全部原图**；**MLLM 自己推理作答**。
**可选（效率/消融）**：只把 query 路径上的视角喂 MLLM（丢无关视角）→ 动视觉 token，碰压缩线，但权衡丢上下文。

**β（parked，future）**：MLLM 充当控制器，每步看当前部分图、提议下一个动作（看哪/桥接谁）。更 agentic 但不稳/贵，暂不做。

**框架图**
```
query ─解析→ 目标物体/轴
   │
   ▼  (best-first 扩展, 按需)
G_0={query物体} → Localize/Bridge/Compose/Verify → G_1 → ... → 连通即停
   │                                      每步把标记/关系画回原图(保留像素)
   ▼
标注后的原图集(稀疏·聚焦·跨视角一致) ──► MLLM 自己推理 ─► answer
```

**伪代码**
```python
def ACVR(views, query):
    targets   = parse_query(query)               # 目标物体 + 目标轴
    G         = init_from_query(targets)         # 仅含 query 物体, 定位到出现视角
    annotated = {}                               # 视角 -> 叠加层
    while not stop(G, query):
        f   = pick_frontier(G, query)            # best-first: 路径相关×置信×短×信息增益
        act = choose_action(f)                   # Localize/Bridge/Compose/Verify
        G, annot = expand(act, f, views)         # 按需感知/对应/组合/验证
        merge_annotation(annotated, annot)       # 把新标记/关系画回原图(GoM式)
    imgs = overlay(relevant_views, annotated)    # 保留像素; 可选只取路径相关视角
    return MLLM(prompt(query), imgs)             # MLLM 在富像素脚手架上自答
```

### 计算成本 / 效率全账（卖点轴，也是唯一碰压缩主线处）
- 报**端到端**：触达视角数/物体数/关系数、视觉 token、FLOPs/延迟——**含几何/辅助骨干**（你一直强调没人算这笔）。
- 对照：全图 GoM 并集（贵+稠密反伤）vs 按需稀疏 → 目标是**又快又准**。
- 专属图表：**Acc/IC vs 触达预算**曲线（极少预算逼近甚至超过全图上限，因避噪）。

### 新颖性 & 区分
- vs **agentic 视觉搜索（V*/SEAL）**：那是单图区域放大；ACVR 是**跨视角结构构建**（扩展靠对应，工具是定性空间关系）。
- vs **工具诱导（VisProg/ViperGPT/NS-VQA）**：那诱导通用视觉工具程序；ACVR 是面向跨视角空间的**固定算子 + 置信驱动搜索**，training-free。
- vs **Coarse Corr/GoM**：无 query 主动构建 / 无渐进视觉脚手架。
- vs **姊妹 X-GoM**：接口同源（都画关系→MLLM 答），分界在**构建策略**——X-GoM 一次性建完整图、重*正确性*（组合演算/降 IC 机理）；ACVR 按 query 渐进按需、重*效率*（少触达、又快又准）。

### 实验设计
- 基准/baseline 同 X-GoM（All-Angles 主 + VSI-Bench 护栏）；**额外效率轴**。
- 核心对照：全图 GoM 并集 vs ACVR 按需 → 证又快又准。
- 消融：去 query 引导（随机/全量扩展）；去信息增益打分；停止阈值；回溯有无；只喂路径视角 vs 全喂（视觉 token×精度权衡）；α vs（将来）β。
- 指标：Acc + IC + 子任务 + **效率（token/FLOPs/延迟/触达量）**。

### 开放问题 / TODO
- 策略走偏 → query 解析（目标物体/轴）要稳；加少量回溯。
- 停止判据置信校准（与多路径一致性共用）。
- **MLLM 仍在环 → IC 保证弱于 X-GoM**（诚实写明；卖点在精度/效率而非 IC 下界）。
- 画回原图的**渐进标注设计**（避免视角间标记冲突，跨视角同 id 同色，ViKey 底角/勿遮目标）——下一步深挖。
- ego-centric：重叠少时退化更优雅（仍有逐视角局部+像素），但核心几何限制仍在；仅作 limitation。

### 引用速查
All-Angles 2504.15280 · VSI-Bench 2412.14171 · Set-of-Mark 2310.11441 · Graph-of-Mark 2603.06663 · Coarse Correspondences 2408.00754 · ViKey 2603.23186 · V* / SEAL（视觉搜索） · VisProg / ViperGPT（工具诱导） · FastV 2403.06764 · VisPruner 2412.01818
