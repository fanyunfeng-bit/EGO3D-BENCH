# 视觉压缩 / 推理加速（Visual Compression）方案库

> 本文件长期记录"多视角 MLLM 视觉 token 压缩 / 推理加速"方向下的所有方案，每个方案含：状态 / Challenge / Motivation / 详尽方案（算法·框架图·伪代码·实验）/ 计算成本 / 新颖性与区分 / 开放问题。目标基准：**Ego3D-Bench**（2509.06266）、**All-Angles**（2504.15280），均为静态多视角图像。设计原则：跨视角空间结构、**不假设时序**、pose-free、training-free 优先。
> 关联：推理方案见 [Spatial-Reasoning.md]，联合方案见 [Joint.md]。
> 共享原语：**pose-free 跨视角对应**（同物体跨视角全局 id + 最佳视角代表）。
> 状态图例：🟢已验证 / 🟡进行中 / ⚪构思 / 🔴存疑。最近更新 2026-06-04。

---

## C1：跨视角对应感知的 token 去重/选择（含 query-agnostic 与 query-aware 两套）⚪
一句话：用 pose-free 跨视角对应识别"同一 3D 物体在多视角的重复编码"，**按是否服务当前 query 决定保留几视角**——而非一重复就删。含 C1-A（query 无关）与 C1-B（query 感知，吸收原 C2）两套及其结合。

### Challenge / 现有缺陷
- 现有压缩把同步多视角当**独立图**（[CLS] per-image → 同物体每视角各留一份）或当**时序相邻帧**（FastVID 2503.11187 / HoliTom 2505.21334 假设相邻相似，对同时刻不同视角误伤互补视角）。
- **Prune2Drive**(2508.13305)：各视角独立 FPS 选 token、**不建模视角间关系**、ratio 靠**验证集**、只能定"裁多少"不能定"留哪些"。
- 跨视角去重的现有工作都各差一个条件：**SeGPruner(2603.29437)/Geo3DPruner(2604.18260) 依赖几何/深度先验**；**ST-Prune(2604.19145) 依赖已知 ring-view 相机布局**；**Prune2Drive/DART(2502.11494)/ToFu 是通用多样性/重复、不认"同一物体"**；**OC-VTP(2511.20439) 是单图 object-centric、需训练**。
- **空位**：「**pose-free + 纯视觉特征匹配 + 同一 3D 物体跨视角 re-id 去重 + training-free**」未见撞车工作（调研裁定新颖性"较高"）。

### Motivation
正确的压缩单元是"**跨视角的同一物体**"，不是"每张图"。删跨视角重复 token、每物体留最佳代表 → 大幅减 token，**且去重本身修复 All-Angles 第一失败模式（跨视角对应/重复计数）→ 压缩即提精度**。但"是否该删某物体的多视角"应**取决于 query 需不需要它的多视角信息**（计数/遮挡/"另一侧"需多视角；属性/颜色一视角足矣）。

### 方案
**C1-A：query 无关（通用紧凑表示，可预计算复用）**
1. 跨视角对应 → 同物体聚成簇。
2. 每簇保留**最佳代表**（见下"最佳代表选择"）；默认保 **top-K 个视角差异最大的代表**（K 为压缩率旋钮，**不默认压成 1**，以保互补外观）。
3. 保 scene completeness（见下）。

**C1-B：query 感知（吸收原 C2 = 几何×query 双感知）**
1. 解析 query → 被指物体 + **query 类型**。
2. **每物体视角预算随 query 变**：
   - 属性/颜色/单体 → 1 个最佳视角；
   - 计数 → 需**全部视角**（遮挡，不能删），但打身份标签防重复计数；
   - 相对方向/遮挡/"X 的另一侧" → 需该物体**多个互补视角**；
   - 相对距离 → 需建立两物体 + 参照系的视角。
3. 非被指物体激进压缩（1 代表或降为上下文锚点）；**被指物体按需 re-densify**（恢复 query 所需视角）。

**C1-A ⊕ C1-B 结合（两层）**：全局 C1-A 基础压缩 + 针对被指物体的 query 感知 re-densification。→ 这层"query 决定保留结构"自然演化为 [Joint.md] U1。

#### 关键子问题①：每簇"置信度/身份感知最佳代表选择"怎么做
两步：
- **(i) 身份核验（先防把不同物体并一起）**：簇内特征余弦一致性 > 阈值 + 布局/几何合理性 + 匹配循环一致性；若一簇含两候选身份则**拆开/保留独立**，避免 All-Angles 式误绑定。
- **(ii) 质量打分（给每视角对该物体的 token 打分，加权求和取 argmax；视角敏感物体取 top-K）**：
  - 尺度/分辨率：该视角中物体的 bbox 面积 / 覆盖 patch 数（越大细节越多）；
  - 可见性/完整度：分割掩码完整度（未被图像边界截断、低遮挡）；
  - 正面/居中度：物体靠近图像中心、低畸变（有弱位姿时取最正面）；
  - 清晰度：高频/梯度能量（非模糊）；
  - 检测/grounding 置信度；
  - 代表性：特征到簇质心距离（最典型），但与质量加权，避免选到"模糊的平均"。
  - **"置信度感知"=按上述置信度加权；"身份感知"=(i) 的门控**。**绝不用算术均值融合**（见 [论文笔记.md] 均值融合失真：会抹平视角外观、无置信度）。

#### 关键子问题②：scene completeness（保剩余信息对场景理解完整）—— 很重要
保留集 = 以下并集：
- **单视角独占 token**：只有一个视角能看到的区域 → 单元素簇 → **自动保留**。
- **逐区域覆盖保证**：用对应图/2D 粗布局把视角并集划成场景区域，**每个被占据区域至少留 1 个存活 token**；若去重会清空某区域则保其最佳 token。
- **上下文锚点**：即便 query 无关，保留一组稀疏的背景/结构 token（墙、地、大地标、地平线）建立空间参照系（删光上下文会伤相对方向/距离）；用 FPS 在剩余 token 上选，保证空间分布。
- **簇内视角多样性**：保 K>1 时取**视角差异最大**的（前/后/侧互补），非 K 个相似。

**伪代码**
```python
def C1(views, query=None):
    clusters = cross_view_match(region_embeddings(views))   # 同物体跨视角簇
    kept = []
    for c in clusters:
        if not verify_identity(c):        # 身份核验, 失败则拆
            c = split(c)
        if query and needs_multiview(obj(c), query_type(query)):  # C1-B
            kept += topk_diverse_views(c, budget(obj(c), query)) # 按query需要恢复多视角
        else:                                                     # C1-A
            kept += topk_diverse_views(c, K=default_K, score=quality_score)
    kept += scene_completeness(views, clusters, kept)  # 单视角独占+逐区域覆盖+上下文锚点
    return MLLM(prompt(query), kept)
```
**框架图**
```
[views] ─特征→ 跨视角对应(簇) ─身份核验→ ┬─ query无关: 每簇top-K优质多样代表
                                          └─ query感知: 被指物体按query预算re-densify
                                       → + scene completeness(单视角独占/区域覆盖/上下文锚点) → MLLM
```

**实验设计（TODO）**：基准 All-Angles（counting 验证"去重提精度"）+ Ego3D-Bench。Baseline：FastV/VisPruner/CDPruner/Prune2Drive/SeGPruner/Geo3DPruner/ST-Prune。指标：Acc、**IC**、保留 token 数/压缩率、**端到端 FLOPs/延迟/显存（含对应原语开销）**。消融：C1-A vs C1-B、K、身份核验开/关、scene-completeness 各组件、均值融合 vs 最佳代表选择。

### 计算成本
对应原语一次性（特征+匹配，见 C3 讨论可 pose-free 廉价化）；去重显著降 LLM 视觉 token（注意力二次复杂度 → 实测加速，**必须含对应开销报端到端**）。

### 新颖性裁定 & 区分
**调研裁定：C1（严格限定 pose-free + 纯视觉匹配 + 同物体跨视角 re-id 去重 + training-free）新颖性"较高"，未见撞车。**
| 工作 | 跨视角去重? | 依赖 | 认"同一物体"? | 区分点 |
|---|---|---|---|---|
| ST-Prune 2604.19145 | 是(training-free) | **已知 ring-view 相机布局** | 否(相似度抑制) | C1 **不要相机布局** |
| SeGPruner 2603.29437 / Geo3DPruner 2604.18260 | 是 | **几何/深度先验** | 否(几何聚类) | C1 **pose/depth-free** |
| Prune2Drive 2508.13305 | 各视角独立 | 验证集定 ratio | 否(多样性/覆盖) | C1 **跨视角联合 + token级 + 显式同物体** |
| DART 2502.11494 / ToFu | 否(通用) | — | 否(通用重复) | C1 **显式同物体 re-id** 而非统计相似 |
| OC-VTP 2511.20439 | 单图 | 需训练 | 单图object | C1 **跨视角 + training-free** |
| **Merge3D**(CVPR26) | 是(VGGT几何相似) | pose-free 但**需训 merger(2-3h)** | 否(token级merging) | C1 **完全免训 + 显式物体 re-id**；攻其 **BLINK Multi-View 卡死 55.6%** 的短板 |
| **Seeing Once**(ICLR26-WS) | 是(几何体素重叠) | **硬依赖 posed RGB-D(深度+位姿)** | 否(几何坐标对齐) | C1 **pose/depth-free 纯视觉**；它在 All-Angles/Ego3D **不可运行**（正交） |
| **Proxy3D**(CVPR26) 2605.08064 | 是(语义分组+几何KNN) | pose-free 但**重训(62 GPU-h+318K数据)** | 否(语义类内多proxy) | C1 **免训 + 实例级 re-id + 单一最佳代表**(它一物体多proxy、几何采样) |

**竞品精读确认 + 重要修正（2026-06-04，详见 [论文笔记.md] 精读 7-10）**：
- **C1 的安全 delta（经四篇竞品验证）= ① pose/depth-free（SeGPruner/Seeing-Once 都硬依赖深度位姿）+ ② 完全 training-free（Merge3D/Proxy3D 都要训练）+ ③ 实例级跨视角同物体 re-id（无人显式做，全是 patch级/几何/语义聚类）+ ④ 单一最佳代表的选择准则**。
- **最佳战场 = Multi-View / All-Angles**：Merge3D 在 BLINK Multi-View 子集**卡死 55.6%≈基线**、作者承认"需 token merging 之外的机制"；这正是 C1"显式跨视角同物体 re-id"最可能拉开差距处。只在单视点 grounding/captioning 上比，Merge3D 已很强、C1 难有空间。
- **⚠️ 修正"避免均值融合"卖点**：**只有 Merge3D 用算术均值（Eq.6，有失真）；Proxy3D 用 FPS/KNN 采样代表、SeGPruner 用 FPS、本就不做均值**。→ "我们不用均值"**不能当主要新颖点**；C1 真正的代表-选择 delta 是"**每簇单一最佳代表的显式优选（按信息量/清晰度/可见性打分）**"，区别于 Proxy3D/SeGPruner 的**几何均匀采样多代表**。
- **可复用论据**：Seeing-Once 的"去重→涨点（缓解 attention dilution）"可作为 C1"压缩即提精度"的有利引用。

### 开放问题 / TODO
- 大基线/遮挡下对应鲁棒性（同 R1）；高置信处才去重，低置信回退保留。
- query 感知的"每物体视角预算"映射表需实验标定（哪类 query 需几视角）。
- 去重伤互补信息的早停监控（用 IC/counting）。

---

## C3：干掉几何骨干"白嫖"开销 →「真加速」⚪
一句话：跨视角对应/去重的合并键只需粗对应，不需重建级几何 → 用 MLLM 自带 ViT 特征匹配（免费）或极小蒸馏头替代 VGGT/Point3R。

### Challenge / Motivation
Geo3DPruner/Cog3DMap/VG-LLM 每次跑重几何模型（VGGT-1B/Point3R），用户已观察 Geo3DPruner"加速很有限"正因 VGGT，且**无人把几何骨干开销计入效率账**。合并键只需"够分桶的对应"，不需稠密几何。

### 方案
- **首个该做的消融**：「仅 ViT 特征匹配 / 仅粗位置」 vs 「用 VGGT」→ 证对应质量相当、成本骤降。
- 主线：training-free 视觉特征跨视角匹配（DINOv2/MLLM 自带 ViT）；可选极小蒸馏深度头（DPT-lite on ViT 特征，教师 VGGT/Point3R）。
- **全程报告含几何开销的端到端 FLOPs/延迟/显存**（差异化指标，现有工作无人报告）。

### 新颖性
首个量化并消除几何骨干开销、让多视角压缩"真的快"。与 SeGPruner/Geo3DPruner（都白嫖 VGGT）区分。

### TODO
- 验证 ViT 特征做跨视角对应的精度上限；蒸馏头的 cost/quality 折中曲线。

---

## 引用速查
All-Angles 2504.15280 · Ego3D-Bench 2509.06266 · Prune2Drive 2508.13305 · SeGPruner 2603.29437 · Geo3DPruner 2604.18260 · ST-Prune 2604.19145 · FastV 2403.06764 · VisPruner 2412.01818 · CDPruner 2506.10967 · DART 2502.11494 · OC-VTP 2511.20439 · FastVID 2503.11187 · HoliTom 2505.21334 · BFA++ 2602.20566 · Point3R 2507.02863 · VGGT(见 Geo3DPruner 依赖)
