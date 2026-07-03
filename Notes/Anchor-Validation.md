# Anchor-Validation — 跨视角"承重 token"是否存在 / 模型是否理解跨视角对应（验证 + 机制 + 新方法）

> 工作版，2026-06-30。本文把"anchor 到底有没有用"从**末端 ACC 的泥潭**里拎出来，改用**干净探针 + 几何 oracle**来回答，并据此规划：
> (1) 验证当前 anchor 方法是否有效；(2) 搞清 MLLM **是否/在哪里**实现跨视角对应；(3) 如果空间信息**存在却没被用**，能否用 **LoRA 等轻量微调**把它激活。
> 关联：[[CVSP-Method.md]]（anchor 分定义 §12）、[[GeoScaffold-Story.md]]（query-aware 转向）、[[Visual-Compression.md]]（§实测发现 D–J + §O 全局裁定）、[[CVSP-Story.md]]（锚点 motivation 历史）。
> ⚠️ 状态：问题分解 + 实验骨架已收敛；**几何真值来源（Path 1/2/3）待用户拍板**；逐探针实现规格待定稿后再写脚本。

---

## 0. 为什么要做这件事（动机）

我们方法里的 **anchor（锚点）** 本意：找出某类特殊 token——它们能在**多视角 MLLM 问答**里帮助建立**跨视角空间关系**（"撑起全局 3D 参照系的跨视角一致 + 结构独特的点"）。当前打分是

```
anchor(t) = cornerness(t) × cross-view_Lowe_margin(t)      # 见 CVSP-Method.md §12, scripts/four_way_extreme.py
```

但有两件**从未被正面验证**的事：
1. **这类 token 到底存不存在**——是否真有一个子集，承载了跨视角空间绑定所需的信息？
2. **我们的打分到底选没选对**——`cornerness × Lowe` 选出来的，是不是那批 token？

整条方法线（anchor → engine → query-aware 两阶段）是在**没有回答这两问**的情况下往前推的。本文补这一课。

---

## 1. 问题分解：两个独立命题 + 一个方法学陷阱

- **H1（存在性）**：是否存在 token，**不成比例地**承载跨视角 / 3D 参照信息？等价问法：有没有**任何**选择能在"跨视角空间绑定"上稳超 random？
- **H2（可识别性）**：我们的 anchor 分是否就抓到了那批 token（优于 random / saliency）？

**陷阱（关键）**：此前所有 anchor 实验，效果都用**末端 MC 准确率**衡量。但已坐实（见 [[Visual-Compression.md]] §O、vision_ablation）：

- 这些任务的 ACC 被**语言先验 + distractor 伪信号**主导，视觉只贡献 ~9–12pp；
- 绝对距离 MC 上，一个**不看图、只挑最接近 ~14m 的纯数值启发式**就能拿 0.55（黑图仍 0.31–0.39 ≫ 0.25 chance）。

所以"anchor ≈ random（按 ACC）"是用一把**被污染、低灵敏度**的尺子量出来的——它**既可能掩盖了真效应，也可能本就没效应，ACC 分不出来**。

**两个此前缺失、本文必须补的东西**：
1. **比 MC ACC 更干净的测量工具**（表征级 / 因果级，而非被先验污染的末端 ACC）；
2. **一个 oracle / 上限**——用真几何构造的"真锚点集"。没有它，null 结果无法区分"H1 假（锚点不重要）"和"H2 假（我们没选对）"。

---

## 2. 前序证据回顾（别重跑已证伪的）

| 研究（脚本） | 结论 | 为什么**不足以**回答 H1/H2 |
|---|---|---|
| P4 全局 FPS（`p4_fps_causal.py`） | 把跨视角冗余 R 压到最低,ACC **没超** random | 用 ACC 测；且优化的是 R 这个错代理 |
| 体检（`leverage_diagnostic.py`） | token 池有效秩 ~84;keep10/25 预算 > 有效秩 → 那里谁都赢不了 random | 只说"战场在 keep≤5%",没测锚点本身 |
| 四方对照 / 实验 A（`four_way_extreme.py`） | keep5,anchor 净 ≈ 0（空间任务 +1~3pp、时序 −7pp） | **ACC 工具污染 + 无 oracle**，是本文要修的 |
| anchor 扩散（`anchor_spread.py`） | Layer-1 锚点已跨视角铺开 | 只看分布,没看"是否承载跨视角信息" |
| 视觉消融（`vision_ablation.py`） | 黑图 0.31–0.39、噪声 0.28–0.38、真图只多 ~9–12pp | 证明了**为什么 ACC 不可信**,正是本文出发点 |

> 教训：以上"anchor 无用"的判决，**全部建立在被污染的 ACC + 缺 oracle 之上**。本文不重跑它们，而是**换尺子 + 加上限**。

---

## 3. 几何真值的现实（决定 oracle 能不能搭）

**仓库自带的几何标签不可靠，不能直接当 GT。** 证据（`utils/cam_info.py` + ego3dvlm 深度路径）：

- **外参是编的**：nuScenes 六相机 `translation=[0,0,0]`（**零基线 → 跨视角重投影在数学上失效**）；内参是假圆整数 `[[800,0,800],[0,800,450]]`（真值 `1266.4…` 被注释掉换成假数）；旋转 = `estimate_rt` = 按视角名猜的纯 yaw。Waymo 只有 front 像真,侧视全靠猜。
- **深度是单目估计 × 手调系数**：Depth-Anything-V2 metric × `scale∈{0.64,0.8,0.86}`。非度量、噪声大。
- 故 `unproject()` 的"3D 坐标" = 伪深度 + 假内参 + 猜的 yaw + 零平移 = **粗糙 pseudo-3D**。当 ego3dvlm 软提示尚可,**当 oracle 是不行的**（自证循环）。

**但真值"存在"，只是不在发布版里——可经文件名 ID 回连源数据集**：

| Benchmark | 源 | 文件名 ID | 源数据集真值 |
|---|---|---|---|
| Ego3D | nuScenes | `n015-…__CAM_BACK__1533201470437525.jpg` | LiDAR 深度 + 逐帧真标定 + 3D 框 |
| Ego3D | Waymo | `…segment-128204…_100_FRONT.jpg` | LiDAR + 标定 + 3D 框 |
| Ego3D | Argoverse | `ring_rear_left_315969629022515800.jpg` | LiDAR + 标定 |
| VSI | ScanNet / ScanNet++ / ARKitScenes | manifest `scene`（如 arkitscenes 41069025）+ 帧序 | **稠密传感器深度 + 位姿 + 重建网格 + 3D 实例** |

补充锚：**QA 答案本身来自真值**（如 `answer=13.7`）→ 任何找回的几何都能用 QA 的 GT 距离交叉校验。

**结论：VSI 是更好的几何底座**（室内 RGB-D,稠密准深度 + 重建位姿 + 实例网格,跨视角重叠大）；Ego3D 更难（稀疏 LiDAR + 零基线 + 巨大下载）。

### oracle / GT 来源 —— 已定 **Path 3（混合）**，2026-06-30
- **Path 2 工作 oracle（先行）**：对多视角图跑 **VGGT / DUSt3R / MASt3R**,直接出跨视角对应 + 相对位姿 + 准度量深度,**不依赖源数据集**。远胜零基线假外参;是 pseudo-GT、且引入外部模型（仅诊断用,不进方法）。
- **Path 1 真值封顶（后做）**：下子集——VSI 走 **ARKitScenes/ScanNet++ 最省事**;Ego3D 走 nuScenes-mini。对齐帧 → 真深度+位姿+对应+3D 实例,在小集上校验 Path-2 结论。
- 决定底座 = **VSI**（室内 RGB-D,真 ego-motion → 真基线,远好过 Ego3D 零基线假外参）。

---

## 3.5 VSI 数据结构 + 覆盖预检（实测 2026-06-30）

**VSI 的 QA 是对整段视频（整个场景扫描）构建的,没有 per-question 片段。** 证据（原始 `nyu-visionx/VSI-Bench` `test.jsonl`）：字段 `id/dataset/scene_name/question_type/question/ground_truth/options`——**无 start/end/segment 任何裁剪字段**;一个 `.mp4` = 一个 `scene_name`,平均 **17.8 题/场景**（max 77）共用同一段视频。题型（counting / room_size / appearance_order / route）本就需整场景。**注意**：原始评测**也是抽帧**喂（video-LLM 普遍 8/16/32 帧）→ "整段 vs 片段"指**范围**（整场景）,实际输入永远是抽帧;我们和原始同类,差别只在**帧数**。

**视频很长**：ARKitScenes 中位 **132s（3962 帧 @30fps）**,max **387s/6.5min**。**6 帧 = 每 ~17–24s 一帧 → 太稀疏**（文献标准 8/16/32;Geo3DPruner 笔记：16/32 帧显著优于 8）。**多房间**：VSI 不按房间切,整段 scan = 一个 QA scope,长 scan 过几个房间会加剧稀疏。**→ 决定改用 16 帧均匀采样。**

**覆盖预检 @ 16 帧均匀**（`scratchpad/vsi_coverage_audit.py`,n=48 场景=43 ARKit+5 ScanNet++,InternVL3 ViT 特征空间,token 级 Lowe 跨帧匹配）：

| 指标 | 值 | 读法 |
|---|---|---|
| within-scene match_rate | **0.320** | ~32% token 在别帧找到自信匹配 |
| **NULL（vs 不同场景）** | **0.188** | floor——通用室内纹理（墙/地/家具）跨**无关**场景也匹配 |
| **真重叠信号**（within−null） | **+0.132** | 真实跨帧对应,在通用 floor 之上 |
| 16 帧图连通 | **79%** 场景 | ~21% 碎裂成 ≥2 不桥接子簇（= "过几个房间"） |

**结论（marginal-go）**：16 帧均匀**可用**——有真重叠信号、79% 连通,oracle 可建;但 (1) ~21% 碎裂场景要丢弃或后续加帧;(2) **关键**：ViT 特征匹配有 **0.19 的通用外观 floor**（~60% naive 匹配是跨场景通用纹理,非几何对应）→ **ViT 特征只能当覆盖筛子,不能当 oracle 的对应真值;必须上 MASt3R/VGGT 真几何对应**（正中"外观≠几何"老担忧,直接坐实 Path-2 的必要性）。
**Caveat**：特征空间代理(阈值依赖,信相对结构);样本无 ScanNet（单房间,预计连通更好,偏保守);各任务类型 match_rate 几乎一致(0.29–0.33,重叠是场景属性)。

---

## 3.6 Path-2 oracle 落地（VGGT，实测 2026-06-30）

**已装好并跑通。** `facebook/VGGT-1B`（1.26B 参数,4.7GB 权重）装在**克隆环境 `geo`**（`conda create --clone ego3d -n geo` + `pip install --no-deps git+…/vggt`,**`ego3d` 原封不动**;`geo` = ego3d 超集,可同时跑 InternVL3 + VGGT）。一次前向吃 16 帧,24GB GPU 装得下。输出（16 帧联合,共享世界系）：`pose_enc(相机位姿)` / `depth` / **`world_points`（逐像素 3D,(16,392,518,3)）** / `*_conf`。`scratchpad/vggt_oracle_smoke.py`。

**首个真几何覆盖 @ 16 帧均匀**（n=4 arkit,体素=场景对角/80,top-50% conf）：

| 指标 | 均值 | 读法 |
|---|---|---|
| **covis_mean** | **1.19** | 每体素平均被 ~1.2 帧看到 → 场景大部分**只被一帧**看到 |
| covis2 | 0.17 | 被 ≥2 帧看到的体素占比（几何跨视角重叠） |
| frame_cov | 0.275 | 一帧约 27% 内容与别帧重叠 |

**读：16 帧均匀的几何重叠"适中偏少"**——帧多在铺新内容,~1/4–1/3 与邻帧重叠;与 ViT 代理的 +0.13 互相印证,且**几何无外观 floor**（同体素 = 真同 3D 位置）。**含义**：跨视角对应**存在但不充裕** → 探针要在**共视 token 对**上测绑定（否则"不绑定"可能只是"无重叠可绑"）;若对应太稀疏再考虑 24–32 帧。
**坑（探针对齐）**：InternVL3 `Resize((448,448))` 把**整帧**压成方图（归一化坐标线性映射,可直接对齐 token 网格）;VGGT 默认 `mode="crop"` 会**中心裁掉上下** → A1 必须用 **`mode="pad"`** 保整帧覆盖,并按 pad 偏移映射。

---

## 4. 实验设计（分层 + 分阶段 gate）

核心轴：**用什么干净工具替代被污染的 MC ACC**。证据强度从弱到强：相关（A1/A3）→ 表征（A2）→ 因果（B）。

### Approach A — 表征 / 对应探针（不走 QA，主力）
分阶段 **A1 → A3 → A2**,前一步结果 gate 后一步。

**A1｜锚点分 vs 真实跨视角对应**（最便宜,连 LLM 都不用前向）
- 输入：各视角 ViT 特征 + oracle 对应（视角 i 的 patch ↔ 视角 j 的同一 3D 点,来自 §3 选定的 GT 路径）。
- 测：我们的 `cornerness × Lowe` 是否把**真对应**排在前面（precision/recall、AUC vs random）。
- 决定性读：分高的 token 若与真 3D 对应无关 → 我们测的是**外观相似,不是几何对应**（正是 [[Visual-Compression.md]] §开放问题 标记的风险）。

**A3｜跨视角注意力质量**（便宜,机制线索）
- 用 `AttentionCapture` / qstage 抓 LLM 注意力,量**视角间**（token_i^viewA → token_j^viewB）注意力质量,以及是否集中在高 anchor token。
- 注意去偏（AdaTP：注意力系统性偏向靠后视角）——看相对集中度,不信原始有偏值。

**A2｜表征可解码性（linear probing,最强存在性证据）**
- 手法：**冻结模型**,取 token 在某层 hidden state,训**极简线性探针**预测几何属性;能线性读出 ⇒ 表征**编码**了它。
- **A2a 解码绝对 3D 位置**：`h → 回归 (x,y,z)`（标签来自 GT 几何）。看 **anchor vs random** 的**相对**可解码性（绝对值被外观混淆,故看相对差）。
- **A2b 解码跨视角对应（核心）**：跨视角 token 对 `[h_i,h_j]（或 |h_i−h_j|）→ 二分类 同点/非同点`（正负样本来自 A1 的真对应）。直接测"表征是否支持**跨视角绑定**"。
- **多层扫描**：ViT 输出 + LLM 早/中/晚层 → 回答"跨视角对应**在哪一层**变得可解码（若有）"。
- 落仓库：复用 ego3dvlm 几何打 3D 标签（坑：InternVL3 2×2 pixel-shuffle,256 token 各对应一个 2×2 patch 区域,标签在该区域对有效深度求平均）;用 hook 抓多层激活;sklearn 线性/逻辑探针 + 交叉验证。
- **控混淆**：相对比较为主;视角均衡采样;三组（anchor/random/oracle）同探针预算;打乱标签 null 当 chance。
- 决定性读：某层可解码且 anchor > random → **H1 成立 + 定位机制层**;**从头到尾解码不出**（不超 null/外观基线）→ **模型没建跨视角几何绑定,H1 倾向证伪**（很硬的发现）。

### Approach B — 因果敲除（封顶,证明"真的用",非"只是存了"）
- 在**干净探针**（curated 跨视角子集,见下）上,移除 {anchor / random / **oracle**} token,看准确率掉多少。
- oracle 助、anchor 不助 → H1 真、**H2 假 → 方法可修**;oracle 也不助 → **H1 假 → 锚点前提死,pivot**。

### Approach C — 合成受控探针（仅当 A+B 模棱两可时）
- 构造**强制跨视角**关系题 + 平衡选项,完全控混淆。代价:构建 + 合成-真实差距。

> curated 干净子集：选"两个被问物体分处不同视角"的题（强制跨视角绑定）,并控语言先验（平衡选项 / 用相对方向题）。

---

## 5. 判定标准（go / no-go 决策矩阵）

| A1/A3 | A2（跨视角可解码） | B（oracle 敲除） | 判定 | 行动 |
|---|---|---|---|---|
| 弱 | 解码不出 | oracle 不助 | **H1 假** | 锚点/几何承重前提死 → pivot（回 query-aware 覆盖型 或 纯语言先验路线） |
| 任意 | 可解码 | oracle 助、ours 不助 | **H1 真、H2 假** | 方法可修 → 用 oracle 反推更好的打分（进入原目标 2） |
| 强 | 可解码 | ours 也助 | **H1 真、H2 真** | 当前 anchor 有效 → 强化为正式精度主张 |
| 任意 | **可解码但 B 中谁都不助** | — | **存而不用** | → 第 6 节（LoRA 激活) |

统计现实：n=200 时 1SE≈3pp,差异低于此即噪声;探针用交叉验证报置信区间。

---

## 6. 前瞻：新问题 + 新方法（本调查的真正价值）

把上面的探针当**透镜**,它们直接催生新课题:

1. **MLLM 到底懂不懂跨视角对应？** → A2b 的二分类正答率直接回答。这是个独立可发表的探测性结论（"多视角 MLLM 是否在内部建立跨视角对应"）。
2. **在哪一层实现？** → A2/A3 的层扫描定位"绑定层 K\*"。若存在 → 一切层级干预（剪枝、re-densify、微调）都该锚在 K\*,而不是拍脑袋的 L/2。
3. **"存而不用"→ 用 LoRA 把信息激活（最有潜力的方法线）**：
   - 逻辑（probing→intervention,Alain-Bengio 式）:若 A2 显示信息**线性可解码**、但 B 显示**行为上没用到**（敲除几何富 token 不掉分）⇒ 这正是"信息在、通路不通"的教科书情形 → **一个小 adapter 就可能把已存在却闲置的信号接进答案**。
   - 具体设计:在**绑定层 K\***（A2 定位）插 LoRA;两种目标——(a) 仅任务 LoRA,看是否**自发**开始用跨视角信号(B 重测);(b) 加**辅助跨视角对应/对比损失**(用 oracle 对应当弱监督),显式逼模型对齐同一 3D 点的跨视角 token。**pose-free**:监督来自 oracle,推理时不需要。
   - 可证伪主张（潜在 paper arc）:**"多视角 MLLM 编码了跨视角对应却不利用;在层 K\* 插一个微小 adapter 即可解锁,clean 跨视角探针 ACC 上升 X pp,token 预算不变。"** 把"压缩"故事升级成"激活未利用空间信息"的故事。
4. **位置编码 / RoPE 干预**：若绑定失败源于跨视角 token 共享/冲突的位置框架（Nüwa PESP/RPME 主题),可单独消融位置重映射对跨视角对应可解码性的影响。
5. **反哺压缩**：一旦知道"哪些 token 承载跨视角绑定、在哪层被用",query-aware re-densify 就有了**有据可依的保护/精修目标**,而非当前的启发式。

---

## 7. 开放决策 + 下一步

- **待拍板**：§3 的几何 GT 路径（推荐 Path 3 混合：VGGT/MASt3R 主力 oracle + ARKitScenes/ScanNet++ 小真值封顶）。
- **拍板后**：把 A1→A3→A2（+B 封顶）连同各探针的"输入/标签/指标/判定阈值/用到的仓库文件"定稿成实现规格 → 再写脚本（`scripts/anchor_probe_*.py`,沿用 `four_way_extreme.collect` + AttentionCapture + ego3dvlm 几何）。
- 实现遵循仓库惯例:`ego3d` 环境、`torch.set_grad_enabled(False)`、per-(task,probe) JSONL 可断点续跑、确定性种子。

## 8. oracle 上限测试(2026-07-01)—— ⚠️ 本节 NO-GO 结论**已被 §9 撤回**(oracle 有去重缺陷)

> ⚠️ 时效:本节用的 oracle **anchor 桶做了跨视角去重(τ=0.85)**,把跨视角一致地标的多重实例删掉了——正是 anchor 的绑定信号。**故这里的 NO-GO 无效**,见 §9 无去重版。保留本节仅作方法演进记录。

> 用户目标 = 找到能支撑 visual-token 压缩的 anchor 计算。**决定性做法**：把"完美几何 anchor"(VGGT 真几何:共视度×表面变化 σ 的 distinctive 跨视角地标) 当**保留 token 的选择器**,在真实 VSI 任务上对比 random——若连**完美 anchor** 都赢不了 random,则没有任何 training-free 打分能救,方向死。`scratchpad/oracle_compress.py`,4 个 VSI MCA 任务(rel_dir easy/hard、rel_dist、route),n=50,logs/oracle/。

| 预算 | oracle 设计 | MEAN ACC vs strat vs random | 裁定 |
|---|---|---|---|
| keep10 | 100% anchor | 0.455 / 0.450 / 0.435 | ≈ random（0/4 胜 >1SE） |
| keep10 | 20% anchor+80% strat-rand | 0.415 / 0.450 / 0.435 | ≤ random（4 任务全 −0.02~−0.04） |
| keep5 | 20% anchor+80% strat-rand | 0.400 / 0.420 / 0.413 | ≤ random（三者最低;赢 easy、输 rel_dist −0.12/route −0.12） |

**结论：完美几何 anchor 在 keep10 与 keep5、无论 100% 还是 20%-mix,都 ≈ 或 < random。** 因此**"找更好的 anchor 打分"没有上限空间**——瓶颈不是"没找到对的分",是**anchor 式选择相对 random 本就无增益**。与全局 verdict（informed ≈ random、视觉盘子小、random 近最优）一致,并**把它从"我们的分不行"升级到"连 oracle 都不行"**。

**Caveat（诚实）**：(1) n=50,SE~0.07,单格差异在噪声内,但**跨 2 预算 × 4 任务 × 2 oracle 设计的平坦/负向格局稳健**,翻盘不太可能;(2) VGGT 是 pseudo-GT(Path-1 真值可细化,难翻平结论);(3) **未测 keep3/keep1**(最极端,最后一线) 与其他 oracle-score / ρ_a;(4) 本 oracle = **query 无关的几何地标**,≠"被问物体 token"——它杀死的是**query-agnostic 几何 anchor** 假设(query-aware 此前 §O 也未整体赢 random)。

---

## 9. 修正版:无去重 oracle → **anchor 存在(任务分层的剂量正效应)**(2026-07-01)

> §8 撤回。关键修正(用户洞察):**anchor 不该跨视角去重**——去重把"同一地标在多视角的一致实例"删了,而多重性正是绑定信号。改用**无去重 mix + 纯 per-view uniform random 基线**(`scratchpad/oracle_compress.py`,`logs/oracle/vsi.*.jsonl`)。oracle anchor 分 = **(covis_degree≥2) × covis_degree × σ**(共视度 × 表面变化,VGGT 真几何),无去重、每视角保留实例。

**演进(keep10,VSI):**
- **去重 mix20** ≤ random(§8);**无去重 mix20** → +0.02(4 任务),且跨 **n=50/150、16/32 帧**三次复现 → **不是噪声,anchor 真实存在**。
- **32 帧没放大**(Δ 仍 +0.02)→ "重叠是瓶颈"证伪;+2pp 对帧数不敏感。
- **ρ_a 扫(0.2/0.4/0.6,6 任务,n≈100–150)**:**平均 Δ ≈ 0**(−0.008/+0.010/+0.008)——**非普适胜利**(4 任务的 +0.02 是任务选择)。**但有连贯分层**(ρ_a=0.6):
  - 三个**跨视角关系任务全为正且随 ρ_a 单调↑**:rel_dir_hard +0.014→+0.028→**+0.042**、rel_distance −0.013→0→**+0.033**、rel_dir_medium(−0.089 异常)→+0.044→**+0.037**。
  - **非关系任务无用/有害**:easy(+0.041→−0.041)、route(→0)、**object_counting(MRA 全负)**(计数要覆盖不要地标)。

**裁定:** **anchor 存在,是"跨视角关系推理"的剂量依赖工具**(越难越吃、ρ_a 越大越好),不是普适压缩杠杆;被覆盖型/简单/时序任务冲平均值。⚠️ 证据强度:每格 SE~0.04–0.05,关系任务 +0.03~0.04 约 ~1SE——**证据 = 三任务方向一致 + 剂量响应,非单格**;且这是 **query 无关几何 anchor 的下界**(VGGT+跨视角匹配本身有损,真 anchor 值可能更高)。

**下一步:** (a) 只在 3 个关系任务上加 n(全量)+ 试 ρ_a=0.8,坐实剂量响应到 >2SE;(b) 据此设计 feature-based anchor(见下)。

> ⚠️ (此处旧"停止打分迭代"的行动已被 §10 推翻——纯特征 anchor 端任务证实有用。)

---

## 10. **纯特征 anchor 端任务证实有用 + A1 是假阴性**(2026-07-01)

> 目标锁定**纯特征**(推理零额外模型)。设计**新特征 anchor**镜像几何 oracle:`anchor_feat = support × sharpness`(纯 InternVL3 特征 Gram 算),
> - `support(t)` = 有多少其他视角有**又强又锐**的匹配(s1>τ 且 Lowe margin>m)≈ covis_degree(多重性);
> - `sharpness(t)` = 这些 matched 视角上 margin 均值 ≈ σ(几何独特性,纹理鲁棒)。
> 核心区别于 a20s40 的 `cornerness×lowe_max`:**多重性计数**替单个最佳匹配、**margin 门控**滤重复纹理、**跨视角对应锐度**替帧内外观独特、**无去重**。

**A1(vs VGGT covis 的 AUC)= 假阴性,别再用。** 新旧特征分**全 ≈ 0.5**(anchor_feat 0.49、cornerness×lowe 0.50、lowe 0.50、random 0.51)。但这**证明的是 AUC-vs-covis 是坏量具**,不是分不行——原因:(1) 标签(裸共视)是墙主导、不是 anchor 目标;(2) **AUC↛任务**(几何 oracle 对自身目标 AUC=1 却只 +0.02);(3) pooled 混场景尺度、AUC 测全域而选择只看 top-k;(4) VGGT 噪声假阴性。**结论:端任务才是真判据。**

**端任务(no-dedup mix vs random,3 关系任务,keep10,n≈135–150):**

| 任务 | random | feat ρ0.2 | **feat ρ0.4** | 几何 oracle ρ0.4(参照) |
|---|---|---|---|---|
| rel_dir_medium | 0.422 | +0.007 | **+0.030** | +0.044 |
| rel_dir_hard | 0.324 | −0.042 | −0.007 | +0.028 |
| rel_distance | 0.353 | +0.020 | **+0.027** | +0.000 |
| **mean Δ** | | −0.005 | **+0.016** | +0.024 |

**裁定:纯特征 anchor 有效。** feat ρ0.4 mean Δ **+0.016**(2/3 正、剂量响应 0.2→0.4↑),达几何 oracle(+0.024)的 **~2/3**,且**推理时零额外模型**。⚠️ n~140、约 <1SE——支撑=一致性+剂量响应+达几何 2/3,非单格。**例外:hard 上特征仍差几何一截**(最难任务更依赖真几何)。`scratchpad/oracle_compress.py --anchor feat`,`logs/oracle/*.featmix*.jsonl`。

**下一步(进行中):** (1) τ/m 扫求最优;(2) `support×sharpness` vs a20s40 原 `cornerness×lowe_max` 同框架头对头(哪种 anchor 算法更好);全 3 源、加 n 保可信。

---

## 11. 头对头(feat vs a20s40)+ τ/m 分布 + support 预算核查(2026-07-01)

**头对头**(全 3 源,n=200/任务,keep10,ρ_a=0.4,no-dedup mix,`scratchpad/anchor_compare.py`,`logs/oracle/*-as.jsonl`):

| 方法 | mean Δ vs random |
|---|---|
| cornlowe (a20s40 原 `cornerness×lowe_max`) | −0.007 |
| feat τ.5/m.05 / τ.6/m.05 | −0.002 / −0.007 |
| **feat τ.6/m.12** | **+0.022** ← 最优 |
| feat τ.7/m.05 / τ.7/m.12 | +0.007 / +0.008 |

- **方向:新 `support×sharpness`(最优 τ0.6/m0.12,+0.022)> a20s40 原 `cornerness×lowe_max`(−0.007);cornlowe ≈ random**。差 ~0.029(~1.5SE)。
- ⚠️ **全在噪声内**(SE~0.02,最优 +0.022≈1.1SE)、**τ/m 敏感 + 跨数据集没复现**(arkitscenes τ0.5/m0.05 曾 +0.016,全源变 −0.002;最优档漂到 τ0.6/m0.12)。**"新分更好"是方向性,非显著。**

**τ/m 真实分布**(`scratchpad/tm_dist.py`,737k token-view 对):
- **s1(最佳跨视角余弦)**:中位 0.53;s1>τ 留存 τ0.5→60% / τ0.6→26% / τ0.7→8% / τ0.8→2%。有效范围 τ∈[0.5,0.75]。
- **margin(s1−s2)极小**:中位 **0.019**;margin>m 留存 m0.05→17% / m0.10→4.5% / m0.12→3.1% / m0.15→2% / m0.20→1.3%。有效范围 m∈[0.01,0.15]。
- **机制洞察**:margin 中位仅 0.019 → **绝大多数跨视角匹配是模糊的**(s1≈s2,重复纹理/语义重复);**只 ~3% 是锐利唯一对应 = 真地标**。→ **验证"sharpness 才是关键"的设计,也解释增益为何只 ~2pp**(锐利 anchor 本就稀少)。

**support 预算核查**(用户 concern,`scratchpad/support_count.py`):**3% 是 per (token,view-对),非 per token。** support 跨 15 视角计数、且锐利匹配集中在真地标上 → 在 τ0.6/m0.12 下 **每视角 42 个 token 有 support>0(~16%),远超 n_a=10 预算 → anchor 桶 100% 真锚点,无零填充**。预算仅在 **m≈0.20**(12/view≈10)才开始吃紧 → **m 安全上限 ~0.15**。

**当前最优(暂定)= τ0.6 / m0.12**;τ0.6 内部合理,m0.12 在网格边界、趋势指向更严但 m≤0.15 为安全上限。

**下一步(坐实)**:固定 τ=0.6,扫 **m∈{0.10,0.12,0.15}** + cornlowe + random,**n≈400 全源**(pooled~1200,SE~0.014),看 (a) 最优 feat 是否稳过 random>2SE、(b) 是否稳过 cornlowe。

---

## 引用 / 相关文件
- 方法/打分：`Notes/CVSP-Method.md §12`、`scripts/four_way_extreme.py`（`cornerness`/`lowe_max`/`collect`）；oracle 上限 `scratchpad/oracle_compress.py`
- 机制工具：`compressors/internvl_adapter.py`（`AttentionCapture`）、`compressors/qstage_llm.py`（层级 hook 范式）
- 几何：`utils/cam_info.py`（⚠️ 外参编造）、`utils/common.py:unproject`、ego3dvlm 深度路径（`models/internvl3_ego3dvlm.py`）
- 证据基线：`logs/ablation/`（vision_ablation）、`Notes/Visual-Compression.md §O`、`Notes/CVSP-Story.md` 实验现状
