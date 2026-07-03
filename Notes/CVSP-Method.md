# CVSP — 详细方法（v1 定型 2026-06-18；v2/v3 更新 2026-06-20，先读 §0）

> 本文件是 CVSP 的**权威方法规格**：abstract、符号、三个信号的算法与公式、预算分配、
> 选择流程、复杂度、消融与实验设计。故事/动机见 [CVSP-Story.md]，实测档案见
> [Visual-Compression.md]。
>
> **与旧设计的关系**：旧版（[CVSP-Story.md] §思路日志）的演进是「压冗余→纯锚点 v1→体检→
> 四方对照→combo」。本版是用户 2026-06-18 拍板的**定型设计**，三处关键改动：
> 1. **配额式三档**：anchor / saliency / coverage 各按预算配额**分别选 token**，不是加权
>    求和成一个分（理由见 §4：query-agnostic 下"保证每类有下限"比"押一个固定权重"鲁棒）。
> 2. **合并挑战(1)(3)**：逐视角"跨视角冗余"与"显著但无关"合为一条，统一由 coverage 治
>    （query-agnostic 无法直接筛相关性，只能去冗余、间接抬高相关 token 的存活概率）。
> 3. **saliency 重定位**：是"**保住前景/物体内容的下限**"，不承诺"筛掉无关"。
>
> **诚实定位**：截至 2026-06-18 的实测（[Visual-Compression.md] §D–J）下，**"informed > random"
> 的精度主张尚无证据**；本方法把赌注押在**极端压缩区（keep≤5%，预算 < 有效秩 ~84）**——
> 体检结论说那里选择质量才开始起作用，是验证留下的唯一缺口。成败由 §8 实验定。

---

## 0. 版本演进与现状（2026-06-20）

> 本节是最新总览。**§1–11 是 v1（三桶配额定型版，2026-06-18）**，其预算默认、★2 阈值实现已被 v2 取代（逐条标注）；**§12 = v3 Block-CVSP（当前主攻）**。实测数据见 [Visual-Compression.md] §K/§L。

**v1（2026-06-18，§1–11）**：三桶配额 anchor/saliency/coverage（ρ=0.4/0.3/0.3）+ 跨视角 dedup + facility-location coverage + φ 保底；★2 锚点阈值用 delta_q 分位。

**v2（2026-06-19/20，实测修正，见 §K/§L）**：
- **★2 改为预算相对 κ**：实测 L **无零值、中位 ~0.03**，delta_q 分位**根本不咬**（B_a/M≈ρ_a·r，要 delta_q>0.98 才生效、且跨档不一致）→ 改 **anchor 合格池 = top ⌈κ·B_a⌉ by L，κ=2**（只提质、count 仍 B_a；"少锚→多覆盖"由 ρ 显式给，不靠阈值溢出）。
- **预算 sweep（keep5）**：anchor-heavy（ρ_a=0.4）**最差**；砍 anchor 都更好。胜出折中 **a20s40 = ρ_a 0.2 / ρ_s 0.4 / ρ_c 0.4**（取代 §4 的 0.4/0.3/0.3 默认）。
- **结果**：a20s40 vs **VisPruner 18/27 赢（p≈0.04，稳）**；vs **plain_random** 原 6 任务 14/18、**新任务 1/9（Localization 输给 random）**→ **赢 VisPruner、过不了 random-general**。
- **关键诊断**：① **固定比例 ≠ 固定数量**，ρ_a 应 ∝1/r（固定比例只在一档对）；② 锚点供给 ≈ **固定**（top~1% by L ≈15）→ 锚点应**固定数量**；③ **纯锚点 ≈ random**（+0.3pp、6/12 —— anchor 单独价值**存疑，必须消融**）；④ 有效秩 ~84 → keep10 预算>秩，谁都赢不了 random；⑤ **task-dependent**：Localization 要覆盖、rel_distance/rel_dir_hard 要锚点 → query-agnostic 固定 ρ 的天花板。

**v3（2026-06-20，§12 Block-CVSP）**：针对 v2 两个病（跨档不通用 + Localization 覆盖差）的重构 —— 锚点改**固定数量**（ratio-invariant）、coverage 改**块内显著（空间硬结构）**。参考 Nüwa（2602.02951）stage-1。**当前主攻方向。**

---

## 1. Abstract

**背景。** 多模态大模型处理多视角输入时要把每个视角的视觉 token 全部喂给
LLM（5–7 视角 × 每视角数百 token），token 数爆炸、主导算力与显存；而视角高度重叠、token
严重冗余（这是我们的实验发现）——压缩是刚需。

**挑战。** 现有视觉 token 剪枝是**逐视角、靠显著性**的（如 VisPruner）。搬到多视角失灵：
① **逐视角各挑显著 token，把同一前景在每个重叠视角各留一遍**，预算砸在跨视角重复上；
（且query-agnostic 分不清哪些显著 token 真有用（这两点本质同源、合为一条），这句话作为注释帮助理解即可）；
② 它**不知道哪些token 是支撑跨视角几何对齐的锚点**，而空间题恰恰依赖这种对齐。

PS：与此同时，**纯随机剪枝是出奇强的基线**（多视角 token 集低秩冗余），任何 informed 方法必须先赢过随机才有意义——而在
keep10/25 这种"预算 > 有效秩"的区间，谁都赢不了随机，差距只可能出现在极端压缩区。

**洞察。** 一个好的跨视角保留集应**同时**满足三件单信号方法各缺一块的事：**锚点**（跨视角
反复出现、落在几何地标上 → 撑空间配准）、**显著**（前景/物体内容 → 保住任务相关内容的下
限）、**覆盖**（铺得开、不塌缩到重复上 → 去冗余、保多样）。三者互补，此前没有方法把三者在
**跨视角层面**合到一起。

**方法。** CVSP（Cross-View Support Pruning）：**training-free、pose-free、query-agnostic** 的
跨视角 token 选择器。给每个 token 算锚点分与显著分，按**三档配额**（锚点档 / 显著档 / 覆盖
档）+ **每视角保底**，在**跨视角去重约束**下分阶段贪心选取，使最终保留集**既是锚点、又显著、
还不冗余**。作为 LLM 前的即插即用模块，降 token 提效率，并以"**把可用压缩率推到比随机更极端**"为额外的优势叙事。

---

## 2. 符号

- 视角 `v = 1..V`；token `t` 属于 `view(t)`；LLM 级特征 `F_t ∈ R^d`（InternVL3 = 像素重排后
  每 tile 256 维 token；Qwen = merger 后 token）。单位化 `f_t = F_t / ‖F_t‖₂`。
- 场景 token 总数 `M = Σ_v N_v`；预算 `B = ⌈r·M⌉`，`r` = keep_ratio。
- 余弦 Gram `G ∈ R^{M×M}`，`G_{ts} = f_t·f_s`。**全程只算这一个矩阵**，三个信号都从它取。
- `imp(t)` = 编码器自带的重要性线索（InternViT CLS→patch 注意力，经模型自己的 pixel_shuffle
  聚到 token 粒度；Qwen 用 attention-received）。adapter 已实现，直接复用。**逐视角算**（每张图
  自身前向产出），用时对**全场景** rank_norm（见下）。
- `rank_norm(x)` = 全场景分位归一：`rank_norm(x)_t = (rank_升序(x_t) − 1) / (M − 1)`，并列取
  平均秩 → 均匀落在 `[0,1]`。比 min-max 抗长尾（锚点/显著分布偏）；作用是让 a、s 量纲可比。

---

## 3. 三个信号

### 3.1 锚点分 a(t) —— 跨视角几何地标
两个因子相乘，**两者缺一不可**：

**(a) 角不角 cornerness c(t)（方法 B：特征 vs 同视角邻居）**

```
c(t) = 1 − mean_{ s≠t, view(s)=view(t) } G_{ts}
```
与本视角其他 token 越不像越高 → 角/边/物体边界（局部独特），而非平坦背景。
（方法 A = 原图 Shi-Tomasi 梯度，留作对照，不默认。）

**(b) 跨视角支持 L(t)（Lowe 比，对视角取 max）**

对每个其他视角 `u`，取 `t` 在 `u` 内的最相似 `s₁(u)`、次相似 `s₂(u)`。**`s₂` 的空间间隔实现**：
每视角 token 在已知 `h×w` 网格上，定位 `s₁` 的网格坐标后屏蔽其周围 `(2ρ+1)×(2ρ+1)` 窗口
（默认 `ρ=1`，即 8 邻域），`s₂` 取窗口外的最大相似——避免相邻 patch 几乎相同把 Lowe margin
自我抵消。`ρ` 可消融，`ρ=0` 退回普通 Lowe（建议先 `ρ=0`，margin 普遍偏小再开 `ρ=1`）。
```
supp_u(t) = [ s₁(u) − s₂(u) ]₊ · s₁(u)          # 独特匹配(s₁≫s₂) 且 匹配强(s₁高)
L(t)      = max_{ u ≠ view(t) } supp_u(t)        # 一个置信跨视角匹配就够
```
**用 max 不用 Σ**：2 个视角即可三角化一个 3D 点；用 Σ 会奖励"被更多视角看到"，系统性埋没
低重叠（驾驶/Ego）数据里只在 2 个视角出现的真锚点（用户早期指出的问题，此为修正）。

**合成并归一：**
```
a(t) = rank_norm( c(t) · L(t) )  ∈ [0,1]
```
逐字解释：**`c(t)·L(t)` 是"与门"（乘积，非求和）**——只有"局部独特 `c` 高"**且**"有置信跨视角
匹配 `L` 高"两条都成立才得高分。平坦区到处能匹配（`L` 高 `c` 低）→ 乘积低；独特但无跨视角回声
（`c` 高 `L` 低）→ 乘积低；唯有真锚点（两者皆高）存活。求和会让单轴特别强的 token 冒头，乘积才
表达"两者皆需"。外层 **`rank_norm`** 把这个偏态乘积压成全场景分位 `[0,1]`，使 `a` 与 `s` 同尺度
可比、抗长尾。

### 3.2 显著分 s(t) —— 前景/物体内容的下限
```
s(t) = rank_norm_scene( imp(t) )  ∈ [0,1]
```
就是 VisPruner 用的那条线索；区别在我们把它当**有上限的一档配额**用，而非主导信号，故不会
像 VisPruner 那样独占预算、跨视角留一堆重复（重复由 coverage 去）。

### 3.3 覆盖算子 —— 去冗余 / 不塌缩
- **跨视角冗余**（候选 `t` 对当前已选集 `S`）：
  ```
  red(t, S) = max_{ s∈S, view(s)≠view(t) } G_{ts}
  ```
- **去重门**：选 token 时若 `red(t,S) > τ`（默认 `τ = 0.85`）则跳过——压跨视角近重复。
- **覆盖档贪心目标（默认设施选址，对"被丢弃集"算覆盖）**：
  ```
  设施选址(默认): argmax_{t∉S} Σ_{u∈被丢弃} [ G_{tu} − max_{s∈S} G_{su} ]₊   # 最大化对丢弃 token 的代表
  最远点  (对照): argmin_{t∉S} red(t, S)                                     # FPS 式，留作消融
  ```
  **为何设施选址而非最远点**：最远点 = FPS，专挑离群极值，易捞到异常/噪声 token（P4 已验证：FPS
  冗余最低却没赢 random）。设施选址挑"代表了还没被覆盖的那堆 token"的点，偏稠密区代表、不偏离群，
  更稳。**坏处（诚实）**：偏多数派质量 → 可能漏掉稀有但真有用的 token（单视角独有真实物体、质量小被
  跳过）——正好是 FPS 的反向毛病，但被锚点档（专挑跨视角独特）与 φ 保底兜住。

---

## 4. 预算分配（三档配额，保底并入覆盖档）

```
锚点档 B_a = round(ρ_a · B)        默认 ρ_a = 0.4
显著档 B_s = round(ρ_s · B)        默认 ρ_s = 0.3
覆盖档 B_c = B − B_a − B_s         默认 ρ_c = 0.3   （含每视角保底，见 §5 阶段6）
每视角保底 φ = 1（至多 2）           小额保险，由覆盖档用覆盖准则满足，不单列预算
```
**为什么是配额而非加权和**（Q1 决策）：query-agnostic 下我们不知道当前任务更要几何还是更要
前景；加权和 `α·a+β·s` 的 `α/β` 是**全局固定盲配比**，且两分量纲不同、易塌向数值大的一边，
把另一类饿死（空间题饿掉锚点、物体题饿掉前景）。**配额 = 保证两类都有下限**，比押固定权重
鲁棒，量纲无关（只用各自档内排序），且每档可单独开关做消融。`ρ` 与 `φ` 是超参，§8 扫。

**ρ 的待办（记下保留意见）**：默认 0.4/0.3/0.3 先跑；但 coverage 是去冗余主力（挑战 1+3 全
靠它），**`ρ_c` 可能仍偏小、`ρ_a` 可能偏大**——效果不好就把预算往 coverage 挪（§8 扫 ρ）。

**保底为何并入覆盖（采纳用户意见）**：旧版"阶段0 先按 a+s 抢 `0.5·B/V` 个"有三病——(1) keep5
时吃掉近半预算、架空三档；(2) `a+s` 混合稀释各信号；(3) 挤占 coverage。改为：锚点、显著先选；
coverage 阶段**先**把低于 φ 的视角用覆盖准则补到 φ，**再**全局填满。锚点+显著都选不到的视角本就
信息少，留 1 个覆盖代表足矣；保留 `φ≥1` 是因为纯设施选址可能把冗余视角整个删掉，而 ≥1/视角的
配准保险很便宜。

---

## 5. 选择算法（三档贪心，保底并入覆盖档）

```
输入: F(M×d), view(·), imp(·), r, ρ_a, ρ_s, τ, φ
1. f ← rownorm(F);  G ← f fᵀ                              # 唯一的 M×M matmul
2. 算 a(t)=cornerness×Lowe-max, s(t)=rank_norm(imp)
3. B ← ⌈rM⌉; 算 B_a, B_s, B_c;  S ← ∅
4. 锚点档: 余下按 a(t)↓ 加入直到本档计满 B_a; red(t,S)>τ 跳过(跨视角去重)
5. 显著档: 余下按 s(t)↓ 加入直到本档计满 B_s; red(t,S)>τ 跳过
6. 覆盖档(含保底):
   6a. 保底: 对每个 token 数 < φ 的视角, 从其余下 token 按设施选址增益挑, 补到 φ
   6b. 填满: while |S| < B: 加入 argmax_{t∉S} 设施选址增益(对被丢弃集)
7. 返回 sort(S)                                            # 保持原始顺序喂 LLM
```
- **溢出规则**：某档在去重门下凑不满（稠密场景），缺额顺延到下一档/覆盖档，保证 `|S|=B`。
- **确定性**：无随机性（各档都是确定排序+贪心）；与现有 random/vispruner 一样按 (sample_id)
  可复现、断点续跑安全。
- **预算口径（更正）**：各方法相等的是**总 token 数 B** → FLOPs/KV/显存相同（LLM 看到的是拉平
  序列，与每视角怎么分无关）。CVSP **跨视角自适应分配**（信息多的视角多给、冗余视角少给，每视角
  数不同，只有 φ 下限），与 VisPruner/random 的**逐视角等额恰恰相反**——这是 feature。

---

## 6. 复杂度 & 工程

- **计算**：一个 `M×M×d` matmul（`M≈1.5k–2.5k` → 一次 GPU matmul，毫秒级）+ 贪心。设施选址
  的边际增益用 **lazy-greedy（子模性）+ 已算好的 G** 维持 ~`O(B·M)`，不必每步 `O(M²)` 重算。
  无训练、无额外前向、无 pose、无 VGGT、无第二个大模型。
- **落地**：在 `compressors/` 加 `CVSPCompressor`（注册进 `__init__.py`）。它需要的不只是
  单图 `select`，还要**跨视角**信息 → 走"先收集全场景 token 再联合选"的路径（参考
  `scripts/four_way_extreme.py` 里已实现的 `cornerness / lowe_max / to_perview /
  build_prompt_var`，把它产品化进 adapter + 一个新 runner，或直接扩 `four_way_extreme` 的
  engine 为正式方法）。显著线索复用 `internvl_adapter.AttentionCapture`。

---

## 7. 每个组件为什么不能删（消融预期）

| 去掉谁 | 退化成 | 已知结果 |
|---|---|---|
| 去锚点 | 显著 + 覆盖 ≈ VisPruner 跨视角版 | §I/J：不稳赢 random |
| 去显著 | 锚点 + 覆盖 ≈ "engine" | §J：净 ≈ 0、时序任务 −7pp |
| 去覆盖 | 锚点 + 显著、无去重 | 挑战(1+3) 回归：跨视角重复挤占预算 |
| 去每视角保底 | 某视角可能整个被删 | 跨视角配准断裂，空间题掉分 |

**novelty 必须押在**：多视角 3D 空间 regime + 几何地标质量分 + "**在 random 击败显著与多样的
地方，三档联合 + 跨视角去重 + 低重叠 max-Lowe 把可用压缩率推得更极端**"。同类工作
（facility-location/FLoC、quality×diversity DPP/CDPruner、leverage/SVD-Prune）都不在此 regime、
不用几何地标、不讲"超过 random"的故事。

---

## 8. 实验设计（瞄准"把可用压缩率推到极端"）

**主曲线（核心证据）**：`ACC vs r ∈ {0.25, 0.10, 0.05, 0.03}`，方法 = {baseline(全量),
plain-random, stratified-random, VisPruner, **CVSP**}。**假设**：曲线在 0.25/0.10 重合，在
**0.05/0.03 分开（CVSP > random）**。这是唯一诚实可立的精度叙事——不赌"keep10 上赢 random"。

**基线选择（两个 random 都报，stratified 是真门槛）**：先测 plain-random 与 stratified-random
谁更弱；headline 可叫"random"（两者都是均匀选取，措辞站得住）。**但 CVSP 必须也赢过
stratified-random**——它已做"每视角均衡 + 视角内均匀"，若 CVSP 只赢 plain-random 而输 stratified，
那"优势"其实只是**自适应分配/保底**，**不是 anchor/显著的 token 质量**，证明不了核心论点；且懂
pruning 的 reviewer（Window-FastV 这类）正会问。所以**两个都进表，stratified 当真门槛**。

- **数据集/任务（只看 ACC，RMSE 是回归到均值的假象）**：
  - Ego3D：`Object_Centric_Absolute_Distance_MultiChoice`、`Ego_Centric_Absolute_Distance_MultiChoice`、
    `Localization`、`Travel_Time`。
  - VSI：`object_rel_direction_{easy,medium,hard}`、`object_rel_distance`、`route_planning`、
    `object_appearance_order`（含时序任务做负面检验）。
  - `n ≥ 200/任务`；跨 **InternVL3-8B + Qwen2.5-VL-7B** 两架构。
- **前提自检（防"模型没看图"攻击）**：目标任务先过 vision-ablation（黑图/噪声 → 掉向
  chance），确认视觉真有用、token 选得好坏才有意义（[Visual-Compression.md] §E 已部分做）。
- **效率表**：每个 `r` 报 FLOPs/KV/峰值显存/CUDA 时间（`utils/efficiency.py`）；同 `r` 下各
  方法相同 → 头条指标 = **固定预算下的精度**。
- **消融**：(i) 逐档开关（§7 四行）；(ii) `ρ_a/ρ_s/ρ_c` 与 `φ` 扫；(iii) **Lowe max vs Σ**
  （验证低重叠修正）；(iv) cornerness 方法 B vs A；(v) 去重阈 `τ` 灵敏度；(vi) 最远点 vs 设施
  选址。
- **诚实指标**：每个数据集统计"能拿到非零锚点分的 token 占比"（低重叠 → 稀疏 → 该信号在高
  重叠 VSI 多、低重叠驾驶少，解释 CVSP 在哪类数据更可能赢）。

**预注册成败门槛**：CVSP 在 `r≤0.05` 于**多数任务**明显 > stratified-random，**且** `r≥0.10`
不退化 → **精度故事成立**。否则 → **回去迭代方法**（调 ρ/信号、或上 query-aware v2），**不走
"效率/发现型论文"退路**（2026-06-18 决策）——反直觉冗余发现只作 CVSP 的 motivation，不单独成文。

> **诚实提示（撤掉退路的代价）**：在"informed ≈ random"的现有证据下，没有了效率/发现型的安全网，
> 项目能否出精度论文**完全押在 CVSP 真能赢 random**。这是个真赌注，门槛只在极端压缩区有缝可钻。

---

## 9. 诚实边界

- CVSP 是 **query-agnostic**：能治"跨视角重复"（含被多视角放大的无关显著），**治不了**"只在
  单视角出现、又恰好无关"的显著物——它和"单视角独有的有用内容"在 query-agnostic 下无法安全
  区分，一视同仁留代表（危害限在少量 token）。彻底去要靠 query-aware v2（future work）。
- "留住有用 token → 精度不降反升"是**假设**，不是已证结论；本方法的精度赌注**只在极端压缩
  区**。keep10/25 预算 > 有效秩(~84)，那里 CVSP 与 random 预期持平，**别在那里宣称胜利**。

---

## 10. 备选优化方向：先分块再压缩（block-first，实验不佳时的后路）

**线索（用户 2026-06-18 提供）**：Nüwa (2602.02951)、AdaTP (2505.20100) 都验证**对单图先分块、
再按块各自压缩**比全局 top-k 更好。这与我们"**coverage 在多视角上很重要**"的发现一致——分块
本质上是把"空间覆盖"做成**硬约束**（每块都必须留代表），而不是像设施选址那样的软目标。

**和现状的关系（诚实）**：
- 真正的增量 = **把分块结构和锚点/显著信号结合**：在**每个视角内按空间块分配配额**（每块保底
  若干 token，块内再按 a/s 排序选），把当前 §5 里"facility-location 软覆盖 + φ 视角级保底"
  升级成"**视角内块级硬配额**"。这把 coverage 从"全局软目标"变成"块级硬约束"，覆盖更稳、不会
  让某个空间区域被整体饿掉。

**触发条件**：若 §8 主曲线显示 CVSP 在 keep5/3 **没稳过 stratified-random**，且诊断指向"覆盖
不够/某些空间区域被丢"，则上**块级配额版**：
1. 每视角切 `b×b` 空间块（先试 2×2、3×3）。
2. 每块给最低配额 `m`（块级保底），块内按 `a`（或 `a`+`s`）排序选 top；剩余预算再走跨视角
   anchor/saliency/coverage 三档。
3. 跨视角去重 `τ` 照旧，避免不同视角的同名块互相重复。

**消融**：块级硬配额 vs 现在的软覆盖(§5)；块大小 `b`；块保底 `m`。**别忘了**：stratified random
已是块基线，所以要报的是"**块结构 + anchor/saliency 信号** > 纯块随机"，否则只是换了个 random。

---

## 11. 候选组件：寄存器 / sink token 保护（high-norm token 进保底）

**线索（用户 2026-06-18 提供）**：ViT 里**高范数 token = register（寄存器）token**，承载全局 /
聚合信息；改动/删除它们会**移动整个特征分布、损害预测**（cf.《TO SINK OR NOT TO SINK: Visual
Information Pathways in LVLMs》；亦见 DINOv2 register tokens）。

**为什么这对 CVSP 是"新信息"（关键）**：我们三个信号全建在**余弦 Gram `G=f·fᵀ`** 上，`f` 已
单位归一 → **范数被归一化掉了**。所以 cornerness / lowe / coverage **天生看不见高范数 token**，
register/sink 很容易被误裁。显式保护 = 补了一路方法现在完全缺失的信息，**几乎零成本**（特征已
在手，`‖F_t‖` 顺手算）。

**怎么整合**：在 §5 的保底阶段加一个**"protected set"** —— 取范数最高的少量 token（先试全局
top-`k_reg`，或每视角 top-1~2），**强制保留**（和 φ 视角保底并列，放在锚点/显著档之前）。
`k_reg` 要小（极端区 keep3 只有 ~46 token，保护几个就够），由消融定。

**机制假设（也许正是 CVSP 赢随机的一条路）**：若删 sink 会移动分布，则**随机在极端压缩区会以
高概率删掉这些 sink → 掉分**；CVSP 显式保护 → 不掉。这给"keep5/3 处 CVSP > random"提供了一个
**具体可检验的机制**。

**诚实caveat**：
- **可能是"水涨船高"**：保护 sink 也许对**所有**方法都有益（给 random 也加就一起涨）→ 那它就
  不构成 CVSP 对 random 的优势。**判别实验**：保护 token **只加给 CVSP** vs **加给所有方法**——
  若只有"random 加保护后追平 CVSP"，说明优势仅来自 sink；若 CVSP 仍领先，才是三档信号的功劳。
- **与显著档可能重叠**：sink 常是 attention 高地 → 显著档（CLS 注意力）也许已抓到一部分；但
  register 高范数 ≠ 必然高 CLS 注意力，多半互补。需消融测增量。
- **预算挤占**：极端区每个保护名额都从 anchor/saliency/coverage 扣，`k_reg` 必须克制。

**触发**：可在 §8 首轮就把"CVSP + 范数保护"作为一个额外变体跑（成本极低），或留作 keep5/3 不
理想时的第一优化。

---

## 12. Block-CVSP（v3，2026-06-20 设计）—— 当前主攻

> **动机**：v2/a20s40 暴露两个病——① 固定比例 ρ 跨 keep 档不通用（§0 诊断①，ρ_a 应 ∝1/r）；② coverage 是**特征空间**软目标，**不保证画面空间覆盖** → Localization 这类覆盖任务输给 random（[Visual-Compression.md] §L）。v3 把 coverage 升成**块内显著（视角内空间硬约束）**，锚点改**固定数量（ratio-invariant）**。机制参考 Nüwa（2602.02951）stage-1 的"区域显著 + 距离惩罚 + 空间合并"（已确认其 stage-1 为 query-agnostic、单图、可移植）。

### 12.1 核心理念：两层 + 全程跨视角去冗余
- **两条正交轴**：**块 = 视角内（intra-view）空间覆盖**；**CVSP 跨视角（inter-view）= 去冗余 / 支持**。块管"每张图画面铺开 + 尺度"，跨视角管"同物去重"。
- **第 1 层 = 锚点**（全局跨视角、**按比例 ρ_a=0.2、★不 dedup 保留锚点对**、最高优先）；**第 2 层 = 块内显著**（覆盖由此涌现、按块"注水"、带 dedup）。anchor + 跨视角去冗余仍是故事核心，但**去冗余只作用在第 2 层(普通内容)，不误伤锚点对**。
- coverage 不再是独立桶：**被块内显著吸收**（每块贡献其最显著 token = 空间覆盖 + 重要性合一）。

### 12.2 符号补充（接 §2）
- 每视角 16×16 网格（InternVL3，256 token/tile）；`blk(t)` = token t 的块号；块 = b×b 空间分块。
- `a(t)=cornerness(t)·lowe_max(t)`（§3.1）；`L(t)=lowe_max` 原始跨视角支持；`s(t)=rank_norm(CLS-attn)`（§3.2）；G 余弦 Gram；`red(t,S)=max_{u∈S,view≠} G_{tu}`。

### 12.3 算法
```
输入: F, view(·), blk(·), a, L, s, G, B=⌈rM⌉, ρ_a=0.2, τ, b
1. 第1层 锚点（按比例，★不做 dedup，保留锚点对，优先）:
   B_anc = round(ρ_a · B)         # 比例 0.2（备选: min(round(α·M), #{L>δ}) 固定数量, 见 §0 诊断②）
   S ← 按 a(t)↓ 取前 B_anc 个       # ★无 dedup → A 与其跨视角匹配 A' 都是高 a, 自然成对保留(候选)
2. 标记: 含锚点的 (视角,块) 记为 "floor 已满足"
3. 第2层 块内显著（注水到 B，★带 dedup）:
   3a. 轮1（保覆盖）: floor 未满足的 (视角,块) 按其最高 s 排序, 各取块内 s 最高且 red(t,S)≤τ 的 token
   3b. 轮2+（重要性）: 所有块竞争, 每轮按全局 s 取剩余最高且 red(t,S)≤τ 的 token（锚点块也可再得）, 直到 |S|=B
   3c. 兜底: 若 dedup 致 |S|<B, 按 s↓ 无视 dedup 补到 B（保证等预算可比）
4. (可选 v3.1 merge): 被删 token 按 relu(特征相似)×距离惩罚 聚合进保留 token（Nüwa 式）
5. 返回 sort(S)，喂 LLM

★ 关键修订(2026-06-20,用户):**锚点层不 dedup**——锚点的价值正是"同一地标跨视角出现",
dedup 会把锚点对拆散、关联建不起来(且可能正是 §0 诊断③"纯锚点≈random"的元凶)。
**dedup 只用于第2层(删普通内容的跨视角重复),锚点反而要保留这种"有意的冗余"。**
```
- **块分辨率自适应**：`每视角块数 = clamp(round(keep_pv / c), 1, 4)`（keep10→2×2、keep5→2×2/2×1、keep3→2×1/1×1），避免极端档块 floor 吃光预算。
- **预算口径**：**锚点 = 比例 `B_anc=round(ρ_a·B)`，ρ_a=0.2（先用比例；固定数量 `min(round(α·M),#{L>δ})` 作备选,见 §0 诊断②）**；其余全部给块内显著（弹性，随 B 涨）。没有 ρ_s/ρ_c（覆盖与显著合一,由块注水自动分）。`block_cvsp 去锚点`消融 = ρ_a=0（全部块显著）。

### 12.4 每个设计点 ↔ 解决的病
| 设计 | 解决 v2 的什么 |
|---|---|
| 锚点按比例 ρ_a=0.2（固定数量作备选） | 先沿用 a20s40 的有效配比；固定数量备选解 §0 诊断①② |
| **锚点层不 dedup、保留锚点对** | 修 dedup 与锚点目的的矛盾；或正是 §0 诊断③"纯锚点≈random"的元凶 |
| 块内显著（空间硬覆盖） | Localization 覆盖差：特征-coverage → 画面-coverage |
| 锚点计入块 floor、但块不锁死（采纳用户意见） | 锚点稀疏不均匀 OK；高显著块仍可在轮2+ 再得 token |
| 块分辨率随预算自适应 | 极端档块 floor 不再吃光预算 |
| 仅第2层 dedup（删普通内容重复） | 保 "去冗余" 核心,但不误伤锚点对 |

### 12.5 实现方案
- **`compressors/nuwa.py`（Nüwa-lite，基线）**：区域显著 + 距离多样性，**逐视角**。移植 `prune_vision_tokens`（Nuwa repo `nuwa/clip_encoder.py`）核心：用我们 16×16 网格 + `internvl_adapter.AttentionCapture` 的 CLS attn + `_create_distance_penalty_matrix`（改 16×16）。**不要** LLaVA monkey-patch / anyres / stage-2。merge 版后补（需 compressor 返回特征而非索引）。
- **`block_cvsp`**：在 `cvsp_curve.py` 加 `sel_block_cvsp(G, a, L, s, view_id, n_views, n_tok, K, alpha, delta, tau, b)` 实现 §12.3；CLI `--method block_cvsp` + `--alpha --delta --b_blocks`（或自适应）。
- **复用**：`four_way_extreme.cornerness/lowe_max`、`internvl_adapter.AttentionCapture/compute_imp`、`cvsp_curve` 的 prompt/generate/resume。

### 12.6 实验与消融
- **方法**：plain_random / **stratified_random（纯块基线，必放）** / vispruner / **nuwa-lite** / a20s40（非块对照）/ **block_cvsp** / **block_cvsp 去锚点（消融）**。Nüwa 全 pipeline 列 related work。
- **任务**：Localization、Ego_AbsDist（低重叠，块该赢）+ rel_dir_hard、rel_distance（锚点任务，别被覆盖杀）+ rel_dir_medium（高重叠 VSI，验风险①）。keep10/5/3，n=200（正式 n=400）。
- **判读（预注册）**：① Localization 超 stratified+plain；② rel_distance/rel_dir_hard 不低于 a20s40；③ VSI 不比 a20s40 掉太多；④ **去锚点掉分** → 锚点配当优先层、核心故事成立。

### 12.7 诚实风险
- **块会把跨视角冗余请回来**：每视角每块都留 → 重叠视角的对应块各留一份 = 重复，dedup 只能压一部分。**块利于低重叠/覆盖任务（Localization、驾驶），可能伤高重叠 VSI**——故 §12.6 必放高重叠 VSI 暴露它。
- **锚点单独价值未证**（纯锚点 ≈ random，§0 诊断③）→ §12.6 去锚点消融是 make-or-break；若去掉不掉分，真正干活的是"块显著+去冗余"，锚点退配角、故事须改。
- 块牺牲 v2 的**跨视角自适应分配**（强制每视角每块覆盖 → 退回较均匀分配）。

### 12.8 已知缺口与待改进（2026-06-21 讨论，对照当前 `sel_block_cvsp` 代码）

当前实现 `cvsp_curve.py::sel_block_cvsp(G,a,s,view_id,n_views,n_tok,K,rho_a,tau,brows,bcols)`
与 §12.3 设计有三处偏差/缺口，已与用户讨论：

**A. 每视角保底（per-view floor）—— 当前缺失，建议补。**
- 现状：`sel_block_cvsp` **没有显式 per-view floor**（不像 v1 的 `sel_cvsp` 有 `phi` 参数）。只有
  **隐式的 per-block 保底**（Layer 2a 每块至少 1 个）。
- 为何隐式保底在极端档失效：每视角预算 `keep_pv = round(256·r)`（keep10=26 / keep5=13 / keep3=8），
  而 8×8 时每视角 **64 块**。`keep_pv ≪ 块数` → Layer 2a 的 round 1 预算覆盖不完所有块；且 uncovered
  块按**全局** top-s 排序 → **整体偏低显著的视角可能拿到极少甚至 0 token**。
- 这个缺口被显著性的**跨视角不可比性**放大：`s=CLS→patch attention` 是**每视角内 softmax 归一化**的
  （和=1）。同视角内跨块可比；跨视角只是"占本视角注意力比例"的近似可比，**有偏**——前景集中的视角
  峰值 s 更高、在全局贪心里抢更多名额，弥散视角吃亏。Layer 2a 的"每块≥1"本就是为抵消这点，但极端档
  预算不够、抵消失效。
- **建议**：在 Layer 2a 前插一个 φ floor（镜像 `sel_cvsp` 的 phi，但**视角内**排序绕开不可比）：
  每视角先按"**视角内** top-s + red_ok"强制保底 φ 个，消耗 φ·n_views 预算；剩下 K−φ·n_views 再走
  现在的"块覆盖 round1 → 全局 water-fill"。**φ=2 建议**（keep3 下 K/视角才 8，φ 不能大，否则退化成均分；
  最多 φ=3）。加 `--phi` 参数，默认 0 = 现状可回退。与锚点层、dedup 不冲突。

**B. 块分辨率：8×8=64 块过细，4×4=16 块更配预算（2026-06-21 用户提议，采纳）。**
- 实测跑的是 `--brows 8 --bcols 8`（64 块/视角），但 §12.3 原设计本就是 `每视角块数=clamp(round(keep_pv/c),1,4)`
  ≤4×4 —— 8×8 其实**越界**了。
- 原理：**块数应与每视角预算同量级**（#blocks ≲ keep_pv），覆盖 round 才"咬得动"。覆盖 round 能完成一轮的占比：

  | 档 | keep_pv | 8×8 覆盖率(完成度) | 4×4 覆盖率 |
  |---|---|---|---|
  | keep10 | 26 | 26/64 = 41% | 26/16 → **全覆盖 + 10 注水** |
  | keep5 | 13 | 13/64 = 20% | 13/16 = **81%** |
  | keep3 | 8 | 8/64 = 12.5% | 8/16 = 50% |

  8×8 时覆盖 round 退化成"全局挑 top-K 块"，块结构几乎不起作用（≈全局 top-s）；**4×4 让覆盖真正生效**。
- 4×4 = 每块 4×4 token（16 token 竞争 1 个覆盖名额），是 2×2(过细) 与 §10 试过的 2×2块=8×8token(过粗)
  之间的中点。**默认切到 4×4**；若要更精细可按档自适应（keep10→5×5、keep5→4×4、keep3→3×3，使 #blocks≈keep_pv）。

**C. 锚点跨视角铺开：机制上无保证，但实测缺口很小（2026-06-22 量化，`scripts/anchor_spread.py`）。**
- 机制：Layer 1 是**纯全局 top-B_anc by a(t)**，无 per-view 配额、无 dedup、无铺开约束 → 理论上不保证铺开。
  唯一"铺开"来源是 `a=cornerness·lowe_max` 里 lowe_max 的**跨视角性 + 近似互惠**（A 强匹配 B → B 回看 A 也高
  → 高 a token 倾向成对出现在两视角）。
- **实测（n=60×3 空间任务，nv≈6，ρ_a=0.2）结论:锚点铺得很好,缺口几乎不存在:**

  | 指标 | keep10 | keep5 | keep3 | random null(keep3) | 含义 |
  |---|---|---|---|---|---|
  | cover(有锚点的视角占比) | 1.00 | 0.97–1.00 | 0.94–0.97 | 0.84 | 几乎每个视角都有锚点 |
  | maxfrac(最挤视角占比) | 0.23–0.30 | 0.25–0.29 | 0.27 | 0.35 | 均匀=1/6≈0.17,轻微集中 |
  | ent(视角分布熵/log nv) | 0.95–0.98 | 0.95–0.96 | 0.93–0.94 | 0.84 | 接近均匀(1.0) |
  | **pair(锚点最佳跨视角伙伴也被选中的占比)** | **0.77–0.84** | **0.88** | **0.89–0.91** | — | **互惠配对已自然发生** |

- **三点判读**：① 锚点**已经铺满几乎所有视角**(cover≈1)、分布接近均匀(ent≈0.95)，**worst-case"扎堆 1–2 视角"实测不发生**——
  因为每个视角都有自己的 corner/match，top-B 自然从各视角抽。② keep3 时锚点甚至**比 random 更均匀**(cover 0.95 vs 0.84、
  ent 0.94 vs 0.84)。③ **pair 0.8–0.9 且越极端越高**——最强锚点(最大 Lowe margin)最互惠,**隐式配对已覆盖 80–90%**。
- **推论(改变结论)**：**显式配对上限很有限**——只能补最后 10–20%,代价却是有效锚点数减半,**大概率不值得**。
  per-view 保底(A)针对的是**显著层(Layer 2)**的跨视角不可比饿死问题,**不是锚点层**——锚点不需要 floor，已自均衡。

### 12.9 Merge（v3.1，2026-06-22）—— Nüwa 真实机制 + 简化验证方案

**A. Nüwa 真实 merge（源码 `Man-PaperRejected/Nuwa` `nuwa/clip_encoder.py::prune_vision_tokens`，2026-06-22 拉取）。**
24×24=576 patch 单图上，两段：
- **选 survivor（"benchmark token"）= 区域分层显著**：metric=CLS→patch attention 各 head 求和；网格切 **2×2 不重叠区**（144 区）；
  每区取 attention top-n（n=1↔keep64，按预算放大）→ 候选；候选里再全局 top-K → survivor，按 index 排序。
  **= 我们 block_cvsp Layer 2（块=2×2、每块保覆盖、全局显著），Nüwa 独立撞同一设计。**
- **merge = survivor 主动 gather 局部相似邻域**（我们之前漏的那步）：
  - 相似特征用**第 16–24 层 hidden state 平均**（深层语义，非投影层）；`sim=cosine`。
  - `distpen(i,j)=max(0,1−‖pos_i−pos_j‖/阈值)` 线性空间衰减，**超阈值=0 → merge 严格局部**。
  - `w(i,j)=relu(sim)×distpen`；**选择性保护**：survivor 中 attention≥自身 55 分位的"高显著"token **整行权重清零、保原始特征**（不被稀释），只有低显著 survivor 吸邻域；self 权重=1，归一化后 `aggregated=W_norm@patch_feat`。
  - 即：每个（非最显著）保留 token = 其**空间邻域内+特征相似** patch 的加权平均；最显著的保纯。**kept 主动 gather、严格局部**——不是 dropped 找全局最近 kept（那版被用户否）。

**B. 我们的简化验证方案（用户定，2026-06-22）：先只验 merge 有没有用，不做跨视角。**
- **选择完全不动**（block_cvsp 锚点+块内显著）。**merge 只作用于显著 token**；**锚点保原样**（不当目标、不稀释，对应 Nüwa 的保护）。
- 公式（同视角、邻近+相似；无跨视角、无保护机制、无硬 sim 阈值）：
  ```
  w(i,j)=relu(cos(f_i,f_j))×distpen(i,j)   j=同视角被删token
  i_new=( f_i + Σ_j w(i,j)·f_j )/( 1+Σ_j w(i,j) )   self权重=1
  ```
  cos 复用已算好的 `G`（=vit 特征余弦，零额外前向；**偏离 Nüwa 的"深层 sim 特征"，记为后续精修点**）。聚合的是 `vit`（即喂 generate 的投影特征）。
- **参数**：`dist_thr=11.0`（=现有 nuwa-lite；16×16 上 thr/maxdist≈0.51，正好 = Nüwa 在 24×24 的比例）。权重均匀 relu(cos)×distpen。
- **实现**：`cvsp_curve.py` 加 `apply_saliency_merge(vit,G,view_id,sel,anchors,n_tok,dist_thr)`（纯特征变换）；`sel_block_cvsp` 加 `return_anchor`；`select()` 多返回 sel_global+anchors；`run()` 在 `--merge sal` 时改 feats；CLI `--merge {none,sal}`、`--merge_dist`。
- **实验（纯 A/B，省算力）**：复用 §M 的 **8×8 无 merge**（`*.block_cvsp-bc8-a20.jsonl`）当对照，只新跑 **8×8+显著merge**（`--tag=-bc8-mSal`）。5 空间任务×keep10/5/3×n=200。判读=聚合 Δ（均值+胜格）。≥0 且多数格不降 → merge 有效，再上跨视角锚点 merge；否则止步。块 4×4 切换暂不混入（先干净测 merge 单变量）。
- **诚实预期**：merge 是少数"增加单 token 信息"的杠杆，最可能破 vision-ablation 平局，但**不保证**；极端档有过平滑风险（靠 distpen 局部化控）。

**C. 结果与判决（2026-06-22，FAIL）。** block_cvsp-bc8-mSal vs 无 merge(-bc8-a20)，5 空间任务×keep10/5/3×n=200：
- **聚合 mean Δ = −0.017，胜 6 / 负 9（共 15 格）；keep10/5/3 每档都负（−.014/−.014/−.023）。→ 未过 go/no-go（标准=mean Δ≥0 且多数不降）→ 不进跨视角 merge。merge 弃用，方法回到 block_cvsp 无 merge。**
- **任务交互（可解释，值得留）**：merge **帮粗方向**（rel_dir_hard +.040/+.065/+.010，三档全赢），**伤精确定位**（Localization −.030/−.045/**−.115**、rel_dist −.040/−.080/+.005、rel_dir_med 三档全负）。
- **根因 = 过平滑**：`dist_thr=11` 的邻近半径≈整张视角（16×16 上 π·11²≈380>256），每个显著 token 几乎把全图被删 token 加权平均进来 → 抹掉空间身份。冒烟时 `mean|Δ特征|=13.58` 已是预警。真"局部"需 dist_thr≈3–5（半径内 ≈11–30%/视角），但 merge 整体方向已否，未再试。
- 数据：`logs/cvsp/*.block_cvsp-bc8-mSal.jsonl`；驱动 `scripts/run_merge_ab.sh`；对比代码见会话。

## 13. 两阶段 Query-Aware Block-CVSP（v4，2026-06-22 定稿）—— GeoScaffold 实现

> **动机**：v1–v3 全程 query-agnostic，实测 ≈ random（天花板：视觉信号小+语言先验主导）。**query-awareness 是唯一没拉过的杠杆。** 借 Nüwa 两阶段：阶段1 query-agnostic 选较多 token（前层享受信息），阶段2 在 LLM 内按 query 裁到很小（"平均"压缩率仍低）。实现记忆里的 GeoScaffold 方向。
> **路径选择**：用户定路径1（过选→query 裁），**不是**路径2（核心+补回）。query 只在裁剪时作用，stage-2 同裁 anchor+显著（layer 已靠后，无需保护）。merge 不入主线（§12.9 已 FAIL）。

### 13.0 预算（keep10/5；K=L/2；Nüwa 1.75/0.25）
- M=n_views×256；L=28（Qwen2.5-7B）；**K=14**（裁剪层，Nüwa 代码 16=L/2，我们 L=28→14）。
- **T = 报告的层平均预算** = keep10/5 · M。N1=1.75T（阶段1，encoder 后），N2=0.25T（阶段2，layer-K 后）。
- 通用：`N2 = T·L/(r·K + L−K)`，`N1 = r·N2`，r=N1/N2。K=L/2 时 → N1=1.75T、N2=0.25T、平均=(N1+N2)/2=T。
- **r=N1/N2 当旋钮扫 {7, 3, 2}**（7=Nüwa，空间任务大概率太狠；小 r 深层留更多）。
- 数：keep10 N1=269/N2=38；keep5 N1=135/N2=19。keep3/keep20 待 10/5 有效再加。

### 13.1 阶段1（pre-LLM，query-agnostic，可缓存）= block-cvsp 选 N1
不变（§12 block_cvsp），只把预算改成 N1：
1. G=Fn@Fnᵀ（跨视角余弦）；a=cornerness×lowe_max；s=CLS-attn。
2. 4×4 块/视角；第1层锚点 top round(ρ_a·N1) by a（ρ_a=0.2，无 dedup）；第2层块内显著注水到 N1（轮1 保覆盖 + 轮2 全局 top-s + 跨视角 dedup + 兜底）。
3. 输出 feats_N1（按视角序）喂 LLM 的 `visual_features`；记 vis_pos（视觉 token 在 LLM 序列的位置）、view_id_kept。
- coverage 在此=**给 query 的候选菜单**（N1 大，覆盖得起；无关候选由 stage-2 query 收回）。

### 13.2 阶段2（in-LLM @ layer K=14，query-aware）——**主线 = Nüwa 代码版（attention）**
忠实复刻 Nüwa `modeling_llama.py`，移植到 InternVL3 的 Qwen2Model：
```
prefill 时，在 decoder 层循环里：
  idx == K-1 (=13): 该层 output_attentions=True（eager，丢 FA 仅此一层）→ 捕获 _last_layer_attention
  idx == K   (=14): 用捕获的注意力裁剪（在跑第 K 层之前）：
     last_attn_avg = mean over heads (_last_layer_attention)
     last_text_idx = 最后一个非视觉 token（=问题末 token）
     score = last_attn_avg[last_text_idx, vis_pos]          # 最后文本 token 对各视觉 token 注意力
     keep = top-N2(score)                                    # 跨视角全局选（不分视角→各视角自适应留不同数）
     keep_idx = 全部文本 token ∪ keep                        # 在完整序列里的原始下标
     # PESP 位置稀疏保留（gather 不重编号）:
     hidden_states = hidden_states[:, keep_idx]
     position_ids  = position_ids[:, keep_idx]
     position_embeddings = (cos[:,keep_idx], sin[:,keep_idx])
     attention_mask/KV 按 keep_idx 收缩；重建 causal mask
  之后 layer K..L-1 在缩短序列上正常前向（FA 恢复）
```
- **跨视角全局选** = Nüwa 的 `topk(score, N2)` 直接作用在我们多视角的 vis token 上（Nüwa 机制本就是全局）。
- **stage-2 同裁 anchor+显著**：无保护，全凭 score 竞争（layer 14 已靠后，锚点早层贡献已用过）。
- **代价**：layer 13 走 eager（output_attentions）丢 FA；其余 27 层保留 FA。

### 13.3 备选（记录，论文版 = cosine + avgpool q̄，FA 全保）
Nüwa **论文**（≠代码）：在 projector 之后的**共享嵌入空间**算 cosine，pre-LLM 可算：
```
q̄  = average-pool( embed(question tokens) )      # 整体查询向量
R_i = cos( proj(v_i), q̄ )                         # 静态（pre-LLM），keep 集进 LLM 前定好
裁剪仍 defer 到 layer 14 执行（拿"平均"红利），patch 只做"按预定 keep 丢 token + PESP"
```
- 优点：**全 28 层保 FA**（cosine 不取注意力权重）；信号 pre-LLM、patch 更轻。
- 旋钮 query_pool：`mean`=q̄（论文）| `last`=最后文本 token 当 q̄。
- **触发条件**：主线（attention）若不行 / FA 代价不可接受 → 切此备选。

### 13.4 效率记账（"平均"）
每样本有效 token = `(N1·K + N2·(L−K))/L`，K=14/L=28 → =(N1+N2)/2 = **T**（报告值）。`n_kept` 写此层平均；FLOPs/KV 用 `utils/efficiency.py`（FastV 约定）。Nüwa 的"matain_token"就是此层平均（112/16@layer16 → 平均 64）。

### 13.5 实现面（中等偏大）
1. **LLM 注意力捕获**：仿 `compressors/internvl_adapter.AttentionCapture`，对 Qwen2 第 13 层强制 output_attentions/eager，stash 注意力。
2. **patch Qwen2Model.forward**：在 decoder 循环插入 §13.2（捕获 + 裁剪 + PESP + KV/mask 重建）。Qwen2 vs Llama 差异：position_embeddings(cos,sin) 外部传入、attention 实现、mask 构造——逐一对齐。
3. **runner**：新建 `models/internvl3_qstage.py`（或扩 cvsp_curve）：阶段1 block-cvsp→feats_N1；在 model 挂 vis_pos/K/N2/r/signal；`n_kept` 写层平均。
4. **冒烟 assert**：序列长度对、生成不崩、PESP 后文本位置不变、N2=N1 时与"无 stage-2"数值一致、有效 token=理论值。

### 13.6 参数与默认
| 参数 | 默认 | 旋钮 |
|---|---|---|
| T | keep10/5 | 后加 keep3/20 |
| K | 14 | =L/2 |
| r=N1/N2 | 7 | 扫 {7,3,2} |
| ρ_a | 0.2 | stage-1 锚点占比 |
| 块 | 4×4/视角 | 3×3 |
| signal | **attention(主)** | cosine+q̄(备选)；query_pool mean/last |
| stage-2 dedup | 关 | 跨视角去冗余开关 |

### 13.7 实验
- 主对比：**两阶段 Q-block-cvsp vs 单阶段 block-cvsp**（同层平均预算 T）。
- keep10/5 × r∈{7,3,2}，5 空间任务，n=200。判读=聚合 Δ（query-aware vs 单阶段）。
- 控制：N2=N1（r=1，退化成单阶段）= 数值自检。

### 13.8 诚实风险
1. **query 信号判别力**：最后文本 token 注意力 / cosine 可能不够锐 → 主线不行切备选 / 反之。
2. **stage-2 全局选深层饿死某视角**（无 per-view 保底）→ 掉分则加深层 floor（旋钮）。
3. **r=7 对空间任务过狠**（深层 N2 太小）→ 扫小 r。
4. **patch 正确性**（position/rotary/KV/mask 重建）最易出 bug → 冒烟 assert 必做。
5. 仍可能撞天花板（视觉信号小）→ query-aware 也未必提点，这是值得一搏、非稳赢。

### 13.9 结果(attn 信号,2026-06-23)—— 两阶段 ≈ 单阶段,query-aware 未破平局

实现见 `compressors/qstage_llm.py` + `scripts/qstage_curve.py`(driver `run_qstage.sh`)。两阶段精确实现确认:use_cache=True + FA2 + per-layer KV(层<K 存全 N1、层≥K 存 N2)+ PESP;自检 active=False ≡ r=1(无剪枝)预测完全一致。
- **两阶段(attn r∈{2,3,7})vs 单阶段基线(cosine r=1,=block-cvsp 4×4 @ T),keep10/5×5 任务×n=200,同层平均 T:**
  - r=2:mean Δ=**−0.006**,2胜/6负;r=3:**−0.011**,4胜/6负;r=7:**+0.0045**,6胜/4负。
  - **全部在噪声内**(n=200,1SE≈3pp;±0.5–1.1pp 的均值差 = 无效应)。**query-aware 两阶段 ≈ 单阶段,没提点。**
- **任务交互(可解释,重复 merge 的规律)**:**Localization 一致被伤**(所有 r、两档全负,−.025~−.080)——query 剪枝在 layer 14 把空间铺开的 token 删掉、向 query 相关集中 → 覆盖任务掉分;rel_dist/rel_dir_med 在 r7 受益(集中有利方向/距离)。覆盖↔集中,净平。
- **vs 真基线(plain/strat/visp 取最好)**:单阶段 4×4 **4/10**、两阶段 attn r7 **5/10** —— **只赢一半格 = 不算赢**(需 ~8/10 才显著)。rel_dist 我方明显赢随机(.40–.44 vs .32–.37),Localization 输 plain/strat(.42–.48 vs .49–.51)。
- **结论**:in-LLM attention 剪枝的 query-awareness **未突破**"视觉信号小+语言先验"天花板;两阶段的"前层多留"对精度无帮助(深层 0.25T 足够 ≈ FastV 观察)。**待 cosine 备选确认(2026-06-23 已起跑)。**
- 数据:`logs/cvsp/*.qstage-{cosine-r1,attn-r{2,3,7}}.jsonl`。

**§13.9 cosine 确认(2026-06-23)**:cosine(layer-14 上下文化)r2 −0.011、r3 −0.017、**r7 −0.000**(部分曾 +0.018,被 Localization −0.06~−0.105 拉回,如预测)。**attn+cosine 全部 ≈ 0,Localization 每个 config 都被伤(−.01~−.105)。两阶段 query-aware 确认无提点。** 最佳 r7 vs strat 仅 5/10=平。→ 按计划上论文精确版 pre-LLM cosine(`cos(proj(vᵢ),q̄)` 输入嵌入空间,裁剪仍 layer14;runner 预计算 keep 集、qs.kept_vis 预置,patch 直接套用,全程 FA2)。

### 13.10 Stage-2 多视角化:任务感知跨视角去冗余/配对(2026-06-24 讨论结论)

**先否掉两条(讨论后判定无用)**:① 去偏跨视角相关性——位置偏置是 **attention 机制**产物(softmax/因果/sink),**cosine 不继承**;`input_cos`(projector 后 ViT 特征、pre-LLM)**结构上几乎无偏**,很可能正是它 > attn/layer-cosine 的原因 → 去偏对主线无用。② 视角间显式预算分配——global top-k 按相似度**已隐式按视角分配**,无偏置可纠时纯属多余、还强加错误先验 → 撤回。

**保留并主推(最具多视角 novelty、贴 3D 本质)：stage-2 任务感知跨视角去冗余/配对。**
- **轻量实现(复用已算的 G,~5 行)**：stage-2 不做纯 top-k,改 **"按 r 降序 + 跨视角 dedup 门"**：按 query 相关度 r(t)↓ 选,加入当且仅当 `red(t,S)=max_{u∈S,view≠} G[t,u] ≤ τ_q`,到 N2。= block-cvsp 的 `red_ok` 搬到 query 相关排序上。
- **任务感知 τ_q(关键,dedup 不是对所有任务都对)**：
  - `relative_distance / localization`（要多实体/广覆盖）→ **dedup 开**(τ_q 小)：同物体不在多视角各留一份,腾预算给另一 query 实体。
  - `absolute_distance`（要三角化估 3D 距离）→ **dedup 关**(τ_q=1)：**故意保同物体 ≥2 视角**。
- **任务类型识别(轻量,两路)**：(1) **类别 oracle**——数据里有 `category`,先用它验证"任务感知 dedup 是否有用",零成本零歧义;(2) **可部署版 = 问题关键词 regex**(`distance/how far/meters`→保多视角;`closer/farther/which`→dedup;`where/locate`→dedup),无类别泄漏、零模型成本。先 oracle 证明,再换关键词当诚实方法。
- **诚实提醒**：stage-1 已做 query-agnostic dedup(τ=0.85),N1 部分去重 → **τ_q 要与 stage-1 的 τ 不同**否则无增量(stage-2 是 query-条件去重、腾给其他 query 内容,与 stage-1 互补但重叠);受总天花板限制,提升可能有限,但这是 stage-2 唯一"单图绝无"的机制。
- **计划**：全量跑完后加 `--stage2_dedup` + 按 category 的 τ_q,对比 `input_cos`(纯 top-k) vs `input_cos+任务感知 dedup`,看 relative/localization 是否提、absolute 是否不被伤。
