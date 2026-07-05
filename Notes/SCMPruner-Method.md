# SCMPruner — 详细方法（v1 定稿 2026-07-03）

**SCMPruner = Spatial-Connected Multi-view token Pruner.** 训练-free / pose-free /
query-agnostic 的多视角视觉 token 压缩器。三桶并集:**anchor**(跨视角 3D 地标,不去重)
+ **saliency**(前景显著,margin 感知去冗余)+ **coverage**(特征多样性/去塌缩)。
**独立于 a20s40(CVSP)方案**:在 `scripts/cvsp_curve.py` 里是一个单独的 `--methods scmpruner`,
a20s40(`cvsp`)的代码与行为不变。本文档是论文写作的权威依据(公式 + 步骤)。

> 由来:`Notes/Anchor-Validation.md` §9–11 用无去重几何 oracle 证明"跨视角载荷 token"存在
> (关系任务剂量正效应 +~0.02),并验证纯特征的 `support×sharpness` 打分能把它挑出来、头对头胜过
> a20s40 的 `cornerness×lowe_max`。SCMPruner 把这个**纯特征 anchor** 放进压缩管线。

---

## 1. 动机（Motivation）

多视角 VLM 输入里视觉 token 高度冗余;**单视角**压缩(saliency + diversity)漏掉两件多视角
特有的事:(1) **跨视角冗余**(同一 3D 点在多个视角重复出现);(2) **支撑跨视角 3D 空间关系的
token**(同一 3D 点的多视角对应,是三角化/布局推理的锚)。这两件事方向相反——(2) 要**留**跨视角
一致的 token,(1) 要**删**跨视角重复的 token。SCMPruner 用**匹配的独占度(margin/sharpness)**
把二者分开:高 margin 的唯一对应 = 地标(留每个副本),高相似但低 margin 的重复纹理 = 冗余(可删)。

---

## 2. 符号（Notation）

- $V$:视角数;每视角 $P=256$ 个 post-pixel-shuffle token(InternVL3 里真正进 LLM 的粒度);$M=VP$。
- $x_t\in\mathbb{R}^C$:token $t$ 的 InternViT 特征(pixel-shuffle 后)。归一化 $\hat x_t=x_t/\lVert x_t\rVert$。
- Gram 矩阵 $G_{tu}=\hat x_t^\top \hat x_u\in[-1,1]$。
- $\mathrm{view}(t)\in\{0,\dots,V-1\}$:token $t$ 所属视角。
- 保留率 $r$;每视角保留 $\mathit{keep\_pv}=\mathrm{round}(P\cdot r)$;总预算 $K=\mathit{keep\_pv}\cdot V$。
- 显著度 $s_t\ge 0$:InternViT 的 CLS→patch 注意力聚合到 token 网格(前景内容下限)。

---

## 3. 三个信号（含公式）

### 3.1 Anchor 分 $a(t)=\text{support}\times\text{sharpness}$ —— 跨视角地标

对 token $t$($v=\mathrm{view}(t)$),遍历每个其它视角 $u\ne v$。设该视角内对 $t$ 的最高、次高
相似度为 $s^{(1)}_{t,u}\ge s^{(2)}_{t,u}$,定义 Lowe 边际

$$m_{t,u}=\max\!\big(s^{(1)}_{t,u}-s^{(2)}_{t,u},\,0\big).$$

"清晰唯一匹配"指示子(用两个门槛 $\tau_a,\,m_a$):

$$\mathrm{matched}_{t,u}=\mathbb{1}\big[\,s^{(1)}_{t,u}>\tau_a \ \wedge\ m_{t,u}>m_a\,\big].$$

于是

$$\text{support}(t)=\sum_{u\ne v}\mathrm{matched}_{t,u},\qquad
\text{sharpness}(t)=\frac{\sum_{u\ne v}\mathrm{matched}_{t,u}\, m_{t,u}}{\max(\text{support}(t),1)},$$

$$\boxed{\,a(t)=\text{support}(t)\cdot\text{sharpness}(t)\,}.$$

- **语义**:$\text{support}$ = 在多少个视角被**清晰重认**(数量);$\text{sharpness}$ = 这些匹配的
  **平均独占度**(质量)。$a(t)$ = 数量 × 质量 = 稳的多视角地标。
- **margin 为什么是关键**:$m_{t,u}$ 大 ⇒ 在视角 $u$ 里**只有一个** token 特别像 $t$(唯一对应=真地标);
  $m_{t,u}\approx 0$ ⇒ 一大片都像(重复纹理,认不准)。实测 $m$ 中位仅 $0.019$,只 ~3% token 够 sharp。
- **sharp / fuzzy 标签**(桶二去冗余要用):$\text{support}(t)\ge 1 \Leftrightarrow a(t)>0 \Rightarrow$ **sharp**;
  $\text{support}(t)=0 \Rightarrow$ **fuzzy**(重复纹理)。
- 代码:`cvsp_curve.py::anchor_scores` 返回 $(a,\ \text{support})$。纯 InternViT 特征,无 VGGT。

### 3.2 Saliency 分 $s(t)$ —— 前景内容下限

InternViT CLS→patch 注意力,聚合到 token 网格(同 `compute_imp`)。仅用于**桶二内部排序**。

### 3.3 Coverage 算子 —— 特征多样性 / 去塌缩

Gram 上的 max-coverage(facility-location)。已选集合 $S$ 对 token $u$ 的覆盖度
$\mathrm{cov}_S(u)=\max_{j\in S}G_{ju}$;下一个挑的 token 是

$$t^\*=\arg\max_{t\notin S}\ \sum_{u}\mathrm{relu}\!\big(G_{tu}-\mathrm{cov}_S(u)\big),$$

即"新增覆盖最多尚未被代表内容"的 token = 特征空间多样性。选中后
$\mathrm{cov}_S \leftarrow \max(\mathrm{cov}_S,\,G_{t^\*,\cdot})$。

---

## 4. 预算分配 + 选择算法（三桶贪心）

**预算(v1)**:$B_a=B_s=\mathrm{round}(K/3)$,coverage = 剩余。桶 1–2 欠额自动流入 coverage,
**总数恒为 $K$**。

```
输入: G, a, support, s, view, K, τ_a      # 常量 ρ_a = ρ_s = 1/3
B_a ← round(K/3);  B_s ← round(K/3);  S ← ∅

# 桶1  anchor —— 按 a 取 top-B_a, 不去重(保留跨视角全副本)
for t in argsort_desc(a):
    if |S| ≥ B_a: break
    S ← S ∪ {t}

# 桶2  saliency —— margin 感知去冗余
c ← 0
for t in argsort_desc(s):
    if c ≥ B_s: break
    if t ∈ S: continue
    if support(t) == 0  and  ∃ j∈S: view(j) ≠ view(t)  and  G[t,j] > τ_a:
        continue                      # fuzzy 且跨视角撞车 → 重复纹理副本, 丢
    S ← S ∪ {t};  c ← c + 1           # sharp token(support≥1)一律照收

# 桶3  coverage —— facility-location 填到 K（+ xview 传播, 默认开）
for t in S: propagate(t)              # 用 anchor+saliency 已选的 sharp 对应初始化覆盖
while |S| < K:
    t* ← argmax_{t∉S}  Σ_u relu( G[t,u] − cov_S(u) )
    S ← S ∪ {t*};  propagate(t*)
return S                              # |S| = K

# xview 覆盖传播（§10 安全变体, sharp-only）:
propagate(t):  for u≠view(t): if matched(t,u): cov[ bm(t,u) ] ← 1
#   同一个 3D 点在别视角只占一个覆盖名额; 只沿 sharp 对应边传播(fuzzy 的 argmax 太噪, 不传)
```

选出的 $K$ 个 token 特征喂给 LLM(prompt 里每视角 `<IMG_CONTEXT>` 数量随之改)。

**去冗余判据 = fuzzy ∧ 跨视角撞车**。sharp token 永远豁免 → 几何信号绝不被当冗余删。
去重阈值 $\tau_\text{dup}$ **复用** $\tau_a$(见 §6)。**注意:去冗余只作用于跨视角**(同视角内的
fuzzy 近重复不在桶二删,靠桶三 coverage 稀释)——这是刻意的设计选择(见 §9.6)。

代码:`cvsp_curve.py::sel_scmpruner`。全程确定性、无 RNG ⇒ resume-safe。

---

## 5. 三桶分工（一句话）

- **anchor**:跨视角**清晰唯一**的点 → **全副本留(不去重)**,保 3D 对应。
- **saliency**:显著前景 → 留,但**fuzzy 的跨视角重复副本删**(sharp 豁免)。
- **coverage**:剩余名额 → 覆盖前两桶没顾上的内容,防塌缩。

---

## 6. 超参 + 默认（**仅 1 个可调**）

| 超参 | 含义 | 取值 | 状态 |
|---|---|---|---|
| $\rho_a,\rho_s$ | anchor / saliency 预算占比(其余给 coverage) | **各 $1/3$** | 固定 |
| $\tau_a$ | 判定"清晰匹配"的相似度门槛(定 support/sharp 标签) | **$0.6$** | 固定(§11 实测 $\tau\in[0.5,0.75]$ 不敏感) |
| $m_a$ | 判定"唯一/sharp"的 margin 门槛 | **$0.12$**(默认) | **唯一可调**(§11 实测敏感,就是它分开 sharp/fuzzy) |
| $\tau_\text{dup}$ | 桶二去冗余的相似度阈值 | **复用 $\tau_a$** | 派生(不独立) |
| $\phi$ | 每视角保底保留数 | — | **删除**(keep10 无视角被饿死) |

CLI:`--anc_tau`(=$\tau_a$)、`--anc_m`(=$m_a$);$\rho$ 在 `scmpruner` 分支写死 $1/3$。

---

## 7. 复杂度 & 工程

- `anchor_scores`:$O(V^2\,P\log P)$(每对视角一次 top-2)。coverage:$O(K\,M)$ 贪心。
- 全部在**视觉侧、LLM 之前**,与 `generate` 解耦;确定性 ⇒ 与"按行数 resume"完全兼容。

---

## 8. 与 a20s40（CVSP）的区别

| | a20s40 (`cvsp`) | **SCMPruner** |
|---|---|---|
| anchor 分 | $\text{cornerness}\times\text{lowe\_max}$(单最佳对) | $\text{support}\times\text{sharpness}$(多视角、独占度) |
| anchor 去重 | **有**(跨视角 $\tau{=}0.85$) | **无**(保跨视角全副本) |
| saliency 去重 | 硬 $\tau{=}0.85$(实测 ~2% 才触发,近乎失效) | **margin 感知**(fuzzy ∧ 撞车,$\tau{=}0.6$) |
| coverage | facility-location | facility-location(**相同**) |
| 预算 | $\rho_a/\rho_s$ 可调 | $1/3$ 固定 |
| 可调超参 | 多($\rho_a,\rho_s,\tau,\phi,\dots$) | **1 个($m_a$)** |

独立性:`--methods scmpruner`,`sel_cvsp` 未被触碰。

---

## 9. 诚实边界 / 已记录的隐患

1. **$\tau_\text{dup}$ 复用 $\tau_a{=}0.6$ 的耦合**〔记录〕:调 $\tau_a$ 会同时移动"匹配门槛"和
   "去重门槛"。v1 固定 $0.6$ 规避;若日后想独立调去重强度,需把两者拆开成两个参数。
   ——用 $p90$ 分位当 $\tau_\text{dup}$ 会太保守(去冗余几乎不触发),故直接复用 $\tau_a$。
2. **$\rho$ 各 $1/3$ 的隐患**〔记录〕:keep10 充足(每视角约 42 token 有 $\text{support}{>}0$,远超
   $B_a$);但在 **keep5/3 或更严的 $m_a$** 下,真 anchor 数 $\#\{\text{support}\ge2\}$ 可能 $< B_a$,
   此时 anchor 桶会用 $a{=}0$ 的 token 补满($a{=}0$ 的 argsort 顺序任意但确定)。
   → 到那步再上**自适应预算** $B_a=\min(\mathrm{round}(K/3),\ \#\{\text{support}\ge2\})$,省下的名额
   给 coverage。**v1 不做。**
3. **跨视角覆盖传播(软 3D 多样性)默认开**〔2026-07-03 决定〕:coverage = facility-location +
   xview 传播(§10 安全变体,仅沿 sharp 对应边)。CLI `--scm_xview 0` 可关做消融(纯特征
   facility-location = a20s40 的 coverage)。风险:sharp 匹配虽相对可靠,仍可能把不相干 token 误标
   已覆盖 → 决定"出问题再定位"(可用 `--scm_xview 0` 二分)。
4. **sharp token 双重偏好**:sharp token 在 anchor 和 saliency 两桶都受偏好,可能过配几何、挤占
   coverage;但被 $B_a,B_s$ 固定预算封顶,不会预算爆炸。
5. **增益本身在噪声内**(§11:~1 SE、$\tau/m$ 敏感、跨数据集未复现)⇒ SCMPruner 是否**稳过 random**
   仍需端任务确认,这是 v1 要跑的第一件事。
6. **去冗余只跨视角**:同视角内的 fuzzy 近重复不在桶二删,只由桶三 coverage 稀释。这是刻意选择
   (跨视角对应=信号,同视角重复=真冗余但另管);"改成同视角去重"是被讨论过、v1 未采纳的备选。

---

## 10. 延后的 v2 扩展:correspondence-cluster / 软 3D 多样性〔记录，供日后开启〕

把 coverage 从"特征多样性"升级到"3D **场景点**多样性",纯计算、不用 block、不用外部几何:

- **松对应建场景点**:token $t$ 与其在视角 $u$ 的最佳匹配 $\mathrm{bm}(t,u)$ 连边,当
  $s^{(1)}_{t,u}>\tau_\text{link}$(**松**,~$0.55$,**非** anchor 门槛)且**互为最近邻**(mutual-NN,
  防均匀区乱连)。连通分量 $\approx$ 一个 3D 点。**cluster $\ne$ anchor**:松门槛、覆盖**全体** token,
  故 anchor 稀缺/集中**不**影响全局多样性。
- **软版(推荐,2a)**:不硬聚类;facility-location 选中 $t$ 后,额外把 $\mathrm{bm}(t,\cdot)$ 的
  $\mathrm{cov}$ 置 $1$(同一 3D 点跨视角只占一个覆盖名额)。**仅对 sharp 匹配传播**以防噪。
- **硬版(2b)**:union-find 场景点 → 每 cluster 先给一个代表(轮询,按 cluster 内最高 $s$)→ 余额填充。
- **下界安全**:最坏退化成纯特征 facility-location(≈ random),不会更差。

---

## 11. 运行（Harness B）

```bash
# SCMPruner 本身(唯一可调 m_a;tag 用 =-scm 形式避免 argparse 把前导 - 当 flag)
python scripts/cvsp_curve.py --methods scmpruner --ratios 0.1,0.05 \
  --anc_m 0.12 --configs ego3d:Object_Centric_Absolute_Distance_MultiChoice \
  --n 200 --tag=-scm
# 对比基线:plain_random 与 a20s40(a20s40 用它自己的 ρ 和 tag)
python scripts/cvsp_curve.py --methods plain_random --ratios 0.1,0.05 --configs ... --n 200
python scripts/cvsp_curve.py --methods cvsp --rho_a 0.2 --rho_s 0.4 --ratios 0.1,0.05 --configs ... --n 200 --tag=-a20s40
```

输出:`logs/cvsp/<ds>.<task>.keep<pct>.scmpruner<tag>.jsonl`;脚本自动按 ACC 打分(Ego3D 看 ACC,不看 RMSE)。

---

## 12. 实测结果与调整方向（2026-07-05）

### 12.1 16 帧 no-think 全量对比（Qwen2.5-VL-7B）
设置:16 帧、no-think(`Output only the final answer … Do not include any reasoning.`,
`max_new_tokens=16`、robust MC 打分 `\b[a-d]\b`)、SCMPruner **20/40/40 + xview on**、
FastV **per-view**、5 个跨视角关系任务全量(rel_dir easy/med/hard、route_planning、
rel_distance,1872 题)。汇总(mean Δ,15 格 = 5 任务 × keep{25,10,5}):

| SCM vs | keep25 | keep10 | keep5 | **overall** |
|---|---|---|---|---|
| random | +0.013 | −0.002 | −0.012 | **−0.000** |
| VisPruner | +0.009 | −0.005 | −0.010 | **−0.002** |
| FastV(per-view) | +0.023 | +0.007 | +0.003 | **+0.011** |
| baseline(全 token) | −0.012 | −0.016 | −0.031 | **−0.020** |

最优方法计数(15 格):baseline **6.5** · FastV 3 · VisPruner 2.5 · random 2 · **SCMPruner 1**。

**结论(诚实):**
1. **SCM ≈ random ≈ VisPruner,击败 FastV(+0.011)。** 20/40/40 修好了 8 帧 with-think 版本
   "keep10 全面输 random(−0.021)"的退化 → 现在打平。
2. **压缩本身就伤精度**:baseline 总体最优(MEAN 0.384 vs 压缩方法 0.35–0.37),在 rel_distance /
   rel_dir_hard 上大幅领先,没有任何 informed selector 追回。
3. **反直觉**:SCM 对 random 的优势 **keep25 最大(+0.013)、keep5 反转 −0.012** —— 与"极端压缩
   selection 才重要"相反;噪声锚点在预算充裕时略帮、预算极小时反害。
4. SCM 唯一相对强项:压缩方法里在 **rel_distance** 三个 ratio 全部最好。
5. **确认项目主线:纯特征 query-agnostic 选择打平、不胜 random。** 原始结果
   `logs/Qwen2.5-VL-7B-{method}-keep{NN}-vsibench/{task}.result.json`。

### 12.2 xview 消融（进行中）
真实 Qwen 特征上 xview 开/关只改变 **keep25 6% / keep10 3% / keep5 0%** 的保留 token(2 样本实测)
→ 预计 ACC 差在噪声内。全量 xview-off 跑在 `Qwen2.5-VL-7B-noxv-*`(`--scm_xview 0`)。

### 12.3 调整方向（2026-07-05,用户判定优先级）
**有潜力(优先推进):**
- **ρ_a/ρ_s**(桶预算)—— 已 1/3→20/40/40;继续试 **ρ_a=0**(诊断:锚点到底有没有用)/ 更小 ρ_a。
- **anc_m**(sharpness 门槛)—— 项目实测最敏感的旋钮;暴露成 CLI 扫 {0.08,0.12,0.15,0.20}。
- **xview** 开/关 —— 正在消融。
- **query-aware(两阶段 / GeoScaffold)** —— **唯一被判断"可能真赢过 random"的路**;纯特征
  query-agnostic 的天花板 = 平 random。

**需进一步研究(暂缓):**
- **coverage → correspondence-cluster / 空间感知**(§10 的 v2 扩展)—— 有潜力,但要先把聚类/空间
  代理的可靠性研究清楚(对应关系噪声大)再上。

**结构性可选(记录,待触发):**
- **per-view floor**(每视角保底)—— 治"全局选饿死视角";改动小,建议优先试。
- 桶2 去重改 within-view;`anc_tau` 与 `τ_dup` 解耦;`match_idx` 用 mutual-NN / 多匹配。

## 13. 引用 / 相关文件

- `Notes/Anchor-Validation.md` §9–11 —— anchor 存在性(无去重 oracle)+ 纯特征 `support×sharpness` 验证 + $\tau/m$ 分布。
- `Notes/CVSP-Method.md` —— a20s40 / block-cvsp 母方法。
- `scripts/cvsp_curve.py` —— `anchor_scores`、`sel_scmpruner`、`_facility_pick`;`four_way_extreme.py` —— `collect`/`build_prompt_var`/`to_perview` 骨架。
