# Query-Aware SCMPruner（QA-SCMPruner）设计（2026-07-06）

> 把现有的 query-agnostic **SCMPruner** 扩展成 **query-aware 两阶段**压缩器,参考 MVPruner
> 的 CCTS/ITS,并**复用项目已有的两阶段基础设施**(qstage / FastV 的 in-LLM 层 K 剪枝)。
> 目标:在 query-agnostic 的天花板(≈ random)之上,看 query 条件能否把极端压缩下的
> ACC 推过 random。关联:[[scmpruner-method]]、[[geoscaffold-direction]]、
> `Notes/SCMPruner-Method.md`、`Notes/GeoScaffold-Story.md`、`Notes/CVSP-Method.md §13`。

---

## 1. 目标与动机

**一句话**:Stage-1 用 SCMPruner 做 query 无关、可缓存的**跨视角骨架**(过采样 N1);
Stage-2 在 **LLM 中间层(K=L/2)**、经过视觉-文本融合后,用 query 相关性把 N1 全局剪到 N2。

**为什么不纯 pre-LLM 做 query 剪枝**:输入嵌入空间里视觉与文本尚未经任何 attention 融合,
`input_cos` 只是词嵌入相似度,不是"该视觉 token 对回答此问题是否有用"。FastV / SparseVLM /
MVPruner 都把 query 剪枝放在 LLM 第 2–16 层(融合之后)。故 Stage-2 必须在 LLM 内。

**为什么这版是合理增量而非全新机制(诚实定位)**:关掉 anchor 保护、K 放到中间层后,本方法
≈ 现有 qstage 两阶段,**唯一实质改动是 stage-1 从 block-cvsp 换成 SCMPruner**(外加可选的
stage-1 软加权)。这是正当增量(SCMPruner 的 `support×sharpness` anchor 在头对头里胜过
block-cvsp 的 `cornerness×lowe_max`,见 `Notes/Anchor-Validation.md §9–11`),但 **§O 的先验
是:两阶段(input_cos r7,block-cvsp stage-1)已胜 VisPruner、但整体不胜 random**。所以这是
一个"换更好的 stage-1 + 调 K/signal/软加权,能否推过 random"的实验性赌注,不是必胜。

**约束(用户拍板)**:training-free、无额外模型;**Qwen2.5-VL-7B 与 InternVL3-8B 双跑**,结果
与现有 SCMPruner(`Notes/SCMPruner-Method.md §12`,VSI 16 帧关系任务)**同 harness、同预算口径
可直接对比**;query 信号 `{attn, cosine}` 留作实验裁决。

---

## 2. 与 MVPruner / 现有两阶段的关系

| | MVPruner (2606.27660) | 现有 qstage(§13) | **本方案 QA-SCMPruner** |
|---|---|---|---|
| Stage-1 | in-LLM 浅层,`Imp=U×R`(唯一性×instruction 余弦),**逐视角** | pre-LLM,block-cvsp,query 无关 | **pre-LLM,SCMPruner,跨视角**;R 以可选软加权注入 |
| Stage-2 | in-LLM 深层(~16),ITS=instruction 注意力,逐视角重分预算 | in-LLM 层 K,query 信号,**全局跨视角** | in-LLM 层 K=L/2,`{attn,cosine}`,**全局跨视角** |
| CCTS(跨阶段贡献) | R 混进 stage-1 打分 | 无 | **软加权开关**(stage-1 `s×relu(cos)`)= CCTS 的弱实现 |
| anchor 保护 | 无(不强留几何集合) | 无 | **无**(认为过 K=14 层已与 text 融合) |
| 粒度 | 逐视角(与本项目主线相反) | 跨视角 | **跨视角** |

MVPruner 是逐视角的——恰是本项目反复证否的方向;我们只借它的**跨阶段贡献**(CCTS)与
**指令引导选择**(ITS)两个思想,骨架仍全程跨视角。

---

## 3. 方法

记 `M = 视角数 × 256` 为总视觉 token 数;`L=28`(两模型 LM 均为 28 层,已核实);`K` 为
Stage-2 裁剪层(默认 `L/2=14`);上报预算 = 层平均 token 数 `T`(与 SCMPruner 的 keepNN 同口径)。

### 3.1 预算拆分(沿用 qstage §13 层平均公式)

```
N2 = round(T · L / (r·K + L − K))        # Stage-2 最终保留(层 K..L-1)
N1 = min(M, round(r · N2))               # Stage-1 过采样保留(层 0..K-1)
layer_avg = (N1·K + N2·(L−K)) / L  ≈ T   # 上报口径
```

`r` = 过采样比(N1/N2),默认 7(headline),建议至少扫 `{3, 7}`。K=14、L=28 时
`r·K+L−K = 112`,得 **N2 = 0.25·T·M、N1 = 1.75·T·M**:

| 上报(层平均) | Stage-1 N1(0–13 层) | Stage-2 N2(14–27 层) | VSI 16帧(M=4096) |
|---|---|---|---|
| keep25 | 43.75%·M | 6.25%·M | N1=1792、N2=256 |
| keep10 | 17.5%·M | 2.5%·M | N1=717、N2=102 |

`r=3` 更温和:keep10 → N1=15%(615)、N2=5%(205)。**注意 r=7 在 keep10 下 Stage-2 只剩
每帧 ~6 个 token,很激进**——r 是调这个拆分的旋钮。

### 3.2 Stage-1 — SCMPruner 过采样(pre-LLM,可缓存)

调用现有 `compressors/scm.py::scmpruner_keep_indices(feats, saliency, …, keep_ratio=N1/M)`,
三桶(anchor / saliency / coverage)不变,统一旋钮 `{scm_rho_a, scm_rho_s, anc_m, anc_tau,
scm_xview}` 沿用。产出 N1 个跨视角 token 的特征(每视角排序拼接)+ per-view 计数,喂给 LLM。

**软加权开关(`--scm_softweight`,CCTS 的弱实现)**:
- OFF(默认先测):`saliency` 原样传入 → Stage-1 纯 query 无关。
- ON:把传给 SCMPruner 的 saliency 替换为 `s' = s · relu(cos_i)`,其中
  `cos_i = cos(投影后视觉特征 v_i, 均值 query 词嵌入 q̄)`(= qstage 的 `input_cos`,pre-LLM)。
  **只改 saliency 桶的桶内排序**(anchor 用 `a`、coverage 用 G,均不受影响)。
  用 `relu(cos)` 而非 `cos`:cos<0(对问题负相关)直接置零、优先丢,且消除 `s×cos` 在负区的
  反向排序隐患(高 saliency×负 cos 会被排到低 saliency×负 cos 之后)。实践风险本就小
  (词嵌入各向异性/"窄锥" → cos 多为正、负区罕见进入),relu 零成本兜底。

### 3.3 Stage-2 — in-LLM 全局跨视角 query 剪枝(层 K,无 anchor 保护)

在 LLM 前向的第 K 层前,把 N1 个视觉 token 按 query 相关性**全局** top-N2 保留、其余丢弃
(含其 KV),保留 token 用 PESP 原位置(RoPE/因果序不变)。**无 anchor 保护**——认为 anchor 经
0..K-1 层已与 text 融合。

query 相关性 `score(v_i)`(`--scm_sig` 切换,实验裁决):
- `cosine`:层 K 隐状态上 `cos(h_K[v_i], mean_{j∈query} h_K[j])`(无需 eager 注意力,更省)。
- `attn`:层 K-1 上 **query token(post-image 提示 token)对每个视觉 token 的平均注意力**
  (`query_reduce='mean'`)——比 FastV 的"最后 token"更贴 MVPruner 的 ITS。

选择恒为全局 top-N2(`per_view=False`),契合项目"跨视角 > 逐视角"主线。

---

## 4. 架构与集成(改动清单,尽量小、且 config-gated 不回归)

主战场 = **两个 VSI runner**(双模型、与 SCMPruner §12 同 harness)。新方法 =
`--compress_method scmpruner_qa`。

1. **`compressors/scm.py`**(共享核心):
   - 加 `input_cos_relevance(vis_feats, query_embeds)` → 返回 `relu(cos)`(供软加权)。
   - 软加权由**调用方在传入前把 saliency 预乘该因子**(`s' = s·relu(cos)`),`scmpruner_keep_indices`
     签名不变——最小改动、核心不动。
2. **`compressors/qstage_llm.py`**(InternVL Stage-2,已支持 `{cosine,attn}`+全局 N2):
   - `_select_keep` 的 `attn` 分支从"最后 query token"改为"**over query_pos 平均**",由新字段
     `qs.query_reduce ∈ {last, mean}` 门控(默认 `last` 以**保持 FastV/qstage 现状不回归**;
     本方法设 `mean`)。
3. **`compressors/fastv.py`**(Qwen Stage-2 patch `make_fastv_forward_qwen`,现仅 `attn`+逐视角):
   - 加 `cosine` 分支(层 K 隐状态余弦,无需 eager 捕获);
   - `attn` 支持 `query_reduce='mean'`(捕获时对 query_pos 行平均,而非仅最后一行);
   - 全局 N2(`per_view=False`)——分支已存在,置位即可。
   - 均由 controller 字段门控,**FastV 默认行为(last/per_view/attn)不变**。
4. **两个 VSI runner**(`models/internvl3_vsibench.py`、`models/qwen2.5_vl_vsibench.py`):
   - `SPECIAL_METHODS` 加 `scmpruner_qa`;
   - 流程:算 N1/N2/K(§3.1)→ Stage-1 SCMPruner 出 N1 特征(软加权可选)→ 安装/配置 Stage-2
     controller(vis_pos=N1 视觉 token 在(缩减)序列中的位置,query_pos=post-image token,
     N2/K/signal/query_reduce)→ 生成(InternVL 走 `model.generate`;Qwen 走现有 `greedy_decode`
     + 已 patch 的 forward)→ off。
   - CLI:`--scm_r`(默认7)`--scm_K`(默认14)`--scm_sig{attn,cosine}`(默认 `attn`,两者都要扫)
     `--scm_softweight{0,1}`(默认0)+ 复用 `--scm_rho_a/--scm_rho_s/--anc_m/--anc_tau/--scm_xview`。
   - **输出目录变体命名**:平行新增 `scm.scmpruner_qa_tag_suffix(r,K,sig,softweight)`,把
     非默认旋钮编入 tag(如 `-r3`、`-sigcos`、`-sw1`),默认配置(r7/K14/attn/sw0)→
     `…-scmpruner_qa-keep<NN>-vsibench`,避免扫参时 resume 撞车污染。
5. **run 脚本**:仿 `scripts/run_vsi16.sh` 加一个把 QA-SCMPruner 与 baseline/plain_random/
   VisPruner/SCMPruner/FastV 并排跑 5 个关系任务的驱动。

> **不做**:不把只支持 InternVL 的 qstage 移植到 Qwen(FastV 那半已把 Qwen 的 in-LLM 层 K 剪枝
> 做好,复用即可);Ego3D 侧(cvsp_curve)本期可选、不阻塞。

---

## 5. 数据流(每模型)

**InternVL3(model.generate + 视觉特征注入)**:
`extract_feature → (可选 input_cos 软加权) SCMPruner 选 N1 → visual_features(N1) + N1 占位 prompt
→ 配置 QStage(K,N2,signal,query_reduce=mean,vis_pos,query_pos,per_view=False)→ model.generate
(patched forward 在层 K 把 N1→N2)`。

**Qwen2.5-VL(自定义 greedy_decode)**:
`visual encoder → (可选软加权) SCMPruner 选 N1 → keep_mask 造 red_embeds(N1 视觉 token)→ 配置
QwenFastV(同上参数,vis_pos=red_embeds 内 N1 视觉位置)→ greedy_decode(patched forward 层 K
N1→N2)`。

两模型 Stage-2 的选择逻辑经统一字段驱动,力求"给定相同 K/N2/signal 行为一致"。

---

## 6. 诚实的局限与风险

1. **新颖性收窄**:headline(软加权 off、无保护)≈ qstage 换 SCMPruner stage-1。§O 先验=两阶段
   不胜 random。这是实验性增量,不保证过 random。软加权/signal/r/K 的消融用来定位"是否有任何
   query 耦合真有用"。
2. **预算口径对两阶段偏宽**:层平均相等,但两阶段在 0..K 层前载 N1(=1.75×T)token,早层信息比
   SCMPruner 的平坦 T 多。结果要按"等层平均"解读,不能宣称"更少 token"。
3. **attn 信号偏置**:注意力有位置偏置(AdaTP);no-think prompt 末尾是格式指令,`query_reduce=mean`
   over 全部 post-image token 会掺入格式 token。缓解=优先用 question span(v2);v1 先跑对比。
4. **软加权 = pre-LLM 弱信号**,可能被 Stage-2 更强的 in-LLM query 剪枝盖过而无增益——这正是开关
   要测的。
5. **K 越深,Stage-2 影响越小**(更多层看 N1),且等层平均下 N2 越小越激进(r=7,keep10 → 2.5%)。
   K 与 r 都需扫。
6. anchor 保护已移除;若结果显示极端预算下几何骨架被 query 误剪,再作为开关加回(不在本期)。

---

## 7. 实验计划与成功判据

- **模型**:Qwen2.5-VL-7B + InternVL3-8B。**数据/任务**:VSI 16 帧、5 个跨视角关系任务
  (`object_rel_direction_{easy,medium,hard}`、`route_planning`、`object_rel_distance`),全量,
  no-think + `\b[a-d]\b` 打分(与 §12 一致)。
- **预算**:keep{25,10,5}(与 §12 同格)。
- **基线(同 harness)**:baseline(全 token)、plain_random、VisPruner、**SCMPruner(§12)**、FastV。
- **QA-SCMPruner 消融**:`softweight∈{0,1}` × `sig∈{attn,cosine}` × `r∈{3,7}`(K=14 固定,先),
  再对最优组合扫 `K∈{7,14,20}`。
- **成功判据(主)**:在关系任务 × 预算的格子上,QA-SCMPruner **稳过 plain_random**(符号检验
  ≥ 多数格),且不低于 SCMPruner。**次要**:哪个 signal 胜、软加权是否加分、K/r 的趋势。
- **打分**:Ego3D 若跑则看 ACC(非 RMSE);VSI 用现有 `compute_metric`。

---

## 8. 测试(无正式测试框架,冒烟为主)

- 语法:`py_compile` 改动文件。
- 冒烟:两模型各 `--compress_method scmpruner_qa --limit 2`,断言
  (a) prompt 内 `<IMG_CONTEXT>` 数 == N1(Stage-1 后),
  (b) Stage-2 后实际过后半段的视觉 token 数 == N2,
  (c) 变体 tag 目录随非默认旋钮变化、默认配置不变。
- 决定性:同 seed 同参两次跑,JSONL 逐行一致(SCMPruner 确定性 + query 信号确定性)。
- 不回归:跑一次现有 `fastv` / `scmpruner`,确认行为与改动前一致(默认字段未变)。

---

## 9. 范围外(YAGNI)

- Ego3D(cvsp_curve/qstage_curve)接入 —— 可后续;本期只 VSI 双模型。
- 真·ITS 的 question-span 精确切分、注意力去偏(AdaTP 式)—— v2。
- anchor 保护开关、双 in-LLM 阶段(MVPruner 严格版)、re-densification(GeoScaffold §2 的
  side-cache 重注入)—— 不在本期。
