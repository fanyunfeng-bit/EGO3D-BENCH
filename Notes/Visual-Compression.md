# 多视角视觉 token 压缩 / 推理加速 —— 问题与挑战

> 本文件记录"多视角 MLLM 视觉 token 压缩 / 推理加速"方向。**2026-06-15 重梳理**：当前版本聚焦**问题定义 + 挑战框架**;具体方案(旧 C1 跨视角去重 / C3 轻量几何等)待挑战定稿后在此之上重建——旧版备份见 `Visual-Compression.archive-20260604.md`。
> 关联：推理方案 [Spatial-Reasoning.md]、联合方案 [Joint.md]、背景调研与竞品精读 [论文笔记.md]。
> 状态图例：🟢已验证 / 🟡进行中 / ⚪构思 / 🔴存疑。

---

## 目标基准（三类 regime，要求普适）
| Regime | 基准 | 形态 | 冗余来源 | 对加速的意义 |
|---|---|---|---|---|
| 单相机室内**视频** | **VSI-Bench** 2412.14171 | 采样多帧、token 密集 | 高帧间重叠(**当无序视角→跨视角同物**收割,不吃时序顺序) | **加速主场**(token 多才省得出绝对收益) |
| 同步多机位 ego **动态** | **Ego3D-Bench** 2509.06266 | 5/6/7 路同步相机 | 视角重叠(室外动态) | 中等 token |
| 宽基线静态**多机位** | **All-Angles** 2504.15280 | ≥3 机位、可达大视角差、同款主体 | 跨视角同物 | 少视角、低重叠 |

**设计原则**：training-free 优先 · pose-free 优先 · **严格不使用时序信息**(2026-06-15 定;视频帧一律当无序视角集 → 同一算子跨三 regime 逐字相同) · **跨三类 regime 普适**。

---

## 问题定义
多模态大模型对**多视角输入**做 3D 空间理解的研究日益增多,但多视角显著放大了视觉 token 数 → 计算量(注意力二次复杂度)激增。**目标:对这一推理过程做视觉 token 压缩 / 加速,且方法须跨 {视频, 同步多机位, 宽基线多机位} 三类 regime 普适,而非只在单一数据集生效。**

> 压缩若**顺带**提升空间推理 = 加分项(此时演化为 [Joint.md] U1);但**当前重点是加速,提精度是另一个方向**。

**普适性的脊柱**：三类 regime 的"冗余"来源不同(时序邻接 / 视角重叠 / 跨视角同物),但**唯一不变的冗余信号 = "同一块 3D 内容被成像了多次"**。现有压缩器各用一个 regime-specific 代理逼近它(FastVID 用时序邻接、SeGPruner/Seeing-Once 用已知位姿的体素重叠、Prune2Drive 用各视角独立多样性),故都不普适;本方向直接对这个不变量下手(pose-free 视觉对应)。**难度梯度(必须正视)**:跨视角去重**价值最高**处(宽基线 All-Angles)恰是**廉价 pose-free 对应最不可靠**处;对应最容易处(视频)又是现有时序方法已做得很好处。

---

## 核心挑战：三类六条

### 一、留什么 token（selection —— 哪些 token 重要）

**① 视角级重要性** —— 不同视角对回答当前 query 的重要性不同,如何判定哪些视角重要。`[query 相关]`

**② 视角内内容重要性** —— 每个视角只有部分信息对回答关键,如何定位这部分内容,而非整张图均等对待。`[query 相关]`

**③ 跨视角信息关联** —— 两类困难:
- **③.1 跨视角对应(互补)**：需多视角联合的任务里,某视角信息不足时,如何确定**其他视角的哪部分**是补充性信息,从而针对性保留。`[query 相关]`
- **③.2 跨视角去冗余**：某视角部分内容确与他视角关联,但若 **A 视角已足够回答**,则 B 视角与之关联的冗余信息可删。`[query 相关]`

**④ ★ 空间几何重要性（与 query 相关性并列的第二重要性轴）** —— 压缩时**不能只看 query 相关度**,还须看对 3D 空间理解的承重度:`importance = f(query相关度, 几何承重度)`,**后者不可为 0**。两类"非 query 却必需"的 token:
- **(i) 参照系锚点**：query 无关的静态结构(墙 / 地 / 地标 / 角点 / 门框)——撑起全局坐标系;删光则相对方向 / 距离崩。`[query 无关 → 可预计算复用]`
- **(ii) 多跳关系桥**：连接被问两物体、但浅层 query 解析没列出的中间锚点(如判"沙发在电视左边"需要连接二者的灯 / 桌)。`[query 相关 → 随题变]`

> **与现有 diversity / scene-completeness 的关系(实事求是,挑战④的 make-or-break)**：空间多样性(Geo3DPruner SDP 的 3D-FPS、CDPruner 的 DPP、逐区域覆盖)会把保留 token 在场景里铺开,**隐式、部分地**覆盖 (i) 类锚点——且对**3D 空间**多样性强、对纯**特征**多样性(CDPruner/DART)弱。**但"隐式"= 不可靠**:① 它优化"铺开 / 不相似"而非"承重",目标错位(铺开的无纹理墙皮被留、关键角点却可能因挨着别的 token 被删);② query 无关,**几乎完全漏掉 (ii)**;③ 某些 saliency 把平整墙地判为低信息而**优先删**,反伤几何。故几何承重应**显式**成第二轴。**诚实保留待验证项**:必须实测"显式几何承重"是否**超过强多样性基线**(Geo3DPruner SDP / CDPruner);若 diversity 已隐式吃掉大部分增益,显式轴可能只是边际改进。

### 二、压多少 token（budget —— 压缩率）

**⑤ 自适应压缩率** —— 不同场景 / query 能容忍的压缩率差异大;压缩率须随**内容 + query** 自适应,而非固定比例或靠验证集标定(Prune2Drive 的硬伤)。"多少 token 才够"是独立于"留哪些"的旋钮。

### 三、元约束（cost —— 别白忙）

**⑥ 净加速：压缩器自身开销 < 它省下的** —— 估重要性 / 对应 / 几何那一步本身有成本;尤其少视角(All-Angles / Ego3D)输入本就不大时,极易把省的吃回去甚至倒亏。须按**含估计开销的端到端 FLOPs / 延迟 / 显存**算账(现有工作集体回避此账)。

> **贯穿性张力(query-aware 的可缓存代价)——直接决定⑥**：①②③④(ii) 都 **query 相关** → 每来一问就得**重算压缩**、辅助开销摊不掉;而 query 无关部分(④(i) 锚点)可**预计算复用**。"针对性 vs 可复用"既影响净加速,也决定方法该放在 **LLM 前**(可缓存、省 prefill、用不上 LLM 注意力)还是 **LLM 内**(随 query、可用注意力但信号不可靠、省得少)。

---

## 挑战 → 现状盲区速查
| 挑战 | 现有方案触及程度 |
|---|---|
| ① 视角重要性 / ② 视角内内容 | 部分(通用 token 剪枝 / query 感知选择已碰,但非跨视角联合) |
| ③.1 互补 / ③.2 去冗余 | ③.2 有(各类跨视角/几何去重);③.1"定位他视角补充信息"少见 |
| **④ 几何承重(第二重要性轴)** | **盲区**:只有隐式 diversity,无"与 query 并列的显式几何承重轴" |
| **⑤ 自适应压缩率** | **盲区**:多为固定 / 验证集标定 |
| ⑥ 净加速(含估计开销) | **盲区**:无人报含几何骨干开销的端到端账 |

## 已定决策（2026-06）
- **时序:严格不用**(06-15),视频帧当无序视角集 → 同一算子跨 regime 逐字相同。代价:VSI 上不吃邻接红利、正面对时序专家(FastVID 等)。
- **算子路线:走"软"(scheme B),否决"硬簇"(A —— 需精准跨视角实例关联,难+引额外模型+吃加速,起步阶段太重)**。核心信号 = **跨视角支持度**(ViT 特征相似度算"内容被多少视角看到")**驱动压缩**:高=冗余(去重)、低=互补(留);**锚点 = 支持度 × 独特性门(Lowe ratio)**,稀疏保护、补④几何承重。全程不碰 VGGT、不建硬簇。
- **基方法:从 VisPruner(2412.01818)出发**(post-encoder、training-free;视觉显著性 = CLS-attn + 多样性去重)。**思路1** = 把它的"**图内**多样性"升成"**跨视角**支持度" + 加**锚点保护**。
- **regime:All-Angles 先搁置**(大基线、ViT 对应最难);起步盯 VSI + Ego3D(高重叠、对应最可靠)。All-Angles 留作后续"只有我能做"的主场。
- **基座:Qwen2.5-VL-7B + InternVL3-8B(双基座验证普适)**。⚠️ saliency 信号须 **encoder-agnostic**:Qwen2.5-VL **无 CLS token**(VisPruner 的 CLS-attn 用不了)、InternViT 是否有可用 CLS **待确认** → 候选 attention rollout / 特征范数 / LLM text-visual attn。**落地前要先定 point① 的 saliency 取法。**
- **数据集:Ego3D-Bench + VSI-Bench(All-Angles 待选)。**

---

## 实测发现（2026-06-16）：极端压缩率下的「反直觉冗余」

> 用现有 harness(`models/internvl3_compress.py` / `internvl3_vsibench.py` + `compressors/`)在 InternVL3-8B / Qwen2.5-VL-7B 上跑出的实证,直接关系到 CVSP 的 make-or-break。

### A. 跨架构 keep 25%（已有）
- VSI(route_planning) + Ego3D(6 类),InternVL3-8B **与** Qwen2.5-VL-7B:剪掉 75% 视觉 token,性能**普遍不掉、常常更好**;VisPruner ≥ random 但仅 ~1pp(在噪声内)。→ "压 75% 几乎无损甚至更好"**跨架构成立**。详见 `RESULTS_qwen_vsibench_keep25.md`、`RESULTS_vispruner_keep25.md`。

### B. ★ 极端率崩溃曲线（新，InternVL3-8B｜VSI object_rel_direction_easy｜ACC｜n=217｜chance=0.25｜baseline 256 tok/帧）
| keep | tok/帧 | random | vispruner | visp−rand |
|---:|---:|---:|---:|---:|
| 100% | 256 | — | 0.4931 (base) | — |
| 25% | 64 | 0.5161 | 0.5069 | −0.009 |
| 10% | 26 | 0.5207 | 0.5484 | +0.028 |
| 5% | 13 | **0.5576** | 0.5253 | −0.032 |

结果(`logs/InternVL3-8B-{random,vispruner}-keep{10,5}-vsibench/object_rel_direction_easy.result.json`)。**判读(对 CVSP 不利,但探针太糙):**
1. **random 从头到尾不崩** —— 压到 5%(13 tok/帧)反而是全场最高(0.5576),每一档都 > baseline。"random 在极端率崩溃"的假设在此任务上**直接证伪**。
2. **random ≈ vispruner,符号乱跳**(−0.9 / +2.8 / −3.2pp,全在 ±3pp 噪声内),5% 处 random 还反超 vispruner 3.2pp → informed 选择**无可见 headroom**。
3. **probe 失效**:relative-direction 只需**粗布局**,13 tok/帧足够答 → 无视觉保真度梯度,谁都不崩;"越压越好" = **attention dilution 缓解**(token 少→更专注),与"选对 token"无关。**这类粗任务对压缩方法没有区分力,是错误战场。**
4. **决定性实验未完成**:Ego3D 绝对距离(RMSE,连续指标、需保真度)在 `random keep25` 中途**人工停止**(2026-06-16)→ "random 是否在保真度敏感任务上崩"**仍开放**。若续:resume 跳过已写 JSONL 行即可。

### C. 由此提炼的研究洞察（2026-06-16）
- **多视角输入冗余/噪声极大,且反直觉**:压缩很少变差、常常变好。多视角把同一 3D 内容成像多次 → 大量冗余 token **稀释注意力、当 distractor** → 删一部分反而提精度。
- **潜在新方向**:把"**多视角去噪 / 降冗余**"本身当作**提精度**的手段(不只是加速)—— 压缩即降噪,呼应 [Joint.md] U1 与 [Spatial-Reasoning.md] R2。**前提**:必须在**保真度敏感任务**(绝对距离 RMSE / counting / IC)上验证,而非 relative-direction 这类粗任务。
- **对 CVSP 的当前裁定**:在已观测到的工作点上,"informed 跨视角选择 > random"的 headroom **尚未出现**。CVSP 要立住,必须先找到 **random 真崩、而 informed 还撑**的工作点/任务(下方 TODO 安全工作点实验的核心);找不到 → 方向需重估,或退守"VSI 多帧**纯加速**账"(token 多才省得出绝对收益)。

---

## 实测发现（2026-06-17）：Ego3D keep10 全任务 + 视觉消融对照（方向定性的关键一组）

> 接续上节。把"保真度敏感任务"重扫做完(上节 B.4 遗留的决定性实验),并加做"模型到底用不用视觉"的消融对照。**代码正确性已三重验证**:源码 `generate()` 传 `visual_features` 即跳过全量 `extract_feature`,且布尔掩码赋值 `input_embeds[selected]=vit_embeds` **无 try/except、数量不符必崩**;单样本实测 LLM 实收序列 = **290(156 视觉+134 文本) vs baseline 1670(1536 视觉)**。反直觉结果是真的,非 bug。

### D. Ego3D keep 10%（剪 90%）6 任务全集，InternVL3-8B（baseline 已有,只跑 random + vispruner）
| 任务 | 指标 | n | base | visp@25 | **visp@10** | **rand@10** |
|---|---|---:|---:|---:|---:|---:|
| Obj_AbsDist | RMSE↓ | 937 | 28.14 | 23.82 | 17.55 | 18.80 |
| Ego_AbsDist | RMSE↓ | 687 | 12.78 | 12.85 | 10.88 | 13.17 |
| Obj_AbsDist_MC | ACC↑ | 937 | 0.495 | 0.475 | 0.459 | **0.474** |
| Ego_AbsDist_MC | ACC↑ | 687 | 0.531 | 0.515 | **0.493** | 0.489 |
| Travel_Time | ACC↑ | 458 | 0.448 | 0.439 | 0.419 | **0.430** |
| Localization | ACC↑ | 770 | 0.351 | 0.358 | 0.370 | **0.401** |

ACC 平均:base **0.456** → rand@10 **0.449**(−0.8pp)→ visp@10 **0.435**(−2.1pp)。结果:`logs/InternVL3-8B-{vispruner,random}-keep10/<task>.result.json`,驱动 `scripts/run_ego3d_keep10.sh`。
1. **"压会降很多"不成立**:剪 90%(1546→157 tok,−82% FLOPs)干净 ACC 平均只掉 0.8/2.1pp。Ego3D 比 VSI **能测出真实下降**(选 Ego3D 对了),但是小降不是崩。
2. **RMSE 是假象**:同一 object-distance,RMSE 说压了更准(28.14→17.5)、ACC 说更差(0.495→0.459)→ 回归均值 / 少了被 clamp 的灾难性过估。**Ego3D 上以 ACC 为准,RMSE 不可信。**
3. **★ 对 CVSP 致命**:干净 ACC 上 informed 打不过 random —— 4 个 ACC 任务 random 赢 3 个(Obj_MC +1.5、Travel +1.1、Loc +3.1pp),vispruner 只在**可疑的 2 个 RMSE** 上领先。

### E. ★ 视觉消融对照（model 到底用不用视觉?）InternVL3-8B｜2 个 4-way MC｜n=250 matched 子集｜chance=0.25
4 条件:`real_full`(1536 真实)/ `keep10_visp`(156 真实)/ `black_full`(全黑图过编码器)/ `noise_full`(scale-matched 高斯噪声替换特征)。脚本 `scripts/vision_ablation.py`,结果 `logs/ablation/`。

| 条件 | Ego_MC | Object_MC |
|---|---:|---:|
| real_full | 0.476 | 0.432 |
| keep10_visp | **0.476** | 0.388 |
| black_full | 0.388 | 0.308 |
| noise_full | 0.380 | **0.280** |

信号拆解(上 chance 部分):

| 任务 | 语言先验地板(black/noise 均值) | 真实视觉贡献 | keep10 vs real |
|---|---|---|---|
| Ego_MC | 0.384(**59%**) | +0.092(41%) | **±0.000** |
| Object_MC | 0.294(24%) | +0.138(**76%**) | −0.044 |

判读(定性,n=250 SE≈±3pp;绝对值与全集 D 有小出入,**定性为准**):
1. **模型真用视觉,不是 (b)**:Object_MC `noise=0.280≈chance`、`black=0.308`,换掉真实内容直接掉到随机;视觉占该任务信号 **76%**。**课题前提有效**(不是在研究"不看图"的模型)。
2. **多视角极度冗余但有边界,(a) 成立**:Ego_MC `keep10=real` 零损;Object_MC keep10 掉 4.4pp(丢约 1/3 视觉贡献)。**90% 压缩在高冗余任务无损、在视觉依赖强的任务轻微有损。**
3. **"压缩不掉" = 视觉冗余 +(尤其 Ego_MC)重度语言先验**(59% 信号黑图就能拿)叠加。

### F. 综合裁定（2026-06-17,对方向影响最大）
- **CVSP 现状**:VSI(B)+ Ego3D(D)两 regime,"informed 跨视角/显著性选择 > random" 都**拿不到证据**,根因(E)已查清:**可争夺的视觉盘子只有 ~9–14pp,且其内高度冗余 + 大量语言先验** → 选择质量没有发挥空间。这是从 saliency(VisPruner)层面的硬约束,CVSP 在其上加跨视角支持度要翻盘,门槛极高。
- **三条路**:(i) 死磕视觉依赖最强 + 压缩确实丢信号的任务(Object_MC 类)+ 更极端压缩率,看 informed 能否翻盘;(ii) **pivot 成分析/发现型贡献**——"多视角输入极度冗余 + 模型重度依赖语言先验"本身是干净、反直觉、可发表的发现;(iii) 放弃 informed 叙事,只主打效率(−82% FLOPs 近乎无损)。

### G. ★ 补证(P1–P3):VisPruner 留的 token 比 random 更冗余、覆盖更差 —— 硬证据
> 故事/方法见 [CVSP-Story.md]。脚本 `scripts/redundancy_analysis.py`,数据 `logs/redundancy_analysis.jsonl`。**纯编码 + 算指标,不跑生成**。指标(在 LLM 实收 token 特征空间,threshold-free):**R = 跨视角冗余**(每个保留 token 对"其它视角保留 token"的最大余弦,越高越冗余);**C = 覆盖**(每个被删 token 对"任一保留 token"的最大余弦,越高越好)。keep10、4 个 ACC 任务、各 n=250。

| 任务 | R_visp | R_rand | ΔR | C_visp | C_rand | ΔC |
|---|---:|---:|---:|---:|---:|---:|
| Object_MC | 0.616 | 0.553 | **+0.063** | 0.555 | 0.613 | **−0.058** |
| Ego_MC | 0.629 | 0.562 | +0.066 | 0.560 | 0.620 | −0.059 |
| Travel_Time | 0.611 | 0.547 | +0.064 | 0.553 | 0.609 | −0.056 |
| Localization | 0.621 | 0.558 | +0.062 | 0.558 | 0.616 | −0.058 |

**A(机制)= 铁证**:每个任务、**每个样本(100%)**,VisPruner 都比 random 更跨视角冗余、覆盖更差;配对 t 全部 |t|>45(p≈0)。诊断从"推测"升级为"硬证据"。(R 与 C 同向,是同一结构事实的两面,不算两个独立证据。)

**B(轴)= 同预算口径下成立**:money 图(同 156 token 预算,R vs 现有 keep10 精度)上,**冗余更低的 random 精度更高**——Object_MC(0.474>0.459)、Travel(0.430>0.419)、Localization(0.401>0.370)3 个任务都符合;Ego_MC 反例(0.493 vs 0.489,0.4pp 在噪声内)。→ **"informed 赢不过 random"的根因 = random 保留集冗余更低、覆盖更全。**
- ⚠️ **诚实边界**:"冗余↓→精度↑"是**同 token 预算**口径。换预算不单调(baseline 1536 token 冗余绝对更高,精度却大多 > keep10——token 多=信息多,是另一根轴)。正确表述:"**给定预算,把预算花在低冗余 / 高覆盖上更好**"。
- **下一步 = P4(因果)**:做一个显式压低 R / 拉高 C 的保留集(全局 FPS),看它精度能否**真的超过 random**(同预算)。A/B 只说明"random 因低冗余而赢",P4 才回答"我们能否更低冗余 + 换来更高精度"——方法配不配发表的硬证据。

### P4 结果(因果,2026-06-17)：压冗余路线被否
全局 FPS(`scripts/p4_fps_causal.py`,`logs/p4/`)把跨视角冗余 R 压到最低(0.44 << random 0.55 << visp 0.62),但**精度没超过 random**(Object_MC 0.404 vs 0.412、Ego_MC 0.468=0.468)。→ **"R 低 → Acc 高"是必要非充分,不是优化目标**;FPS 用"挑特征离群点"压低 R,低 R 没换来高 Acc。**别再直接优化 R**。真问题 = "什么 query-agnostic 信号能选出比均匀 random 更有用的 token"(已知死路:显著性、特征多样性)。方法设计转向"地标信号 + 招考察队引擎",见 [CVSP-Story.md]。

### H. ★ 体检:leverage / 有效秩诊断(2026-06-17)—— 重定向了工作点
> 脚本 `scripts/leverage_diagnostic.py`,数据 `logs/leverage_diagnostic.json`。纯算不跑生成。每场景取 ~1536 块特征 → 余弦 Gram 的谱形状。Ego3D + VSI 各 n=150,结果几乎一致:

| 指标 | Ego3D | VSI | 读法 |
|---|---:|---:|---|
| 有效秩/M | 0.055(**≈84 维**) | 0.053(**≈81**) | 远 ≪ 噪声 0.807 → 高度冗余 ✓ |
| 单视角 有效秩/M | 0.168(≈43) | 0.162(≈41) | 多视角比单视角低 3× → 冗余**跨视角** ✓ |
| leverage Gini | 0.110 | 0.118 | > 噪声 0.002(结构真实)但绝对值低 = **只轻微集中** ⚠️ |
| coherence | 1.99 | 1.97 | 最突出块仅 ~2× 均值,非少数 VIP ⚠️ |
| corr(leverage, 地标) | 0.014 | 0.095 | **≈0 → 高 leverage 块 ≠ 地标** 🔴 |

**三条判读**:
1. 🟢 巨大、跨视角的冗余 → CVSP 前提成立。
2. 🎯 **最关键**:**有效秩 ≈84,而 keep10 预算 ≈156 > 84** → 留的块比有效维度还多 → **这就是 keep10/25 上谁都赢不了 random 的原因**。**真正的战场在"预算 < 有效秩"= keep5(~78)/keep3(~46)/keep2(~31)**。→ **之前在 keep10/25 的选择实验全在"谁都赢不了"区间。**
3. ⚠️🔴 即便到极端率,油水也小(Gini 0.11、coherence 2 = 温和的尖);且**地标 ≠ 高 leverage 块**,纯锚点假设少了"高 leverage 正好是地标"这张牌。

**下一步 = keep5/keep3 四方对照**(stratified random / 地标 / leverage / 招考察队引擎),在预算 < 有效秩处看 informed 能否终于小超 random;不能 → 承认是效率故事(~84 维、巨量无损压缩),pivot。

### I. ★ keep3/keep5 四方对照(2026-06-18,Ego3D + VSI,n=200)
> 脚本 `scripts/four_way_extreme.py`,数据 `logs/fourway/`。4 方法:stratified-random / anchor(地标 top-K) / leverage(ridge) / engine(质量加权 log-det)。Ego3D{Obj_MC,Ego_MC} + VSI{rel_dir_easy,rel_dist},chance=0.25。

| | strat_rand | anchor | leverage | engine |
|---|---:|---:|---:|---:|
| **keep5** Ego3D Obj_MC | 0.370 | 0.370 | 0.330 | 0.380 |
| keep5 Ego3D Ego_MC | 0.495 | 0.500 | 0.435 | 0.450 |
| keep5 VSI rel_dir | 0.450 | **0.525** | 0.495 | 0.495 |
| keep5 VSI rel_dist | 0.330 | **0.385** | 0.255 | 0.355 |
| keep3 平均(4 任务) | 0.431 | 0.430 | 0.381 | 0.433 |
| keep5 平均(4 任务) | 0.411 | **0.445** | 0.379 | 0.420 |

**判读**:
- 🔴 **leverage 死透**:几乎每个任务垫底 → "数学上独特"的 token 没用(同 FPS 在挑离群)。砍。
- 🟡 **anchor 唯一亮点 = VSI keep5**(rel_dir +7.5pp、rel_dist +5.5pp,两任务同向;Ego3D 全平)。engine 没赢过 plain anchor → **信号(地标)才关键,花哨引擎没用**。
- ⚠️ **大概率噪声**:strat_random 自己跨预算就晃 8pp(VSI rel_dir keep3=0.53 vs keep5=0.45)> 我们看到的所有 win;n=200 SE~3.5pp。**没有 informed 方法稳定赢过 stratified random。** 体检"温和的尖、油水极小"被兑现。
- **唯一可赌**:anchor 在**高重叠 VSI**、keep5、两任务同向冒头(理论说 anchor 在高重叠信号最多)。真假分不清 → **实验 A**:anchor vs strat-random,多 VSI 任务、大 n(400)、keep5,把 random 噪声压下去定真伪。真→窄但实的故事;假→pivot 效率叙事。

### J. ★ 实验 A:anchor vs stratified random,VSI 全 MC、大 n(2026-06-18)
> 验 §I 的 VSI 苗头真假。只跑 anchor vs strat-random,6 个 VSI MC 任务、keep5、n 拉到 400/全量,压低 random 噪声。脚本 `four_way_extreme.py --methods strat_random,anchor`,数据 `logs/fourway/`。

| 任务 | strat | anchor | Δ | n | 类型 |
|---|---:|---:|---:|---:|:--|
| rel_dir_easy | 0.447 | 0.525 | **+7.8pp** | 217 | 空间 |
| rel_dir_hard | 0.322 | 0.349 | +2.7pp | 373 | 空间 |
| rel_distance | 0.307 | 0.330 | +2.3pp | 400 | 空间 |
| rel_dir_medium | 0.431 | 0.444 | +1.3pp | 378 | 空间 |
| route_planning | 0.304 | 0.273 | **−3.1pp** | 194 | 空间 |
| obj_appearance_order | 0.343 | 0.273 | **−7.0pp** | 400 | 时序 |

**全 6 任务平均 Δ=+0.7pp≈0;仅空间 5 任务 +2.2pp 但不全正(route −3.1)。** 裁定:
- §I 的 VSI 苗头**没扛住大 n**:rel_distance 从 +5.5(n=200)缩到 +2.3(n=400);唯一大赢只剩 rel_dir_easy(+7.8)。
- anchor 是**空间专属信号**:帮方向/距离(多为 +1~3pp),**伤时序(appearance_order −7)**;固定 query-agnostic 配比无法按任务自适应。
- **query-agnostic 路已穷尽**:显著性 / 多样性(FPS) / leverage / 锚点 / engine 全部 ≈ 或 < stratified random(跨 keep5/10/25、跨 Ego3D/VSI、跨架构)。**没有 informed 方法稳定赢随机。**

**最后一搏(进行中):combo = anchor + 显著性 + 多样性**(`q = norm(anchor)+norm(saliency)` 喂 engine)。假设:显著性当"非几何任务安全网",救回 appearance_order/route 的负数。**判读**:combo 平均明显 > random、救回负数、不杀 rel_dir 的赢 → 窄但实的鲁棒小赢;否则 → 三种组合(VisPruner / engine / combo)全验完 → **pivot 到效率+发现型论文(故事 A)**。query-aware 是唯一没碰的精度杠杆,作 future work。

---

### K. ★ CVSP 主曲线(§8 定型方法,`cvsp_curve.py`,2026-06-19 跑完)
> combo 已取消(2026-06-18,见 [CVSP-Story.md] 实验现状),方法定型为 [CVSP-Method.md] 的**配额式三档 anchor/saliency/coverage + 跨视角去重 + φ 保底**。本节是它的主曲线:5 方法 = baseline / plain_random / **stratified_random** / vispruner / **cvsp(ρ_a/ρ_s/ρ_c=0.4/0.3/0.3, τ=0.85, φ=1)**,r∈{25,10,5,3}%,8 任务(6 空间+2 时序),n=200。数据 `logs/cvsp/`。

**跨 8 任务平均 ACC**(informed 三家全挤在 1.5pp 内):

| keep | plain | strat | visp | cvsp | baseline |
|---:|---:|---:|---:|---:|---:|
| 25 | .394 | .397 | **.407** | .405 | .427 |
| 10 | .364 | .396 | .393 | **.397** | .427 |
| **5** | .364 | **.386** | .385 | .381 | .427 |
| 3 | .379 | .374 | .376 | **.393** | .427 |

**cvsp − strat,配对 McNemar:32 格仅 3 格显著**——rel_dir_hard keep3 **+12pp(p<0.01,赢)**、rel_dir_medium keep3 +7.5pp(p=0.09)、appearance_order keep3 **−8pp(p=0.04,输)**。其余 29 格 p>0.10=噪声。按档:keep5 平均 **+0.000、3/8、0 显著**(预注册命门**是平的**);keep3 平均 +1.8pp 但只 1 赢 1 输。

**裁定**:
1. **§8 预注册门槛未达标**——keep5 完全平、keep3 靠单点。**CVSP 与 stratified_random 统计上不可区分**(除 rel_dir_hard keep3 一格)。
2. **空间/时序劈分坐实**(同实验 A):极端压缩下 anchor 帮难空间题(rel_dir_hard +12)、伤时序题(appearance_order −8)。→ **聚焦空间、剔除时序。**
3. **vs VisPruner(真正对手)**:keep3 空间 **3 赢 3 输**;keep5 空间 **2 赢 4 输**。要赢得"多数"还需调。
4. **RMSE 仍是假象**,只看 ACC(同 §D)。

**由此定的下一步(2026-06-19 讨论,见下方 §sweep 计划)**:① 不再要求超 baseline,目标=多数空间任务 > plain_random **且** > vispruner;② ★2 **anchor 原始 L 阈值**(阈值设在 rank_norm 前的 L 上,砍假锚点、缺额顺延 coverage);③ **budget sweep**(用户假设 coverage 偏小:试 cov40/50);④ 工作流=**在 keep5 选设置 → 同设置验 keep10/5/3**;⑤ n=200 探索、判据=聚合平均Δ+符号检验(单任务 +1~2% 在 n=200 下测不出,要 n≥400 才声称)。范数保护若加,只作显著模块的计算方式(`s=rank_norm(imp·‖F‖)`),不单列模块。

**★ L(t) 实测分布(2026-06-19,InternVL3-8B,VSI+Ego3D 各 6 样本)**:`L=lowe_max` **无零值**(每 token 都有正的跨视角支持),右偏——中位 ~0.03–0.037、q90 ~0.08–0.10、q99 ~0.31、max ~0.7,两数据集几乎一致。**含义**:`delta_q=0.5`(中位阈)在 keep5/3 **完全不咬**(B_a/M≈ρ_a·r≈0.012–0.02,要 eligible<B_a 需 delta_q>0.98)→ ★2 阈值应改**预算相对**(锚点池=按 L 取 top κ·B_a)或绝对 L≈0.1–0.2,且这样跨 keep10/5/3 行为一致。

---

### L. ★ budget sweep + ★2(κ)→ a20s40 是首个赢过双基线的设置(2026-06-20)
> 接 §K 的"下一步"。决策:只看空间任务、目标=赢 plain_random + vispruner(不要求赢 baseline)、★2 改预算相对 κ(top κ·B_a by L,κ=2,只提质不减量;"少锚多覆盖"由 ρ 显式给)、工作流=keep5 选设置→keep10/5/3 验证。脚本 `cvsp_curve.py(+--kappa/--tag)` / `run_cvsp_{auto,stage2,stage3}.sh` / `eval_{sweep1,stage2}.py`,数据 `logs/cvsp/*-k2.jsonl`。

**Stage1(keep5 选 budget,3 任务,n200,κ=2)**:砍 ρ_a 从 0.4 都更好,anchor-heavy(a40s30)最差。并列最佳(平均 +2.2pp vs visp):a20s50(显著重)、a30s20(覆盖重)。用户取折中 **a20s40 = ρ_a0.2 / ρ_s0.4 / ρ_c0.4**。

**Stage2(a20s40,6 空间任务 × keep10/5/3,n200)= GOOD**:
| | keep10 | keep5 | keep3 | 全档≥visp |
|---|---|---|---|---|
| ≥visp 格数 | 3/6 | **5/6** | 4/6 | **12/18** |

- **vs plain:14/18(符号检验 p≈0.015,稳)**;**vs vispruner:12/18、平均 +0.67pp(p≈0.12,占优未显著)**。
- keep5 最强(调参档);稳赢点 = rel_distance、rel_dir_hard 极端档;弱点 = rel_dir_easy、Obj_AbsDist。
- **首个在多数空间任务、跨三档同时赢过 plain+visp 的设置**(此前所有 informed 全 ≈ random)。量级小、靠聚合,**vs visp 要显著需 n=400**。
- 解读:用户"anchor 给太多(0.4)、砍到 0.2 + 显著/覆盖各半"的判断成立。

**泛化验证(GOOD 分支,2026-06-20 完成)**:a20s40 在 3 个新 Ego3D 空间 MC 任务(Localization / Ego_Centric_Relative_Distance / Object_Centric_Relative_Distance)× 4 方法 × 3 档跑完。**结果给 a20s40 泼冷水**:新 3 任务 ≥visp 6/9 但 **≥plain 仅 1/9**——尤其 Localization,plain_random 全胜(random 把"压缩去噪"红利吃得比 informed 更干净,base 0.37→plain 0.49–0.51)。

**最终裁定(全 9 空间任务 × 3 档 = 27 格,绝对 ACC 表见下方附录)**:
- **vs VisPruner:18/27(p≈0.04),原+新都占优 → "跨视角选择 > 逐视角 VisPruner"成立、可复现。**
- **vs plain_random:15/27,但 = 原 6 任务 14/18(稳)+ 新 3 任务 1/9(塌)→ "赢 random"不泛化。** 原任务的"赢 plain"大概率是调参邻域过拟合;一换任务类型(Localization 这类去噪友好)就回到"informed ≈ 或 < random"。
- **结论:a20s40 能稳过 VisPruner,过不了 random。** 全会话的硬骚扰(random 极难超)依旧。三条后路:① 改主张为"赢 VisPruner"(数据扎实,但须诚实交代 random;补 n=400 + stratified_random);② 继续追赢 random(大概率死胡同);③ 分析/发现型(用户已否)。

#### 数据附录 — a20s40 终版绝对 ACC(InternVL3-8B,n=200,κ=2,ρ=0.2/0.4/0.4)

Stage1 keep5 选 budget(3 任务):
| ρ_a/ρ_s/ρ_c | rel_dir_easy | rel_dir_med | Obj_AbsDist | 平均Δvisp | 赢visp |
|---|---|---|---|---|---|
| vispruner | .510 | .420 | .405 | — | — |
| plain | .490 | .370 | .385 | — | — |
| a40s30 | .530 | .380 | .385 | −.013 | 1/3 |
| a30s20 | .530 | .475 | .395 | +.022 | 2/3 |
| a30s30 | .490 | .440 | .390 | −.005 | 1/3 |
| a20s50 | .560 | .435 | .405 | +.022 | 3/3 |

9 空间任务 × 3 档(base / plain / visp / **cvsp(a20s40)**):
| 任务 | 组 | k10 base/pl/vp/**cv** | k5 base/pl/vp/**cv** | k3 base/pl/vp/**cv** |
|---|---|---|---|---|
| rel_dir_easy | 原 | .495/.465/.535/**.500** | .495/.490/.510/**.515** | .495/.535/.530/**.505** |
| rel_dir_med | 原 | .525/.380/.445/**.465** | .525/.370/.420/**.445** | .525/.360/.435/**.390** |
| rel_dir_hard | 原 | .315/.260/.330/**.290** | .315/.260/.360/**.395** | .315/.275/.290/**.340** |
| rel_distance | 原 | .355/.355/.330/**.350** | .355/.365/.320/**.370** | .355/.370/.295/**.335** |
| Obj_AbsDist | 原 | .375/.360/.380/**.370** | .375/.385/.405/**.375** | .375/.350/.375/**.405** |
| Ego_AbsDist | 原 | .495/.430/.490/**.490** | .495/.420/.485/**.510** | .495/.500/.495/**.500** |
| Localization | 新 | .370/.490/.350/**.430** | .370/.510/.445/**.490** | .370/.490/.505/**.465** |
| Ego_RelDist | 新 | .500/.495/.470/**.465** | .500/.470/.465/**.465** | .500/.460/.450/**.455** |
| Obj_RelDist | 新 | .720/.750/.745/**.725** | .720/.750/.725/**.755** | .720/.780/.750/**.750** |

汇总:原6任务 ≥visp 12/18 ≥plain 14/18;新3任务 ≥visp 6/9 ≥plain 1/9;合计 ≥visp 18/27(+0.8pp)≥plain 15/27(+1.6pp)。

---

### M. ★ Block-CVSP（v3 §12,8×8 细块,`run_blockcvsp.sh`,2026-06-21 跑完）
> 验证 [CVSP-Method.md] §12 的块状 CVSP。方法:**block_cvsp** = 第1层锚点(ρ_a=0.2,a=cornerness×lowe,**无 dedup 被动保对**)+ 第2层 **8×8 细区(2×2-patch/区)块内显著 water-fill(每区≤1 保覆盖 + 跨视角 dedup)**。对照:plain / **stratified** / vispruner / **nuwa-lite**(复现 Nüwa stage1:CLS注意力+距离惩罚,逐视角,无merge)/ a20s40 / **block_cvsp 去锚点(ρ_a=0)**。5 空间任务 × keep10/5/3,n=200。数据 `logs/cvsp/*.block_cvsp-bc8-{a20,noanc}.jsonl`、`*.nuwa.jsonl`。
> 注:粗块 2×2 版试过即弃(每块 8×8 patch、1 token 覆盖不了,用户判定无用),改 8×8 细区。

**绝对 ACC(plain/strat/visp/nuwa/a20s40/BLK8(锚)/BLK8(去锚)):**

| keep10 | plain | strat | visp | nuwa | a20s40 | BLK8 | BLK8nA |
|---|---|---|---|---|---|---|---|
| rel_dir_hard | .260 | .305 | .330 | .260 | .290 | .270 | .260 |
| rel_dir_med | .380 | .425 | .445 | .455 | .465 | .455 | .450 |
| rel_distance | .355 | .330 | .330 | .345 | .350 | .380 | .430 |
| Localization | .490 | .490 | .350 | .465 | .430 | .475 | .460 |
| Ego_AbsDist | .430 | .485 | .490 | .485 | .490 | .505 | .520 |

| keep5 | plain | strat | visp | nuwa | a20s40 | BLK8 | BLK8nA |
|---|---|---|---|---|---|---|---|
| rel_dir_hard | .260 | .335 | .360 | .335 | .395 | .275 | .300 |
| rel_dir_med | .370 | .415 | .420 | .410 | .445 | .420 | .410 |
| rel_distance | .365 | .335 | .320 | .375 | .370 | .405 | .375 |
| Localization | .510 | .500 | .445 | .495 | .490 | .500 | .570 |
| Ego_AbsDist | .420 | .480 | .485 | .530 | .510 | .450 | .465 |

| keep3 | plain | strat | visp | nuwa | a20s40 | BLK8 | BLK8nA |
|---|---|---|---|---|---|---|---|
| rel_dir_hard | .275 | .275 | .290 | .320 | .340 | .325 | .315 |
| rel_dir_med | .360 | .335 | .435 | .415 | .390 | .425 | .405 |
| rel_distance | .370 | .355 | .295 | .315 | .335 | .350 | .380 |
| Localization | .490 | .475 | .505 | .485 | .465 | .570 | .500 |
| Ego_AbsDist | .500 | .485 | .495 | .490 | .500 | .490 | .460 |

**汇总(15 格,≥基线数 / 平均Δ):**
| vs | BLK8(锚) | BLK8(去锚) |
|---|---|---|
| plain | 11/15 +3.1pp | 13/15 +3.1pp |
| stratified | 10/15 +1.8pp | 9/15 +1.8pp |
| vispruner | 10/15 +2.0pp | 8/15 +2.0pp |
| **nuwa(竞品)** | **13/15 +0.8pp** | 8/15 +0.8pp |
| a20s40(自己) | 8/15 +0.2pp | 8/15 +0.2pp |
| **锚点消融 锚 vs 去锚** | **9/15 −0.0pp** | — |

**BLK8(锚)分档胜出**(每档5任务):vs nuwa 5/3/5(keep10/5/3)、vs a20s40 3/2/3、vs strat 3/3/4、vs plain 4/4/3。**keep5 最弱**(vs a20s40 仅 2/5——keep5 是 a20s40 主场)。

**结论**:
1. **block_cvsp 稳过所有标准基线 + 已发表竞品**:vs plain/strat/visp **+1.8~3pp**(10–13/15),vs **nuwa 13/15(+0.8pp)**。→ "跨视角块状 CVSP > 随机/VisPruner/Nüwa-lite" 可发表。
2. **但 ≈ a20s40(8/15、+0.2pp)**,没超过自己的简单版。细块的价值=**修好覆盖任务**(Localization:a20s40 keep10 输 plain −6 → BLK8 仅 −1.5、keep3 BLK8 0.570 反超 plain +8),代价=**伤集中型 anchor 任务**(rel_dir_hard keep5:BLK8 0.275 vs a20s40 0.395)。覆盖↔集中两端,净打平。
3. **★ 锚点中性**:消融 anchor vs noanc 9/15、−0.0pp。**注意**:此处 anchor 仅"被动去 dedup",**未做显式配对**——之前所有 anchor 实验都带 dedup(会拆锚点对),显式配对是 anchor 没被干净测过的最后一枪(待做)。
4. nuwa-lite 是复现核心、**无 merge**,故"赢 nuwa-lite"≠"赢完整 Nüwa"。

**待做(2026-06-21)**:① 实现**显式锚点配对**(选 A 时强制带其最佳跨视角匹配 A*)→ 跑 anchor30(ρ_a=0.3 显式配对)vs noanc vs passive 的干净消融(keep3/keep5,先 go/no-go);② 配对若救活 anchor 再扫 ρ_a∈{0.2,0.3,0.4} 找最优比例(n=400,n=200 测不出 ±0.1 差)。诚实预期:当前 anchor≈0,不配对则比例扫不出峰。
**更新(2026-06-22)**:①显式配对实测**不值得**——锚点跨视角集中度量化(`scripts/anchor_spread.py`,n=60×3 空间)显示锚点已铺满几乎所有视角(cover≈0.94–1.00、熵≈0.95)、隐式互惠配对已覆盖 80–90%(pair 0.77–0.91),显式配对只能补最后 10–20% 却让有效锚点减半。详见 [CVSP-Method.md] §12.8。

## §N · Merge 验证(2026-06-22):FAIL —— 显著点局部聚合整体伤精度

> 验证 [CVSP-Method.md] §12.9。**block_cvsp-bc8-mSal**(选择=§M 的 8×8 不变,仅对**显著 token**做 Nüwa 式视角内局部聚合 `relu(cos)×distpen`,dist_thr=11,锚点保纯)vs **无 merge**(复用 §M 的 `-bc8-a20`,选择完全相同)。5 空间任务×keep10/5/3,n=200。数据 `logs/cvsp/*.block_cvsp-bc8-mSal.jsonl`。

**ACC(无merge → merge,Δ):**

| 任务 | keep10 | keep5 | keep3 |
|---|---|---|---|
| rel_dir_hard | .270→.310 **+.040** | .275→.340 **+.065** | .325→.335 +.010 |
| rel_dir_med | .455→.420 −.035 | .420→.380 −.040 | .425→.405 −.020 |
| rel_dist | .380→.340 −.040 | .405→.325 **−.080** | .350→.355 +.005 |
| Localization | .475→.445 −.030 | .500→.455 −.045 | .570→.455 **−.115** |
| Ego_AbsD | .505→.500 −.005 | .450→.480 +.030 | .490→.495 +.005 |

**聚合:mean Δ = −0.017,胜 6 / 负 9;keep10/5/3 每档都负(−.014/−.014/−.023)。→ 未过 go/no-go(标准=mean Δ≥0 且多数不降)。merge 弃用,不进跨视角 merge,方法回到 block_cvsp 无 merge。**

**结论:**
1. merge **整体伤精度**,不是"增加单 token 信息破平局",而是抹平。
2. **任务交互(可解释,留存)**:merge **帮粗方向**(rel_dir_hard 三档全赢),**伤精确定位**(Localization keep3 −.115、rel_dist keep5 −.080、rel_dir_med 全负)。平滑利于粗判断、害于定位。
3. **根因=过平滑**:dist_thr=11 的邻近半径几乎覆盖整张视角(16×16 上 π·11²≈380>256)→ 显著 token 把全图平均进来。真"局部"需 dist_thr≈3–5,但 merge 方向已否,未再试。

---

## 待做实验 / TODO

> **⚠️ 时效(2026-06-18)**:下方"支持度算法 / DINOv2 / 锚点保护"等条目属**旧 4 组件 CVSP**(§思路1 历史草案),已被"anchor/leverage/combo + 体检 + 四方对照"取代(见 §H/I/J)。**当前唯一活跃 TODO = combo 对照(§J)**;其结果决定走"精度方法"还是"效率+发现型论文(故事 A)"。下方条目保留备查。
- 🔬 **[安全工作点实验 —— 先记录,后续做] 软冗余核心的 make-or-break**：VSI/Ego3D 上,**MLLM 自带 ViT vs DINOv2** 两档支持度,在**精度中性(安全)阈值**下各能买到多少压缩率 + 各自开销。判定"VGGT-free 的跨视角支持度够不够用":安全点压得动 → 核心成立;压不动 → 回头重想(再议要不要引轻几何)。
  - **2026-06-16 部分进展(见上方「实测发现」B)**:在 VSI relative-direction 上扫了 keep 100→25→10→5%,random 不崩、random≈vispruner → 该任务**太糙、无区分力**,不能当安全工作点探针。
  - **2026-06-17 已完成(见「实测发现」D/E/F)**:Ego3D keep10 全 6 任务跑完 + 视觉消融对照。结论:**informed 仍打不过 random**(干净 ACC 上 random 赢 3/4),根因 = 可争夺视觉盘子仅 ~9–14pp、其内高度冗余 + 大量语言先验。**安全工作点这条路对 saliency 选择已基本证伪**;若仍要做 DINOv2 支持度轴,需先回答"凭什么跨视角支持度能拨动比 saliency 更大的盘子"。
- **挑战④ make-or-break**:锚点保护(显式几何承重)vs 强多样性基线(VisPruner 自带多样性 / Geo3DPruner SDP / CDPruner)。
- 决策:支持度具体算法(哪层特征、互最近邻 vs 软聚合、阈值)。
- 决策:代表(幸存者)选择准则;剪枝 × MLLM 视角/位置结构的兼容。

## 思路库（持续扩展）
- **思路1 = VisPruner-based 跨视角支持度压缩(CVSP)**(query 无关、可预计算)——详见下方 §思路1。
- (思路2+ 留位:query 感知、覆盖式、学习极小预算器…)

---

## 思路1 · 跨视角支持度感知压缩（CVSP, Cross-View Support Pruning）⚪
> 工作名 CVSP｜从 VisPruner 出发｜training-free · post-encoder · **query 无关(可预计算复用)** · pose-free · 无 VGGT · 无硬簇 · 不用时序。最近更新 2026-06-16。
> **⚠️ 2026-06-17 故事/方法已大幅简化演进**:经实测(§实测发现 D/E/F)+讨论,CVSP 收敛为 **2 组件版(跨视角冗余折扣 + 全局覆盖保独有)**,删掉了下文的"支持度显式计算 / 锚点 / Lowe ratio / 旁证去假"。**最新 motivation + 方法以 [CVSP-Story.md] 为准**;下文这版(支持度网格/4 组件)是历史草案,保留备查。

### 故事 / 动机（= 本课题主叙事）
单视角视觉 token 压缩已收敛为两条轴:**显著性**(重要性派:FastV / VisPruner 的 CLS-attn)+ **多样性**(去冗余派:CDPruner / ToFu);VisPruner 把两者合一。但多视角 3D 理解爆发、token 暴涨,**把单视角压缩器逐视角套用,会从这两条轴双双失效**:
- **显著性逐视角独立** → ① 跨视角冗余删不掉(同物体显著 token 每视角各留一份 → 过度 token 化 / 过计数);② 跨视角互补丢失(在 B 视角不显著、却补足 A 视角的 token 被丢)。
- **多样性逐视角算** → 不认"跨视角锚点 / 帮 MLLM 建立空间理解的 token"。

仅有的两个多视角尝试各有硬伤:**Prune2Drive**(选视角 + 每视角预算,**不建模视角间关系**)、**Geo3DPruner**(用**重 VGGT 几何骨干**建关系,吃掉加速)。**CVSP 的突破口 = 一个 training-free 信号同时破两家。**

### 核心信号:跨视角支持度（驱动压缩 + 锚点加独特性门）
**支持度(t) = token t 的内容在多少个其他视角里有相似对应**(MLLM 自带 ViT / DINOv2 特征余弦)。一个标量驱动压缩,但锚点要加一道门:
- **高支持 = 冗余**(压)。
- **高支持 *且独特* = 可靠空间锚点**(被多视角共同确认的*独特*点 → 位置可信 → 保护)。⚠️ **修正(2026-06-16)**:高**表观**支持也可能来自**重复纹理**(天空/路面/地砖/重复墙面)——对压缩无害(本就该压),但**不是锚点**。故 **锚点 = 支持度 × 独特性门控**(匹配锐度 / Lowe ratio:真对应最近邻 ≫ 次近邻=峰尖;重复纹理一堆 near-equal=平),**且稀疏选**。"一信号两用"据此降级为"一信号驱动压缩 + 锚点加独特性门"。
- **低支持 = 单视角独有 = 互补信息**(留;压不掉也不该压)。

### 方法:saliency × 支持度 网格（CVSP 全貌）
| | 高支持(冗余) | 低支持(独有) |
|---|---|---|
| **显著** | 留最佳代表、删余份 | 留(显著且互补) |
| **非显著** | **候选锚点 → 仅*独特*者(峰尖)稀疏保护;通用纹理照常去重** | 丢(压缩预算主要来自这格) |

### 执行流程
| # | 环节 | 方法 | 注意 |
|---|---|---|---|
| 0 | 编码 | N 视角过 vision encoder 取 **post-encoder patch token** | ViT 2D 位置已焊进特征 → 后续"保位"靠它 |
| 1 | **显著性①** | **encoder-agnostic**:attn rollout / 特征范数(**非 CLS**) | 跨双基座统一;Qwen2.5-VL 无 CLS,先定取法 |
| 2 | **支持度②** | ViT/DINOv2 特征余弦 + 互最近邻;support = 有把握匹配的其他视角数;附 **Lowe-ratio 锐度** | **全局、独立于①、对全部 token 先算**(否则漏锚点);大 NT 用 ANN;**开销计入⑥** |
| 3 | 冗余分组 ③.2 | 高支持**软分组**(不建硬簇) | 表观相似的伪匹配对压缩无害 |
| 4 | **代表选择③** | 组内留最佳(暂:显著最高) | 准则 **(a) 待实验**;显著 ≠ 最佳几何视角 |
| 5 | **锚点保护④** | 非显著高支持中,**仅 Lowe-ratio 峰尖者稀疏保护** | 必须**稀疏 + 独特性门**;通用纹理不护 |
| 6 | 互补保留 ③.1 | 低支持(单视角独有)**直接留** | = scene completeness |
| 7 | **全局选择/预算⑤** | 全局池:留 {独有} ∪ {每组1代表} ∪ {稀疏锚点},余删;**单一阈值=安全工作点** | **每视角 ≥1 token**;预算/比例涌现 |
| 8 | **保位重排(b)→projector→LLM** | 幸存者按原视角分组 + 原 position embedding,**不重编号** | 保视角分隔;RoPE「空洞 vs 压实」消融 |

**四组件**(前两为 VisPruner 原装,后两新增,互不打架):①显著性(**encoder-agnostic,非 CLS**)②**跨视角支持度**(把图内多样性升成跨视角)③**代表选择**(冗余组留哪个幸存者)④**锚点保护**(高支持**且独特**(峰尖/Lowe ratio)的稀疏代表即便低显著也保;通用重复纹理不特殊保护)。
**冲突消解**:支持度**分组**(谁和谁冗余) → 显著性**组内选幸存者** → 锚点保护**兜底**(低显著高支持组也留一个),三者分工不撞。
**全局池化**:所有视角 token 汇成一个池统一处理 → 视角间关系进得来、独有内容多的视角自然多留 → ⑤自适应预算涌现(破 Prune2Drive 逐视角独立 + 验证集比例)。
**query 无关 → 全程可预计算、跨 query 复用** → 化解⑥"query-aware 不可缓存"代价。

### 三个 solve-point（+ 精炼）
1. **跨视角 token 建立关联**(支持度怎么算)—— 全局、独立于 saliency **先算**。
2. **显著 token 判冗余 / 互补** —— 高支持→冗余(留代表)、低支持→互补(留)。
3. **非显著 token 里哪些助空间理解** —— 高支持的救回当锚点;**必须稀疏**(绝大多数非显著要丢,捞多了就不压)。
> 精炼:**point2 与 point3 是同一台机器** —— 用支持度**双向修正 saliency 默认**(显著但冗余↓删、非显著但高支持↑救),非两套机制。

### 实现要点
- **(b) 位置结构已解(保位剪枝)**:删 token 后**不重排不重编号**,幸存 token 仍按原视角分组 + 视角内原顺序 + 原 position embedding 排好,过 projector→LLM。成立因 **ViT 2D 位置已焊进 post-encoder 特征**。
  - 小钉子:**每视角至少留 1 token**(保视角分隔、别让某视角整个消失);每视角 token 数不均是轻微分布 shift(实测确认);LLM RoPE「留空洞 vs 压实」消融各试。
- **(a) 代表选择准则暂搁**:显著 ≠ 最佳几何视角,留哪个幸存者待实验定。
- **成本(⑥)**:支持度全算 O((NT)²D),但相对它省的 LLM prefill **微不足道**(N=6 约 0.05%、视频 N=32 <0.3%);大 NT 用 ANN/互最近邻/降维。**必须计入端到端账,不假装免费。**

### 六挑战覆盖
| ① 视角 | ② 内容 | ③.1 互补 | ③.2 冗余 | ④(i) 锚点 | ④(ii) 桥 | ⑤ 自适应 | ⑥ 净加速 |
|---|---|---|---|---|---|---|---|
| ◐ 全局池化 | ✅ 显著性 | ✅ query无关 | ✅ 核心 | ✅ 核心(独特性门) | ✗ 留思路2 | ◐ 留阈值旋钮 | ✅ 强(可缓存) |

### 新颖性 / 区分 & "incremental" 防守
- vs **VisPruner 逐视角**:加跨视角支持度(它图内) + 锚点保护(它没有)。
- vs **Prune2Drive**:token 级 + 跨视角关系 + 自适应预算(它视角级 / 独立 / 验证集比例)。
- vs **Geo3DPruner**:几何-without-VGGT(支持度当锚点),training-free 真加速。
- **防"只是 VisPruner 跑 N 遍"**:核心对照 = **VisPruner-全局多样性池**(diversity 直接在跨视角全局池上算)——**不是** VisPruner-逐视角(那是稻草人,会高估 CVSP)。⚠️ **2026-06-16 反思**:"高支持→删、低支持→留"在全局池上 ≈ 全局 diversity 本身的行为 → **支持度作为压缩驱动可能只是 diversity 换皮**;CVSP 真正独有 = (a) 锚点门 + (b) 刻意代表选择(diversity 只是贪心随机留一个)。消融必须 isolate 这两项相对 global-diversity 的增量;再逐项 +支持度 / +锚点;并打三 regime 普适。
- **bonus**:按**外观**判冗余(非 3D 位置)→ 动态场景不会"同位异物错并"(几何路线 Cog3DMap/Point3R/Geo3DPruner 的硬伤)→ 对动态反而更安全。

### 关键实验 / 消融
- 🔬 **安全工作点实验**(见 TODO):MLLM 自带 ViT vs DINOv2 在安全阈值下的压缩率 + 开销(软核心 make-or-break)。
- **挑战④ make-or-break**:锚点保护 vs 强多样性基线(VisPruner 多样性 / Geo3DPruner SDP / CDPruner)。
- baseline:VisPruner-逐视角 / Prune2Drive / FastV / CDPruner;消融:+support / +anchor / 全局 vs 逐视角池化 / RoPE 两式。
- 指标:Acc + 压缩率·保留 token + **含估计开销的端到端 FLOPs/延迟/显存**;基准:VSI + Ego3D(All-Angles 后续)。

### 开放问题 / TODO
- (a) 代表(幸存者)选择准则 —— 实验定。
- **独特性门控**:用匹配锐度(Lowe ratio)区分"真同点支持" vs "重复纹理伪支持",只在峰尖处保护锚点;阈值待定(可与安全工作点一起标)。⚠️ **2026-06-16 扩展(门控不止用于锚点)**:**冗余分组那步本身**也会被"语义相似 vs 几何对应"骗——DINOv2/ViT 特征语义聚类,两把不同椅子 / 两辆不同车余弦很高 → 被判冗余删一份 = **删错实例、丢互补**(比重复纹理更严重:后者删了无害,这是真错)。故冗余"是否真冗余"也要过一道锐度/独特性门,或明确把 CVSP 的 scope 限定为"appearance 冗余"(与 §新颖性 line 动态场景 bonus 的 appearance 假设保持一致)。
- 支持度算法细节:哪层特征、互最近邻 vs 软聚合、阈值 / 安全工作点;大 NT 用 ANN/降维省算(开销计入⑥)。
- query 感知版(②细化 / ④(ii) 桥 / ③.1 query 相关)→ 思路2。

## 引用速查
VSI-Bench 2412.14171 · Ego3D-Bench 2509.06266 · All-Angles 2504.15280 · Geo3DPruner 2604.18260 · CDPruner 2506.10967 · Prune2Drive 2508.13305 · FastVID 2503.11187 · HoliTom 2505.21334 · SeGPruner 2603.29437 · Merge3D(CVPR26) · Proxy3D 2605.08064 · Point3R 2507.02863

## §O · Ego3D-Bench 全量 4 方法对比(2026-06-24)—— 两阶段 input_cos r7 vs 不压缩/plain/visp

> 7 个空间任务**全量数据**(5 MC 按 ACC、2 绝对距离按 RMSE),keep10/5。方法:FULL(不压缩)/plain_random/vispruner/**OURS=两阶段 input_cos r7**(stage1 block-cvsp 4×4 过选 N1=1.75T → stage2 LLM layer-14 query-cos 剪到 N2=0.25T)。runner: `cvsp_curve.py`(基线)+ `qstage_curve.py`(OURS);驱动 `run_ego3d_full.sh`。数据 `logs/cvsp/ego3d.*`。

| 任务 | 指标 | FULL | k10 plain/visp/**OURS** | k5 plain/visp/**OURS** |
|---|---|---|---|---|
| Ego_AbsD | RMSE↓ | 12.06 | 11.74 / 10.88 / 11.81 | 12.34 / 11.99 / **11.93** |
| Ego_AbsD_MC | ACC↑ | .523 | .485 / .493 / **.508** | **.492** / .476 / .480 |
| Ego_RelD | ACC↑ | .488 | **.479** / .464 / .463 | **.480** / .470 / .464 |
| Localization | ACC↑ | .360 | **.405** / .370 / .390 | **.423** / .421 / .412 |
| Object_AbsD | RMSE↓ | 26.68 | 17.30 / 17.53 / **13.93** | 15.94 / 14.80 / **14.25** |
| Object_AbsD_MC | ACC↑ | .501 | .479 / .459 / **.487** | **.461** / .435 / .440 |
| Object_RelD | ACC↑ | .714 | **.714** / .683 / .674 | **.702** / .675 / .644 |

**胜负(14 格):OURS 胜 plain 5/14、胜 vispruner 8/14。**

**结论:**
1. **OURS 稳过 VisPruner(8/14)**,且**绝对距离类大幅提升**(Object_AbsD RMSE 13.9 vs visp 17.5);但**整体不过 plain_random(5/14)**。
2. **任务族规律(一致)**:OURS **赢绝对距离族**(Ego/Object AbsD + AbsD_MC),**输 relative/localization 族**(Ego_RelD/Object_RelD/Localization → plain 全胜)= query 剪枝牺牲覆盖。
3. **⚠️ RMSE 假象**:压缩在 Object_AbsD 上 RMSE(14–17)远好于 FULL(26.7),是回归均值/少 clamp 的产物;"OURS 最低 RMSE"序对但非干净精度主张。
4. 与全线天花板一致:**informed 整体平/输 random;可发表的实证 = "过 VisPruner + 改善绝对距离",不是"过 random"。**

### §O 补充(2026-06-24):a20s40 全量 keep10 并入 → 五方对比

a20s40(`cvsp-a20s40-k2`,ρ_a0.2/ρ_s0.4/κ2)全量 keep10,7 任务。`run_a20s40_full.sh`(keep5 暂未跑)。

| 任务 | 指标 | FULL | plain | visp | a20s40 | OURS(2-stage) |
|---|---|---|---|---|---|---|
| Ego_AbsD | RMSE↓ | 12.061 | 11.740 | **10.883** | 11.299 | 11.809 |
| Ego_AbsD_MC | ACC↑ | .5226 | .4847 | .4934 | **.5226** | .5080 |
| Ego_RelD | ACC↑ | .4877 | **.4792** | .4642 | .4728 | .4632 |
| Localization | ACC↑ | .3597 | .4052 | .3701 | **.4091** | .3896 |
| Object_AbsD | RMSE↓ | 26.676 | 17.301 | 17.531 | 18.128 | **13.927** |
| Object_AbsD_MC | ACC↑ | .5005 | .4792 | .4589 | **.4867** | .4867 |
| Object_RelD | ACC↑ | .7101 | **.7143** | .6835 | .6765 | .6737 |

**a20s40 胜负(keep10):vs plain 4/7、vs visp 4/7、vs OURS-两阶段 5/7。**
- **a20s40(单阶段)keep10 整体 ≥ 两阶段(5/7)**;两阶段只在 Object_AbsD RMSE(假象档)大赢。两阶段 query 剪枝未带来增量。
- a20s40 覆盖型任务强:**Localization .409 连 plain 都赢**(显式 coverage 桶补两阶段最弱短板);Ego_AbsD_MC .5226=FULL(零掉分)。
- a20s40 vs 随机:≈plain(4/7)、稳过 visp(4/7)。与"a20s40 最会赢 visp"一致。
- keep5 未跑(用户暂缓)。
