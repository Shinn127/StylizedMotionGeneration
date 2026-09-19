# StylizedMotionGeneration 全项目实现评估

日期：2026-09-19。评估基于当前 dirty worktree、最新 N00–N06 工件、训练 checkpoint、测试与静态检查。

## 1. 总体结论

项目已经从原型构想推进到一套**可训练、可回放、强绑定工件身份的研究系统**。数据、NEF tokenizer、masked transport、三类 operator、reference encoder、区域/时间 mask、多步采样、物理度量、渲染和分阶段训练均有实际实现和产物。它不是“只有计划和脚手架”。

但当前更接近**成熟的研究基础设施 + 尚未证明核心效果的方法候选**，还不是可投稿完成态。最可靠的科学事实是：

1. NEF 的 token/feature ownership 和硬 support 局部性成立；
2. context masked completion 可学习；full generation 仍弱；
3. style-ID 在三类 seen style 上提供小而稳定的条件似然收益；
4. current birth-death CTMC 在相同预算下没有胜过 logit field；
5. reference encoder 经监督预训练后能够影响 operator，但独立运动学 evaluator 尚未在生成动作上识别出目标 style；
6. 最终test、unseen style、跨表示operator对照、多区域不同style和感知评估尚未完成。

建议将项目当前状态定义为：

```text
工程成熟度：研究级 beta
表征主张：部分支持
局部控制主张：结构层支持，感知层待验证
style-ID 条件建模：弱正信号，两个 operator seed 一致
reference style transfer：机制恢复，效果未验收
CTMC 主贡献：当前不支持
SIGGRAPH 主线：尚未达到投稿证据门槛
```

## 2. 分系统成熟度

| 子系统 | 当前成熟度 | 已有证据 | 主要缺口 |
|---|---|---|---|
| SEED 数据/packed store | 高 | 142,220 clips、v4 store、可恢复构建、SHA/split/norm绑定、actor holdout | token store内actor轴需外部对齐；标签高度不平衡 |
| NEF-FSQ contract | 高 | 40×9、13 stream、layout hash、ownership/causality/alias/交换测试 | 仅NEF进入MTS，跨representation接口未泛化 |
| NEF-FSQ motion quality | 中 | 日常动作厘米级，局部性强，level probe支持邻近性 | root朝向、接触、极端姿态；正确物理指标下需重做完整评估 |
| Base transport | 中 | 2,000步后context四类显著胜过unigram | full-mask差于action unigram；采样jerk大、自由生成弱 |
| Logit operator | 中 | hard mask/identity/strength/多步实现；style-ID两seed有小收益 | 生成style fidelity尚未独立确认 |
| CTMC operator | 中低 | 数值稳定、概率约束和matrix-exp oracle完整 | 同预算NLL更差；未显示质量/ordinal收益 |
| Reference encoder | 中 | 监督encoder 95.2% seen-style；冻结/low-LR/auxCE均恢复reference响应 | seen-style三类；生成动作仍未被独立evaluator识别 |
| 物理/局部评估 | 中高 | bind skeleton、mirror、单位、metric version、oracle与反例 | 旧报告混有错误ref_pos；需要统一重跑主表 |
| 实验追踪/复现 | 中高 | checkpoint SHA、源码digest、协议hash、逐行manifest、stage records | 大量关键run脚本仍在gitignored outputs；工作区未提交 |
| 测试 | 高（定向）/中（全量） | 578个显式test函数，多轮MTS矩阵和真实工件测试 | 全套有外部100STYLE失败；packed prefetch慢测在当前沙箱可卡住 |
| 文档 | 中 | 计划、进度、勘误、工件说明丰富 | 历史状态与最新结果并存，入口分散，README未收口最新结论 |
| 代码维护性 | 中低 | 核心contract已模块化，错误多数有回归反例 | MTS约2.8万行；train/eval四脚本近4千行；约9500行变更未提交 |

## 3. 数据与划分

### 已完成

- SEED 全量 v4 feature store约60 GB、token store约2.5 GB；逻辑clip与物理shard解耦。
- split按take group隔离；actor holdout为52个test actor，train/test actor交集0。
- tokenizer/token store通过checkpoint SHA、normalization、feature schema、split manifest和layout绑定。
- 配对器排除same clip/take、错误split、无窗口、held-out style，并记录拒绝原因。
- 100STYLE后缀已纠正为action而非performer。
- canonical content schema v1合并 `Basic Locomotion Neutral/Styles`，去掉最明显的style标签捷径。

### 风险

- `neutral`占92%；只有 `neutral / injured leg / injured torso` 三类具备稳定的same-style/different-content证据。
- canonical映射只去掉一个显式捷径。`Basic Locomotion`内走/跑/转向、速度、接触阶段没有真正匹配。
- 当前val eligible只覆盖训练action词表的一部分；未知action被正确排除，但结果是seen-action subset。
- SEED没有unseen-style轴。`held_out_styles=[]`意味着没有zero-shot style实验。
- style和content分布仍相关；同style paired likelihood可被动作统计、目标可见token和多数类解释一部分。
- 当前operator开发协议反复用于选best、选strength、筛方法；它是开发集，不是未触碰test。

### 判断

数据工程足够支撑下一轮实验，但SEED本身不足以独立支撑“广泛跨内容/未见风格迁移”。需要：

- 固定balanced dev与一次性test；
- 明确content×style覆盖；
- 引入100STYLE或另一套真正多style×content数据做style轴；
- 对每个泛化轴分别报告上游tokenizer/transport是否看过数据。

## 4. NEF-FSQ 表征

### 已完成

- canonical 40-coordinate、9-level、13 Node/Edge stream contract稳定。
- family共享、左右side identity、stream ownership、decoder temporal RF和token layout均有测试。
- adjacent/far probe初步支持ordinal结构：历史128窗口中40/40 coordinate相邻变化更小；新版probe又分离pulse/span与tail完整性。
- part与NEF局部token交换区域外FK为0；NEF支持集更小、contact side effect历史上更低。

### 物理质量的新事实

bind skeleton/mirror修复揭示旧 `stats['ref_pos']` 是数据均值，不能作为SOMA bind骨架。错误路径可产生约30 cm系统误差，错误镜像约60 cm。因此旧绝对米制NEF physics表不能直接作为新论文表。

正确几何oracle下，特征→FK坐标实现误差约0.005–0.16 cm；tokenizer往返在四组真实pair上约4.9–25.7 cm。误差按clip在root流与局部关节旋转流之间变化；不能笼统归因root。

可视化的9段动作表明日常动作较好，盘腿、躺地、jumping jack有明显placement/pose误差。Kabsch后的pose残差变小只能说明最佳刚体对齐能解释一部分误差，不是root stream的因果证明。

### 判断

NEF作为局部离散字母表成立，作为高物理保真decoder仍有限。当前MTS使用actor-holdout的recon+delta tokenizer，不是尊重holdout划分的physical v1.1。正式物理/感知结论前，应在正确metric下比较：

- holdout v1；
- 在holdout split上从合法起点训练的physical checkpoint；
- part/flat同split对照；
- source→round-trip与round-trip→edited两层误差。

是否重训tokenizer需由通道反事实和operator上限决定，当前还不能认定它是operator主瓶颈。

## 5. Base transport

canonical base完成2,000步训练；context mask学习明显，full-mask仍弱。

第一轮pilot中，context类相对数据unigram改善约0.37–0.70 nat，full-generation只比无action unigram好约0.012，却比action-conditioned unigram差。canonical轮同样：base full-mask 2.0103，action-conditioned unigram 1.9486。

生成benchmark显示sampling本身造成很大jerk和运动分布偏移，operator只在此基础上增加一部分变化。因此当前base更适合作为局部infill上下文模型，不适合作为任意动作生成器。

实现上已具备：

- coordinate-aware stream embedding/head；
- sinusoidal time encoding；
- action condition；
-五类mask固定协议；
- token-weightedCE、固定验证、checkpoint/协议身份；
- monotonic多步filling与CRN。

继续增加base预算的价值低于先修生成分布和评估。若目标仍是局部编辑，可以降低full-generation主张，增加真实编辑mask比例和保持source anchor；若论文需要free generation，则需要单独提升base或接入trajectory/phase条件。

## 6. Style operator 结果

### Style-ID logit

canonical内容映射后，两个operator seed均在同一213行协议上优于constant：

| seed | style-ID相对constant改善 | take-group bootstrap 95% CI | 正确ID相对错误ID |
|---|---:|---|---:|
| 3407 | 0.00749 nat | [0.00469, 0.01012] | 0.01600 |
| 3408 | 0.00907 nat | [0.00613, 0.01181] | 0.01859 |

三个style与五类mask方向均为正；生成侧更换ID会改变约7.4%的token，区域外泄漏为0。这支持“模型使用style-ID”这一窄结论。

但constant已解释大部分相对base的提升，独立style evaluator在生成动作上对style-ID、constant和reference的目标准确率都只有0.25，低于1/3 chance；所有生成臂被判为同一类。style fidelity和感知效果仍未成立。

### Birth-death CTMC

- uniformization、概率质量、rate界、identity和hard mask数值实现可靠；
- 同budget的NLL在各style/mask均差于logit；
- current ordinal分析曾有边界重复计数和marginal-vs-transition定义问题。

因此CTMC可以保留为稳定ablation，不应作为主贡献。若以后重启，只能在matched style fidelity下比较物理代价/strength稳定性，并用1D Wasserstein或真实kernel transition cost。

### Reference encoder

round-1 reference encoder接近常量；梯度确实到达，问题不是断图。fresh encoder其实含style可读信息（约92% seen-style线性读出），纯NLL训练把输入依赖衰减。

seen-style监督预训练后：

- encoder balanced accuracy约95.2%，unseen-content约95.4%；
- 冻结encoder operator的reference输入效应约+0.00826 nat（143配对行），80/120生成行换reference会改变draw；
- low-LR fine-tune约+0.01751，是三种reference修复中最强；
- 单auxiliary CE也阻止坍缩，约+0.01088。

这说明reference控制链能工作，但只在三个seen labels上。其95%分类准确率很可能同时利用动作/风格统计；它不是zero-shot style descriptor的证明。独立生成style evaluator仍未确认风格被正确表现。

## 7. 局部、时序和组合控制

已实现并测试：

- Node/Edge region与graph radius；
- selected frame range；
- visible/anchor/valid/hard mask统一语义；
- locked edit区域外token逐位不变；
- steps=1/4多步采样；
- strength 0 identity与递增响应；
- decoder temporal spill probe。

修复过一个重要bug：多步生成先把待采block标为visible，导致styled draw静默使用base。回归测试已锁住。

仍未实现/证明：

- 不同区域使用不同style descriptor。当前多个region只共享同一个style输入；这不等于disjoint multi-style。
- overlap conflict规则；
- long-horizon稳定性；
- 局部编辑的感知style fidelity；
- relaxed support下root/contact协调与strict support的公平对比。

## 8. 评估与可视化

评估体系相比初版已经有质变：固定manifest、每行seed、真实wrong/random reference、三种NLL聚合、base/source/styled三对比、bind/mirror物理context、逐样本CSV/JSON、CRN和版本化协议均已实现。

仍有四个核心不足：

1. 独立style evaluator在real motion上ceiling仅0.587，injured leg recall为0.284；在生成motion上所有臂都判成injured leg。它目前不能认证也不能反驳style fidelity。
2. 无正式人类感知评分；flipbook存在，但style是时间现象，静帧不足。
3. 开发val被多次使用，最终test未锁定评估。
4. 外部方法baseline未复现；没有Motion Puzzle/MoST/STyMo等可比较结果。

应优先完成一个能作用于生成域的裁判：

- 使用base draw/edited draw做domain-aware校准，或只选采样后仍稳定的特征；
- train/val/test按take/actor隔离；
- 同时保留real-motion ceiling与generated calibration；
- 再加固定动画的盲评表。

## 9. 代码质量与工程债务

### 优点

- shape、mask、概率、checkpoint和layout契约集中；
- 数据与模型身份强校验；
- 关键错误几乎都转成回归反例；
- dry-run/preflight/预算/输出防覆盖做得好；
- 数值oracle覆盖CTMC、CRN、padding、batch partition和mirror geometry；
- stage工件与失败版本保留，研究过程可审计。

### 当前风险

- 工作区相对HEAD约9500行新增/修改，且大量关键文件untracked；丢失工作区即丢实现。
- `train_mts_operator.py`与`train_mts_transport.py`各约1192行，`evaluate_mts_operator.py`861行，编排、数据、模型重建、checkpoint、报告耦合过重。
- MTS核心约2.8万行、181个顶层class/function；作为单一研究分支已偏大。
- 关键实验脚本/analysis/manifest仍有一部分保存在gitignored `outputs/`，不适合长期复现。
- README和`mts_operator_stage_status.md`主要描述旧P0–P5；最新N04/N05b/N06结论分散在长进度日志。
- requirements依赖范围宽，运行时虽记录环境JSON，但没有精确可重建lockfile/container。
- exact resume与AMP未实现；短实验影响小，长实验风险大。
- test suite含依赖外部100STYLE布局的失败；`packed_downstream`的prefetch测试在当前沙箱CPU可运行数十秒，导致全套测试体验差。

建议在继续大型实验前做一次不改变科研行为的收口：

1. 将当前工作区分成可审查commit：data/contract、model、training/checkpoint、evaluation/physics、experiments/docs、tests。
2. 把正式analysis与run命令从`outputs/`迁入`scripts/`与`data/configs/experiments/`；outputs只存结果。
3. 抽出CLI共有的artifact loading、protocol construction、run finalization；目标是三个train/eval脚本各保留参数和流程，不再复制校验细节。
4. 建一个`CURRENT_STATUS.md`只写最新可信结论、canonical artifacts和失效清单；历史日志继续保留。
5. 将100STYLE外部布局测试改成fixture或显式integration marker；slow/prefetch/GPU测试分组，默认单元集应稳定完成。
6. 生成精确environment lock或container说明，并记录CUDA/driver。

## 10. 论文主张成熟度

| 候选主张 | 当前状态 |
|---|---|
| 结构化离散motion alphabet支持硬局部编辑 | **支持**（token/feature层） |
| decode后局部关节无off-target变化 | **部分支持**，需正确metric下扩大样本/长时 |
| global style descriptor可在任意区域工作 | **机制支持**，reference已能影响输出；style fidelity未证 |
| 跨content style transfer | **未成立**，标签/动作匹配和独立style评价不足 |
| unseen performer泛化 | 数据划分存在，**operator最终test未完成** |
| unseen style泛化 | **未实现实验** |
| CTMC优于普通adapter | **当前反证**（相同预算更差） |
| NEF优于part/flat style operator | **未实现公平operator矩阵** |
| disjoint multi-style组合 | **未实现** |
| strength提供可控感知风格 | 分布响应单调，**感知语义未证** |
| physical/contact质量优于基线 | **未成立**；base sampling仍粗糙，旧绝对指标部分失效 |

因此投稿叙事应暂时收缩为：“NEF上受硬support约束的离散概率编辑框架，以及style conditioning的初步证据”。达到SIGGRAPH主线还需要生成域style评估、人类感知、最终test、跨表示/外部baseline和多seed。

## 11. 建议后续顺序

### P0：冻结当前研究版本

- 提交当前实现与配置；迁移outputs内关键分析脚本；生成最新status/失效清单。
- 修复测试分层和100STYLE fixture问题。
- 不启动新模型训练。

### P1：建立有效的style质量裁判

- 改造独立evaluator使其在generated/base域有非退化准确率；
- 固定最终test manifest；
- 对现有style-ID、constant、reference low-LR checkpoint重评；
- 完成人工动画盲评小试。

如果现有checkpoint在可靠裁判和盲评下没有style提升，停止扩训练，回到数据/目标定义。

### P2：完善物理与decoder上限

- 正确bind/mirror下全量重跑NEF/part/flat重建与locality；
- 训练或重评actor-holdout physical tokenizer；
- 修base采样jerk/接触，分别针对context editing与full generation设目标。

### P3：核心实验

- style-ID logit与constant至少3个operator seed，并考虑≥2个base seed；
- reference low-LR至少第二seed；
- 固定协议与生成域style evaluator；
- matched style fidelity下比较content/physics/locality。

### P4：论文差异化

- 实现part/flat operator adapter，完成NEF公平对照；
- 与外部style transfer baseline比较；
- 实现不同区域不同style及冲突规则；
- 建立held-out style或改用100STYLE等数据。

CTMC只有在新的matched-fidelity物理实验出现优势时再恢复为主方法，否则保留ablation。

## 12. 最终判断

项目当前最强的资产不是CTMC，也不是已经成功的reference style transfer，而是：

- 一套结构清楚、可验证的NEF离散局部编辑表示；
- 一条可靠的数据与工件绑定链；
- 一个能快速否证想法的训练/评估框架；
- 对方法失败机制已有相当细的实证定位。

当前最大风险不是“代码完全不可用”，而是**工程进度领先于研究证据**：系统能跑很多实验，但style是否真的被人和独立指标感知，仍未回答。下一轮应该把预算集中在测量与公平对照，而不是继续增加算子复杂度。

## 13. 本次验证范围

- `git diff --check`通过；`compileall stylized_motion scripts`通过。
- 当前MTS + NEF probe定向集合：**304 passed / 2 skipped / 2 warnings，48.14秒**。skip为条件性设备/工件路径；warnings为测试中requires-grad tensor转float和一个退化四元数除零告警，均需后续清理但本轮没有断言失败。
- 完整CPU套件在`test_packed_downstream.py::test_prefetching_does_not_inflate_the_resume_position`附近耗时很长，当前工具环境内未等到全套结束。该测试单文件在前17项通过后进入慢测；未观察到断言失败。
- 最近有记录的完整套件为进度工件中的571 passed / 1 skipped / 1 failed；唯一失败是本机100STYLE目录布局。N00–N06之后新增测试需要在分层测试方案下重新形成一次完整基线。
- 本次是实现和证据审计，没有修改模型代码、训练checkpoint或原实验结论文件。
