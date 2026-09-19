# MTS-FSQ 训练规划准入复核

日期：2026-09-18；依据：[最新进度报告](MTS_FSQ_Code_Revision_Progress_zh.md)与当前工作区。

## 结论

**可以开始规划并准备有预算的真实数据训练。优先进入 S0 profile，随后按验收结果进入 S1/S2；无需再把项目整体退回代码重构阶段。**

本结论限定于当前 SEED、冻结 holdout tokenizer、训练词表内 action、三类可配对 style 的先导实验。它不表示已验证风格效果、unseen style、unseen action或论文创新。正式长训练仍应分阶段批准，不直接一次性启动所有operator。

## 本次独立验证

1. 重新运行 `test_mts_cli_matrix.py`、`test_mts_preflight.py`、`test_mts_checkpoint.py`、`test_mts_eval_protocol.py`：**94 passed、1 skipped，35.71秒**。这是本次选定集合，不与历史95项gate set混作同一口径；本次未重跑全量套件。
2. 对当前profile配方运行真实只读 `transport_train` 预检，读取最多2窗口：**exit 0，ready=true，required_not_passed=[]**。
3. 预检确认tokenizer/store匹配，真实action条件、val split、五种mask、每类2行共10行、20步预算均成立。证据：`outputs/training_readiness_audit_20260918/review_profile_preflight/preflight.json`。
4. 核对代码：正式transport protocol确实由val窗口构建；tokenizer文件SHA强制校验已进入transport loader；blocked不再作为ready；独立算子配方和constant descriptor对照已存在。
5. 核对冻结清单50个文件：实现文件及profile/pilot/四份独立operator配方与所核对清单一致；三份通用配置的hash不同：`mts_revision2_transport.yaml`、`mts_revision2_style.yaml`、`mts_revision2_style_smoke.yaml`。不同冻结记录也存在摘要差异，开跑应重新生成唯一run manifest，不直接沿用历史gate的总摘要。

上一轮最关键的“train窗口冒充val”“content条件断链”“预检blocked放行”等阻塞，在此次核对范围内已经解决。

## 规划时必须明确的限制

### 数据覆盖

- 当前全store有20个action标签，train词表只有17类。
- 14,230个val满窗clip中，5,680个因action不在训练词表被排除，占 **39.9%**；实际eligible pool为8,550个clip。
- 因此本轮结果必须写为 **seen-action validation subset**，不能称整个SEED验证集性能或unseen-action泛化。
- 该排除不是训练泄漏，本身不阻止profile/pilot；但应在S0期间只读核对label provenance、各split标签频次，确认是数据分布事实而非字段解析偏差。禁止将未知action强制映射到任意已知ID。
- 同style跨content的有效风格当前只有 `neutral / injured leg / injured torso`。八类标签不等于八类有效训练风格；`held_out_styles=[]`不能报告unseen-style实验。

### 上游与设备

已有tiny链路与设备工件支持工程可运行；tiny 2步结果不是生成质量证据。本次设备相关跳过项仍记录为未在本轮执行，S0必须验证目标训练设备上的实际运行。

operator依赖尚未完成的真实pilot transport，这是正常阶段依赖。operator启动前仍要用真实上游做完整dry-run与 `operator_train` 预检；不能用tiny transport的passed记录替代。尤其逐行核对operator val/eval的action都在冻结词表内，未知action处理范围须与base主实验一致。

### 复现边界

不要求为开训重写架构或先完成AMP/exact resume。使用现有fp32与新目录；保存code/config/数据/协议hash。warm-start重置optimizer，不能把它当成连续resume拼接训练曲线。

## 推荐训练顺序

| 阶段 | 配方/预算 | 验收后才进入下一阶段 |
|---|---|---|
| S0 profile | `mts_revision2_transport_profile.yaml`；20步，B8，T64，seed3407 | loss/gradient finite、无OOM、步数准确、计时/峰值显存完整，checkpoint回读通过 |
| S1 拟合检查 | 1–8个固定窗口，建议≤200步；使用已有overfit入口，独立配置/输出 | 有明确context的infill可学，无输入泄漏；不要求同action全mask多模态目标逐token拟合到零 |
| S2 base pilot | `mts_revision2_transport_pilot.yaml`；2,000步，暂定B16，10×200步 | 320行固定val、分mask指标、初始/最终数据基线对比、8–16组固定动画；有质量改善后才能作为operator上游 |
| S3 风格信号 | style-ID logit vs no-reference logit，各≤1,000步 | 同base SHA、同pair与完整validation hash、同采样预算；条件臂的收益不能仅是修补base |
| S4 方法筛查 | style-ID CTMC，再reference logit，各≤1,000步 | CTMC与简单算子分开判断；reference正确/错误对照明确；无信号不自动加算力 |

S0/profile的10行协议与S2/pilot的320行协议不可直接比较best值。S0临时模型不作为正式operator上游；S2从明确的新初始化开始，或另行标注warm-start方案。

S1是便宜诊断，不必发展成新benchmark。S2若2,000步有正面趋势但未收敛，再基于吞吐、val曲线和动画单独决定追加10k/20k步，不预先把2,000步当最终质量标准。

S3/S4只做单seed筛查；确认有信号后才规划三seed、part/NEF、FiLM、shuffled/full kernel、独立style/content evaluator、长序列和用户研究。

## 开跑前的最小准备清单

1. 使用profile/pilot和独立operator配方，避免混用发生过hash变更的通用配置。
2. 为实际拟用配方生成新的resolved config、完整protocol hash、源码digest及唯一输出目录；开跑前重跑对应stage预检。
3. 每个run固定 `seed / max_steps / B / T / device / validation规模 / 输出目录 / 墙钟上限`；S0测量后再核定后续GPU时间，不猜总时长。
4. 所有方法保存raw逐样本结果、supervised token数、训练/验证/保存分项时间。相同protocol_id还不足以证明行集一致，必须比较完整hash。
5. 准备train-only per-coordinate或action-conditional unigram基线；动画同时看速度、幅度、接触与边界，不用“低foot slide”单独宣布质量好。
6. operator加载真实pilot后，先dry-run、再预检、再2步链路检查；每个验证目标的style/action都必须能查到合法map。

停止条件：任何nonfinite、身份不匹配、协议漂移、实际步数不足或非法条件均停该run。只有质量尚低但数值与协议正常时，才依据曲线决定延长；不要边训练边改模型/数据而沿用同一个run身份。

## 本次工作边界

本次新增复核文档，运行定向测试和只读预检；未修改实现、未启动S0–S4真实学习。建议已从“继续补基础代码”转为“按上述阶段准备并执行受控实验”，性能与研究结论由后续结果决定。
