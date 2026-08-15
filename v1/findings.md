# 检查发现

## it25 确定性泛化验证结果（2026-08-14）

- 新目录包含metrics、manifest和JSONL，`stop_reason=validation_only`正常结束；checkpoint SHA为it25 best的`ac210d75...85cbb`，环境seed=20263000，policy seed为空。
- 策略语义核对通过：`policy_mode/assignment_policy_mode/execution_policy_mode`均为`deterministic`，assignment mode为actor，24v4/5/6各32场、32个process worker；没有随机采样混入。
- 总体全成功率为95/96=98.96%，平均毁伤率99.83%，无效损失17.80%，成功完成时间153.35s，控制消耗0.0013515。
- 分规模结果为24v4 32/32、24v5 32/32、24v6 31/32；唯一失败来自24v6，且该规模平均毁伤仍为99.48%。
- 与选优固定集20262000的92/96相比，独立环境集反而多成功3场；分规模差异仅24v4从29/32提升到32/32，24v5保持32/32，24v6保持31/32。确定性argmax高表现跨环境块复现，不支持“只记住固定集”的解释。
- 当前证据把问题分成两部分：确定性策略的任务效果和跨环境复现性很好；随机assignment分布仍只有18/96（同20263000），因此步骤六闸门3仍失败，但“受控确定性试运行”的可用性判断应明显上调。
- 独立复算：20263000确定性成功率Wilson 95% CI为94.33%–99.82%；两个确定性环境块合并187/192=97.40%，Wilson 95% CI为94.05%–98.88%。
- 两个确定性环境块差+3.13 pp，Newcombe 95% CI -4.03~+10.04 pp、保守独立Fisher p=0.368，未见跨环境退化证据。
- 同一20263000上，确定性actor比capacity-aware高10场（+10.42 pp，Newcombe 95% CI +0.85~+19.18 pp，Fisher p=0.00493），也比随机actor高77场（+80.21 pp，p=2.30e-16）。
- 两环境块按规模合并：24v4 61/64=95.31%、24v5 64/64=100%、24v6 62/64=96.88%；高表现不是由某一个较容易的目标数规模单独支撑。
- 结论置信度为“Share with caveats”：当前仿真配置范围内的确定性受控试运行已有充分证据；但只有一个训练policy seed，且传感器噪声/位置速度扰动为0，不能外推到随机执行、强域外扰动或正式无保护部署。
- 可视化采用新增的三系列分组柱图：两个确定性环境块与20263000 capacity-aware，横轴为总体/24v4/24v5/24v6；保留成功计数、分母、毁伤率、损失率和Wilson区间作为可审计字段。原随机/确定性/capacity五系列图保持不变。

## assignment-only 随机验证结果（2026-08-14）

- 两个目录都包含 metrics、manifest、JSONL，均以 `stop_reason=validation_only` 正常结束；checkpoint SHA 均为 it25 best 的 `ac210d75...85cbb`，policy seed 均为20265001。
- 语义核对通过：`assignment_policy_mode=stochastic`、`execution_policy_mode=deterministic`、assignment actor、24v4/5/6各32场、并行32 worker；因此结果隔离的是高层随机分配，不混入低层动作采样。
- 环境 seed 20262000：全成功19/96=19.79%，平均毁伤71.79%，无效损失70.23%；分场景为24v4 15/32=46.88%、24v5 3/32=9.38%、24v6 1/32=3.13%。
- 环境 seed 20263000：全成功18/96=18.75%，平均毁伤75.24%，无效损失70.05%；分场景为24v4 12/32=37.5%、24v5 6/32=18.75%、24v6 0/32=0%。
- 两个环境集合仅差1场，说明约19%的随机策略成功率不是单一环境 seed 偶然；目标数增加时成功率系统性下降，24v6合计仅1/64成功。
- 当前确定性 it25 在20262000为92/96=95.83%，同环境随机结果低73场、差76.04 pp；接口诊断强烈支持“argmax模式优秀但策略分布过宽”为步骤六闸门3失败的主因。
- 同场景 capacity-aware 对照分别为20262000的75/96=78.13%和20263000的85/96=88.54%；随机 actor 分别低56场（-58.33 pp）和67场（-69.79 pp），不是“探索稍弱”，而是远低于启发式基线。
- it25训练 iteration 的随机 rollout 成功率为27.08%，与两次独立 assignment-only 随机验证的18.75%–19.79%处于同一低水平量级；此前训练/确定性 gap 不是评估接口把低层也随机化造成的假象。
- 当前 it25 checkpoint 尚未在环境 seed 20263000上做确定性 actor 验证；旧 v1 actor 的20263000结果不可替代当前模型。该缺口不影响“随机策略分布过宽”的结论，但影响当前模型的独立确定性泛化判断。
- 标准库复算：两组随机结果合并37/192=19.27%，Wilson 95% CI 14.32%–25.43%；20262000随机对确定性it25差-76.04 pp，Newcombe 95% CI -85.32~-60.92 pp，保守独立Fisher p=3.25e-16。
- 按目标数合并：24v4 27/64=42.19%、24v5 9/64=14.06%、24v6 1/64=1.56%。性能随目标数增大呈阶梯式崩落，说明宽分布在容量约束更紧时尤其致命。
- 两个随机环境seed相差仅1.04 pp，Fisher p≈1、差值区间跨0；环境块差异远小于随机/argmax及随机/capacity差异，主结论稳健。
- 报告将基于此前20-block完整artifact做全量修订：保留原4图/2表/所有章节，新增随机验证指标strip、分场景分策略分组柱图和统计对照表，并更新技术摘要、闸门3、局限与下一步。
- 统计结论足以判定步骤六闸门3失败：继续给当前T=1随机策略追加环境seed的边际价值很低，也不能进入步骤七。
- 最小成本下一步应为同一it25与环境seed 20262000上的assignment采样温度扫描（T=0.5、0.25）。若成功率明显恢复，再把训练后段改为entropy退火/target-entropy；若仍失败，则优先检查自回归早期错误导致的容量连锁偏移。
- `20264000`应继续保留给步骤七独立holdout；当前it25在`20263000`的确定性actor对照可作为后续argmax泛化诊断，但不是确认随机分布问题所必需。
- 完整修订报告已通过Data Analytics MCP artifact结构/数据验证并成功渲染；报告中明确记录只有一个policy sampling seed、缺逐场配对结果，因而不能做McNemar，但两个环境块的一致结果不改变主结论。

## assignment-only 随机验证接口（2026-08-14）

- 现有 `validate_checkpoint.py --stochastic` 把 assignment 与 execution 同时设为随机，且96场走串行路径，不适合隔离高层分配策略的宽分布问题。
- `train_env.py --validation-only` 已有32进程固定验证池，但 `_fixed_validation_metrics`、`evaluate_parallel_episodes` 与串行 `HierarchicalPolicyRuntime` 当前都用一个 `deterministic` 同时控制两层。
- 新接口应严格限定于 validation-only，避免随机验证误用于 checkpoint 选优；新增 `--validation-assignment-stochastic` 与必需的 `--validation-policy-seed`，语义固定为 assignment 随机、execution 确定性。
- checkpoint 恢复会覆盖命令入口最初设置的 Torch RNG，因此策略采样 seed 必须在 checkpoint restore 之后、固定验证之前重新设置并落入报告；只改变环境 seed 不足以控制随机策略复现。
- 为兼容现有调用，运行时和并行评估函数继续保留总 `deterministic` 参数，并增加可选的 assignment/execution 独立覆盖；默认值保持现有两层同态行为。
- 新随机开关只允许与 `--validation-only --validation-assignment-mode actor` 组合，且必须显式给非负 `--validation-policy-seed`；这样不会让训练期固定验证/选优意外变成随机，也不会对 capacity-aware 产生虚假的“随机”标签。
- validation-only 报告将同时记录 `policy_mode`、`assignment_policy_mode`、`execution_policy_mode` 与 `validation_policy_seed`；validation config 也持久化两层确定性布尔量，便于命令、manifest 和 metrics 交叉审计。
- 接口已实现并通过验证：4项定向测试通过；完整 `pytest -q tests` 为89 passed。两个文档命令的 Bash 语法与 CLI help 参数均通过检查，且都锁定 best checkpoint SHA256 `ac210d75...85cbb`。
- 两个诊断命令复用同一 policy seed `20265001`，分别评估环境 seed `20262000`（步骤六固定集）和 `20263000`（已有 capacity-aware 匹配集）；未使用 `20264000`，继续保留其作为步骤七全新 holdout。

## 步骤六修复后重训结果（2026-08-14）

- 用户已按评审文档重新执行步骤六，结果仍位于 `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/`；本轮必须先用时间戳、manifest、日志与 checkpoint 证明它是修复后重训，而不能沿用 2026-08-12 的失败结论。
- 控制证据为评审文档步骤六、`TEST_CLI.md` 重训命令、当前源码和本次输出目录；补充对照包括 Stage 1、capacity-aware 基线、旧 Stage 2 v1 以及修复前分析记录。
- 分析路线为 metric diagnostics → visualization → technical MCP report → validate-data；关键目标是判断四项步骤六闸门是否通过，以及是否可进入步骤七。
- 文件谱系已证实为修复后重训：周期 checkpoint 从 2026-08-12 21:44（it5）连续写到 2026-08-14 01:38（it45），latest/metrics/manifest 同期收尾；这晚于 2026-08-12 的代码修复与旧失败运行清理。
- 当前 run 产出 it5–45 共 9 个周期 checkpoint，best 的写入时间为 2026-08-13 12:12（与 it25 同时），日志 359,106 bytes、metrics 466,666 bytes；需进一步确认停止原因是调度早停而非异常中断。
- Manifest 确认本次从原始 `stage1_low_best.pt` 转换开始，Stage 2 局部计数从 0 重置，`completed_iterations=45`、`stop_reason=early_stop_validation_patience`；不是从修复前 Stage 2 checkpoint 续训。
- 运行配置与步骤六一致：high-only、48 个进程环境、rollout 64、高层成功奖励 512、学习奖励 scale 1/512、entropy coef 0.001、stickiness 1.0、potential 512、每 5 次做 96 场固定验证、plateau patience 2、early-stop patience 4。
- 调度链路正常执行：9 个验证点、4 次 best 更新、2 次 assignment LR 下调、终止前回滚到 it25 best；it45 固定验证仍有 93.75% 全成功率，说明 no-target 自锁已消失。是否达标需与 it25 best、基线和统计噪声精确比较。
- 9 个固定验证点全成功率依次为 63.54%、42.71%、86.46%、88.54%、95.83%、89.58%、92.71%、92.71%、93.75%；best=it25（92/96 场），后续虽有波动但没有复现旧 v1 从 96.88% 连续回落到 89.58% 的形态。
- `capacity_aware` 基线为同固定 seed 75/96=78.13%、独立 seed 85/96=88.54%；it25 best 分别高 17/96=17.71 pp 和 7/96=7.29 pp，均明显超过文档定义的 1 场=1.04 pp 噪声阈值。正式统计显著性和区间仍需独立复算。
- 新 run 的 it45 不是最终部署权重：终止流程先把 assignment actor/optimizer 回滚到 it25 best，再以 2.5e-5 保留 LR 写入 latest；best 文件自身仍是 it25。需分别报告“best 任务效果”和“终止时固定验证观测”，避免混淆。
- 当前 `actor.py`、`train_env.py`、`mappo.py`、`rollout.py` 的 SHA256 与 manifest 记录逐字一致，说明分析时源码仍是本次训练使用的修复后版本，未发生训练后代码漂移。
- Stage 1 独立质量门是 400 场 `many_to_one` 低层制导验证（96% 全成功），它证明低层 checkpoint 可用，但与 Stage 2 的 24v4-6 `many_to_many` 分配评估不是同一总体，不能直接拿 96% 当作 Stage 2 基线。
- capacity-aware 基线的弱点在 24v6 最明显：同 seed 75.0%，独立 seed 81.25%；新 best 的分场景结果将决定其优势是全面改善还是由某一规模贡献。
- 源码口径确认：固定验证按每场 `full_success / damage / ineffective / success_time / control_effort` 取均值；best 使用五级字典序分数，且调度在每个验证点基于是否改善推进。早停回滚仅恢复 assignment actor/Adam 并保留当前降后 LR，因此 latest 与 best 的 actor 权重应相同，但完整 checkpoint 不应逐字节相同。
- 独立复算的严格闸门结果：闸门1通过、闸门2通过、闸门3只通过“entropy 下降”而未通过“train/eval gap 收窄”、闸门4通过；因此步骤六按原文的合取标准总体仍不通过，当前不应直接进入步骤七。
- 闸门1：it10 后所有连续 4 点窗口均无显著负 Spearman；全 7 点也没有支持单调退化。闸门2：best=92/96，对同 seed capacity-aware 75/96 的差为 +17.71 pp（Newcombe 95% CI +4.55~+29.51 pp，保守独立 Fisher p=0.00040）；对独立 seed 85/96 的差为 +7.29 pp，但其区间跨 0、p=0.104，说明达到了文档的 1.04 pp 规则，却还不是独立 holdout 上的确定性证据。
- 闸门3：assignment entropy 前5次均值 20.897、后5次 18.958，下降 9.28%，Spearman ρ=-0.598；但固定验证减训练 rollout 的全成功率差，前3个验证点均值 57.29 pp、后3个 68.06 pp，反而扩大 10.76 pp，未收窄。
- 闸门4：仅 it27 出现 1 次孤立 KL stop（最大连续次数=1），此后恢复；全程不是“持续触发”。KL p95=0.00523、最大=0.01237，故按文档措辞判为通过，而不是要求绝对零次。
- PPO/冻结契约：45 行数值全有限，critic EV 前5次 0.064→后5次 0.612；execution 更新、loss、梯度全为 0，best/latest 的低层 actor/critic 与 Stage 1 位级相同。终止后 latest 的 assignment actor 与 it25 best 位级相同，assignment critic 不同，符合只回滚 actor/Adam 的设计。
- 新 best 的分规模全成功率为 24v4 90.63%、24v5 100%、24v6 96.88%；相对同 seed capacity-aware 分别 +9.38、+21.88、+21.88 pp，相对独立 seed capacity-aware 分别 +3.13、+3.13、+15.63 pp。优势不是只由单一规模贡献，但 24v4 的余量最小。
- entropy 口径是每个高层决策对 24 枚导弹自回归分类熵求和后按有效高层时间步平均，不是单枚导弹熵；后5次 18.96 nats 约等于每弹 0.79 nats。训练明确使用随机 assignment，固定验证使用确定性 argmax，因此 60–70 pp 的落差指向“分布仍宽、argmax 很好”而非物理/低层失效。
- 与旧 v1 同固定集峰值相比，新 best 92/96=95.83%，旧 it10 actor 93/96=96.88%，新方案少 1 场（-1.04 pp，区间完全重叠）；分规模上新方案 24v4 少 3 场、24v5 多 2 场、24v6 相同。它解决的是“持续退化与保护机制”，不是已经证明峰值更高。
- 趋势差异是实质改善：旧 v1 在 it10 后 5 点 Spearman ρ=-1、精确单侧 p=0.0083；新 run it10 后 7 点 ρ=+0.631、负趋势单侧 p=0.938。新 run 的 peak 略低，但不再呈统计显著单调回落，并由 LR/回滚/早停保存 it25。
- 最终判断：no-target 修复、稳定性保护、容量基线优势和冻结契约均得到验证，但步骤六闸门3是“entropy 下降且 train/eval gap 收窄”的合取条件；后半项失败，因此严格总体结论为 `FAIL`，不建议直接进入步骤七。
- 已生成并复核技术报告 canonical artifact：20 blocks、4 charts、2 tables、7 datasets、8 sources；artifact validator 两次通过，最终 renderer 仅调用一次且成功，独立 QA 对全部数据集行数、来源和图表数量断言通过。
- 下一步优先做 it25 best 的随机 actor 验证（同96场与新 seed 各一组），并补 per-missile entropy、同场景 stochastic/deterministic 配对结果和逐 trial 明细；若随机策略仍显著偏弱，再小步降低 entropy 或引入 target-entropy schedule，先做 2–3 update smoke 后重跑步骤六闸门。

## 步骤六缺陷修复（2026-08-12）

- 修复边界：`assignment_stickiness_logit_bonus` 只作用于 `current_assignment ∩ target_mask` 的物理目标槽；保留 no-target 槽 0 强制清零。
- 该修复不改变 checkpoint 参数结构、actor 权重、容量约束、自回归顺序或 PPO log-prob 口径，因此必须从原始 Stage 1 checkpoint 重新训练，但无需重训 Stage 1。
- 回归覆盖三种语义：有效物理当前目标概率上升；no-target 当前槽对输出完全无影响；已不可用的历史物理目标不获得 bonus。
- 原始 Stage 1 best 在 24v4、seed 20262000 首决策上验证：修复前初始化 no-target 自锁已消失，确定性动作计数为 `[8,4,4,4,4]`，恰好满足每个物理目标容量 4。
- 修复前的 `seed_20260810/`（约 298 MiB）与分析派生目录 `analysis/`（约 284 KiB）已移入系统回收站；Stage 2 目标根目录为空，可由步骤六命令安全新建。

## 步骤六正式训练结果分析（2026-08-12）

- 用户已执行复核文档步骤六，结果位于 `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/`。
- 本轮先按文档约束核实配置、产物完整性和验收边界，再独立复算指标；不把训练 rollout 与固定验证混用。
- 预期关键契约来自步骤四/五：`high_only`、48 env、每 5 iteration 固定验证、assignment 侧 LR/回滚/早停、成功奖励与 critic scale 修复、低熵系数、目标滞回、potential 权重 512、低层位级冻结。
- 报告受众选择为 technical：需按“技术摘要→关键证据→口径/数据→方法→局限/稳健性→下一步→开放问题”组织。
- 当前为 Codex Desktop、未识别为 Work Mode；Data Analytics artifact 的 `validate_artifact` 与 `render_artifact` 均可调用，因此单一交付面选 MCP app report，不另行并行生成 HTML。
- 步骤六文档规定 4 个闸门：it10 后连续 4 个验证点不显著负相关、best 超过 capacity-aware 且超出 96 场噪声、entropy 明显下降且 train/eval gap 收窄、KL-stop 不持续。
- 实际目录只有 seed `20260810`；周期 checkpoint 到 iteration 25，best 文件时间与 iteration 5 相同，latest/metrics/manifest 于 2026-08-12 15:23 写完；当前无训练进程，说明本次 run 已在 25 次 update 后结束，而非仍在追加。
- `stage2_high_training.jsonl` 共 67 行；需从事件类型与 run manifest 继续确认是 early stop 正常触发还是异常中断。
- 运行配置与步骤六一致：GPU 9 映射为 `cuda:0`、high-only、48 env、50 上限、5 次验证、成功奖励 512、assignment reward scale 1/512、entropy 0.001、stickiness 1.0、potential 512、assignment plateau patience 2、early-stop patience 4。
- 本次是正常 early stop：共 25 update/5 次固定验证；it15 与 it25 各降一次 assignment actor LR（1e-4→5e-5→2.5e-5）并恢复 best，it25 再执行 terminal best restore 后停止。
- 阻断性异常：it5/10/15/20/25 的 96 场 actor 固定验证全部为 `full_success=0`、`average_damage=0`、`ineffective_loss=1`，且 24v4/5/6 各 32 场均完全相同。best=it5 仅因为它是首次验证点，不代表任何有效性能。
- 因五个验证点恒为 0，步骤六的四项验收目前至少第 2 项明确失败；第 1 项不能用“非负 Spearman”形式机械通过，因为恒零代表彻底退化而非“不再回落”。
- 验证路径明确使用 `assignment_mode=actor` + `deterministic=True`；执行侧仍是冻结 actor。全零结果不是 capacity-aware 入口误选。
- 高层 actor 的滞回实现把 `assignment_stickiness_logit_bonus * encoded.current_assignment` 直接加到所有槽位 logits，包括 no-target 槽。环境重置时若 `current_assignment` 对 no-target 为 1，则首次确定性决策会给 no-target 固定 +1；选中后又持续获得 +1，形成自锁。该机制与“96 场控制消耗=0、平均毁伤=0”高度一致，是当前首要根因假设，待用观测编码和 checkpoint 前向实测闭环确认。
- 观测代码已闭环确认前半段：每枚未分配导弹在 reset 时 `current_slot=0`，并设置 `current_assignment[...,0]=1`；actor 当前实现确实对该 no-target one-hot 加 1.0。故实现违反了步骤五“给仍然有效的当前目标增加滞回”的语义——no-target 不是有效目标，却被当成可黏住的当前目标。
- best checkpoint 的 `model_config.assignment_stickiness_logit_bonus=1.0`，其内部保存的 best 验证也为全零；不是日志汇总层单独写错。
- 步骤二基线精确值：同一选优集 20262000 的 capacity-aware=75/96=78.125%，旧 actor it10=93/96=96.875%；独立 20263000 的 capacity-aware=85/96=88.542%，旧 actor=89/96=92.708%。当前 v2 best=0/96，较同集合 capacity-aware 少 75 个成功场，远超 1 场（1.04 pp）噪声。
- 旧 v1 的固定验证 it5 已有 23/96 成功、77.62% 毁伤，it10 达 93/96；当前 v2 it5 就是 0 毁伤，说明失败在“早期确定性策略可用性”层面，不能归因于旧 run 那种 it10 后缓慢目标错配。
- 代表性 24v4、seed 20262000 的首决策实测完成：五个周期 checkpoint 在 bonus=1.0 时均选择 `[no-target=24, target1..4=0]`；只把同一 actor 的配置临时改为 bonus=0.0 后，五个 checkpoint 均选择 `[no-target=8, target1..4=4]`，立即满足每目标容量 4 的覆盖。
- 该对照使用完全相同的权重和观测，只改滞回项，因此已把根因从“网络可能学坏”提升为“已验证的 no-target 滞回自锁”。概率均值不能跨两次自回归路径直接比较（无滞回路径先填满目标后，末尾被容量约束强制 no-target），但最终动作计数是直接证据。
- 回归测试为何漏检已定位：`test_assignment_hysteresis...` 把 `current_assignment[...,1]=1`，只验证“有效物理目标概率上升”，没有覆盖真实 reset 时 `current_assignment[...,0]=1` 的未分配/no-target 情形。因此 84/84 测试全绿并不能反驳该生产路径 bug。
- 分析 QA 结论口径：固定验证是 96 场确定性 actor 结果，训练趋势是 48-env 随机采样 rollout；两者必须分开。0/96、0 毁伤、0 控制消耗同时发生，并由首决策动作复现，不是平均口径、样本权重或统计显著性造成的假象。
- 步骤六最终判定：4 个闸门中 1/2/3 失败、4 通过，总体失败；不得进入步骤七，也不应从当前 v2 best/latest 续训。
- 训练随机采样侧前5→末5批：全成功 2/240→11/240、平均毁伤 36.60%→51.77%、无效损失 91.79%→86.01%；entropy 21.27→20.52，Spearman ρ=-0.556，说明有学习信号但 deterministic policy 被实现 bug 截断。
- P0/P1 中可保留的有效部分：critic 梯度中位数 1911.41→0.0889（范围 0.0225-0.3103）、25 次 KL-stop=0、最大 KL=0.00760、全 rollout 完整、execution actor/critic 位级冻结、两次 LR 下调/三次 actor restore/early stop 均按设计生效。
- 当前必须修复：stickiness bonus 只能施加到 `target_slot>0` 且仍有效的当前物理目标；需新增 reset/no-target 单元回归和 deterministic 首决策端到端回归。修复后应从 Stage 1 重新跑步骤六，不能继承本轮错误行为分布下的高层权重/optimizer。
- 可复核报告已通过 artifact validator（7 datasets、7 sources、ready）并成功渲染；最终 QA 行数为 1/7/75/25/10/4/6，19 blocks、4 charts、2 tables。


## 步骤五执行前置确认（2026-08-11）

- 用户要求执行复核文档步骤五，验收后只写步骤六命令，由用户自行正式训练。
- 上次中断后未发现本仓库 `red_swarm_policy.train_env` 残留进程；系统中其他用户的训练/评估进程不属于本任务，不作干预。
- 当前目录没有 Git 元数据，继续通过显式文件核对、测试和产物校验保护既有工作。
- 本轮沿用步骤四禁改边界：不覆盖 Stage 1、旧 Stage 2 v1 产物；步骤六命令使用新的 Stage 2 v2 输出目录。

## 步骤五文档约束与初始差距

- 5a 要把 Stage 2 正式批量从 16 提到 48-64 env，并把 rollout 观测缓存放在 CPU pinned memory、PPO minibatch 时再搬 GPU；步骤六指定 48 env。
- 5b 允许“下调 entropy coef”或 target-entropy 调度；当前 `assignment_entropy_coef=0.01`，尚无 target-entropy scheduler。
- 5c 允许“切换惩罚”或“滞回”；当前高层奖励没有切换项，实测切换率 0.53-0.63，切换会重置 seeker 与低层 GRU。
- 5d 要把 `high_potential_weight` 提到与 damage 项可比量级；当前默认 1、damage 权重 512。虽然 `gamma=1` 时整条 episode 塑形望远镜和为零，但 `lambda_high<1` 时中间稠密信号仍会影响 GAE。
- 5e 要删除 assignment actor `_distributions()` 的死计算。
- 步骤五验收固定为完整 `pytest tests` 以及 2-3 iteration `high_only` smoke；执行侧更新数/loss 必须为 0，冻结权重必须位级不变。
- 步骤六要求 `high_only`、48 env、40-50 iterations、每 5 iteration 验证并显式配置早停；正式训练由用户自行执行。

## 5a 实现设计

- 当前 rollout 在 GPU 上执行策略前向，并把每个 high/low 时间步的 observation、hidden、action、log-prob、reward 全部继续留在 GPU 列表中；48 env 会随轨迹长度持续占显存。
- 当前 `MAPPOTrainer.update(high_only)` 仍无条件计算 execution critic values/GAE，虽然该结果完全不会被高层更新使用；这会抵消 CPU 缓存收益。
- 采用两层迁移：策略推理输出在每步记录时立即 `.detach().to("cpu")`；完整 batch 组装后在 CUDA 训练时递归 `pin_memory()`。训练器按时间 slice/chunk 使用 `non_blocking=True` 搬到网络设备。
- `high_only` 只计算 assignment values/GAE；冻结 execution 分支返回形状一致的零占位，不执行低层 critic 前向。反之纯低层模式跳过 assignment critic。
- CPU-only 测试保持普通 CPU storage（不强制 pin，避免无 CUDA 驱动环境的锁页错误）；CUDA 路径新增 pinned/storage-device 断言。

## 5b-5e 实现决策

- 5b 采用直接下调而非新增未校准的 target-entropy 控制器：新训练默认 assignment entropy coefficient 从 0.01 降至 0.001；Stage 1→2 命令显式传值并允许仅在该阶段转换覆盖，普通 Stage 2 resume 禁止改变。
- 5c 采用 actor logit 滞回，不改任务奖励的五级优先关系：给仍然有效的当前目标增加可配置 `assignment_stickiness_logit_bonus`；步骤六使用 1.0（赔率乘数约 e≈2.72），容量/探测 mask 仍具有最终约束权。Stage 1→2 可覆盖，普通同阶段 resume 禁止改变。
- 5d 新训练的 `high_potential_weight` 默认从 1 提到 512，与 `high_damage_weight` 同量级；步骤六显式传 512，且只允许 Stage 1→2 转换覆盖旧 checkpoint 的值。
- 5e 已删除 `_distributions()` 以及整批 target-head 预计算；自回归分配现在只计算真正使用的逐弹 logits，并移除不再需要的 encoded activation 字段。
- 5a 第一版代码已完成：rollout 快照逐步回 CPU，CUDA batch 最终锁页；PPO 只迁移当前训练层级的 observation，`high_only` 不再执行低层 critic/诊断前向。核心文件已通过 `py_compile`。

## 步骤五回归测试

- 首轮完整测试：79 passed、4 failed；4 项均为旧测试对“batch 与网络同 device”的假设或 CUDA 上构造值，核心训练更新未失败。
- 更新 CPU-pinned storage 测试契约后，新增滞回概率、dead-head call count、high-only execution 零前向以及阶段转换三项配置落盘/继承覆盖。
- 最终源码状态完整测试：84 passed in 41.82 s。

## 步骤五 48-env smoke 验收

- 运行设备：GPU 9（RTX 3090 24 GiB，开跑前 0 MiB / 0%）；CPU 为 64 物理核，使用 48 个 process workers、每 worker 1 thread。
- 有界 smoke：`high_only`、48 env、3 iterations、24v4-6 随机场景、真实 actor/critic/PPO 路径，缩短物理时域用于工程验收；10.4 s 完成。
- 三轮均为 assignment actor/critic 各 1 次更新；execution actor/critic update、loss、preclip grad norm 全为 0；所有浮点日志有限。
- 以相同 seed 和模型初始化顺序重建初始网络后，最终 checkpoint 中 execution actor 与 execution critic 逐 tensor 位级一致；assignment actor 与 critic 均已变化。
- 新配置落盘正确：entropy=0.001、stickiness=1.0、potential=512；entropy 三轮为 37.7518→37.7352→37.2136。
- 最终 streamed smoke 产物位于 `outputs/stage2_v2_p1_batch48/step5_smoke_streamed_20260811/`；进程池已退出，无残留 worker。较早的 selective-level smoke 保留在相邻 `step5_smoke_20260811/` 作对照。
- 禁改边界复核：Stage 1 best SHA256 仍为 `29396d32...ac6a`，旧 Stage 2 v1 best 仍为 `5e805de3...c388`。
- `TEST_CLI.md` 已改写为步骤六命令：单 seed 20260810、48 env、50 iteration 上限、validation interval 5，并显式携带 P0 + P1 配置；新训/续训/首轮检查/日志四个 Bash 围栏均通过 `bash -n`。

## 步骤四执行前置确认（2026-08-11）

- 步骤二阻塞性闸门已完成：capacity-aware 两批合并全成功率 83.33%，按复核文档 `<90%` 分支直接进入步骤四。
- 当前没有 `high_only` 训练或基线验证残留进程。
- 步骤四范围固定为：4a 奖励/字典序语义，4b assignment 验证调度、best 回滚和早停，4c assignment critic reward scale；不得触碰网络结构、阶段一产物、环境物理与既有 Stage 2 输出。

## 步骤四当前实现差距

- `RewardConfig.validate_lexicographic_priority` 把成功/失败/超时三项总和全部计入 damage 与 waste 的低优先级跨度；默认 `terminal_success_reward=0`。
- `PPOConfig` 只有 `execution_reward_learning_scale`；高层 GAE 直接使用原始 `batch.rewards_high`，低层 GAE 才先做 reward scale。
- assignment advantage 后续会单独标准化，因此新增高层 reward scale 将改变 critic 的回归目标量级，而不会改变 actor 的有效归一化优势步长。
- 现有验证调度函数和 CLI 参数都命名为 execution；需检查 checkpoint/恢复/事件落盘全链路后再决定抽象为通用调度还是增加 assignment 对称通路。
- 调度调用当前不区分训练模式，任何 `policy_updated` 的固定验证都会走 execution optimizer；`high_only` 因而错误地下调/回滚冻结的低层 actor。
- best checkpoint 保存的是四个网络与四个 optimizer，已有 execution 专用恢复函数可作为 assignment 对称实现的模板；训练状态、迭代日志和 manifest 目前也只持久化 execution 调度计数/配置。
- 阶段一现有调度/恢复测试必须继续保持兼容；步骤四宜新增 assignment 对称 helper 与按 `update_mode` 路由，避免改变已验收的 low-only 行为。
- 现有 `_restore_execution_policy_from_checkpoint` 会校验 best score、恢复 actor 与 Adam state，并保留调度后 LR；assignment 通路应逐项镜像此契约。
- PPO checkpoint 反序列化使用 dataclass 关键字构造，新增带默认值的 assignment scale 可向后读取旧 checkpoint；但 stage1→stage2 时必须允许只覆盖此前未训练的高层 scale，普通同阶段 full resume 仍应禁止覆盖。
- 现有测试已覆盖低层 reward scale 的“return/raw advantage 等比例缩放、归一化 advantage 不变”；4c 可增加结构对称的高层用例。

## 步骤四实现决策

- 4a 采用语义修复而非整体放大奖励：从 damage/waste 的低优先级跨度中只剥离 `terminal_success_reward`，失败与超时惩罚仍参与跨度校验；Stage 2 新实验显式使用成功奖励 512。
- 成功奖励 512 使文档反例中“80% 全杀、20% 全空”的期望任务回报（819.2）高于“恒定杀 5/6”（426.67），同时原 512/64 damage/waste 权重在 24v4、24v5、24v6 下仍通过字典序校验。
- 4c 采用 `assignment_reward_learning_scale=1/512`，与低层已验收尺度一致；普通同阶段 full resume 禁止覆盖，只有 `low_only→high_only` 阶段转换允许显式覆盖此前未训练的高层配置。
- 4b 增加 assignment 独立 CLI/状态字段并在 `high_only` 路由至 assignment actor optimizer；execution 旧字段与 low-only 行为保持兼容。Stage 2 命令拟使用 plateau patience 2、factor 0.5、min LR 1e-5、LR reduction/early-stop 都回滚 best、early-stop patience 4。
- 新训练必须写入 `stage2_v2_*` 新目录；不复用或覆盖 `outputs/stage2_v1_high_batch16/**`。

## 步骤四实现进展

- 4a 核心语义已落地：成功奖励从 damage/waste 的低优先级跨度中剥离，失败/超时仍保留；新增 24v4-6 校验与 5/6 对 80% 全杀反例测试。
- 4c 已新增 PPO `assignment_reward_learning_scale` 并在高层 GAE 前缩放；CLI、checkpoint 向后兼容、阶段转换受控覆盖和同阶段拒绝覆盖均已接线。
- 4b 已新增 assignment scheduler、actor+Adam best restore、CLI、manifest/checkpoint/metrics 状态，并在 `high_only` 时路由 assignment、保留 execution 旧路径。
- 新增高层端到端退化测试：连续两次固定验证时应降低 assignment LR、两次回滚 best（LR reduction 与 early stop）、触发 early stop；best/latest 的 assignment actor、execution actor/critic 位级一致，低层更新数为零。
- 修改后的 4 个核心 Python 文件与 2 个测试文件均通过 `py_compile`。

## 步骤四最终验收

- 定向测试 9/9 通过；补充 critic 梯度测试确认同一 batch、零初始化高层 critic 下，`assignment_critic_grad_norm_preclip` 随 1/512 reward scale 精确缩小 512 倍，且 explained variance 有限。
- 完整测试套件最终为 83 passed in 40.74 s；Stage 2 新训与续训两个 Bash 命令块均通过 `bash -n`。
- 真实 `stage1_low_best.pt` + 既定质量门的零更新过渡预检通过：source iteration 25 → Stage 2 start 0；不恢复 RNG；不恢复旧 assignment critic；重建 latent-sum 5 分量高层 critic。
- 预检产物确认：`terminal_success_reward=512`、`assignment_reward_learning_scale=0.001953125`、assignment plateau patience/factor/min-LR 为 2/0.5/1e-5、LR reduction 与 early stop 均启用 best restore、early-stop patience 4。
- 禁改边界复核：阶段一 best SHA256 仍为 `29396d32...ac6a`；旧 Stage 2 best SHA256 仍为 `5e805de3...c388`；没有写入或覆盖旧 `outputs/stage2_v1_high_batch16/**`。
- `assignment_explained_variance` 是否维持旧训练的 0.8-0.93 属于步骤 6 重训后的经验验收项；步骤四已完成其代码、尺度回归和真实 checkpoint 迁移前置验收，不提前声称训练曲线结果。

## 步骤二基线验证初步结果（2026-08-11）

- 四组 validation-only 产物齐全：两个 seed 集合（20262000 选优集、20263000 新 holdout）× 两种分配模式（actor、capacity_aware），每组 24v4/5/6 各 32 场，共 96 场。
- 四组均加载同一 `stage2_high_best.pt`，记录 SHA256 `5e805de35fb119c8c53cc96a5b23282001662c597b84ca937ea11ebc0de4c388`；策略均为 deterministic，配置指纹均为 `dc42c2...312d6`。
- 口径复现门禁通过：actor@20262000 的全成功率为 `93/96 = 96.875%`，精确复现复核文档中的 96.88%。
- capacity-aware 全成功率：20262000 为 `75/96 = 78.125%`，20263000 为 `85/96 = 88.542%`，均低于步骤二 `<90%` 分支阈值；按预设决策闸门应判为“it10 的高层增量真实，后续直接进入步骤 4”，不是“场景过配置”。
- actor 全成功率：20262000 为 `93/96 = 96.875%`，20263000 为 `89/96 = 92.708%`；相对 capacity-aware 分别高 `18.75 pp` 与 `4.167 pp`。
- 两个 seed 合并的描述性结果：actor `182/192 = 94.792%`，capacity-aware `160/192 = 83.333%`，差 `11.458 pp`。合并仅用于稳健性描述，正式推断仍需说明两个 seed 集合、同场景配对但缺逐 episode 原始结果的限制。
- 三种目标规模上 actor 的合并全成功率均更高：24v4 为 61/64 vs 54/64，24v5 为 62/64 vs 56/64，24v6 为 59/64 vs 50/64；增量不是由单一规模驱动。
- actor 同时提高平均毁伤率并降低无效损失率；完成时间与控制消耗属于更低优先级指标，需按字典序在成功/毁伤/无效损失之后解释。
- 统计复算（仅依赖标准库）显示：20262000 上 actor-capacity 差 `+18.75 pp`，Newcombe 95% CI 约 `[+9.67,+28.24] pp`、Fisher 双侧 `p=1.10e-4`；20263000 holdout 差 `+4.17 pp`，95% CI 约 `[-4.40,+12.90] pp`、Fisher 双侧 `p=0.459`，单个 holdout 尚不足以独立证明优势显著。
- 两模式使用相同 episode seed，理论上应做配对 McNemar 检验，但产物未保存逐 episode 成败。由边际计数可知：20262000 在任何可能配对下 McNemar 双侧 `p≤2.8e-4`；20263000 在任何可能配对下 `p≥0.125`，所以“原集合强优势、单个 holdout 方向一致但未解析”的结论不依赖未知配对结构。
- capacity-aware 合并为 `160/192 = 83.33%`，Wilson 95% CI `[77.42%,87.94%]`，单侧相对 90% 阈值 `p=0.00285`；两个批次点估计也都 `<90%`。因此步骤二预注册的 `<90%` 决策分支有充分依据。
- 合并描述性计数：actor 毁伤 `950/960` 个目标、漏 10 个；capacity-aware 毁伤 `925/960`、漏 35 个。actor 无效损失 `959/4608`，capacity-aware `1329/4608`，少 370 枚（`-8.03 pp`）。这些是聚合计数，因缺逐 episode 分布不对 damage/loss 另作独立性显著检验。
- 分规模的 actor vs capacity-aware 合并结果：24v4 成功 `61/64 vs 54/64`，24v5 `62/64 vs 56/64`，24v6 `59/64 vs 50/64`；无效损失分别为 `36.00% vs 42.45%`、`19.53% vs 25.85%`、`6.90% vs 18.23%`。v6 的效率增益最大，但三种规模方向一致。
- 代码口径确认：总体指标是 192 个 episode 等权均值；成功完成时间只在全成功 episode 上均值；选优顺序是全成功率→平均毁伤率→无效损失率→成功完成时间→控制消耗。两模式唯一策略差异是高层 assignment：capacity-aware 为按固定导弹索引顺序、覆盖奖励+负载惩罚+pair quality 的贪心分配；两者共用同一低层 actor、环境与确定性验证。
- 四份 manifest 的 `source_sha256` 映射完全相同；实际 checkpoint SHA256 与四份记录一致；console 每次均启动 32 个 process worker 并写出 validation/metrics，无错误事件。
- 旧训练曲线的 actor 固定验证为 it5 `23/96`、it10 `93/96`、it15 `93/96`、it20 `91/96`、it25 `89/96`、it30 `88/96`、it35 `86/96`；capacity-aware@20262000 为 `75/96`。因此 it10 明显超过启发式，且训练确实学到了高层能力；但 it10 后持续退化的既有诊断仍成立，基线结果不能替代奖励/早停修复。
- capacity-aware 是按导弹固定索引逐个执行的贪心：即时 `assignment_pair_quality` + 首次覆盖奖励 − 负载惩罚；该 quality 只由可见性、时间余量、离轴角、能量和可用过载构成，源码明确注明“不是命中概率”。actor 则可利用完整集合/弹目关系/当前分配/GRU 与自回归联合容量状态。当前结果与“actor 学到了贪心启发式缺失的全局/时序匹配”一致，但没有逐 episode 分配轨迹，不能把该机制表述为已证实因果。
- 24v4/5 的容量 4 分别导致每场至少 8/4 枚无法分配，对应无效损失理论地板 33.33%/16.67%。合并后 actor 仅高出地板 2.67/2.86 pp，capacity-aware 高出 9.11/9.18 pp；24v6 无此地板时 actor 6.90%、capacity-aware 18.23%。这说明 actor 优势不只是“填满容量”，还体现在减少额外无效损失，且 v6 最明显。

## 产物
- `stage1_v3_batch64/A2` 含 3 个独立种子：20260703、20260713、20260723。
- 三个种子均有 `stage1_low_best.pt`、`stage1_low_latest.pt`、迭代检查点、训练日志、manifest 和 100 场固定 holdout 验证结果。
- 检查点约 29.35 MB，未发现零长度或明显截断文件；后续需核对内部 schema、模型状态和 holdout 字典序结果。
- 独立 holdout 报告判定三个种子全部通过；阶段二检查点已在 holdout 前预注册为 `A2/seed_20260713/stage1_low_best.pt`，不能用 holdout 事后重选。
- 选定种子 holdout：全目标成功率 96.00%，无效损失率 4.00%；1/2/3/4v1 成功率 86/99/99/100%。
- 选定 best 检查点 SHA256 为 `29396d32fb9ed75a4531c56888e3653aa89a1ce52d7a8fe0423d7cd18272ac6a`，与验收报告记录一致；schema 为 13，best iteration 为 25。
- 四个网络参数张量全部有限：assignment actor 849,411 参数、execution actor 668,293、assignment critic 1,030,661、execution critic 1,129,732。
- manifest 源码哈希仅 `validate_stage1_low_checkpoint.py` 和 `tests/test_training_readiness.py` 与训练时不同；所有训练核心源码哈希一致，变化来自训练后的独立 holdout/门禁测试扩展。

## 代码
- `CLI_COMMAND.md` 当前只有阶段一训练、恢复和验证命令，没有阶段二命令；分析报告中“第 4 节阶段二命令”的引用已过时。
- `train_env` 已提供 `high_only` 模式、低层到高层转换专用质量门参数、模块冻结函数和完整 checkpoint restore；仍需核对其执行顺序与 CLI 覆盖规则。
- 高层固定验证计划硬编码为 24v4/5/6、actor 分配，符合最终规模，但训练采样分布由 CLI 的 red/blue counts 决定。
- `high_only` 明确只启用 assignment actor/critic；rollout 中 assignment 随机采样而 execution actor 确定性执行，符合“冻结低层”的策略定义。
- 阶段切换使用 full restore：模型、四个优化器、RNG、累计 iteration 都会恢复；`--reset-best-on-resume` 只清空阶段 best/调度状态和阶段 policy update 计数，不重置总计数或 RNG。
- checkpoint 的 model/PPO/env 配置在 resume 时整体覆盖 CLI 构造值；阶段二专用超参必须要么已预埋在阶段一 checkpoint，要么代码需提供受控覆盖机制。
- checkpoint 已预埋高层 PPO：assignment actor/critic LR `1e-4/3e-4`、clip `0.10`、target KL `0.01`、entropy `0.01`、`lambda_high=0.95`、sequence length 32。
- 高层 actor 将随机自回归顺序随 rollout 保存，PPO evaluate 时复用；逐弹条件 log-prob 求和形成联合 log-prob，容量计数随每步采样更新，训练与推理使用同一 assignment matrix。
- 高层 PPO ratio/clip/KL/entropy 都在团队联合动作粒度计算；高层优势不做逐弹平均，符合团队标量价值与联合分配定义。
- GAE 使用每段实际持续时间相对 5s 决策周期对 gamma/lambda 做幂缩放，且 episode_active mask 排除终止后的 padding。
- **阻断问题**：stage1 checkpoint 的共享 `critic_value_head_mode=scalar` 令 `d_value_components=1`，`TargetAssignmentCritic` 和 `OverloadBiasCritic` 都读取同一维数；直接 high_only resume 会让高层 critic 只有 1 个输出，违反高层 5 维潜在团队价值分量要求。
- 修复方向：解耦 assignment critic 的 head mode；阶段一到阶段二转换时仅重建此前未训练的 assignment critic 为 5 维 latent-sum，保留 execution actor/critic 的 checkpoint 状态和低层冻结语义。
- 高层奖励为毁伤增量 512、无效损失增量 -64、终端归一化时间 -2、potential 权重 1；24v4-6 下 `validate_lexicographic_priority` 会逐场景拒绝破坏优先级的权重。
- 高层 potential 使用 gamma=1 且终态强制 0，整条 episode 上塑形项望远镜抵消，不改变总任务目标；best 仍用五级指标字典序直接选择。
- 高层观测未把目标编号作为连续特征，当前分配通过 pair 的 one-hot relation 注入；全局 critic 保留每实体 alive 状态及完整弹目关系、分配矩阵和上下文。

## 验证
- 当前目录不是 Git 工作树，无法用 Git diff/status 追踪；所有本次变更需在计划文件中逐项记录。
- 定向回归测试 5/5 通过：critic 头解耦、短时失锁高层目标槽、stage1→stage2 门禁/重置/转换。
- 完整测试套件通过：74 passed in 34.38s。
- GPU 9（RTX 3090 24 GiB）真实 Stage 2 基准：`parallel-envs=4`、`rollout-steps=64`、24v4-6 随机采样的一次完整 high_only 更新成功；实际 39 个高层步/1930 个低层步，墙钟 1154 s，峰值显存 2360 MiB，峰值利用率 26%。
- 该基准的 Stage 2 过渡元数据正确：质量门哈希匹配、Stage 1 RNG 未恢复、assignment critic 重建、best/scheduler 基线重置。
- 24v4-6 的瓶颈主要是 0.005 s 物理步进，4 个 worker 仅使用约 3 个 CPU 核；本机有 64 个物理核、约 485 GiB 可用内存，正式训练应提高环境并行度。
- 32 环境、2 个高层步的 CUDA 对照成功：49 s、峰值显存 1340 MiB；进程池、24v4-6 随机场景批量和高层 PPO 更新均正常。
- 32 环境 × 40 步完整更新成功：1475 s、峰值显存 18016 MiB；但仅 30/32 轨迹终止，2 条仍处于非终态。
- 高层决策除 5 s 周期外还会因目标毁伤/导弹失效立即触发，因此决策步数不等同固定物理时长；仍应使用 64 步覆盖事件驱动重决策较多的完整 episode。
- 32×40 已使用 18 GiB，32×64 缺少可靠显存余量；精确正式候选调整为 16 环境 × 64 步。
- 精确 16×64 CUDA 候选成功：实际 41 个高层步/2025 个低层步，16/16 episode 终止，墙钟 1369.6 s，峰值显存 9212 MiB，峰值 GPU 利用率 40%。
- 候选 checkpoint 深检通过：schema 14；高层 critic 实际输出 5 个潜在分量；全部模型张量有限；低层 actor/critic 相对 Stage 1 位级不变；高层 actor 已变化；高/低层更新数分别为 4/8 与 0/0；联合 KL 0.002201 < 0.01。

## 修复
- checkpoint schema 升至 14，同时继续支持 11/12/13；新增 assignment critic 独立 head mode。
- low_only→high_only 必须同时提供质量门和 `--reset-best-on-resume`；转换不恢复旧 RNG，重置 trainer update step。
- scalar stage1 转换时保留低层 scalar critic，重建 5 维 latent-sum 高层 critic。
- 高层观测在当前目标短时失锁且估计有效时使用预测位置/速度和置信度保留目标槽；其他不可见目标仍被 mask。

## Stage 2 阶段计数修正（2026-08-10）
- 修正前，`low_only → high_only` 转换会把 Stage 1 `completed_iterations=25` 赋给 `start_iteration`，导致新 Stage 2 日志从 26 开始。
- 修正前，`--iterations 80` 仍执行 80 个新 update，验证已经按清零后的 `completed_stage_policy_updates` 调度，但训练日志、checkpoint 总计数和场景 RNG 使用继承编号。
- `--reset-best-on-resume` 已重置 best、调度器状态和阶段策略更新数；stage transition 也已不恢复 RNG并重置 trainer 内部 update step。
- 正确契约：阶段转换将 Stage 2 的 iteration/optimizer update/policy update 清零；同阶段 resume 保持连续；Stage 1 来源计数进入显式 provenance 元数据。
- 当前目录没有 Git 元数据，无法使用 `git diff`；本轮继续逐文件审计并以测试验证。
- 周期 checkpoint 文件名和触发条件直接使用 `iteration + 1`；重置 `start_iteration` 后会自然生成 Stage 2 的 `iteration_000005.pt` 等文件。
- 同阶段恢复测试已覆盖 `completed_iterations` 从 1 继续到 2，必须保持不变。
- 来源元数据需要写入 checkpoint 的 `training_state`，否则 Stage 2 latest 再恢复后只剩当前阶段计数，Stage 1 来源会丢失。
- 实现采用 `stage_origin` 持久化来源 checkpoint、Stage 1 模式以及 iteration/optimizer/policy/stage-policy 四类计数；首次转换的 `stage_transition` 额外记录目标模式和各项重置事实。
- `completed_optimizer_updates` 与 `completed_policy_updates` 仅在真正的 `low_only → high_only` 时清零；普通 resume 仍从 checkpoint 读取。
- 新增定向测试已真实验证：Stage 2 首次 update 为 iteration 1、周期文件为 `iteration_000001.pt`，该 Stage 2 checkpoint 零步恢复后 `start_iteration=1` 且 `stage_transition=None`。
- `tests/test_training_readiness.py` 全部 22 项通过；完整测试套件 74 项全部通过。
- 真实 Stage 1 best 零步转换验证：Stage 2 `start_iteration/completed_*` 全为 0，`stage_origin` 保存来源 25/25/20/20，质量门和 critic 重建正常。
- 真实 Stage 1 best 的 assignment actor/critic optimizer state 均为空；转换时 trainer update step 清零且 assignment critic 重建。仅被冻结的 execution optimizer 状态保留，不会影响 high-only 更新。
- 修正后的首批 `(24v4,24v5,24v6)` 数量：seed 20260810 为 `(5,6,5)`，20260820 为 `(7,3,6)`，20260830 为 `(3,7,6)`。
- 若各 seed 完成 80 次 update，修正后的训练环境总数分别为：20260810 `(421,440,419)`、20260820 `(437,412,431)`、20260830 `(425,402,453)`。

## Stage 2 日志指标分析（2026-08-11）
- 用户需要理解训练期间应重点关注的实际日志字段、含义、正常趋势和异常信号。
- 分析必须区分随机训练 rollout 与每 5 次 update 的固定 24v4/5/6 验证，并遵守五级字典序目标。
- 当前 seed 20260810 已有 iteration 1–14，固定验证位于 iteration 5 和 10；正式训练仍可能继续写入日志，因此分析采用当前快照。
- 顶层 `hit_count`、`miss_distance_m`、`reward_components` 仅是 env0 最后一步，日志已经标为 deprecated；整批结果应使用 `rollout_diagnostics` 和 `episode_*` 聚合字段。
- high_only 的核心 PPO 字段为 `assignment_*`；`execution_actor/critic_*` 应保持 0，因为低层冻结。低层行为仍可通过 bias、导引头模式、ZEM 和 episode low return 监控。
- 固定验证汇总直接计算 32×3 场的 full success、damage、ineffective loss、仅成功场景完成时间和全场控制消耗；best score 为 `(success, damage, -loss, -time, -effort)` 的严格字典序。
- `no_target_ratio` 是所有有效导弹-高层决策样本中 target slot=0 的比例；`target_switch_rate` 是本次目标槽与观测中的当前槽不同的比例；二者均是训练 rollout 行为诊断，不是固定验证指标。
- PPO 默认高层 clip epsilon=0.10、target KL=0.01、max grad norm=0.5、每 actor epoch 配 2 次 critic update；实际当前日志每 update 为 4 actor/8 critic steps。
- 当前训练 rollout iteration 1–14 的 full success 全为 0、每批 16 个环境均 timeout，但确定性固定验证由 iteration 5 的 23/96 成功（23.96%）提升到 iteration 10 的 93/96（96.875%）；说明训练采样指标与部署式确定性验证必须分开看。
- iteration 10 固定验证：平均毁伤 99.41%、无效损失 19.57%、成功完成时间 160.17s、控制消耗 0.001391；24v4/5/6 成功分别为 32/32、30/32、31/32。
- 当前高层 PPO 稳定性：KL 最大 0.004754 < 0.01；clip fraction 0.0016–0.315；entropy 22.34–25.24；critic EV 从约 0 上升至 iteration 14 的 0.584。
- 当前分配行为：no-target 0.387–0.454，switch 0.588–0.633；冻结低层 bias RMS 仅 0.2606–0.2619g、饱和率始终 0。
- high_only 冻结契约在 14 次 update 全部满足：execution actor/critic update、loss 均为 0；所有 rollout `done=true`。
- assignment entropy 是每个高层联合动作样本的自回归联合熵均值，不是单枚导弹分类熵；绝对值应看趋势，不能直接套单分类最大熵阈值。
- `mean_assignment_count_by_target` 在混合 24v4/5/6 batch 上对第 5/6 槽天然被不存在该槽的环境稀释，不能直接用槽间大小判断目标偏置。
- 监控顺序确定为：固定验证五级任务指标 → 分场景指标 → 分配行为 → 高层 PPO 稳定性/critic EV → 冻结低层与 rollout 完整性。
- 经验异常规则：KL 接近/超过0.01或频繁 KL stop、clip fraction 连续高于约0.3、entropy 快速塌缩并伴随 no-target 上升、EV 长期负值、执行层出现非零更新、`done=false`，均需优先排查；单次波动不单独下结论。
- 固定验证每个蓝方规模仅32场，单场对应3.125个百分点；应结合连续验证和三个 seed，不能把一个样本的变化当成稳定趋势。

## Stage 2 结果合理性复核（2026-08-11 08:47 CST 快照）
- 当前只有 `A2_stage1_seed_20260713/seed_20260810` 一个 Stage 2 seed；训练进程仍在运行，目标 80 updates。
- 一致快照边界为日志前 69 行、最新完整 iteration 28；稳定的周期 checkpoint 已到 iteration 25，`stage2_high_best.pt` 仍来自 iteration 10。
- 训练命令与计划一致：`high_only`、CUDA 0、16 并行环境、64 个高层 rollout 步、训练场景 24v4/5/6 随机分层采样、每 5 updates 做 3×32 场固定验证。
- iteration 28 的随机训练批次仅 1/16 全目标成功、平均毁伤 57.71%、无效损失 82.29%；这再次说明训练 rollout 与确定性固定验证不可混为一谈。
- iteration 28 的高层 KL 0.002338、clip fraction 0.1272、assignment EV 0.7896，且 execution actor/critic 更新数和 loss 为 0；单看该 update，PPO 数值与冻结契约正常。
- 因训练仍在追加，所有最终统计必须显式限定在 iteration 1–28；报告不得把当前 best 或趋势表述为 80-update 最终结果。
- 五次固定验证（iteration 5/10/15/20/25）的全目标成功率为 23.96%/96.88%/96.88%/94.79%/92.71%；iteration 10 之后没有继续改善第一优先级，且 iteration 20–25 出现回落。
- iteration 15 与 iteration 10 的全目标成功率同为 96.875%，但平均毁伤率 99.323% 略低于 99.410%；按严格字典序，后续无效损失、时间、控制消耗即使更优也不得覆盖这一差异，因此 best 保留 iteration 10 符合代码契约。
- iteration 10 的分场景成功率为 24v4=100%、24v5=93.75%、24v6=96.875%；iteration 25 为 100%/96.875%/81.25%，总体回落主要来自 24v6，提示后期策略在更高目标数上退化，而不是所有规模均同步变差。
- 固定验证每个规模只有 32 场，成功率单场步长为 3.125 个百分点；iteration 10 与 15 的同总成功率、以及 20/25 的小幅差异仍需跨 seed 或更大 holdout 判断，但 24v6 从 31/32 降至 26/32 已超过单场噪声量级，属于明确监控信号。
- 训练 rollout 从 iteration 17 开始偶尔出现 1/16 成功，毁伤率总体由前 10 次约 22%–33%上升到后期约 52%–60%，无效损失从约 93%–96%降至约 82%–90%；这与策略学习方向一致，但训练采样动作具有随机性，不能替代固定验证选择。
- 5-update 分块均值进一步确认探索 rollout 的方向性改善：平均毁伤约 27.21%（1–5）→49.31%（21–25）→58.30%（26–28），无效损失约 94.43%→86.88%→84.46%；同时 no-target 仅从前5次 43.06%降至后8次 40.87%，切换率从61.18%降至55.81%。
- 448 个训练 episode 的蓝方规模分布为 24v4=133、24v5=165、24v6=150，随机采样有波动但未严重失衡；总计 7 次成功、441 次 timeout，训练动作的高探索性与固定验证确定性执行造成巨大 train/eval gap。
- 固定验证每次复用同一 `seed_start=20262000` 的96个场景，利于 checkpoint 间配对比较，但同一集合又用于反复选 best；完成80次训练将最多在同一验证集上比较16次，存在 checkpoint-selection overfitting，最终必须用未参与选优的新 seed holdout。
- 当前命令文档只规划三 seed 训练和恢复，没有 Stage 2 独立 holdout 命令；本目录也没有 Stage 2 holdout 产物，这是当前泛化结论的主要缺口。
- checkpoint 语义深检：best 与 iteration 10 的四个网络状态逐张量完全相同；Stage 1→best/iteration25 的 execution actor 和 critic 位级不变，高层 actor 已变化；iteration25 四个网络均有限，冻结与更新契约成立。
- iteration 26 出现唯一一次 KL 超阈值：0.011588 > 0.01、clip fraction 0.5118、`assignment_kl_stopped=1`；但仍记录满4次 actor update，说明阈值很可能在最后 epoch 才越过，没有回滚该高层步。iteration 27–28 恢复正常，属于单点警报而非持续发散。
- critic EV 从接近0提升到最高0.832、后期约0.64–0.80，说明价值排序能力在形成；assignment critic pre-clip grad 28/28 次都超过0.5（最高10349），但代码每次执行0.5梯度裁剪，且奖励尺度为512，因此绝对 critic loss/grad 大不能单独判为爆炸。
- 高层 reward 实际对 `mission_completion`（毁伤比例）做线性奖励，且 `terminal_success_reward=0`；因此它严格鼓励平均毁伤，却不在策略分布层面严格优化“全目标成功率第一”。反例：6目标下，始终毁伤5/6的策略获得约426.7的期望毁伤奖励，高于80%场景全毁、20%场景零毁策略的409.6，但后者全目标成功率为80%、前者为0。这与 checkpoint 选择的第一优先级存在目标口径差异，可能解释后期训练毁伤上升而固定全成功率回落。
- Wilson 95%区间：iteration 10/15 的 93/96 为[91.21%,98.93%]，iteration20的91/96为[88.38%,97.76%]，iteration25的89/96为[85.71%,96.42%]，区间明显重叠；因此总体回落是监控信号，不足以单凭96场宣称统计显著退化。
- 24v6 的 iteration10 31/32 区间[84.26%,99.45%]，iteration25 26/32区间[64.69%,91.11%]；同种子固定场景使checkpoint比较具配对性质，但日志没有逐场结果，无法做McNemar等配对检验。
- 报告可视化契约：①5个固定验证点不画折线，使用按iteration分组的成功/毁伤/无效损失柱状图；②28个训练update使用随机rollout毁伤与无效损失双折线；③28个update使用联合KL与0.01阈值双折线。三图均保留试验数、场景、PPO伴随指标等审计字段，最终在MCP技术报告中检查。
- 技术报告总体评级拟定为 `Share with caveats`：管线、冻结、checkpoint选优和数值学习证据可信；当前模型泛化与最终优先级达成尚未验证。
- 已生成SQLite兼容复算脚本 `outputs/stage2_v1_high_batch16/analysis/iteration_1_28_report_source.sql`；内存执行成功，得到 iteration_metrics=28、固定验证分场景=15、验证总览=15、训练趋势=56、PPO趋势=28、headline=1，五个固定验证汇总与JSONL原值一致（仅浮点末位舍入差异）。
# Stage 2 最终 CLI 与产物清理（2026-08-14）

- `CLI_COMMAND.md` 仍把 Stage 2 正式入口写成旧的 `stage2_v2_p0_batch16` 三种子方案；当前最终已训练产物实际位于 `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/`，必须以真实步骤六配置替换旧章节。
- `TEST_CLI.md` 保存了最终单种子 48-env 训练/续训命令，也混有 assignment-only 随机诊断命令；正式 `CLI_COMMAND.md` 应保留前者和最终确定性验证，不能把随机诊断路径误标为部署验证。
- 当前目录不是 Git 工作树；本轮继续依赖逐文件内容、hash、manifest 和语法检查记录变更，不使用 Git 回滚。
- `CLI_COMMAND.md` 旧 Stage 2 第 7–9 节配置为 16 env、三 seed、80 updates，未包含 P1 的 entropy/stickiness/high-potential 修复，不能作为最终复现命令。
- `TEST_CLI.md` 中实际最终训练配置为 seed `20260810`、48 env、最多 50 updates、P1 参数 `assignment_entropy_coef=0.001`、`assignment_stickiness_logit_bonus=1.0`、`high_potential_weight=512`，最终训练因早停实际保留 iteration 25 best。
- 最终正式验证应是 actor assignment + execution 均 deterministic；`TEST_CLI.md` 末尾两条 stochastic 命令仅用于诊断概率分布，不应列入最终部署验证命令。
- 最终训练 run 正常以 `early_stop_validation_patience` 结束于 iteration 45；验证最优 checkpoint 来自 iteration 25，SHA256 为 `ac210d751f1ae226ca6681a201eed09cc867f2f6822f7d604c5f8d9f9ac85cbb`。`stage2_high_latest.pt` SHA256 为 `01fdee83...5322`，应保留用于审计/安全续训，推理与验证只使用 best。
- env 20263000 正式 validation-only manifest 明确记录 `assignment_policy_mode=deterministic`、`execution_policy_mode=deterministic`、`validation_policy_seed=null`，结果为 `95/96 = 98.9583%`。
- 最终综合报告的数据已经物化到三份 SQLite：`report_data.sqlite`、`assignment_stochastic_report_data.sqlite`、`deterministic_generalization_report_data.sqlite`。因此可删除旧 v1/baseline/smoke 原始目录而保留最终报告证据，不需要保留所有中间 Python 分析脚本和随机验证原始目录。
- 清理完成后 Stage 2 只保留 P1 batch48 最终根；训练目录不再保留 `iteration_*.pt`，但保留 best（推理/验证）与 latest（审计/续训）。最终报告仍可从三份完整 SQLite 数据快照读取旧基线、随机诊断和确定性泛化证据。
