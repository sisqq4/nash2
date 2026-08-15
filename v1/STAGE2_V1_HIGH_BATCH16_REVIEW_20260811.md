# 阶段一/阶段二训练结果与源码合理性复核

日期：2026-08-11
对象：`src/red_swarm_policy/`、`outputs/stage1_v3_batch64/`、`outputs/stage2_v1_high_batch16/`
快照边界：`stage2_high_training.jsonl` 前 93 行，最新完整 iteration 39（训练仍在写入，本报告所有阶段二统计严格限定在 iteration 1-39）
测试状态：`pytest tests` 74 passed in 38.18s

## 0. 结论

- **阶段一（低层残差）合理，可以接受，可以作为阶段二的输入。**
- **阶段二（高层分配）的工程实现是正确的，但"结果"不合理。**问题不是数值发散，而是策略在稳定地最大化一个与验收标准不同构的目标，并且已经出现统计显著的验证退化；同时缺少早停保护和基线对照。

`stage2_high_best.pt`（iteration 10，96.88%）是本次运行的合法产物，但它是训练早期偶然到达的点，不是训练收敛的结果。在补上基线对照之前，它不能作为"高层网络有效"的证据。

### 是否需要修改并重新训练

| 阶段 | 结论 | 依据 |
|---|---|---|
| **阶段一** | **不修改，不重训** | ①三种子独立 holdout 验收干净；②**低层不是瓶颈**——阶段二 it10 的 96 场验证平均毁伤率 99.41%（480 个目标只有约 3 个存活），重训低层对最终指标的改善空间接近零，阶段二丢失的 19.6% 无效损失来自分配层；③阶段二是针对某个具体 `stage1_low_best.pt` 的制导动力学学出来的，改低层会作废阶段二，顺序上必须先修高层 |
| **阶段二** | **必须修改并重训，但不要立即开始** | 目标函数与验收标准不同构且被校验函数锁死（§2.3），无任何早停保护（§2.4-1）。但改什么取决于 `capacity_aware` 基线结果（§2.5），先跑基线再动代码 |

阶段一唯一需要补的是**验证而非训练**：一次 24v4-6、固定分配下的低层单独 holdout（§5 步骤 3）。只有当基线测试后决定提高场景难度、导致低层重新成为瓶颈时，才回头考虑放开 `bias_log_std` 重训阶段一。

内容绑定：

| 对象 | SHA256 |
|---|---|
| stage1 best（阶段二输入） | `29396d32fb9ed75a4531c56888e3653aa89a1ce52d7a8fe0423d7cd18272ac6a` |
| stage2 best（= iteration 10） | `5e805de35fb119c8c53cc96a5b23282001662c597b84ca937ea11ebc0de4c388` |
| stage2 iteration_000010 | `520d34db79ebe853e2455eafa787ea71a1559a62cadf056f0cb548c762dc97ab` |

---

## 1. 阶段一：合理

### 1.1 结果站得住

独立 holdout（400 trial，seed 20271000 起连续，`many_to_one`，1/2/3/4 红 vs 1 蓝）：

| 策略 | 全成功率 | 无效损失 | 平均末端脱靶 | timeout |
|---|---:|---:|---:|---:|
| PN 基线（残差恒为 0） | 86.00% | 14.21% | 466.5 m | 53/400 |
| A2 seed_20260703 | 94.25% | 5.88% | 106.4 m | 19/400 |
| A2 seed_20260713 | **96.00%** | 4.00% | 78.5 m | 14/400 |
| A2 seed_20260723 | 94.75% | 5.38% | 151.2 m | 16/400 |

这是一次干净的消融：同一环境、同一 PN 增益 3.5、同一 `capacity_aware` 分配、同一批 seed，唯一差别是残差是否为零（基线报告 `zero_residual_all_valid: true`，检查点报告 `residual_bound_all_valid: true`）。三个独立种子一致高出基线 8.25-10.00 pp（非配对两比例检验约 5σ）。增益主要来自把 timeout 转成命中（53→14/16/19）。

### 1.2 选优泄漏可量化且很小

内部验证轨迹（400 trial，seed 20261000）：

```
it  5 10 15 20 25 | 30 35 40 45 50 55 60 65
   .865 .870 .9175 .935 .9775 | .9625 .970 .965 .960 .9375 .9625 .970 .9625
```

峰值 97.75%（it25），it30-65 平台约 96.3%，独立 holdout 96.0%。**选择性过拟合约 1.7 pp。** 早停（patience 8）在 it65 触发并回滚到 it25，`early_stop_best_restored` 行为正确。PN 基线内部 86.5% / holdout 86.0%，说明两个验证集统计上可比。

### 1.3 PPO 健康度正常

critic EV 0.05→0.75；`execution_approx_kl` 1e-5~4e-3；前 5 次为纯 critic warmup（32 critic / 0 actor）；`assignment_actor_updates` 全程 0，`low_only` 冻结契约成立。

### 1.4 两个必须写进结论的限定

**（1）阶段一只在 1-4 红 vs 1 蓝上训练和验证过**（`--red-counts 1,2,3,4 --blue-counts 1`），最终任务是 24v4-6。

这个跨尺度迁移在设计上有辩护理由：低层观测是"自身 + 同目标友弹（容量 4 上限 ≤3 个）+ 1 个目标"，维度与规模无关；`env/observation.py:279-286` 的 4 维执行上下文全部按 `max_missiles_per_target` 和全局时限做了归一化，取值域与规模无关。所以这不是实现缺陷。但它意味着**低层从未在 24v6 的几何/能量分布下被单独验证过**，应补一次 24v4-6、固定分配下的低层 holdout。

**（2）低层残差容量基本没被使用。**

`OverloadBiasActor.bias_log_std` 初始化为 -2.5，65 次迭代后 `execution_entropy` 从 -2.1618 只走到 -2.1601，即 log_std 移动 0.00085（下界 -4.0，未触界）——**探索标准差实际是冻结的**。后果：

- holdout 的 `guidance_bias_rms_g = 0.257`，**整个 400 场中 bias 范数最大值只有 0.275 g**（上限 5 g，饱和率 0）；
- `execution_clip_fraction` 全程 ≈ 0，PPO 的 clip 从未生效，等价于极小步长的普通策略梯度。

学到的是"PN + 一个近乎恒定的 0.26 g 微偏置"。它确实有效（120 s 飞行下 0.26 g 持续偏置足以产生公里级弹道整形，末端脱靶 466→78 m），但这是局部搜索的产物，不是残差策略空间被探索过的产物。三个种子的 bias RMS 差 2 倍（0.205/0.257/0.384）而成功率只差 1.75 pp，也支持这一判断。

---

## 2. 阶段二：结果不合理

### 2.1 观测事实

固定验证集（96 场，seed 20262000，`assignment_mode=actor`，7 个检查点）：

| it | 全成功率 | 平均毁伤 | 无效损失 | 24v4 | 24v5 | 24v6 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 23.96% | 0.7762 | 0.6988 | 11/32 | 7/32 | 5/32 |
| 10 | **96.88%** ← best | 0.9941 | 0.1957 | 32/32 | 30/32 | 31/32 |
| 15 | 96.88% | 0.9932 | 0.1931 | 31/32 | 30/32 | 32/32 |
| 20 | 94.79% | 0.9898 | 0.4332 | 31/32 | 30/32 | 30/32 |
| 25 | 92.71% | 0.9875 | 0.3828 | 32/32 | 31/32 | 26/32 |
| 30 | 91.67% | 0.9818 | 0.2305 | 27/32 | 32/32 | 29/32 |
| 35 | 89.58% | 0.9811 | 0.2535 | 31/32 | 30/32 | 25/32 |

同期训练 rollout 的 `episode_high_reward_mean`：**70 → 314（+349%）**，毁伤率 0.22→0.70，无效损失 0.95→0.67，`target_switch_rate` 0.63→0.55。

**策略在稳定地最大化它的奖励，同时验收指标在稳定地下降。**

### 2.2 退化是统计显著的

- 平均毁伤率在 it10 之后 6 个点**严格单调递减**：Spearman ρ = −1.000，置换检验 p = 1/720 ≈ 0.0014。
- 全成功率合并对比：it(10,15) = 186/192 = 96.88% vs it(30,35) = 174/192 = 90.63%，差 6.25 pp，**非配对两比例检验 z = 2.55，p ≈ 0.011**。这 96 场是同一批固定 seed，配对检验只会更显著。

> 修订说明：`findings.md` 中"区间重叠、不足以宣称显著退化"的结论基于 iteration ≤28 的 4 个验证点做逐点 Wilson 区间。补上 it30/it35 之后，用趋势检验和合并检验，退化已经显著。逐点区间不是趋势检验的正确工具。

### 2.3 根因：目标函数与验收标准不同构，且被校验函数锁死

高层 episode 回报的解析形式已用日志直接验证：

```
episode_high_reward_mean  vs  512·D − 64·W − 2·T
it 1:   69.97  vs   69.97   (差 0.00)
it19:  129.74  vs  129.73   (差 0.01)
it39:  314.29  vs  314.27   (差 0.02)
```

残差只有浮点舍入。说明两件事：

**(a) 势函数塑形对回报的贡献严格为 0。** 入场时刻尚无分配，`assignment_feasibility_potential` 的 coverage = 0 ⇒ Φ₀ = 0；终端强制 Φ = 0；`high_potential_gamma = 1` ⇒ 整条 episode 望远镜抵消到零。而且 `high_potential_weight = 1`，量级只有毁伤项（512）的 1/300。**高层实际是在稀疏奖励下训练的**，规范里设计的稠密塑形在数值上等于关闭。

**(b) 被优化的是 E[D]，验收用的是 P(D=1)。** `terminal_success_reward = 0`。反例（6 目标）：恒定杀 5/6 的策略拿 512×5/6 = 426.7、全成功率 0；80% 概率全杀、20% 全空的策略拿 0.8×512 = 409.6、全成功率 80%。**奖励偏好前者。** 数据完全吻合：it10→it35，平均毁伤只掉 1.3 pp（0.9941→0.9811），全成功率掉 7.3 pp——策略正在往"平均多杀、但更少全杀"漂移。

**这个缺陷被 `RewardConfig.validate_lexicographic_priority` 从结构上锁死。** 实测当前权重在 24 红下的允许范围：

```
terminal_success_reward = 0.66 : ACCEPTED
terminal_success_reward = 0.67 : REJECTED (ineffective-loss priority)
terminal_success_reward = 20   : REJECTED (damage priority)
terminal_success_reward = 512  : REJECTED (damage priority)
```

约束来自 `lower_after_waste = time_weight + control_weight + terminal_span`，要求 `high_waste_weight / red_count = 64/24 = 2.667 > 2.001 + R` ⇒ **R < 0.666**。

该校验保证的是"多杀一个目标永远比省弹省时间值"，但它把 terminal 项归入低于毁伤优先级的量，于是**从设计上排除了用终端事件表达"全目标成功优先"**。要允许 R 达到与毁伤项可比的量级，`high_damage_weight` / `high_waste_weight` 需整体放大约两个数量级，并同时给高层加 reward scale——或者修改校验语义，把"全目标成功"作为独立的最高优先级事件排除在 `terminal_span` 之外。

### 2.4 其余实现层缺陷

**（1）阶段二完全没有自适应控制。**
`--early-stop-validation-patience` 默认 0（`train_env.py:2590`）且命令行未传；LR plateau 与 restore-best 只作用于 `trainer.execution_actor_optimizer`（`train_env.py:2019, 2059`），而 `high_only` 下该网络被冻结，等于空操作。**高层没有任何早停、LR 衰减或回滚机制。** 于是 it10 之后的 update 全部在持续劣化的方向上进行。`stage2_high_best.pt` 本身是安全的（仍是 it10），但按检查点时间戳约 36 min/iteration 计算，当前 39/80，**剩余约 25 小时是在已知退化的轨迹上消耗**。

**（2）训练/部署分布严重脱节。**
`assignment_entropy` 20-24 nats / 24 弹 ≈ 每弹 0.9 nats（约 2.4 个有效选项），39 次 update 只从 23.8 降到 21.9。采样策略的 rollout 全成功率 0-25%，确定性 argmax 96.9%。PPO 优化的是采样策略的回报，与部署的 argmax 只是松散耦合——这正是"训练指标涨、验证指标跌"能同时成立的机制。

**（3）团队标量优势 × 24 智能体 × 16 episode/update。**
每个高层步只有一个共享标量 advantage 被 24 个自回归条件共用，per-agent 信用分配极弱；每次 update 只有 16 env × 1 episode，39 次 update 累计约 624 个 episode（阶段一是 64 env × 65 = 4160）。组合动作空间叠加这个批量，方差主导。
资源上完全有余量：GPU 峰值 9.2/24 GiB、利用率 0-40%，64 物理核只用了 16 个。瓶颈是 0.005 s 的 Python 物理步进而非显存。把 rollout 缓存移到 CPU pinned memory、env 并行提到 48-64，是当前最大的样本效率杠杆。

**（4）高层 critic 没有 reward scale。**
execution 侧有 `execution_reward_learning_scale = 1/512`，assignment 侧无对应项。`assignment_critic_loss` 2500-10000，`assignment_critic_grad_norm_preclip` 260-10349 而 `max_grad_norm = 0.5`，裁剪 500-20000 倍，退化成纯归一化梯度下降。EV 仍能升到 0.8-0.93 所以可用，但条件数很差。

**（5）规范要求的"保持目标分配稳定"没有任何机制兑现。**
奖励中没有切换惩罚，也没有滞回。实测 `target_switch_rate` 0.53-0.63，而每次切换会在 `env/physics.py:_reset_target_tracking` 清空导引头锁定、在 `training/rollout.py:575` 清空 execution GRU hidden。唯一的隐式约束是那个量级只有 1/300 的势函数。

> `no_target_ratio ≈ 0.40` **不宜**当作病态证据：24v4 时总容量只有 16，强制 33% 无目标；目标陆续被毁后存活容量继续缩小，该统计量被后期状态主导。

**（6）次要问题。**
- `env/scenario.py` 中 5 种 `ScenarioStyle` 只写入 `parameters` 字典，**不改变任何几何**——训练/验证/holdout 共用同一个场景族（60° 扇区 / 140-160 km / 20 km 蓝方盘），泛化结论范围需相应收窄。
- `policy/actor.py:_distributions()` 计算的 `target_logits` 被 `_capacity_constrained_actions` 完全重算并丢弃（仅用其 shape/dtype/device 与 `zeros_like`），是纯浪费的前向与激活显存。

### 2.5 最大的评估缺口：阶段二没有基线

阶段一做了 PN 基线对照，阶段二没有做任何对照。而 24 弹打 4-6 目标、单目标容量 4——**24v6 时总容量恰好等于弹数**，"覆盖所有目标"接近平凡。成功完成时间 155-162 s / 180 s 上限，说明弹道基本是直线飞满射程，战术分配的自由度很小。

**不跑这个基线，96.88% 无法证明高层网络提供了任何增量价值。** 这是目前最该优先补齐的一个数。

启发式分配算法本身已经存在并已接入 `HierarchicalPolicyRuntime`（`training/rollout.py:416 _capacity_aware_assignment`、`:451 _capacity_aware_assignment_output`、`:552`），**但目前没有任何命令行入口能在 24v4-6 下并行跑它**：

- `validate_checkpoint.py`：`_run_trial` 有 `assignment_mode` 形参（`:279`），但 argparse 未暴露、`:484` 的调用处也未传参，实际恒为 `"actor"`；且 `:483` 是串行 `for` 循环，无 worker 池。
- `validate_stage1_low_checkpoint.py`：`assignment_mode="capacity_aware"` 已写死（`:208`），但 `blue_count=1` 同样写死（`:136, :198`），且 `:496` 强制要求 red_counts 为 `[1,2,3,4]`。
- `train_env.py:_validation_plan`（`:861`）：只在 `low_only` / `low_critic_only` 分支返回 `capacity_aware`；`high_only` 分支硬编码为 24v(4,5,6) + `actor`。

按单 episode 约 20-30 min 墙钟计（`--parallel-envs 4` 基准 1154 s、16 env 迭代约 36 min），串行 96 场需 30 h 以上，不可接受。**因此基线需要一处小改动来复用已有的并行验证池**，见 §5 步骤 2。

---

## 3. 建议

### P0

1. **停止当前 run。** it39/80，验证已连续 5 次下降，best 是 it10 且已落盘，剩余约 25 小时为纯消耗。
2. **跑 `capacity_aware` 基线**，在同一 96 场验证集 + 一个未参与选优的新 seed holdout 上对照。需要一处小改动打通并行验证入口（§2.5、§5 步骤 2）。该结果决定后续是否值得继续投入，是**阻塞性决策闸门**。
3. **修目标函数**：提高 `high_damage_weight` / `high_waste_weight` 量级并放开 `terminal_success_reward`；必须同时修改 `validate_lexicographic_priority` 的语义（把"全目标成功"从 `terminal_span` 中排除），否则任何 R > 0.666 都会被拒。同时给高层加 reward scale。
4. **为高层补早停 / best-restore / LR 调度**——现有机制挂在 execution optimizer 上，`high_only` 下是空操作。

### P1

5. env 并行从 16 提到 48-64（rollout 缓存移到 CPU pinned memory），批量 ≥48 episode/update。
6. 下调 `assignment_entropy_coef` 或改用 target-entropy 调度，缩小 stochastic / argmax 落差。
7. 增加分配切换惩罚或滞回，兑现规范中的分配稳定性要求。
8. 将 `high_potential_weight` 提到与毁伤项可比的量级，否则稠密塑形形同虚设。
9. 阶段二三种子 + 独立 holdout；阶段一补一次 24v4-6 固定分配下的低层单独验证。

### P2

10. 让 `ScenarioStyle` 真正影响几何，或从配置中删除以免误导。
11. 删除 `_distributions()` 的死计算。
12. 放开低层 `bias_log_std`（当前 5 g 残差预算只用了 0.27 g）。

---

## 4. 复核方法与证据路径

- 源码：`src/red_swarm_policy/` 全量阅读，重点为 `env/{physics,guidance,seeker,adjudication,reward,observation,environment,scenario,types}.py`、`policy/{actor,critic}.py`、`training/{mappo,rollout,gae}.py`、`core/{config,masks,networks}.py`、`train_env.py`。
- 阶段一数据：`outputs/stage1_v3_batch64/A2/seed_*/stage1_low_training.jsonl`（65 个 iteration 事件）与 `holdout_100_seed_20271000/*.json`，PN 基线 `outputs/stage1_v3_batch64/pn_baseline/`。
- 阶段二数据：`outputs/stage2_v1_high_batch16/A2_stage1_seed_20260713/seed_20260810/stage2_high_training.jsonl` 前 93 行（iteration 1-39，含 7 次固定验证）。
- 回报解析式核对：逐 iteration 比较 `episode_high_reward_mean` 与 `512·rollout_diagnostics.average_damage_rate − 64·ineffective_loss_rate − 2`，残差 ≤ 0.02。
- 奖励校验边界：以 `RewardConfig` 实例在 24 红 / 6 蓝下二分 `terminal_success_reward`，确定上界 0.666。
- 统计检验：全成功率 Wilson 95% 区间、合并两比例 z 检验、平均毁伤率对 iteration 的 Spearman 秩相关（置换零分布）。
- 回归：`pytest tests` 74 passed。

已核实为正确、不构成问题的实现（避免后续重复排查）：三自由度积分与 ISA 大气/马赫阻力表、PN 指令与重力补偿的机体系分解、35 g 合成限幅、导引头 35°/60° 双视场与 0.75 s 丢锁保持、线段最近点命中判定（5 m 杀伤半径下必需）、过最近点 600 m/40 m/s 脱靶判定、目标切换时 `min_range_m` 重置、高层自回归容量约束采样与联合 log-prob（`evaluate_actions` 复用采样顺序）、GAE 按实际时长对 γ/λ 做幂缩放、`high_only` 冻结契约（execution 更新数与 loss 恒为 0，权重位级不变）、阶段转换的计数清零与 `stage_origin` 溯源。

---

## 5. 执行计划（供后续 agent 按序执行）

### 5.0 禁改边界（任何步骤都不得触碰）

违反以下任一项都会使已有验收结论作废或引入回归：

1. **不得修改网络结构。** `TargetAssignmentActor` / `OverloadBiasActor` / 两个 critic 的层次、注意力拓扑、GRU、自回归容量约束采样、5 维潜在价值分量均与规范一致且已核实正确。
2. **不得修改阶段一产物。** `outputs/stage1_v3_batch64/**` 只读。`stage1_low_best.pt`（SHA256 `29396d32…`）是阶段二的既定输入，不得替换、不得重训、不得按 holdout 事后重选种子。
3. **不得修改已核实正确的环境物理与判定逻辑**（§4 末尾清单），除非某一步骤明确要求。
4. **不得覆盖 `outputs/stage2_v1_high_batch16/**` 现有内容。** 新实验写入新目录（如 `outputs/stage2_v2_*`）。
5. **不得放宽或绕过 `validate_lexicographic_priority` 的调用**（`env/types.py:476`、`env/environment.py:197`、`train_env.py:954`）来回避报错。步骤 4 要求的是**修改该函数的语义并同步更新其单元测试**，不是删除调用或 try/except 吞掉异常。
6. **不得在没有完成步骤 2 的情况下开始步骤 4-7。** 基线结果决定改动内容。
7. 本目录**没有 Git 元数据**，无法 `git diff` 回滚。所有改动必须逐项记录到 `task_plan.md` 的变更记录表。
8. 每步结束必须 `pytest tests` 全绿（当前基线 74 passed）。

### 5.1 顺序总览

```
步骤 1  停止当前 run                                   立刻      无代码改动
步骤 2  打通并行 capacity_aware 基线入口并运行          ~0.5 天   小改动      ← 阻塞性决策闸门
步骤 3  低层 24v4-6 固定分配单独 holdout                ~2 h      小改动      可与步骤 2 并行
        ├─ 基线 ≥95% → 先做步骤 3.5（改场景难度），再进入步骤 4
        └─ 基线 <90% → 直接进入步骤 4
步骤 4  P0 代码改动（奖励语义 / 高层早停 / critic scale） ~0.5 天
步骤 5  P1 代码改动（批量 / 熵 / 切换惩罚 / 势函数权重）  ~0.5 天
步骤 6  单种子重训验证修复是否生效                       ~30 h
步骤 7  三种子 + 独立 holdout                            ~4 天
```

### 步骤 1 — 停止当前 run

**做什么**：`kill 355256 355268`（PID 以实际 `ps` 结果为准，匹配 `--training-mode high_only --seed 20260810`）。
**为什么**：it39/80，验证自 it10 起连续 5 次下降且无早停保护，剩余约 25 h 为纯消耗。
**安全性**：`stage2_high_best.pt` 已固定为 it10，不会被后续 update 覆盖。
**验收**：`ps aux | grep train_env` 无残留；`stage2_high_best.pt` 的 SHA256 仍为 `5e805de3…`。

### 步骤 2 — capacity_aware 基线（阻塞性决策闸门）

**目的**：确定高层网络相对启发式分配是否存在增量价值。在拿到这个数之前，96.88% 既不能证明高层有效、也不能证明无效。

**改动点（二选一，推荐 A）**：

- **方案 A（推荐）**：在 `train_env.py` 增加 `--validation-assignment-mode {auto,actor,capacity_aware}`（默认 `auto` 保持现状）与一个 `--validation-only` 路径（`args.iterations == 0` 时执行一次验证后退出，绕开 `train_env.py:1857` 的训练循环）。`_validation_plan`（`:861`）在显式指定时用该值覆盖返回的 `assignment_mode`。**优点**：直接复用已有的 `--validation-parallel-envs 32` 并行池与 `_summarize_validation_values` 统计口径，与训练期验证逐字段可比。
- **方案 B**：给 `validate_checkpoint.py` 加 `--assignment-mode` 并接到 `:484` 的 `_run_trial` 调用，同时把 `:483` 的串行循环换成 worker 池。**工作量更大**，且统计口径需另行对齐。

**运行**：对 `stage2_high_best.pt` 与同一 96 场固定验证集（`--validation-seed-start 20262000`、`--validation-trials-per-blue-count 32`、`--red-counts 24 --blue-counts 4,5,6`），分别以 `capacity_aware` 和 `actor` 各跑一次；再用一个**未参与选优的新 seed**（如 20263000）各跑一次。

**验收标准**：产出 4 组 `full_success_rate / average_damage_rate / ineffective_loss_rate / successful_completion_time_s / control_effort`；`actor` 模式在 20262000 上必须复现 96.88%（否则说明新入口与训练期验证口径不一致，先修口径）。

**分支判据**：

| capacity_aware 全成功率 | 判定 | 下一步 |
|---|---|---|
| ≥ 95% | 高层 RL 当前无增量价值；24v4-6、容量 4 的过配置使"覆盖所有目标"接近平凡 | 先做步骤 3.5 提高场景难度，再进入步骤 4 |
| 90-95% | 增量有限，但目标函数修复后仍可能拉开 | 进入步骤 4，并在步骤 6 后重新对比 |
| < 90% | it10 的增量真实，只是被错误目标函数带偏 | 直接进入步骤 4 |

### 步骤 3 — 低层 24v4-6 固定分配单独 holdout

**目的**：低层唯一的独立验证是 1-4v1，从未在目标规模下单独测过；同时为步骤 2 提供"低层地板线"。
**改动点**：`validate_stage1_low_checkpoint.py` 的 `blue_count=1` 写死在 `:136` 与 `:198`，`:496` 强制 red_counts 为 `[1,2,3,4]`。需要参数化 blue_count 并放宽该断言（**不要删除断言，改成"按传入的 scenario 集合校验"**）。若步骤 2 采用方案 A，此步可直接复用同一入口。
**运行**：`stage1_low_best.pt`，24v4/5/6 各 32 场，`capacity_aware` 分配，seed 20262000（与步骤 2 同集合以便逐场配对）。
**验收**：产出与步骤 2 同口径的 5 项指标；`residual_bound_all_valid`、`pn_gain_all_valid`、`capacity_all_valid`、`finite_state_all_valid` 全为 true。
**注意**：本步骤**不训练**，不得写入 `outputs/stage1_v3_batch64/**`，结果另存新目录。

### 步骤 3.5 — 仅当步骤 2 判定"场景过配置"时执行

**目的**：让分配真正具有决策空间，否则再怎么修 RL 也无从体现价值。
**候选手段**（选择并记录理由）：提高蓝方数量上限、降低 `max_missiles_per_target`、缩短 `max_guidance_time_s` 或增大 `red_cluster_radius_range_m`、加入蓝方协同规避、让 `ScenarioStyle` 真正影响几何（当前 `env/scenario.py` 中 style 只写入 `parameters` 字典，不改变任何几何）。
**约束**：任何场景改动都会使阶段一的 1-4v1 holdout 与阶段二现有验证集不可比，必须重新建立基线（回到步骤 2、3）。

### 步骤 4 — P0 代码改动

**4a. 目标函数与字典序校验语义**
- `env/types.py:RewardConfig.validate_lexicographic_priority`（`:373`）：当前 `terminal_span = success + failure + timeout` 被并入 `lower_after_damage` / `lower_after_waste`，导致 24 红下 `terminal_success_reward < 0.666`。需把"全目标成功"作为**独立的最高优先级事件**从 `terminal_span` 中剥离；或将 `high_damage_weight` / `high_waste_weight` 整体放大两个数量级。
- `env/reward.py:_finish_high`（`:546-559`）：`terminal_outcome_adjustment` 的接线本身正确，只需要一个非零权重。
- **必须同步更新** `tests/` 中覆盖字典序校验的用例，并新增一条用例锁定"策略 A（恒定杀 5/6）的回报必须低于策略 B（80% 全杀）"这一反例（§2.3）。
- **验收**：新权重下 `validate_lexicographic_priority` 在 24v4、24v5、24v6 三种规模均通过；反例用例通过。

**4b. 高层的早停 / best-restore / LR 调度**
- `train_env.py:_step_execution_validation_scheduler`（`:669`）当前只操作 `trainer.execution_actor_optimizer`（`:2019`、`:2059` 的 restore 同样只还原 execution actor），`high_only` 下全部是空操作。需要一条对称的 assignment 侧通路。
- `--early-stop-validation-patience` 默认 0（`:2590`），阶段二命令行未传。新的阶段二命令必须显式传入。
- **验收**：构造一个高层验证连续劣化的小规模用例，断言触发 LR 下调、best 回滚与早停，且 execution 侧权重位级不变。

**4c. 高层 critic 的 reward scale**
- `core/config.py:141` 有 `execution_reward_learning_scale`，`training/mappo.py:573` 使用；assignment 侧缺对应项。在 `_advantages_and_returns`（`:560` 附近，`generalized_advantage_estimation(batch.rewards_high, ...)`）对 `rewards_high` 施加同类缩放。
- **注意**：assignment advantage 已在 `:589` 做归一化，所以缩放只影响 critic 回归目标，不改变 actor 的有效步长。
- **验收**：`assignment_critic_grad_norm_preclip` 从当前 260-10349 降到与 `max_grad_norm=0.5` 同量级；`assignment_explained_variance` 不低于现状（0.8-0.93）。

### 步骤 5 — P1 代码改动

5a. **批量 16 → 48-64 env**：把 rollout 观测缓存移到 CPU pinned memory，minibatch 再搬 GPU。GPU 峰值仅 9.2/24 GiB、利用率 0-40%，64 物理核只用了 16 个，真实瓶颈是 0.005 s 的 Python 物理步进而非显存。**这是最大的样本效率杠杆。**
5b. **`assignment_entropy_coef` 下调或改 target-entropy 调度**（现状 39 次 update 熵仅 23.8→21.9，argmax 与采样策略差距 0-25% vs 96.9%）。
5c. **分配切换惩罚或滞回**（实测 `target_switch_rate` 0.53-0.63，每次切换清空导引头锁定 `env/physics.py:_reset_target_tracking` 与 execution GRU hidden `training/rollout.py:575`）。
5d. **`high_potential_weight` 提到与毁伤项可比的量级**（现状塑形对回报贡献严格为 0，见 §2.3a）。
5e. 顺带清理：删除 `policy/actor.py:_distributions()` 的死计算（结果被 `_capacity_constrained_actions` 完全重算并丢弃）。

**验收**：`pytest tests` 全绿；一次 2-3 iteration 的 smoke run 完成且 `high_only` 冻结契约仍成立（execution 更新数与 loss 恒为 0、权重位级不变）。

### 步骤 6 — 单种子重训验证

**配置**：`--training-mode high_only`，48 env，40-50 iteration，`--validation-interval 5`，显式传入早停 patience，输出到 `outputs/stage2_v2_*`。
**成本**：约 27-30 h。
**验收标准（决定是否进入步骤 7）**：
1. 固定验证全成功率**不再在早期见顶后单调回落**——it10 之后 4 个连续验证点的 Spearman ρ 不显著为负；
2. 最终 best ≥ 步骤 2 得到的 `capacity_aware` 基线，且差距超过 96 场的统计噪声（单场 1.04 pp）；
3. `assignment_entropy` 出现可见下降趋势，训练 rollout 与确定性验证的落差收窄；
4. 全程无 `assignment_kl_stopped` 持续触发。

### 步骤 7 — 三种子 + 独立 holdout

**配置**：3 个独立 seed。注意 **CPU 核是真实预算**：64 核只能是"1 seed × 48 env"或"3 seed × 16 env"二选一，不能兼得；考虑到方差问题，建议步骤 6 用大批量单种子、步骤 7 再评估。
**必须**：新增一个**未参与任何选优**的 holdout seed。当前 80 次 update 会在同一 96 场上比较 16 次，checkpoint-selection 过拟合无法排除（阶段一实测该泄漏约 1.7 pp）。
**验收**：三种子在独立 holdout 上一致超过 `capacity_aware` 基线，并按五级字典序（全毁伤→无效损失→时间→控制消耗）给出对照表，格式对齐 `STAGE1_V3_A2_INDEPENDENT_HOLDOUT_ANALYSIS_20260805.md` 第 3 节。
