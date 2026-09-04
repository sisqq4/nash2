# 蓝方强化学习子系统

## 双方均不学习的规则基线

蓝方测试新增独立基线场景，并通过与 Rainbow 测试相同的 `BlueEscapeEnv` 适配层运行：蓝方使用 v1 现有的
`BlueEvasionController(BlueEvasionRuleMachine)`，红方使用容量约束的规则分配与
零残差比例导引（PN 系数默认 3.5）。该入口不加载双方任何 checkpoint，不创建优化器或
回放缓存，也不进行参数更新，专门作为衡量智能博弈策略增益的无学习对照组。入口独立于原有
蓝方训练、Rainbow 评估和机理消融流程，因此不会改变已有训练及测试设置。

以下命令在 1～4 枚来弹场景各运行 100 回合，并输出逐回合 CSV 和汇总 JSON；汇总配置会显式
记录 `baseline=true`、双方学习开关均为 `false`，以及双方 checkpoint 均为空：


先进入仓库的 `v1` 目录（PowerShell 使用 `cd v1`），再执行：

```bash
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rule_baseline \
  --missiles 1,2,3,4 --episodes-per-scenario 100 \
  --seed-start 20271000 --decision-interval 0.1 \
  --log-interval 1 \
  --output outputs/blue_rl/rule_baseline/holdout_100_seed_20271000
```

命令启动后会立即创建输出目录和 `blue_rule_baseline_progress.jsonl`，并在控制台输出
`baseline_start`；默认每完成一个回合输出一条 `baseline_progress`（可用
`--log-interval N` 调整频率），结束时输出 `baseline_complete`。正式汇总 JSON 和逐回合
CSV 在全部回合完成后写入。


与智能策略比较时应使用相同的来弹数量、每场景回合数和连续 seed 起点，以确保初始化样本配对；
基线结果分别写入 `blue_rule_baseline_summary.json` 和
`blue_rule_baseline_trials.csv`。

## 旧 checkpoint 的仅测试机理塑形与消融

`evaluate_blue_rl` 可在完整 29 维 C51 动作价值上独立启用四类物理评分：
`--mechanism-threat`（短时威胁缓解）、`--mechanism-timing`（带迟滞的 P0/P1/P2
时机）、`--mechanism-direction`（LOS 法向逃逸方向）和 `--mechanism-overload`
（随威胁及阶段变化的过载）。`--mechanism-weight` 调整每项归一化评分的融合权重，默认
为 `0.35`。

所有开关仍默认关闭，只用于复现实装四项训练奖励之前的旧 checkpoint 消融。新
`normalized_v4` checkpoint 已经在训练奖励和观测中内生化四项机理，正常评估不应再打开这些
动作整形开关，以免重复施加偏好。无开关为策略本体基线，评估 JSON 记录启用项、
网络原始/最终动作、阶段、威胁和介入率；启用任一机理时，平台结构过载、速度和高度物理
包线组成唯一硬动作掩码，其他机理偏好均为软评分。

当前机理实现还包括：单弹威胁与 LOS 角速率、多弹角域覆盖/到达时序同步/安全走廊压缩、
带持续确认的主威胁切换、按仿真时间驻留的三阶段状态机、LOS 旋转法平面候选方向、随平台
过载/速度余度变化的目标过载，以及速度、高度和结构过载硬约束。可用
`--mechanism-{threat,timing,direction,overload}-weight` 分别调节相对权重；
`--mechanism-detail-log` 会额外保存每个决策周期的 29 维原始/融合 Q 值和各评分通道，适合
复盘介入原因，但会显著增加 JSON 文件大小。

消融实验可用批处理入口一次顺序执行，并在输出根目录写入可恢复审计的
`ablation_manifest.json`。`core` 套件包含 Rainbow-only、四个单项、威胁+时机、
威胁+时机+方向和完整融合共 8 组；`full-factorial` 会运行全部 16 种开关组合：

```bash
PYTHONPATH=src python -m red_swarm_policy.run_blue_rl_ablations \
  outputs/blue_rl/curriculum_normalized_v3/blue_rainbow.pt \
  --suite core --seeds 10042,20042,30042 \
  --missiles 1,2,3,4 --episodes 400 --device cuda:0 \
  --parallel-envs 16 --acmi-interval 0 \
  --output outputs/blue_rl/ablations/core
```

先加 `--dry-run` 可只生成并打印全部命令；默认遇到首个失败即停止，加入
`--continue-on-error` 后会继续其余组合，并在 manifest 中记录每次运行的返回码。


`red_swarm_policy.blue_rl` 将 nash1.6 的离散蓝方逃逸强化学习结构移植到 v1：采用 29 个 v1
蓝机动作、Rainbow DQN（Dueling、NoisyNet、C51、PER、n-step、Double DQN）并提供固定长度的
单机/1–4 枚来弹观测。当前训练输入使用下述 `normalized_v4` 物理/机理状态；联合训练按所选最大来弹数
补零到固定长度，从而由同一个策略随机学习多个场景。该子系统只复用 v1 的场景生成、三自由度动力学、导引头、裁决和参数；
没有复制 nash1.6 的场景数值。

训练环境 `BlueEscapeEnv` 与红方分层训练环境相对独立。所有红弹固定分配给唯一蓝机且残差过载恒为
零，因此只运行 v1 的比例导引，不会调用红方高层或低层网络。训练和测试每回合都写 Tacview ACMI。

蓝机在 9–11 km 高度内生成，初始滚转角和航迹倾角均为 0°，初速度只在水平面内随机取向。
在任一存活来弹与蓝机的三维距离严格小于 60 km 之前，环境固定执行 0 号平飞动作，不调用蓝方
策略网络、不产生学习奖励、不写 replay，也不计算 loss 或梯度；首次越过探测边界的过渡只用于建立
第一帧 RL 状态，从下一决策开始机动和训练。触发状态在本回合内不可逆。

训练、课程内验证、独立评估和普通 `BlueRLController` 现在共用同一个预测飞行包线约束层。每次决策先用
与三自由度飞机一致的载荷/重力模型模拟全部 29 个候选动作 0.1 s；预测会从状态快照中的当前连续指令
开始，在这 0.1 s 内采用与执行端相同的指令线性插值，再用候选末端垂直速度外推 2 s。
物理高度范围沿用 `AircraftConfig` 的 8000–12000 m；默认安全余量 500 m，因此预计高度离开
8500–11500 m 后施加二次软代价：`4 * min(越界深度 / 500, 2)^2`。因此 8250/11750 m 的代价为 1，
8000/12000 m 的代价为 4；预计越过物理高度边界的动作则直接
进入硬掩码。该代价只参与同一状态下的动作排序，不重复写入环境奖励，避免改变势函数奖励的终局语义。

通用包线还预测 1 s 内的水平速度、水平/总速度比例、航迹倾角和水平速度衰减，并检查 2 s 内能否恢复到
航向有效状态。恢复检查不再使用冻结加速度的 `v+a*t` 近似，而是同时展开 29 个当前候选与 29 个恢复动作，
每 0.02 s 根据变化后的速度方向重建机体系、重新计算加速度；恢复动作的前 0.1 s 同样连续插值，并在整个
恢复路径上检查最大载荷、载荷/滚转变化率、物理高度和物理速度。默认软目标为水平速度 150 m/s、水平速度
比例 0.70、绝对航迹倾角 45°；硬下限/上限分别为
100 m/s、0.35、70°。动作相对上一个**实际执行动作**的载荷矢量变化率和滚转指令变化率也分别设软/硬门槛
30/100 g/s 与 240/1200 deg/s，用于阻止瞬时左右翻转或正负载荷翻转。硬动作排除后，以
`Q - 高度软代价 - 包线软代价 - 指令变化软代价` 选择实际动作；若初始状态已经使安全集为空，则只开放
确定性的最小风险恢复动作。P1/P2 紧急门只把包线软代价最多降到 50%、指令变化软代价最多降到 20%，
高度软代价和全部硬掩码始终不变。执行端把约束后的离散目标在一个 0.1 s 决策区间内逐 0.005 s 物理帧
线性插值。

新 checkpoint 会保存完整的飞行包线配置，评估时自动恢复，并拒绝与保存值不一致的决策周期。PER/n-step
回放保存约束后的实际动作以及 n-step 末状态的硬动作掩码和软代价；Double-DQN 的下一动作选择也使用它们，
不会把回报归因给未执行的 RL 原始动作。日志和飞行品质轨迹分别保留 `requested_action_index`、
`constrained_action_index`、`executed_action_index`、目标载荷和物理层实际载荷。

新训练使用 `normalized_v4` 观测。除蓝机状态和 32 维实际动作上下文外，每个来弹槽包含相对位置、
相对速度、距离、闭合速度、`t_go`、LOS 角速率、能量、制导状态和有效位；全局部分包含滤波威胁及
变化率、多弹包围指标、P0/P1/P2 阶段及驻留/确认状态、主威胁切换迟滞、机体系期望方向、离散参考
过载、包线余度和紧急门。1v1 输入为 74 维，补齐到 1v4 的联合输入为 119 维。这样奖励状态机使用的
状态不再隐藏在环境内部。旧 `normalized_v3`（42/46/50/54 维）、`normalized_v2` 和 `legacy_v1`
checkpoint 仍按保存的输入契约加载。

蓝方奖励在每个 **0.1 s 决策边界**计算一次。旧的远离/切向/俯冲三项势函数全部保留，尺度由 2
下调到 1；另加入由距离、闭合速度、`t_go`、低 LOS 角速率碰撞特征、导弹能量/制导状态和多弹包围度
构成的威胁结果势函数。四个势分量统一使用 `gamma * Phi(next) - Phi(current)`，终止状态统一令
`Phi=0`。P1 紧急阶段只把旧战术势权重平滑降到最低 25%，避免“继续飞远/俯冲”压过眼前规避，但不删除
旧奖励内容。

其余三项采用同一个 P0（低威胁）、P1（规避）、P2（释放）迟滞状态机，并作为小预算的直接代价：
时机代价比较实际净加速度激活度与阶段目标；方向代价只比较净加速度单位方向与多弹 maximin 方向，
留有 20° 死区；过载代价用速度、水平速度比例和航迹倾角形成的包线余度计算连续参考值，再投影到可行
离散动作。三项分别按整回合预算 0.60/0.45/0.55 除以剩余时域，不按物理帧重复计罚；硬约束只剩一个
可选动作或进入 fallback 时全部关闭。最终仍以生存裁决为主：脱靶/物理失效 `+10`（另有不超过 `+1`
的快速完成奖励）、蓝机被击中 `-10`（按生存进度最多减轻 `1`）、timeout `+2`。

建议奖励重构后的训练计划先做 terminal-only 与新 threat-potential 的相同 seed 消融，再逐步扩展到 1v2～1v4；
每阶段使用独立 evaluation seeds 比较生存率、终止/失效原因、脱靶距离、完成时间与动作分布。C51 support
随新奖励调整为 61 atoms、`[-14, 12]`；正式长跑仍应依据投影前 n-step return 分位数复核边界，而不是依据 episode
总回报机械扩大 support。

蓝方训练和评估入口统一把 mission timeout 与 missile guidance timeout 设置为 200 s。针对平均上千个
决策步的轨迹，Rainbow 默认采用 `gamma=0.999`、`n_step=20`，并将学习率降低到 `2.5e-4` 以减轻后期
策略漂移。每个 decision 的 `reward_components` 分别记录旧三项势差、`threat_outcome_shaping`，以及
`timing_penalty`、`direction_penalty`、`overload_penalty`；`reward_diagnostics` 另外记录威胁、变化率、
包围度、阶段、紧急门、三类误差和可选动作门。训练窗口同时报告 C51 上下界 clamp 比例。

```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl --missiles 1,2,3,4 --episodes 1000
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl outputs/blue_rl/train/blue_rainbow.pt --missiles 1,2,3,4
```

## 当前 normalized_v4 推荐命令

以下命令均从 `v1/` 目录执行。新启动的训练会自动使用 `normalized_v4`，无需额外指定 schema；checkpoint
会保存 schema，评估入口会自动校验，不能手工混用旧的 `legacy_v1` 输入。

先做快速 smoke，确认环境进程、119 维联合观测、replay、更新和 checkpoint 全链路可运行：

```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl \
  --missiles 1,2,3,4 --episodes 8 --seed 42 --device cpu \
  --parallel-envs 2 --batch-size 8 --updates-per-transition 0.25 \
  --checkpoint-interval 8 --log-interval 2 --acmi-interval 0 \
  --output outputs/blue_rl/smoke_normalized_v4
```

推荐的课程训练（默认课程完整长度为 7500 回合）：

```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl \
  --curriculum --episodes 7500 --seed 42 --device cuda:0 \
  --parallel-envs 16 --env-worker-threads 1 \
  --batch-size 256 --updates-per-transition 0.5 \
  --checkpoint-interval 500 --log-interval 10 --acmi-interval 0 \
  --output outputs/blue_rl/curriculum_normalized_v4
```

固定独立 seed 对最终 checkpoint 做 1v1～1v4 联合评估：

```bash
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl \
  outputs/blue_rl/curriculum_normalized_v4/blue_rainbow.pt \
  --missiles 1,2,3,4 --episodes 4000 --seed 10042 --device cuda:0 \
  --parallel-envs 16 --env-worker-threads 1 \
  --log-interval 100 --acmi-interval 0 \
  --output outputs/blue_rl/eval_normalized_v4
```

代码回归测试和命令行检查：

```bash
PYTHONPATH=src pytest -q tests/test_blue_rl.py
PYTHONPATH=src pytest -q tests/test_smoke.py tests/test_training_readiness.py
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl --help
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl --help
```

## 课程学习（可选，原训练方式不变）

课程模式通过单独的 `--curriculum` 开关启用。新训练统一使用版本化 `normalized_v4` 观测，因此课程模式
从第一回合起固定使用 `max_missiles=4`、119 维观测和 29 维动作；无效来弹槽的 13 个物理特征严格补零。
旧训练环境接口仍默认 `legacy_v1`，评估和常规运行会根据 checkpoint 输入维度自动选用旧的 9/12/15/18 维
schema、`normalized_v2` 的 10/14/18/22 维、`normalized_v3` 的 42/46/50/54 维，或
`normalized_v4` 的 74/89/104/119 维 schema。
旧 checkpoint 不会被静默解释为新输入。

默认八阶段共 **7500** 回合（原建议表各行相加是 7500，而不是文字中的 8500），保留所有旧难度复习，
并在每一新阶段前 500 回合线性改变抽样概率。若希望达到建议的 10,000–12,000 回合，应在完成默认课程后，
以均衡采样继续常规联合训练；当前入口刻意拒绝超过课程定义的回合数，避免悄悄使用未定义概率。


```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl \
  --curriculum --episodes 7500 --parallel-envs 16 --device cuda:0 \
  --acmi-interval 0 --output outputs/blue_rl/curriculum
```

默认每 500 个训练回合关闭 NoisyNet 探索，以固定且独立的 seed 对每个已引入场景测试 300 回合，并打印
`curriculum_evaluation` JSON。评估同时记录各场景生存率、平均生存时间、历史最佳和五个百分点遗忘约束。
`best_new_scenario.pt` 保存当前最难场景的历史最佳，`best_balanced.pt` 仅在所有旧场景未突破遗忘约束时
按阶段权重改善才更新（1v4 阶段使用 0.1/0.2/0.3/0.4，最终阶段恢复等权）。全部评估写入
`training.jsonl` 和 `training_metrics.json` 的 `curriculum_evaluations`，逐回合结果另含 `curriculum_stage`。
可用 `--curriculum-eval-episodes 500` 增强阶段评估，最终候选仍建议用独立评估入口每场景测试 1000 回合；
仅做快速调试时可设为 0 禁用内嵌评估。

`--missiles` 接受逗号分隔的任意子集（例如 `1,3,4`）。每个 episode 从该集合中均匀随机抽取一个场景；
训练日志记录每个窗口和全程的实际抽样数量，评估结果另外按场景报告生存率。随机序列仅由 `--seed`
决定，不受并行 worker 完成先后影响。测试联合检查点时，所选集合的最大来弹数必须与训练时一致；例如
用 `1,2,3,4` 训练的检查点可以测试 `2,4`，但不能只传 `1`（后者是 9 维单场景观测）。

常规仿真中，现有 `BlueEvasionController`（规则机）保持不变；将载入的 Rainbow agent 包装为
`BlueRLController` 即可作为 `RedBlueEngagementEnv(..., blue_policy=controller)` 并列替换。算法与环境通过
`DiscreteBluePolicy` 协议以及 `PolicyRegistry` 解耦，之后实现离散 PPO 时注册新工厂即可，无需修改环境。

## 参数设置

命令行参数可通过 `--help` 查看。常用参数为 `--missiles`（1–4 的逗号分隔子集）、`--episodes`、`--seed`、
`--device`、`--decision-interval`、`--replay-size`、`--checkpoint-interval`、`--log-interval`、`--acmi-interval` 和 `--output`。物理与场景参数默认完整使用
`EnvironmentConfig`；如需覆盖，向 `--env-config` 传入只包含改动项的 JSON。例如：

```json
{
  "max_steps": 24000,
  "scenario": {"red_cluster_radius_range_m": [140000.0, 150000.0]},
  "missile": {"proportional_navigation_gain": 3.5}
}
```

字段名必须与 `env/types.py` 中的 v1 配置一致，未知字段会立即报错。训练和测试必须使用同一个
环境配置、来弹数量和决策周期。完整运行示例：

原有单场景接口完全保留：`--missiles 4` 仍生成原来的 18 维观测并训练/测试 1v4 专用检查点，
`--missiles` 省略时仍默认 1v1。联合模式只在传入多个值时启用补零，因此已有单场景检查点及命令无需迁移。

```bash
cd v1
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl \
  --missiles 4 --episodes 1000 --seed 42 --device cuda:0 \
  --parallel-envs 16 --env-worker-threads 1 --batch-size 256 \
  --replay-size 500000 \
  --updates-per-transition 0.5 \
  --log-interval 10 \
  --acmi-interval 10 \
  --env-config blue_env.json --output outputs/blue_rl/train_1v4

PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl \
  outputs/blue_rl/train_1v4/blue_rainbow.pt \
  --missiles 4 --episodes 100 --seed 10042 --device cuda:0 \
  --parallel-envs 16 --env-worker-threads 1 \
  --env-config blue_env.json --output outputs/blue_rl/test_1v4
```

`--parallel-envs N` 启动 N 个持久 CPU 进程并行推进独立场景，主进程将这些场景的观测合并后只执行一次
GPU 前向推理。每个环境维护独立的 n-step 轨迹，结束的环境会立即用新的全局 episode 编号和 seed 重置，
不会等待最长回合。`--updates-per-transition` 控制每收集一条 transition 安排多少次梯度更新；并行训练建议先从
`0.25`–`0.5` 开始，并通过 `--batch-size 128` 或 `256` 增加 GPU 工作量。训练和评估的默认值仍为
`--parallel-envs 1`，用于保持原有资源占用习惯；性能运行建议关闭 ACMI。

训练会在启动时输出 `experiment_config` 和 `environment_workers` JSON 行，此后每
`--log-interval` 个已完成 episode 输出一个 `iteration` 聚合行，其中包含窗口胜率、回报统计、命中数、
脱靶距离、终止原因、采样吞吐、动作直方图与熵、预测价值、loss、梯度范数、PER、replay、学习率、
optimizer/target 更新次数以及 CUDA 显存。所有运行时 JSON 行同步追加到输出目录的 `training.jsonl`；
训练结束后，完整配置、区间指标、逐 episode 结果和最终汇总写入 `training_metrics.json`。可分别使用
`--jsonl-path` 和 `--metrics-path` 覆盖路径。

评估会在启动时输出 `evaluation_config` JSON 行，此后每 `--log-interval` 个已完成 episode 输出一条
`evaluation_progress`，包括完成进度、窗口生存率、回报、命中数、脱靶距离、终止原因和运行吞吐；结束时输出
`evaluation_complete`，随后保留原有的完整评估 JSON 输出。以上流式事件同时写入输出目录的
`evaluation.jsonl`（可用 `--jsonl-path` 覆盖），最终逐 episode 结果仍写入 `evaluation.json`。最终结果的
`statistics` 以及各 `by_scenario` 项包含生存/毁伤计数、终止与动作分布，并对回报、脱靶距离、仿真时长和
决策步数给出均值、标准差、最值、5/25/50/75/95 分位数及直方图；逐 episode 结果另外记录动作直方图、
累计奖励分量和平均奖励诊断。每个 episode 的 `initialization` 完整记录飞机和各枚导弹的初始位置、高度、
航向、航迹倾角与速度。飞机初始航向相对弹群中心按 90° 扇区分为朝向弹群、正 90°、负 90°和远离弹群；
`by_blue_orientation` 分别给出四类的胜率和完整统计（即使某类无样本也保留）。整体及各朝向统计均包含以
1 m 为固定组距的稀疏脱靶量概率直方图 `miss_distance_probability_histogram_1m`，其中仅输出非空区间。
评估推理固定使用 `torch.inference_mode()`，冻结在线网络参数，并在结束时校验
步数、optimizer/target 更新计数、replay 大小及参数版本均未改变；若有任何训练状态变化会直接报错。

`--acmi-interval N` 表示只保存第 N、2N、3N……个回合；默认值 `1` 表示每回合保存，设为 `0`
会完全关闭 ACMI 并且不在内存中累计轨迹。训练 ACMI 位于训练输出目录的 `acmi/`，测试 ACMI
位于测试输出目录的 `acmi/`。常规场景中可直接选择：

```bash
PYTHONPATH=src python -m red_swarm_policy.run_blue_evasion \
  --blue-policy rainbow --blue-checkpoint outputs/blue_rl/train_1v4/blue_rainbow.pt \
  --red-count 4 --blue-count 1 --device cuda:0
```

将 `--blue-policy` 改为 `rule` 即使用原有规则机。

## 蓝机飞行品质验收报告

训练和评估现在都会在输出目录的 `flight_quality/` 下维护独立验收产物：

- `flight_quality.json`：逐回合完整时序、异常事件起止时间、硬门槛结论和总体生存率对比；

- `flight_quality_episodes.jsonl`：每个回合结束时立即追加一行完整诊断，即使训练中断也保留已完成回合；

- `flight_quality_summary.csv`：一回合一行的汇总表，便于回归比较和导入表格工具；
- `episode_XXXXXX.png`：最低分代表回合的六联图（3D/俯视轨迹、高度、航迹倾角与航向角速度、速度与转弯半径、策略/执行动作及安全介入）。

报告按 `x=北、y=高度、z=东` 计算航迹倾角、水平速度占比、航向角速度、转弯半径、螺旋、短窗折返、
轨迹自回访、高度边界驻留、动作切换和安全过滤器介入率，并保留所有原始指标及异常时间段，而不只输出加权分。
水平速度低于 150 m/s 时航向无定义，诊断不会计算该段航向角速度或水平转弯半径，也不会把它计为螺旋；
若同时绝对航迹倾角不小于 30°，则单独记录 `steep_low_horizontal_speed`（陡峭低水平速度）事件。
默认仅绘制品质分最低的 10 个回合，可用 `--flight-quality-plot-limit` 调整或设为 `0` 关闭图片。
传入修改前测试的 `--baseline-survival-rate` 后，报告还会直接标记当前生存率是否下降超过 5 个百分点；
未提供基线时该结论为 `null`，避免把缺失对照误报成“未下降”。
