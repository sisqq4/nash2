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

## 仅测试启用的机理塑形与消融

`evaluate_blue_rl` 可在完整 29 维 C51 动作价值上独立启用四类物理评分：
`--mechanism-threat`（短时威胁缓解）、`--mechanism-timing`（带迟滞的 P0/P1/P2
时机）、`--mechanism-direction`（LOS 法向逃逸方向）和 `--mechanism-overload`
（随威胁及阶段变化的过载）。`--mechanism-weight` 调整每项归一化评分的融合权重，默认
为 `0.35`。

所有开关默认关闭，且训练入口不提供这些选项，因此不会改变已有训练、奖励或回放数据。无开关
为 Rainbow-only 基线，单独开启可做逐项消融，全部开启为完整融合。评估 JSON 记录启用项、
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
  outputs/blue_rl/curriculum_normalized_v2/blue_rainbow.pt \
  --suite core --seeds 10042,20042,30042 \
  --missiles 1,2,3,4 --episodes 400 --device cuda:0 \
  --parallel-envs 16 --acmi-interval 0 \
  --output outputs/blue_rl/ablations/core
```

先加 `--dry-run` 可只生成并打印全部命令；默认遇到首个失败即停止，加入
`--continue-on-error` 后会继续其余组合，并在 manifest 中记录每次运行的返回码。


`red_swarm_policy.blue_rl` 将 nash1.6 的离散蓝方逃逸强化学习结构移植到 v1：采用 29 个 v1
蓝机动作、Rainbow DQN（Dueling、NoisyNet、C51、PER、n-step、Double DQN）并提供固定长度的
单机/1–4 枚来弹观测。单场景观测与 nash1.6 一致，为蓝机绝对位置、速度和每枚红弹相对位置；
联合训练则按所选最大来弹数补零到固定长度，从而由同一个策略随机学习多个场景。该子系统只复用 v1 的场景生成、三自由度动力学、导引头、裁决和参数；
没有复制 nash1.6 的场景数值。

训练环境 `BlueEscapeEnv` 与红方分层训练环境相对独立。所有红弹固定分配给唯一蓝机且残差过载恒为
零，因此只运行 v1 的比例导引，不会调用红方高层或低层网络。训练和测试每回合都写 Tacview ACMI。

蓝方奖励在每个 **0.1 s 决策边界**计算一次，而不再在 0.005 s 物理帧上反复惩罚不可避免的弹目接近。
其有界多弹威胁势函数采用 soft-min 权重：远距阶段奖励速度指向远离来弹的方向；进入 30 km 附近后，
平滑转为奖励与来袭方向近似切向的速度以及俯冲分量。势函数尺度默认为 2，并使用与 DQN 一致的
`gamma * Phi(next) - Phi(current)`；终止状态的势函数严格置零，避免终局残留势函数改变奖励方向，因此 shaping 不会掩盖
终局的生存优先级。裁决结果分别映射为明确脱靶/物理失效 `+10`（另有不超过 `+1` 的快速完成奖励）、
蓝机被击中 `-10`（按生存进度最多减轻 `1`）、mission timeout `+2`。日志逐回合保存
`reward_components` 和每枚导弹的 `red_loss_reasons`，避免把超时和真正脱靶混为同一成功类型。

建议奖励重构后的训练计划先做 terminal-only 与新 threat-potential 的相同 seed 消融，再逐步扩展到 1v2～1v4；
每阶段使用独立 evaluation seeds 比较生存率、终止/失效原因、脱靶距离、完成时间与动作分布。C51 support
随新奖励调整为 `[-12, 12]`；正式长跑仍应依据投影前 n-step return 分位数复核边界，而不是依据 episode
总回报机械扩大 support。

蓝方训练和评估入口统一把 mission timeout 与 missile guidance timeout 设置为 200 s。针对平均上千个
决策步的轨迹，Rainbow 默认采用 `gamma=0.999`、`n_step=20`，并将学习率降低到 `2.5e-4` 以减轻后期
策略漂移。每个 decision 的 `reward_components` 分别记录 `far_away_shaping`、`near_tangent_shaping` 和
`near_dive_shaping`；`reward_diagnostics` 另外记录 `range_blend_weight`、`softmin_threat_distance`、
`potential_before`、`potential_after`。训练窗口同时报告 C51 上下界 clamp 比例。

为抑制原地盘旋、旋转爬升/下降以及相反机动的高频切换，训练奖励还在同一决策边界加入六个可配置的
非正正则项。`action_switch_penalty` 按相邻动作的机动过载矢量差连续计罚（而不是把任意换挡一律等价
处理）；`opposite_maneuver_penalty` 仅在去除平飞重力补偿后的相邻过载矢量夹角余弦低于阈值时计罚，
专门约束突然反向。爬升与下降根据超过垂直速度死区的幅度分别计罚，默认下降权重较低，以免完全抵消
近距阶段的战术俯冲收益。结构过载采用超过 6g 软阈值后的平方罚项，仍由动力学的 9g 上限负责硬裁剪；
横向约束以决策前水平速度为随动前向，惩罚一个决策周期内超过 5m/s 的法向速度增量，并以平台在该
决策周期内的最大理论横向速度增量归一化。它约束的是
过快转弯而不是相对初始航向的累计偏角，因此不会错误惩罚已经稳定在新航向上的正常平飞。所有严重度
均归一化并截断到 `[0, 1]`，不会随一个决策内的物理子步数重复累计。

每步 `reward_components` 记录六个带负号的罚项；`reward_diagnostics` 同时记录动作切换严重度、相邻
机动余弦、垂直速度、指令过载和决策周期横向速度增量，便于区分究竟是哪类异常机动导致回报下降。
这些正则项会改变最优策略，并非势函数塑形；开始长训前应以相同 seed 分别做全关闭、逐项开启和完整
组合消融，并联合检查生存率、动作切换率、反向切换率、垂直速度/过载/横向速度分位数，而不能只看
episode return。

上述权重定义为“整段任务在严重度始终为 1 时的最大累计预算”，每步实际罚值还会乘以
`decision_interval_s / policy_horizon_s`。这样更改决策间隔或物理子步数不会无意放大奖励尺度，并保证
默认六项预算总和小于生存/被击中的终局回报差。动作切换奖励依赖上一动作，因此新训练使用
`normalized_v3`：在原 `normalized_v2` 后附加归一化的上一动作 `[轴向过载, 法向过载, 滚转角]`，使该
奖励保持马尔可夫性；评估和常规运行仍兼容旧的 `legacy_v1` 与 `normalized_v2` checkpoint。

```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl --missiles 1,2,3,4 --episodes 1000
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl outputs/blue_rl/train/blue_rainbow.pt --missiles 1,2,3,4
```

## 当前 normalized_v3 推荐命令

以下命令均从 `v1/` 目录执行。新启动的训练会自动使用 `normalized_v3`，无需额外指定 schema；checkpoint
会保存 schema，评估入口会自动校验，不能手工混用旧的 `legacy_v1` 输入。

先做快速 smoke，确认环境进程、25 维联合观测、replay、更新和 checkpoint 全链路可运行：

```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl \
  --missiles 1,2,3,4 --episodes 8 --seed 42 --device cpu \
  --parallel-envs 2 --batch-size 8 --updates-per-transition 0.25 \
  --checkpoint-interval 8 --log-interval 2 --acmi-interval 0 \
  --output outputs/blue_rl/smoke_normalized_v3
```

推荐的课程训练（默认课程完整长度为 7500 回合）：

```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl \
  --curriculum --episodes 7500 --seed 42 --device cuda:0 \
  --parallel-envs 16 --env-worker-threads 1 \
  --batch-size 256 --updates-per-transition 0.5 \
  --checkpoint-interval 500 --log-interval 10 --acmi-interval 0 \
  --output outputs/blue_rl/curriculum_normalized_v3
```

固定独立 seed 对最终 checkpoint 做 1v1～1v4 联合评估：

```bash
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl \
  outputs/blue_rl/curriculum_normalized_v3/blue_rainbow.pt \
  --missiles 1,2,3,4 --episodes 4000 --seed 10042 --device cuda:0 \
  --parallel-envs 16 --env-worker-threads 1 \
  --log-interval 100 --acmi-interval 0 \
  --output outputs/blue_rl/eval_normalized_v3
```

代码回归测试和命令行检查：

```bash
PYTHONPATH=src pytest -q tests/test_blue_rl.py
PYTHONPATH=src pytest -q tests/test_smoke.py tests/test_training_readiness.py
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl --help
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl --help
```

## 课程学习（可选，原训练方式不变）

课程模式通过单独的 `--curriculum` 开关启用。新训练统一使用版本化 `normalized_v3` 观测：蓝机水平位置置于
相对原点、绝对高度除以 20 km、速度除以 2000 m/s、弹机相对位置除以 200 km，并在每个导弹槽后附加一个
有效位，并附加 3 维上一动作。因此课程模式从第一回合起固定使用 `max_missiles=4`、25 维观测和
29 维动作；无效槽严格补零。
旧训练环境接口仍默认 `legacy_v1`，评估和常规运行会根据 checkpoint 输入维度自动选用旧的 9/12/15/18 维
schema、`normalized_v2` 的 10/14/18/22 维，或 `normalized_v3` 的 13/17/21/25 维 schema。
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
默认仅绘制品质分最低的 10 个回合，可用 `--flight-quality-plot-limit` 调整或设为 `0` 关闭图片。
传入修改前测试的 `--baseline-survival-rate` 后，报告还会直接标记当前生存率是否下降超过 5 个百分点；
未提供基线时该结论为 `null`，避免把缺失对照误报成“未下降”。
