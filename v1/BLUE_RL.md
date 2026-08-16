# 蓝方强化学习子系统

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

```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl --missiles 1,2,3,4 --episodes 1000
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl outputs/blue_rl/train/blue_rainbow.pt --missiles 1,2,3,4
```

`--missiles` 接受逗号分隔的任意子集（例如 `1,3,4`）。每个 episode 从该集合中均匀随机抽取一个场景；
训练日志记录每个窗口和全程的实际抽样数量，评估结果另外按场景报告生存率。随机序列仅由 `--seed`
决定，不受并行 worker 完成先后影响。测试联合检查点时，所选集合的最大来弹数必须与训练时一致；例如
用 `1,2,3,4` 训练的检查点可以测试 `2,4`，但不能只传 `1`（后者是 9 维单场景观测）。

常规仿真中，现有 `BlueEvasionController`（规则机）保持不变；将载入的 Rainbow agent 包装为
`BlueRLController` 即可作为 `RedBlueEngagementEnv(..., blue_policy=controller)` 并列替换。算法与环境通过
`DiscreteBluePolicy` 协议以及 `PolicyRegistry` 解耦，之后实现离散 PPO 时注册新工厂即可，无需修改环境。

## 参数设置

命令行参数可通过 `--help` 查看。常用参数为 `--missiles`（1–4 的逗号分隔子集）、`--episodes`、`--seed`、
`--device`、`--decision-interval`、`--checkpoint-interval`、`--acmi-interval` 和 `--output`。物理与场景参数默认完整使用
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

`--acmi-interval N` 表示只保存第 N、2N、3N……个回合；默认值 `1` 表示每回合保存，设为 `0`
会完全关闭 ACMI 并且不在内存中累计轨迹。训练 ACMI 位于训练输出目录的 `acmi/`，测试 ACMI
位于测试输出目录的 `acmi/`。常规场景中可直接选择：

```bash
PYTHONPATH=src python -m red_swarm_policy.run_blue_evasion \
  --blue-policy rainbow --blue-checkpoint outputs/blue_rl/train_1v4/blue_rainbow.pt \
  --red-count 4 --blue-count 1 --device cuda:0
```

将 `--blue-policy` 改为 `rule` 即使用原有规则机。
