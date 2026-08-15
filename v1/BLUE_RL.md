# 蓝方强化学习子系统

`red_swarm_policy.blue_rl` 将 nash1.6 的离散蓝方逃逸强化学习结构移植到 v1：采用 29 个 v1
蓝机动作、Rainbow DQN（Dueling、NoisyNet、C51、PER、n-step、Double DQN）并提供固定长度的
单机/1–4 枚来弹观测。观测与 nash1.6 一致，为蓝机绝对位置、速度和每枚红弹相对位置，
所以不同红弹数量需要分别训练检查点。该子系统只复用 v1 的场景生成、三自由度动力学、导引头、裁决和参数；
没有复制 nash1.6 的场景数值。

训练环境 `BlueEscapeEnv` 与红方分层训练环境相对独立。所有红弹固定分配给唯一蓝机且残差过载恒为
零，因此只运行 v1 的比例导引，不会调用红方高层或低层网络。训练和测试每回合都写 Tacview ACMI。

```bash
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl --missiles 4 --episodes 1000
PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl outputs/blue_rl/train/blue_rainbow.pt --missiles 4
```

常规仿真中，现有 `BlueEvasionController`（规则机）保持不变；将载入的 Rainbow agent 包装为
`BlueRLController` 即可作为 `RedBlueEngagementEnv(..., blue_policy=controller)` 并列替换。算法与环境通过
`DiscreteBluePolicy` 协议以及 `PolicyRegistry` 解耦，之后实现离散 PPO 时注册新工厂即可，无需修改环境。

## 参数设置

命令行参数可通过 `--help` 查看。常用参数为 `--missiles`（1–4）、`--episodes`、`--seed`、
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

```bash
cd v1
PYTHONPATH=src python -m red_swarm_policy.train_blue_rl \
  --missiles 4 --episodes 1000 --seed 42 --device cuda:0 \
  --acmi-interval 10 \
  --env-config blue_env.json --output outputs/blue_rl/train_1v4

PYTHONPATH=src python -m red_swarm_policy.evaluate_blue_rl \
  outputs/blue_rl/train_1v4/blue_rainbow.pt \
  --missiles 4 --episodes 100 --seed 10042 --device cuda:0 \
  --env-config blue_env.json --output outputs/blue_rl/test_1v4
```

`--acmi-interval N` 表示只保存第 N、2N、3N……个回合；默认值 `1` 表示每回合保存，设为 `0`
会完全关闭 ACMI 并且不在内存中累计轨迹。训练 ACMI 位于训练输出目录的 `acmi/`，测试 ACMI
位于测试输出目录的 `acmi/`。常规场景中可直接选择：

```bash
PYTHONPATH=src python -m red_swarm_policy.run_blue_evasion \
  --blue-policy rainbow --blue-checkpoint outputs/blue_rl/train_1v4/blue_rainbow.pt \
  --red-count 4 --blue-count 1 --device cuda:0
```

将 `--blue-policy` 改为 `rule` 即使用原有规则机。
