
# 红方三枚导弹 vs 蓝方逃逸 —— 博弈论 + 强化学习demo架构

本项目给出一个**可训练的代码架构**，用于模拟如下场景：

- 红方有 **3 枚导弹**，导弹的**发射位置**由一个“博弈论决策器”给出；
- 发射位置被限制在三维空间中的一个立方体区域内；
- 蓝方飞机的初始位置在该区域 **外部**，目标是**尽量逃离红方导弹的追击**；
- 蓝方策略由一个 **强化学习（DQN）智能体**学习，动作集为简单机动（在 \(\pm x, \pm y, \pm z\) 方向加速，或保持加速度为 0）。

整体结构满足“代码职责隔离”的要求：

- `env/`：环境与物理/博弈逻辑
  - `escape_env.py`：可训练的环境类 `EscapeEnv`（Gym 风格 API）
  - `game_theory_launcher.py`：红方博弈论式发射位置决策模块
  - `missile_dynamics.py`：蓝机与导弹的简单三维点质点运动学
- `agent/`：蓝方强化学习智能体
  - `dqn_agent.py`：DQN 智能体（含 Q 网络与训练逻辑）
  - `replay_buffer.py`：经验回放缓冲区
- `config.py`：环境与训练超参数配置
- `train_blue_agent.py`：训练脚本（将环境与智能体连接起来）

---

## 一、核心理念与接口设计

### 1. 红方：博弈论发射位置决策（GameTheoreticLauncher）

文件：`env/game_theory_launcher.py`

- 定义 `LaunchRegion`：
  - `region_min, region_max`：三维立方体发射区域的边界；
  - `num_missiles`：导弹数量（此处为 3）；
  - `candidate_launch_count`：每次复位时候采样的候选发射点数量。
- 定义 `GameTheoreticLauncher`：
  - 接口：
    ```python
    launcher = GameTheoreticLauncher(region_cfg)
    launch_positions = launcher.compute_launch_positions(blue_initial_pos)
    # launch_positions.shape == (num_missiles, 3)
    ```
  - 当前实现中：将该问题简化为一个静态博弈中的“红方最佳反应”：
    - 在发射区域内均匀采样若干候选点；
    - 以“蓝机初始位置到候选点的距离”作为威胁指标（距离越小越有利于红方拦截）；
    - 选择距离最小的若干点作为 3 枚导弹的发射位置。
  - 将来如需更复杂的博弈（例如考虑蓝机可能的初始机动策略），可以在此模块内部替换为真正的收益矩阵/混合策略求解，而不会影响 RL 训练代码。

### 2. 蓝方与导弹运动学

文件：`env/missile_dynamics.py`

- 蓝方：
  - 动作集：`{0: 无加速度, 1..6: 在 \pm x, \pm y, \pm z 方向加速}`；
  - 速度有上限 `blue_max_speed`；
  - 使用简单的欧拉积分更新位置与速度。
- 导弹：
  - 速度为常数 `missile_speed`；
  - 每个时间步朝向当前蓝机位置的方向进行纯追迹（pure pursuit）机动。

### 3. 环境：EscapeEnv

文件：`env/escape_env.py`

- Gym 风格接口：
  ```python
  env = EscapeEnv(EnvConfig())
  obs = env.reset()
  obs, reward, done, info = env.step(action)
  ```
- 状态向量结构：
  - 蓝机位置 \( (x_b, y_b, z_b) \)；
  - 蓝机速度 \( (v_{bx}, v_{by}, v_{bz}) \)；
  - 每枚导弹相对于蓝机的位置向量（共 3 枚导弹 → 9 维）；
  - 因此总维度：`3 + 3 + 3 * num_missiles`。
- 奖励与终止：
  - 若任一导弹与蓝机距离 `<= hit_radius`：判定为被击中，`reward = -1`，episode 结束；
  - 若时间步数 `>= max_steps`：视为成功逃脱，`reward = +1`，episode 结束；
  - 中间时间步使用轻微的 shaping：最小导弹距离越大，奖励越高，引导学习“远离导弹”。

---

## 二、蓝方强化学习智能体（Rainbow DQN）

文件：`agent/rainbow_dqn_agent.py`, `agent/replay_buffer.py`

- 采用 Rainbow DQN 组合增强项：
  - Dueling 架构 + NoisyNet 提升探索；
  - Distributional C51 预测回报分布；
  - Double DQN + 目标网络降低过估计；
  - Prioritized Replay + n-step return 提升采样效率。

---

## 三、训练脚本

文件：`train_blue_agent.py`

运行方式（在项目根目录）：

```bash
python train_blue_agent.py
```

逻辑：

1. 创建 `EnvConfig` 与 `TrainConfig`；
2. 使用 `make_env_and_agent` 生成 `EscapeEnv` 与 `RainbowDQNAgent`；
3. 循环若干 episode：
   - 调用 `env.reset()`；
   - 反复 `agent.select_action -> env.step -> agent.store_transition -> agent.update`；
   - 根据 `TrainConfig.print_interval` 打印最近若干 episode 的平均奖励。

---

## 四、后续扩展建议

1. **更真实的博弈论部分**  
   在 `GameTheoreticLauncher` 中：
   - 将蓝方“可能的初始机动策略”离散化为若干候选；
   - 定义红蓝策略的收益矩阵（例如拦截时间、命中概率）；
   - 对该矩阵进行静态博弈求解（Nash/Stackelberg），输出红方的混合策略并抽样发射位置。

2. **更精细的动力学模型**  
   - 引入速度与加速度限制、转弯半径等，更接近真实空战；
   - 用 3DoF/PN 制导替代当前的纯追迹模型。

3. **多蓝机 / 多红机扩展**  
   - 扩展环境，使得多个蓝机协同逃逸（多智能体 RL），
   - 红方仍使用博弈论导引区域 + 简单导弹模型。

当前版本旨在给出一个**干净、模块化、可运行训练的最小架构**，方便你在此基础上不断替换/升级各个模块。

---

## 五、测试时切换蓝方策略（DQN / BT）

各个评估脚本都提供 `--blue-eval-policy` 参数，用来选择蓝方飞机的评估策略：

- `--blue-eval-policy dqn`：使用 checkpoint 中训练好的 DQN 强化学习策略。该模式会加载 `checkpoint` 内的 `blue` 参数，并要求 checkpoint 的观测维度与当前场景的 `num_missiles` 匹配。
- `--blue-eval-policy bt`：使用规则/行为树（BT）基线策略。该模式不会加载 DQN 网络权重，适合在没有匹配 DQN checkpoint 或只想看规则策略表现时使用。

参数默认值为 `dqn`。因此，如果要评估强化学习策略，可以显式写出：

```bash
python test/test_1v3_coordination.py --blue-eval-policy dqn
```

如果要切换到 BT 规则策略：

```bash
python test/test_1v3_coordination.py --blue-eval-policy bt
```

1v1 与 1v2 测试脚本使用同样的切换方式，例如：

```bash
python test/test_1v1_escape_direction.py --blue-eval-policy dqn
python test/test_1v2_coordination.py --blue-eval-policy bt
```

输出目录会根据策略自动区分，避免 DQN 和 BT 的结果互相覆盖：

- 1v3 DQN：`outputs/tests_1v3_coordination_dqn/`
- 1v3 BT：`outputs/tests_1v3_coordination_bt/`

对于 1v2/1v3 这类带逐步诊断的评估，`results/step_diagnostics.csv` 中会记录：

- `blue_eval_policy`：本次实际使用的策略类型；
- `blue_action`：每一步蓝方选择的离散动作编号；
- `bt_state`：BT 模式下的行为树状态（如 `CRUISE`、`BEAM`、`BREAK`）；DQN 模式下为空。

如果你期望使用强化学习训练结果，但看到蓝机动作很像规则策略或机动很弱，建议优先检查：

1. 运行命令中是否写成了 `--blue-eval-policy dqn`；
2. 输出目录是否是 `*_dqn`；
3. `step_diagnostics.csv` 里的 `blue_eval_policy` 是否为 `dqn`；
4. checkpoint 是否与当前测试场景的导弹数量一致，例如 1v3 场景需要匹配 `num_missiles=3` 的 DQN checkpoint。
