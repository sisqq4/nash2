# 蓝方测试结果离线提取与对比

`red_swarm_policy.analyze_blue_evaluations` 只读取已经生成的测试 JSON，不加载模型、不创建环境。
`evaluate_blue_rl` 和 `evaluate_blue_rule_baseline` 在原有测试和保存全部结束后默认调用它，
额外输出到测试 `--output` 目录下的 `evaluation_analysis/`。原有 JSON、CSV、JSONL、ACMI、
flight_quality 文件的已有字段继续保留，模型测试的状态记录扩展见下节。它同时支持 Rainbow 测试的 `evaluation.json`、规则基线汇总，
以及消融批次中每个 case 的 `evaluation.json`。

## 模型测试中的逐回合实体状态记录

`evaluate_blue_rl` 默认保存以下扩展，不需要增加命令行参数；关闭 ACMI、汇总图或逐回合图片也不会关闭数据记录。
此扩展用于模型测试（包括调用同一入口的消融测试），训练及规则基线入口保持原有输出。

每回合完成后写入 `flight_quality/flight_quality_episodes.jsonl`；全部测试完成后，相同记录按回合编号排序，
汇总到 `flight_quality/flight_quality.json` 的 `episodes` 中。扩展记录标记 `schema_version: 2`。

| 字段 | 含义 |
| --- | --- |
| `trace.red_ids` | 当前回合的固定实体编号，从 0 开始，与该回合红弹数组槽位对应；失效后不移除、不重排 |
| `trace.time_s` | 红蓝双方共用的仿真时间，保留环境的实际起始时间，不强制从 0 开始 |
| `trace.step_count` | 与每个采样时刻对应的物理步编号 |
| `trace.sample_interval_s` | 与上一条记录的实际时间差；初始记录为 0，回合提前结束时可短于常规决策间隔 |
| `trace.red_positions_m` | `[采样点][固定槽位][x,y,z]`，惯性坐标，x 向北、y 向上、z 向东，单位米 |
| `trace.red_velocities_mps` | 同一结构的速度向量，单位米/秒 |
| `trace.red_alive` | `[采样点][固定槽位]`，实体是否仍存活；缺失状态为 `null` |
| `trace.red_loss_reasons` | 同一结构的失效原因；没有已记录原因为 `null` |
| `trace.red_position_valid` / `red_velocity_valid` | 对应向量的三个分量是否齐全且为有限数值 |
| `trace.red_state_valid` | 存活且位置、速度均有效；失效后即使保留冻结数值，此标记仍为 `false` |
| `red_termination_events` | 各实体失效时的物理步事件，包含编号、时间、步编号、原因、最终位置和速度，以及同一时刻的蓝机位置和速度 |

常规记录频率与环境决策步一致，包含初始帧和回合结束帧；终止事件在每个物理步后单独观察并记录，
所以事件时间可以位于两个常规采样点之间。事件时间表示物理步判定时刻，并非更细的连续时间估计。
仍存活但因整个回合结束而停止记录的实体不会被伪造为失效事件。

缺失或非有限的向量分量保存为 JSON `null`，不以 0 填充；原始有效性与存活状态分开保存。
轨迹只保存当前回合实际存在的实体，不保存模型观测中的补齐槽位。逐回合图片读取有效标记，
并使用单独保存的终止位置结束相应轨迹，避免把失效后的冻结位置作为后续活动轨迹。

运行开始时，`flight_quality/evaluation_metadata.json` 保存一次完整环境配置、适配器配置、测试选项、
坐标及采样约定、Python/NumPy/PyTorch 版本、模型文件路径及 SHA-256，以及源码文件与源码整体 SHA-256。
源码摘要标识实际运行时文件内容，包含尚未提交的源码改动。每回合 `metadata` 保存运行编号 `run_id`、
元数据文件名 `manifest`、实际随机种子（`--seed + episode`）、实体数量、初始化信息和回合终止原因；
初始状态向量可从该回合 `trace` 的第 0 条读取。元数据文件与逐回合文件应一起保留。

旧结果不会自动补出缺失字段；旧版记录仍可被既有绘图和分析逻辑读取。

## 基线与强化学习对比

两个测试命令新增 `--no-result-plots` 用于关闭本次新增的汇总绘图，
`--result-plots-dir PATH` 用于指定独立输出目录。原 `--flight-quality-plot-limit` 仅控制原逐回合
轨迹图，不控制新增汇总图。自动绘图异常只输出 stderr 提示，不改变测试返回结果或已保存文件。

例如默认 Rainbow 测试目录为 `outputs/blue_rl/test`，图片就位于
`outputs/blue_rl/test/evaluation_analysis/`。独立调用仍使用 `--output` 指定分析目录。

Windows PowerShell 独立绘图示例（在 `v1` 目录执行）：

```powershell
$env:PYTHONPATH = "src"
python -m red_swarm_policy.analyze_blue_evaluations rainbow=outputs/blue_rl/test --output outputs/blue_rl/test/evaluation_analysis
```

Python 调用：

```python
from red_swarm_policy.analyze_blue_evaluations import write_evaluation_analysis_safely
write_evaluation_analysis_safely("outputs/blue_rl/test/evaluation.json")
```

```bash
PYTHONPATH=src python -m red_swarm_policy.analyze_blue_evaluations \
  rule=outputs/blue_rl/rule_baseline \
  rainbow=outputs/blue_rl/test \
  --baseline rule --output outputs/blue_rl/analysis/rule_vs_rainbow
```

默认按红弹数量 `missile_count` 和蓝机初始方向 `blue_orientation` 联合分层，并额外给出总体结果。
如需纳入其他场景维度，传入行中已有的字段（嵌套字段使用点号）：

```bash
--dimensions missile_count,blue_orientation,initialization.some_scenario_field
```

## 四种逃逸机理及消融对比

先使用 `run_blue_rl_ablations --suite full-factorial` 生成 16 种组合，再把需要比较的 case
逐个命名传入。例如：

```bash
PYTHONPATH=src python -m red_swarm_policy.analyze_blue_evaluations \
  base=outputs/blue_rl/ablations/seed_10042/00_rainbow_only \
  threat=outputs/blue_rl/ablations/seed_10042/01_threat \
  timing=outputs/blue_rl/ablations/seed_10042/02_timing \
  full=outputs/blue_rl/ablations/seed_10042/15_threat_timing_direction_overload \
  --baseline base --output outputs/blue_rl/analysis/ablation
```

标签可任意命名；完整的添加/移除机理消融应把 16 个 case 都传入。所有策略均按相同分层汇总，
便于控制场景构成后比较。

## 产物

* `analysis.json`：可追溯输入路径、分层条件、完整描述统计、终止/红弹损失原因计数，以及相对基线差值。
* `metrics.csv`：扁平表格，包含逃脱率及 95% Wilson 区间、脱靶量均值/标准差/最小值/最大值和
  25%/50%/75% 分位点、主威胁切换次数、奖励、仿真时间、决策步数与命中数。
* `episodes.csv`：逐回合关键字段，包括策略标签、场景维度、逃脱结果、脱靶量、主威胁切换、
  命中数、终止原因与机理介入率，便于自行重分组或进行显著性检验。
* `escape_rate_overall.png`、`escape_rate_by_missile_count.png`、`escape_rate_by_blue_orientation.png`：
  总体、按红弹数量、按初始朝向的蓝方逃脱率，标出样本数和 95% Wilson 区间。
* `escape_rate_by_missile_count_and_orientation.png`：红弹数量 × 初始朝向的逃脱率热力图，
  每格显示逃脱率和样本数，未观测组合标注 N/A。
* `miss_distance_distribution.png` 及 `_by_missile_count`、`_by_blue_orientation`、
  `_by_missile_count_and_orientation` 版本：总体及三类分组的**固定 1 米分箱**脱靶量概率图。
  每组包含长尾在内的全部样本；有不小于 50 米的记录时额外输出对应的 `_0_50m.png` 细节图。
* `miss_distance_histogram_1m.csv`：每个策略、分组的精确箱下界、上界、数量、概率和分母。
  按 `[0,1)、[1,2)、[2,3)…` 计数，恰好 1 米归入 `[1,2)`，恰好 2 米归入 `[2,3)`。
  仅存储非空箱，未列出的箱计数为 0；不会丢弃远距离样本或将它们合并成一个尾箱。
* `flight_quality_score_distribution.png`：飞行质量评分分布；`flight_quality_score_by_*.png`
  分别给出按弹数、朝向、弹数 × 朝向的评分箱线图。
* `flight_quality_metrics.png`：已有飞行质量指标的箱线图，包括评分、最大航迹倾角、最小水平速度、
  水平速度比例、估计实际过载、动作切换频率、安全介入率、近高度边界时间、螺旋和陡峭低水平速度时长。
* `flight_quality_failure_rate_*.png`：总体及三类分组的已有验收项不通过比例；
  `flight_quality_event_rates.png`：出现各类已记录异常事件的回合比例。
* `plot_statistics.json`：上述图表清单、总体及所有分组的精确统计，包含飞行质量指标的有效/缺失数、
  验收项观测分母、不通过数和区间、事件次数与涉及回合数。

每张图最多放 6 个子图，更多分组自动分页为 `_02.png`、`_03.png` 等；以
`plot_statistics.json.figures` 为本次生成的完整清单。重复绘制覆盖同名派生文件；旧分页图不会自动删除，
重新分析不同输入集合时建议使用新分析目录。

图片和 JSON/CSV 分别保存：图片用于直观检查，数值文件用于后续统计分析，不互相替代。
若服务器没有 matplotlib，可用 `--no-plots` 仅生成全部数值产物，包括新增的 1 米分箱和飞行质量统计。
`--miss-distance-bins` 保留兼容旧命令，但不再改变箱宽；所有脱靶量分布固定为 1 米一档。

脱靶量沿用逐回合 `miss_distance_m`，每回合一个观测，包含被命中和逃脱回合，不按来弹数量重复计数。
纵轴是该组全部回合中落入对应 1 米箱的比例；0–50 米细节图也使用同一个分母，不重新归一化。
完整范围跨度较大时，点标记用于标明很窄的非空 1 米箱，完整计数可查 CSV。

飞行质量只读取已有的 `flight_quality.metrics/verdicts/events`，不重新计算轨迹、不改变阈值。
原 `verdicts=True` 表示通过，图中“不通过率”统计 False；缺少该项记录的回合不计入分母。
旧测试文件和当前规则基线可能没有逐回合飞行质量记录，此时生成缺失提示图，不解释为评分 0、
评分 100 或全部通过。原 `flight_quality/episode_*.png` 逐回合轨迹图继续按原选项保存。

## 数据完整性说明

工具会在分析前检查逐回合数据是否确实包含 `blue_survived`、`miss_distance_m`、
`missile_count` 和 `blue_orientation`，避免把缺失值错误地当成“未逃脱”或零脱靶量。
检查结果写入 `analysis.json` 的 `inputs.*.data_quality`。

现有 `evaluate_blue_rl` 仅在至少启用一种测试逃逸机理时跟踪主威胁序列；未启用机理时写出的
零切换来自空序列，不能解释为真实的“没有切换”。分析器会将这种情况标记为缺失，并在
`data_quality.warnings` 中说明。因此，旧的规则基线或纯 Rainbow 产物仍可正确比较逃脱率和
脱靶量，但不能凭空恢复主威胁切换次数；若该指标需要覆盖所有策略，必须由后续测试输出真实的
逐决策主威胁编号或顶层 `main_threat_switches` 字段。

总体逃脱率会受到各红弹数量、初始方向样本占比影响。不同评估运行的场景构成不一致时，应优先
使用 `level=stratified` 的同场景结果，而不是只根据总体差值下结论。

## 运行方式、场景差异与异常处理

所有命令都应从 `v1` 目录运行，并设置 `PYTHONPATH=src`。每个输入使用 `标签=文件或目录`；
标签会原样写入表格的 `policy` 列。`--output` 是分析产物目录，若省略则默认写到
`outputs/blue_rl/analysis`。

### 单一测试结果

```bash
PYTHONPATH=src python -m red_swarm_policy.analyze_blue_evaluations \
  rainbow=outputs/blue_rl/test --output outputs/blue_rl/analysis/rainbow
```

### 规则基线与强化学习

```bash
PYTHONPATH=src python -m red_swarm_policy.analyze_blue_evaluations \
  rule=outputs/blue_rl/rule_baseline rainbow=outputs/blue_rl/test \
  --baseline rule --output outputs/blue_rl/analysis/rule_vs_rainbow
```

### 指定不同场景划分

默认把红弹数量和初始方向作为一个**联合分组**。只按红弹数量分析、只输出总体、或者加入更多
初始化条件的方式分别为：

```bash
--dimensions missile_count
--dimensions none
--dimensions missile_count,blue_orientation,initialization.some_scenario_field
```

维度越多，`metrics.csv` 和 `analysis.json` 中的 `stratified` 行越多、每组样本通常越少；总体行
不受维度选项影响。新增图表和 `plot_statistics.json` 固定覆盖总体、弹数、朝向和弹数 × 朝向，
不受 `--dimensions` 影响；额外自定义维度仍在原数值表格中体现。

### 多种机理/消融场景

每个消融 case 都作为一个独立输入传入，并把不含机理的基础 Rainbow case 指定为 baseline。
不同 case 的统计结构相同，区别体现在 `policy`、各指标数值和 `comparison_to_baseline` 差值。
建议各 case 使用相同种子和回合数，避免把样本构成差异误认为机理收益。

### 空值、错误和跳过规则

* 输入没有逐回合数组，逐回合数组为空，或者缺少/损坏逃脱结果、脱靶量、红弹数量、初始方向时，
  默认立即报错并且不生成一个看似有效的不完整报告。
* 奖励、仿真时间、决策步数、命中数等可选数值缺失时不会报错；该指标的 `count` 为 0，其他统计量
  为 JSON `null`、CSV 空单元格。
* 某策略没有与基线相同的分层组合时，该组相对基线的差值为 `null`，该策略自身统计仍会保留。
* 未记录主威胁序列时按缺失处理，不会把空序列错误解释成 0 次切换。
* 批量处理时可加 `--skip-invalid-inputs` 跳过坏文件；跳过项及错误原因写入
  `analysis.json.skipped_inputs`。若所有输入均无效，或者被跳过的是 `--baseline` 指定的输入，仍会
  报错，因为此时无法生成有意义的比较结果。
