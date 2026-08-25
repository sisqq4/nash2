# 蓝方测试结果离线提取与对比

`red_swarm_policy.analyze_blue_evaluations` 只读取已经生成的测试 JSON，不加载模型、不创建环境，
因此不会改变现有训练和测试过程。它同时支持 Rainbow 测试的 `evaluation.json`、规则基线汇总，
以及消融批次中每个 case 的 `evaluation.json`。

## 基线与强化学习对比

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
* `miss_distance_distribution.png`：各策略脱靶量概率密度分布。
* `escape_rate_by_missile_count.png`：不同红弹数量下各策略逃脱率柱状图。

图片和 JSON/CSV 分别保存：图片用于直观检查，数值文件用于后续统计分析，不互相替代。
若服务器没有绘图库，可用 `--no-plots` 仅生成数值产物。

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
不受维度选项影响。两张图片仍分别展示所有输入的总体脱靶分布，以及按红弹数量划分的逃脱率，
不会因为额外维度而重复生成大量图片。

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
