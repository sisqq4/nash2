# 蓝方训练/测试速度、过载与单回合轨迹分析

项目保存的飞行质量轨迹已经包含按回合分析蓝机速度、过载、位置和动作所需的数据。正常训练结束时会自动生成训练速度与过载统计；测试结果则可以流式生成全部回合及各来弹数的监测统计。也可以使用以下离线入口重绘已有结果：

- `red_swarm_policy.analyze_blue_training_load`：统计训练过程中各回合的蓝机速度与过载，并按来弹数量绘制变化曲线。
- `red_swarm_policy.analyze_blue_evaluation_dynamics`：流式统计测试阶段全部回合以及 1、2、3 等不同来弹数下的速度、过载、峰值和时间剖面。
- `red_swarm_policy.export_blue_episode_trace`：从测试结果中选择一个回合，导出完整时序表和九张独立中文图片。
- `red_swarm_policy.plot_blue_results`：完整批量重绘训练、测试、飞行质量和训练速度过载图，也可仅导出一个测试回合。

所有离线入口只读取已有结果，不启动仿真、不加载模型，也不更新训练参数。

## PowerShell 运行环境

以下命令都应在项目的 `v1` 目录中执行：

```powershell
cd C:\Users\sisqq4\PycharmProjects\nash_origin\v1
$env:PYTHONPATH='src'
```

PowerShell 设置环境变量时必须写成 `$env:PYTHONPATH='src'`。其中 `$` 不能省略，也不要写成 `Env: Pythonpath='src'`。

## 现有数据和采样粒度

训练和测试目录的 `flight_quality/flight_quality_episodes.jsonl` 每行保存一个完整回合。主要时序字段如下：

| 字段 | 含义 | 单位或取值 |
| --- | --- | --- |
| `time_s` | 仿真时刻 | s |
| `sample_interval_s` | 当前样本所代表决策区间的长度；新格式首项为 0 | s |
| `step_count` | 当前状态对应的物理积分步序号 | 整数 |
| `position_m` | 蓝机惯性坐标位置 `[x, y, z]` | m |
| `velocity_mps` | 蓝机惯性坐标速度 `[vx, vy, vz]` | m/s |
| `speed_mps` | 蓝机总速度 | m/s |
| `horizontal_speed_mps` | 蓝机水平速度 | m/s |
| `vertical_speed_mps` | 蓝机垂直速度 | m/s |
| `flight_path_angle_deg` | 航迹倾角 | 度 |
| `heading_rate_deg_s` | 航向角速度；不可计算时为空 | 度/s |
| `turn_radius_m` | 转弯半径；不可计算时为空 | m |
| `commanded_load_g` | 记录当时动作表得到的名义合成指令过载 | g |
| `estimated_actual_load_g` | 由相邻速度采样差分并补偿重力得到的估算过载 | g |
| `policy_action` | 网络原始选择的机动序号 | 0–28；尚未启用策略时可能为空 |
| `executed_action` | 安全约束处理后实际执行的机动序号 | 0–28；初始采样可能为空 |
| `safety_intervened` | 安全约束是否修改了策略动作 | 布尔值 |
| `safety_reasons` | 安全约束修改动作的原因 | 字符串列表 |

记录粒度为“环境重置状态加每个已完成的环境决策步”。当前默认决策间隔约为 0.1 s，而物理积分步长为 0.005 s；JSONL 不保存每一个物理积分步。因此，应以文件中的 `time_s` 和 `sample_interval_s` 为准，不要用数组下标推算时间。样本 `i` 表示此前的决策区间 `[t_i-dt_i, t_i]`；首个状态样本的 `dt_0` 为 0，不给统计量增加一段虚构时长。旧轨迹没有 `sample_interval_s` 时，脚本按 `[0, t_1-t_0, t_2-t_1, ...]` 重建。部分回合的首个绝对记录时刻可能位于助推或策略激活之后，并不保证为 0 s；单回合导出的相对时间仍以首样本为 0。

测试轨迹还保存来弹位置、速度、存活状态和固定槽位标识。新格式测试记录的 `metadata` 包含全局回合编号、`scenario_episode`、`missile_count`、随机种子、初始化状态和终止原因。当前训练目录中的旧轨迹没有逐行 `metadata`，训练分析会按 `episode` 与训练目录的 `training_metrics.json` 和 `episodes.json` 关联，补充来弹数量、逃逸结果和终止原因。关键元数据冲突会报错：若两份文件对 `missile_count`、`blue_survived`、`simulation_time_s` 或 `termination_reason` 给出互相冲突的非空值，脚本会停止，避免静默采用错误分组；后一个文件中的缺失值或空值不会覆盖已有值。

## 训练速度与过载统计

### 训练结束自动生成

`red_swarm_policy.train_blue_rl` 在正常训练结束并写完 `episodes.json` 和 `training_metrics.json` 后，自动生成训练速度与过载表格和图片。默认输出目录为：

```text
<训练输出目录>/training_load_analysis
```

若训练命令指定了 `--training-plots-dir "<绘图目录>"`，则速度与过载结果写入 `<绘图目录>/speed_load`。自动分析的滚动窗口复用 `--plot-smoothing-window`，分段大小为 100 回合，图片 DPI 为 180。`--no-training-plots` 会同时跳过该分析。调用使用安全包装器隔离；统计或绘图失败只会输出错误信息，不会修改模型、检查点或已保存的训练结果。

### 已有训练结果的离线命令

对当前 2000 回合训练结果执行：

```powershell
python -m red_swarm_policy.analyze_blue_training_load `
  --training-dir "outputs/blue_rl/train_1v1-1v3_train_1v1-1v3_missle_around_60_ep2000" `
  --output "outputs/blue_rl/train_1v1-1v3_train_1v1-1v3_missle_around_60_ep2000/training_load_analysis"
```

省略 `--output` 时也会写入训练目录下的 `training_load_analysis`。可追加 `--rolling-window 50`、`--block-size 100`、`--dpi 180` 或 `--font "<中文字体文件>"` 调整滚动窗口、分段大小、图片分辨率和中文字体。

输出 `逐回合速度过载统计.csv`、`分段速度过载统计.csv` 和 `训练速度过载分析清单.json`。逐回合表包含轨迹完成次序、回合编号、来弹数量、回合结果、记录时长和采样数，以及速度、两种过载、安全介入和动作切换统计。

速度统计包括：

- 总速度的有效样本数、有效时长、时间加权均值、样本 P05、样本中位数、样本 P95、最小值、最大值和时间加权均方根。
- 水平速度的时间加权均值、最小值和样本 P05。
- 垂直速度绝对值的时间加权均值、样本 P95 和最大值。
- 分段表继续按有效时长合并速度均值和均方根，并保存逐回合速度分位数的平均值及分段极值。

过载统计口径为：

- 估算实际过载均值、均方根（RMS），以及名义指令过载均值，均按每个样本代表的决策区间长度进行时间加权；首样本权重为 0。
- 估算实际过载的 P50 是样本中位数；该 P50、估算实际过载 P95 和名义指令过载 P95 均为有效样本值的分位数，不按持续时间加权。最大值同样直接取有效样本最大值。
- 超过 3g、6g、9g 的时长按样本区间累计，相应比例以有效估算过载的总持续时间为分母。
- 分段表默认每 100 个完成回合汇总一次。合并均值和均方根仍以有效持续时间加权，同时保留逐回合样本 P50、P95 的平均值、成功规避率、安全介入和动作切换统计。

训练变化图以轨迹实际完成顺序为横轴。每张图片只绘制一个指标：散点颜色区分 1v1、2v1、3v1，每一种来弹数量分别计算自己的最近若干个同类完成回合滚动均值，不会把其他场景夹入同一滚动窗口。十一张独立图片为：

1. `speed_mean.png`：每回合总速度的时间加权均值。
2. `speed_min.png`：每回合最低总速度。
3. `speed_max.png`：每回合最高总速度。
4. `horizontal_speed_min.png`：每回合最低水平速度。
5. `abs_vertical_speed_p95.png`：每回合垂直速度绝对值的样本 P95。
6. `estimated_load_max.png`：每回合最大估算实际过载。
7. `estimated_load_p95.png`：每回合估算实际过载的样本 P95。
8. `estimated_load_mean.png`：每回合估算实际过载的时间加权均值。
9. `estimated_load_above_6g_duration.png`：每回合估算实际过载超过 6g 的持续时间。
10. `commanded_load_max.png`：每回合最大名义指令过载。
11. `commanded_load_mean.png`：每回合名义指令过载的时间加权均值。

图例中的“1 枚来弹”“2 枚来弹”“3 枚来弹”分别对应 1v1、2v1、3v1；每个场景均只有一架蓝机。由于并行环境的回合编号与完成顺序可能不同，滚动均值按 JSONL 中实际写入的完成顺序计算；表中同时保留原始回合编号。

## 测试全批次速度与过载统计

测试目录中的 `flight_quality/flight_quality_episodes.jsonl` 已经包含每个完整回合的速度、两种过载以及 `metadata.missile_count`。以下命令不会重新运行仿真或加载 checkpoint，而是逐行扫描现有 JSONL；即使文件很大，也不会把所有完整轨迹同时载入内存：

```powershell
python -m red_swarm_policy.analyze_blue_evaluation_dynamics `
  --evaluation-dir "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）" `
  --output "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）/evaluation_dynamics"
```

省略 `--output` 时，默认也写入 `<测试输出目录>/evaluation_dynamics`。可用 `--time-bin-s 1.0` 调整跨回合时间剖面的箱宽，用 `--dpi 180` 或 `--font "<中文字体文件>"` 调整绘图；加 `--no-plots` 时只写数值表和 JSON。

输出包括：

- `逐回合速度过载统计.csv`：每个测试回合一行，保留来弹数、生存结果、总/水平/垂直速度统计、估算实际过载、名义指令过载、3g/6g/9g 以上时长、安全介入和动作切换等指标。
- `全部及分弹数速度过载汇总.csv` 和 `.json`：第一组为全部测试回合，随后分别给出每种实际来弹数。速度、估算实际过载和名义指令过载都包含有效样本数与时长、时间加权均值、RMS、样本 P05/P50/P95、全局峰值、逐回合峰值的均值与 P95；两种过载另含超过 3g、6g、9g 的总时长、比例及逐回合平均时长。
- `速度过载时间剖面.csv`：以每回合首样本为相对 0 秒，按时间箱保存全部回合及各弹数组的均值、P05、P50、P95和实际贡献回合数。回合结束后的时间箱不会用末值补齐。
- `速度时间剖面.png` 和 `过载时间剖面.png`：分别展示“全部来弹数”以及各弹数组随相对时间的总体变化；每个时间箱先在单回合内求均值，再让各回合等权参与组统计，避免采样点多的回合获得额外权重。
- `测试速度过载分析清单.json`：记录输入、输出、回合总数、各弹数回合数、时间箱宽、全部产物及统计口径。

跨回合均值和 RMS 按 `sample_interval_s` 加权；P05/P50/P95 是组内有效采样值的分位数，不做时间加权。“全局峰值”是该组所有回合峰值中的最大值，“逐回合峰值均值/P95”则先求每回合峰值，再对回合等权汇总。估算实际过载和名义指令过载的物理含义仍受本文末尾所述限制。

## 完整批量重绘

使用当前训练目录和测试目录完整重绘时，在 `v1` 目录执行：

```powershell
$env:PYTHONPATH='src'
python -m red_swarm_policy.plot_blue_results `
  --training-dir "outputs/blue_rl/train_1v1-1v3_train_1v1-1v3_missle_around_60_ep2000" `
  --evaluation-dir "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）"
```

完整模式会同时生成常规训练曲线、训练分组结果、`training_dynamics` 训练速度/过载结果、
`evaluation_dynamics` 测试总体及分弹数速度/过载结果、测试分组结果，以及训练和测试的飞行质量图。
未指定 `--output-root` 时，测试统计写入测试目录的 `evaluation_dynamics`；指定统一输出根目录时写入
`<output-root>/evaluation_dynamics`。可用 `--evaluation-time-bin-s` 调整其时间箱宽。批量清单的
`figures.evaluation_dynamics` 和 `tables.evaluation_dynamics` 会列出对应图片和表格。

完整模式也可在命令末尾加入 `--episode 1`，或加入 `--missiles 2 --scenario-episode 1`，在同一次运行中附带导出所选测试回合。

## 导出一个测试回合

### 使用完整批量入口但只给测试目录

`plot_blue_results` 可以省略训练目录，只导出一个测试回合。按当前测试目录的全局回合编号导出第 1 回合：

```powershell
$env:PYTHONPATH='src'
python -m red_swarm_policy.plot_blue_results `
  --evaluation-dir "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）" `
  --episode 1
```

按来弹数和场景内回合编号导出 2v1 的第 1 个回合：

```powershell
python -m red_swarm_policy.plot_blue_results `
  --evaluation-dir "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）" `
  --missiles 2 `
  --scenario-episode 1
```

默认分别输出到测试目录的 `episode_details/episode_000001` 或 `episode_details/missiles_2_scenario_000001`。可用 `--episode-output "<目录>"` 指定最终目录；若使用 `--output-root`，默认写入其 `evaluation_episode_details` 子目录。只给测试目录的模式不会扫描训练目录，也不会生成完整批量图。

### 按全局回合编号选择

以下命令导出当前测试结果的第 1 回合：

```powershell
python -m red_swarm_policy.export_blue_episode_trace `
  --evaluation-dir "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）" `
  --episode 1 `
  --output "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）/episode_trace_000001"
```

### 按场景内回合编号选择

如果更方便按来弹数量和该场景内部编号定位，可使用 `--missiles` 与 `--scenario-episode`。例如导出 2v1 场景的第 1 个测试回合：

```powershell
python -m red_swarm_policy.export_blue_episode_trace `
  --evaluation-dir "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）" `
  --missiles 2 `
  --scenario-episode 1 `
  --output "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）/episode_trace_2v1_000001"
```

`--episode` 与 `--missiles` 加 `--scenario-episode` 是两种选择方式，一次只使用一种。脚本按行扫描大型 JSONL，只把目标回合载入内存。

也可以用 `--input-jsonl "<flight_quality_episodes.jsonl 路径>"` 直接指定轨迹文件；`--evaluation-dir` 和 `--input-jsonl` 二选一。可用 `--dpi 180` 设置图片分辨率，或用 `--font "<中文字体文件>"` 指定中文字体。默认不在任何图片上标出安全介入；确需复盘安全约束时，显式追加 `--show-safety-interventions`。`plot_blue_results` 的单回合导出也支持同名选项。

## 单回合输出

脚本输出以下数据表：

- `蓝方逐时刻数据.csv`：保留绝对及相对时刻、物理步序号、采样间隔、三轴位置和速度、总/水平/垂直速度、航迹倾角、航向角速度、转弯半径、过载、动作和安全接管信息。名义过载同时保存“绘图采用值”“轨迹历史记录值”和“当前动作表重建值”三列。
- `动作区间.csv`：把策略动作和执行动作均相同的连续样本合并为区间，并保存指令分量、安全介入次数和原因。单个样本 `i` 的动作区间是 `[t_i-dt_i, t_i]`，首个零时长状态样本不会使阶梯图向右错移；合并区间沿用这一端点语义。
- `动作映射.csv`：列出导出时当前代码中 0–28 全部机动序号及其轴向过载、法向过载、滚转角和名义合成过载，便于解释重建列。
- `回合摘要.json` 和 `回合摘要.csv`：记录全局回合编号、场景内编号、来弹数量、随机种子、初始状态、终止原因、是否成功规避、样本数和时刻范围；同时保存红弹轨迹数、图片是否标注安全介入、当前动作表 SHA-256 指纹，以及轨迹记录值与当前动作表重建值的可比较样本数、不一致样本数和最大绝对差。

同时生成九张互相独立的中文图片：

1. `01_速度随时间.png`：总速度、水平速度和垂直速度绝对值。
2. `02_速度分量随时间.png`：北向 `vx`、垂直 `vy` 和东向 `vz`。
3. `03_过载随时间.png`：估算实际过载和名义执行动作过载；后者优先使用轨迹记录值，记录缺失的样本才回退到当前动作表重建值。
4. `04_XY平面轨迹.png`：蓝机及各枚红弹的 X-Y 投影，横轴为北向位置 x，纵轴为高度 y。
5. `05_YZ平面轨迹.png`：蓝机及各枚红弹的 Y-Z 投影，严格按 Y-Z 次序，横轴为高度 y，纵轴为东向位置 z。
6. `06_ZX平面轨迹.png`：蓝机及各枚红弹的 Z-X 投影，严格按 Z-X 次序，横轴为东向位置 z，纵轴为北向位置 x，表示水平面态势。
7. `07_位置分量随时间.png`：x、y、z 三个位置分量。
8. `08_动作序号随时间.png`：策略选择动作和实际执行动作的阶梯线。
9. `09_动作指令分量随时间.png`：动作对应的轴向过载、法向过载和滚转角。

三张投影图中蓝机轨迹固定使用蓝色，所有红弹轨迹固定使用红色；蓝机起点和终点仍分别用绿色圆点和黑色叉号表示。新格式轨迹会根据 `red_state_valid` 去掉红弹失效后的冻结状态，并在有物理步终止事件时把该弹的精确终止位置接到轨迹末端；不含红弹遥测的旧结果仍可导出，只是投影图中仅有蓝机。

默认情况下，九张图都不标注安全介入区间，以免遮挡主轨迹和时序曲线。只有显式传入 `--show-safety-interventions` 时，轨迹图才会用小三角和白心小圆分别标记连续介入区间的起止位置，时间图才会显示低透明度淡红背景。无论是否显示这些视觉标记，CSV 和回合摘要中的安全介入原始数据都始终保留。九张图的标题会附加可用的 1v1、2v1 或 3v1、全局回合编号及场景内回合编号；动作序号和动作指令采用按 `[t_i-dt_i, t_i]` 绘制的区间阶梯线。

## 坐标定义

轨迹使用环境的惯性坐标系：

- x 轴指向北方。
- y 轴垂直向上，表示高度。
- z 轴指向东方。
- 坐标单位为米，速度单位为米每秒。

因此，XY 和 YZ 是两个垂直剖面，ZX 是水平面俯视轨迹。图片严格按平面名称排列坐标：XY 为横轴 x、纵轴 y，YZ 为横轴 y、纵轴 z，ZX 为横轴 z、纵轴 x。它们不是蓝机随动坐标系中的相对位置。

## 两种过载的解释限制

`commanded_load_g` 是生成该轨迹时按实际执行动作序号记录的名义合成指令值。它反映当时离散动作库要求的载荷等级，但不包含物理积分步之间的执行器插值、气动响应和所有限制过程，不能直接当作机体真实瞬时过载。单回合过载图和“名义执行动作过载”主列优先采用这份历史记录；只有历史标量缺失的样本才使用当前代码动作表重建值。CSV 会并列保留两种来源，摘要会保存当前动作表的 SHA-256 指纹及两者一致性检查，动作表后来发生变化时仍可识别历史口径。

`estimated_actual_load_g` 使用决策级相邻速度样本计算加速度，再补偿重力并求模。它会受到约 0.1 s 采样间隔、数值差分噪声、速度限幅和高度边界状态修正的影响；短时尖峰可能明显高于真实物理过载。因此，该值适合比较训练趋势和排查数值异常，不应用来单独证明飞机的实际过载能力。

分析时应同时查看两种口径：使用 `commanded_load_g` 判断策略选择了什么载荷等级，使用 `estimated_actual_load_g` 观察保存的决策级运动状态发生了怎样的变化。所有统计和图片均保留这两个名称，避免把二者混为同一测量值。
