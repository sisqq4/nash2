# 训练结果绘图

`src/red_swarm_policy/analyze_training.py` 是独立的训练日志后处理模块。
读取已有 JSON/JSONL，不执行环境步进、模型推理或参数更新。

## 默认调用

原有 `train_blue_rl` 和 `train_env` 命令无需修改。训练结束、原有文件全部保存、
环境 worker 关闭后，默认额外生成图表。原有 checkpoint、ACMI、逐回合结果、
metrics、JSONL、flight_quality、run_manifest 的保存代码和格式保持原样；
测试入口及 `train_env --validation-only` 不调用新模块。

- `train_blue_rl`：训练曲线默认保存到 `--output` 目录下的 `training_analysis/`，
  按弹数、朝向及联合场景的结果分布图保存在其中的 `results/` 子目录。
- `train_env`：默认保存到 metrics 文件同级的 `<metrics文件名去扩展名>_analysis/`。
  例如 `outputs/env_training_metrics.json` 对应 `outputs/env_training_metrics_analysis/`。
  关闭 metrics 保存时，使用已有内存指标，保存到 run_manifest 同级的 `training_analysis/`。

两个训练入口新增的可选参数：

| 参数 | 含义 |
| --- | --- |
| `--no-training-plots` | 关闭新增的训练结束后绘图 |
| `--plot-smoothing-window 20` | 滑动平均窗口，正整数，默认 20 条记录 |
| `--training-plots-dir PATH` | 自定义独立的分析输出目录 |

自动绘图异常只输出 stderr 提示，不改变训练返回结果，也不影响已经保存的文件。
绘图不改变训练随机数状态，不设置全局 pyplot 后端，不打开交互窗口。
自动调用发生在训练正常结束或原有提前停止收尾后；意外中断时，可独立读取已经落盘的 JSONL。

## 蓝方训练结果分组与飞行质量图

蓝方训练结束后还会调用 `red_swarm_policy.analyze_blue_training_results`，在
`training_analysis/results/` 中生成以下汇总图及精确统计：

- 总体、按弹数、按初始朝向的逃脱率图，以及弹数 × 朝向热力图。
- 总体及上述分组的固定 1 米分箱脱靶量分布，包含所有远距离样本；有长尾时另附 0–50 米细节图。
- 飞行质量评分分布、分组评分箱线图、关键指标分布、验收项不通过比例、异常事件涉及回合比例。
- `plot_statistics.json` 和 `miss_distance_histogram_1m.csv`，保存各图精确数据及样本分母。

这些统计使用训练中实际采集的回合，包含探索以及训练期间不断变化的策略，不能当作固定 checkpoint
的独立测试成绩。详细图表文件名和 1 米分箱口径见 [BLUE_EVALUATION_ANALYSIS.md](BLUE_EVALUATION_ANALYSIS.md)。

原训练 JSON 没有初始朝向字段，因此额外将已有 `reset` 返回的初始朝向按 episode ID 保存到
metrics 文件旁的 `<metrics文件名去扩展名>_episode_metadata.json`，默认是
`training_metrics_episode_metadata.json`。不修改原 JSON 行或重新生成场景。
附加文件记录 metrics 的 SHA-256，离线绘图时会核对，避免旧文件误配到新的训练结果。
`--no-training-plots` 同时关闭训练曲线和结果汇总绘图，但仍保存这个附加文件供以后使用。

独立绘制已有蓝方训练结果（在 `v1` 目录运行）：

```powershell
$env:PYTHONPATH = "src"
python -m red_swarm_policy.analyze_blue_training_results outputs/blue_rl/train/training_metrics.json
```

可用 `--output PATH` 指定汇总图目录；移动过的朝向附加文件可用 `--episode-metadata PATH` 指定。
使用自定义 `--metrics-path` 时，附加文件始终跟随 metrics 路径；自动生成的图片仍跟随
`--training-plots-dir` 或训练 `--output`。旧日志没有朝向数据时标注 `<not recorded>`，
仍生成总体、弹数及飞行质量统计；不会凭空推断初始朝向。只有 JSONL 聚合记录而没有逐回合数据时，
请使用原训练曲线模块，不生成缺乏分组依据的结果统计。

原有 `flight_quality/episode_*.png` 继续按 `--flight-quality-plot-limit` 选择评分最低的回合。
其中转弯半径轴已改为覆盖全部有效记录的自适应范围；没有有效值时显示 unavailable。
指令过载与估计实际过载具有完整图例，右轴统一标为 `Load (g)`；航迹倾角与航向角速度分别使用
度和度/秒坐标轴。这些修正也适用于蓝方测试使用的同一逐回合绘图函数，原数值评估不变。

## 独立调用

在 `v1` 目录执行。绘图依赖 `matplotlib`，可通过 `python -m pip install matplotlib` 安装。
直接运行脚本只需要标准库和 matplotlib，无需导入训练包及 PyTorch：

```powershell
python src/red_swarm_policy/analyze_training.py outputs/blue_rl/train/training_metrics.json --output outputs/blue_rl/train/training_analysis --smoothing-window 20
```

已有训练环境也可按项目原来的模块方式运行：

```powershell
$env:PYTHONPATH = "src"
python -m red_swarm_policy.analyze_training outputs/blue_rl/train --output outputs/blue_rl/train/training_analysis
```

可传入训练目录、最终 metrics JSON、`episodes.json` 或训练 JSONL：

```powershell
python src/red_swarm_policy/analyze_training.py outputs/blue_rl/train/training.jsonl --output outputs/blue_rl/train/partial_analysis --smoothing-window 10 --dpi 180
python src/red_swarm_policy/analyze_training.py outputs/env_training_metrics.json
```

目录按顺序寻找 `training_metrics.json`、`env_training_metrics.json`、`training.jsonl`、
`episodes.json`。自定义文件名需要显式传入文件路径。若同一目录存在旧的完整 metrics 和新的
进行中 JSONL，分析当前进度时请显式传入 JSONL 路径。没有 `--output` 时，独立调用默认
使用输入文件同级的 `<输入文件名去扩展名>_analysis/`。

JSONL 中设备、配置、checkpoint 和最终汇总事件不作为训练迭代；读取时仅容忍末尾
尚未完成写入、没有换行的 JSON 记录，中间损坏或已换行的损坏记录会报错。
分析输出应使用独立目录；重复调用会覆盖该分析目录内的同名派生图表和摘要。

Python API：

```python
from red_swarm_policy.analyze_training import analyze_training

report = analyze_training(
    "outputs/blue_rl/train/training_metrics.json",
    "outputs/blue_rl/train/training_analysis",
    smoothing_window=20,
    dpi=160,
)
# 也可传入 {"iterations": [...], "episodes": [...]}，此时必须指定输出目录。
```

## 输出及统计口径

| 文件 | 内容 |
| --- | --- |
| `reward.png` | 逐回合 reward、日志窗口均值；分层训练中各已有奖励指标单独成图 |
| `loss.png` | 优先使用窗口 loss 或各 actor/critic loss；仅有逐回合日志时标注为 learner 快照 |
| `blue_escape_rate.png` | 逐回合结果的滑动逃脱率与累计逃脱率、训练窗口逃脱率、已有验证/快照存活比例 |
| `validation_escape_rate.png` | 存在课程验证结果时，绘制日志中的各场景验证曲线 |
| `learning_rate.png` | 已记录的学习率 |
| `gradient_norm.png` | 已记录的梯度范数 |
| `optimizer_diagnostics.png` | 已记录的 entropy、KL、clip fraction、C51 clamp 等技术指标 |
| `training_progress.png` | 已记录的吞吐、时间、replay 大小、更新数和显存指标 |
| `episode_length.png` | 已记录的回合步数、仿真时长 |
| `analysis_summary.json` | 来源、记录数、绘图文件、各指标有效/缺失数与基础统计、口径说明 |

前三张核心图始终生成；没有可用值时写明缺失，不绘制虚假的零曲线。其他图表按可用字段生成。
旧训练日志没有某项指标时无需修改或补写原文件。

逐回合曲线按 episode ID 排序；蓝方迭代曲线横轴使用已完成回合数，分层训练使用原迭代编号。
滑动窗口是最近 N 条记录（含当前记录），仅平均其中有效值；当前值缺失时曲线保留断点。
窗口均值之间的平滑不按回合数再加权。灰色为原始记录，蓝色为滑动平均。

蓝方完整回合中 `blue_survived` 的真值记为 1；蓝色为滑动逃脱率，橙色为有效回合累计逃脱率。
分层日志中的 `1 - average_damage_rate` 表示蓝方个体存活比例，**不是**
`1 - full_success_rate`。固定验证和 rollout 末的存活快照使用不同子图；
rollout 可能尚未终局，其存活比例不视为最终逃脱率。
逐回合 `mean_loss` 在原日志中实际是 learner 完成时的快照，模块不会将它解释为回合平均 loss。
预热期的 `None` loss、非有限数和越界比率均保留为缺失值。
