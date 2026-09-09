# 训练结果绘图

`src/red_swarm_policy/analyze_training.py` 是独立的训练日志后处理模块。
读取已有 JSON/JSONL，不执行环境步进、模型推理或参数更新。

## 默认调用

原有 `train_blue_rl` 和 `train_env` 命令无需修改。训练结束、原有文件全部保存、
环境 worker 关闭后，默认额外生成图表。原有 checkpoint、ACMI、逐回合结果、
metrics、JSONL、flight_quality、run_manifest 的保存代码和格式保持原样；
测试入口及 `train_env --validation-only` 不调用新模块。

- `train_blue_rl`：默认保存到 `--output` 目录下的 `training_analysis/`。
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
