# 蓝方训练与测试中文图表

项目提供两个离线入口：

- `red_swarm_policy.plot_blue_results`：批量重绘训练曲线、训练分组结果、测试分组结果和飞行质量图。
- `red_swarm_policy.export_blue_report_charts`：按第二版报告章节批量导出中文 PNG、SVG、Word、Markdown、CSV 和 JSON。

两个入口都只读取已有结果，不启动仿真、不加载检查点，也不改变训练或测试数据。

## 运行

需要 Python 3.10 或更高版本，以及 NumPy、Matplotlib、python-docx 和 lxml。不要安装旧包 `docx` 或 `exceptions`；若已安装旧 `docx`，请先卸载，再强制重装 `python-docx`：

```powershell
python -m pip uninstall -y docx
python -m pip install --upgrade "numpy>=1.24,<3" "matplotlib>=3.7,<4"
python -m pip install --upgrade --force-reinstall "python-docx>=1.1,<2" "lxml>=4.9,<7"
```

在 `v1` 目录执行。当前训练和测试结果的完整重绘命令为：

```powershell
$env:PYTHONPATH='src'
python -m red_swarm_policy.plot_blue_results `
  --training-dir "outputs/blue_rl/train_1v1-1v3_train_1v1-1v3_missle_around_60_ep2000" `
  --evaluation-dir "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）"
```

默认覆盖输入目录中的派生图片：训练目录的 `training_analysis/`、`training_analysis/results/`、`flight_quality/`，以及测试目录的 `evaluation_analysis/`、`flight_quality/`。原始 JSON、JSONL、CSV、ACMI 和模型参数不会被改写。若要保留旧图，可指定统一的新输出目录：

```powershell
python -m red_swarm_policy.plot_blue_results `
  --training-dir "<训练结果目录>" `
  --evaluation-dir "<测试结果目录>" `
  --output-root "<中文图片输出目录>" `
  --flight-quality-plot-limit 10 `
  --smoothing-window 20
```

`--training-dir`（别名 `--train-dir`）和 `--evaluation-dir`（别名 `--eval-dir`）就是批量绘图的两个输入接口。测试目录名称含括号、中文或空格时，请保留双引号。

如需同时导出第二版报告图表，在上面的命令末尾增加：

```powershell
  --include-report-charts `
  --report-output "outputs/blue_rl/train_ep2000_eval_pt1660_charts_zh"
```

也可以单独调用报告导出模块：

```powershell
python -m red_swarm_policy.export_blue_report_charts `
  --train-dir "outputs/blue_rl/train_1v1-1v3_train_1v1-1v3_missle_around_60_ep2000" `
  --evaluation-dir "outputs/blue_rl/eval_1v1-1v3_ep500_missle_around_60（pt1660参数下的训练结果）" `
  --output "outputs/blue_rl/train_ep2000_eval_pt1660_charts_zh"
```

Linux 等环境需要安装中文字体，或增加 `--font "中文字体文件路径"`。也可以设置环境变量 `RED_SWARM_PLOT_FONT`。Windows 默认优先使用微软雅黑或等线；找不到中文字体时会直接报错，避免生成方框字符。原生分析图的 `--dpi` 默认 160，报告图的 `--report-dpi` 默认 220。

第二版报告导出还需要 `python-docx` 和 `lxml`；仅重绘原生训练、测试图时不需要 Word 依赖。

## 覆盖范围

| 报告章节 | 输出 |
| --- | --- |
| 1 测试条件 | 当前运行配置及统计口径表 |
| 2.1 与 3.1 指标要求 | 指标要求与核对口径表 |
| 2.2.1 飞行状态 | 六项已有飞行质量判据图；明确未执行专门标志性机动试验 |
| 2.2.2 回合终止 | 回合终止原因与来弹最终状态图 |
| 2.2.3 过载及高度边界 | 全轨迹数值统计与最大差分尖峰附近短片段图 |
| 3.2.1 训练损失 | 窗口曲线、后向平均及更新次数加权阶段表 |
| 3.2.2 训练回报 | 回报曲线、后向平均及阶段表 |
| 3.2.3 训练逃逸率 | 每窗口准确存活计数、逃逸率及后向平均 |
| 3.2.4 学习率 | 梯度范数与真实学习率双面板 |
| 3.2.5 回合时间 | 仿真结束时刻曲线；不误称墙钟耗时 |
| 3.3.1 总体结果 | 总体逃逸率及置信区间、脱靶量累计分布，各一张图 |
| 3.3.2 来弹数量 | 逃逸率及脱靶量分组表 |
| 3.3.4 初始朝向 | 逃逸率、脱靶量、过程时间、列联表及适用时的卡方统计 |
| 3.3.5 联合分层 | 来弹数与朝向热图；各组计数及置信区间 |
| 3.3.6 过程时长 | 激活与结束时刻、决策步数图；运行效率及学习冻结表 |
| 3.3.7 奖励构成 | 固定模型评估区块及奖励直方图、五项独立奖励图 |
| 4.1 与 4.2 | 技术指标核对表、训练前后阶段及独立测试分析数据表 |

明确排除 3.3.3 基线对比和 3.4 完整回合分析。2.2.3 的约两秒片段只用于数值边界诊断，不包含完整回合轨迹。

## 每张图的文件夹

```text
3.2.3_训练逃逸率/
  图.png
  图.svg
  关键数据.docx
  关键数据.md
  数据表01.csv
  数据表02.csv
  ...
  数据.json
```

所有可见图标题、轴名、图例和文档图注均为中文。Word 文档逐点保留训练曲线窗口值和平滑值；逃逸率图同时保存准确分子、分母和置信区间。Word 小数显示十位有效数字，CSV、JSON、Markdown 保留原数值精度。奖励直方图文档包含每个箱的精确计数和边界，额外 CSV 保存逐回合奖励；脱靶累计分布的全部坐标也另存 CSV。仅需表格的项目不会生成空图片。

根目录 `图表索引.md` 给出完整目录，`导出清单.json` 保存输入路径、文件摘要、实际样本数和产物列表。脚本会覆盖同名派生文件；不同实验请指定新输出目录，避免旧结果混入。失败会返回非零退出码，完整性检查失败不得当作成功导出。

## 当前数据的口径

- 当前训练目录记录 2000 回合、100 个日志窗口；原生训练曲线默认使用 20 条记录的后向平滑。第二版报告图默认使用 10 个日志点的后向平均，数据文档逐点给出实际覆盖回合数。
- 训练按日志的完成顺序统计，不能把并行任务编号排序后重新拼成窗口。阶段回报和逃逸率按完成回合数加权，阶段损失和梯度范数按优化器更新次数加权。平滑线是日志点均值，不是更新加权的阶段统计。
- 当前固定模型测试使用 pt1660 参数，1、2、3 枚来弹各 500 回合，共 1500 回合。逃逸率直接按实际存活标记计算，置信区间为 95% 威尔逊区间。
- 当前记录的命中半径为 3 米，报告中的“不大于 5 米”是另一个待核对指标，两者分别列出。
- 回合终止分类与来弹末态分母不同；失效原因缺失不能当作仍存活，必须核对 `red_alive`。
- `commanded_load_g` 是名义执行动作合成指令；速度差分过载用于数值诊断，不代表真实机动能力。
- 五个独立奖励项来自累计 `reward_component_sums`。战术塑形及其子项已包含在联合势函数内，不重复累加。
- 评估区块按回合编号排序，仅反映固定模型在不同样本中的波动，不判定训练收敛。

数据只读；不启动仿真，不加载检查点，不改变训练或测试逻辑。大型评估轨迹按行读取，按回合编号关联并核验运行标识，避免依赖并行完成顺序。

## 验证

```powershell
$env:PYTHONPATH='src'
python -m pytest tests/test_plot_localization.py tests/test_plot_blue_results.py `
  tests/test_training_analysis.py tests/test_evaluation_plots.py `
  tests/test_blue_flight_quality.py tests/test_report_chart_export.py
```

测试覆盖统计置信区间、后向平滑、并行完成顺序、重复编号与缺失值处理。实际导出时还校验输入回合数、轨迹关联及奖励加和。
