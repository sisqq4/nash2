# Stage 2 训练日志指标分析

## 目标
基于实际 Stage 2 日志与指标计算代码，给出按任务优先级组织的监控指标说明、正常趋势、异常信号和联合诊断方法。

## 当前阶段
Stage 2 最终 P1 训练/确定性验证命令已固化，过程测试产物已完成可恢复清理

## 阶段
### Phase 1：检查现有 Stage 2 日志与字段集合
**Status:** complete

### Phase 2：核对任务指标、分配行为和 PPO 指标计算口径
**Status:** complete

### Phase 3：建立正常趋势、异常阈值和联合诊断表
**Status:** complete

### Phase 4：完成面向训练监控的分析说明
**Status:** complete

### Phase 5：复核 `stage2_v1_high_batch16` 当前训练结果
- [x] 盘点全部 seed、日志、固定验证与 checkpoint 产物
- [x] 对照训练/奖励/验证/最佳 checkpoint 代码复算关键指标
- [x] 检查高层 PPO、冻结低层、分配行为和数值稳定性
- [x] 按五级任务优先级判断结果是否合理并给出后续建议
**Status:** complete

### Phase 6：分析步骤二 `stage2_baseline_20260811` 结果
- [x] 对照复核文档还原步骤二假设、命令、分支判据和验收口径
- [x] 盘点 baseline 目录的运行完整性、场景覆盖、随机种子与产物谱系
- [x] 独立复算 capacity-aware / actor 基线的任务指标、方差与配对差异
- [x] 结合旧 Stage 2 固定验证与源码口径解释退化来源
- [x] 输出技术分析报告并完成计算、证据和结论校验
**Status:** complete

### Phase 7：执行复核文档步骤四 P0 改动
- [x] 4a 修正全目标成功的最高优先级奖励语义并增加反例测试
- [x] 4b 增加 assignment 侧 LR 调度、best 回滚与早停并验证低层位级不变
- [x] 4c 增加 assignment critic reward scale、回归测试与训练 smoke 验证
- [x] 运行完整 `pytest tests` 并记录变更与验收结果
**Status:** complete

### Phase 8：执行复核文档步骤五 P1 改动并交付步骤六命令
- [x] 核对步骤五/六完整约束、当前代码与步骤四已验收配置
- [x] 实现 5a 批量与 CPU pinned rollout 缓存、5b 熵调度、5c 切换惩罚/滞回、5d 势函数权重、5e 死计算清理
- [x] 增补/更新回归测试并运行完整 `pytest tests`
- [x] 在可用 GPU 上完成 2-3 iteration high-only smoke，验证冻结契约与数值稳定性
- [x] 确认步骤五验收通过后，将步骤六单种子训练命令写入 `TEST_CLI.md`（不启动正式训练）
**Status:** complete

### Phase 9：分析步骤六 `stage2_v2_p1_batch48/A2_stage1_seed_20260713` 结果
- [x] 对照步骤六命令与验收口径，锁定本次运行配置和完整性边界
- [x] 盘点训练日志、固定验证、checkpoint、早停/回滚和 manifest 产物
- [x] 独立复算任务指标、分场景趋势、PPO/分配行为与冻结契约
- [x] 对比 Stage 1、旧 Stage 2 v1 和 capacity-aware/actor 基线，判断 P0/P1 改动效果
- [x] 输出可复核的技术分析报告，明确是否可验收及下一步动作
**Status:** complete

### Phase 10：修复步骤六 no-target 自锁并准备重训
- [x] 修正高层 actor：no-target 槽不获得目标滞回 bonus，且仅对仍有效的物理目标施加 bonus
- [x] 增加 reset/no-target 与物理目标滞回回归测试
- [x] 运行定向测试、完整测试与同权重首决策复核
- [x] 清理本轮诊断派生产物及会阻塞同路径重训的失败运行产物
- [x] 复核并交付可直接执行的步骤六命令
**Status:** complete

### Phase 11：分析修复后重新执行的步骤六结果
- [x] 对照评审文档步骤六、重训命令和上轮缺陷边界确认本次运行谱系
- [x] 盘点训练/验证/调度/checkpoint/manifest 产物并验证运行完整性
- [x] 独立复算任务效果、分场景变化、训练—验证差距、分配行为与 PPO 稳定性
- [x] 对比 Stage 1、capacity-aware、旧 Stage 2 v1 和修复前失败 run，执行四项验收闸门
- [x] 生成并验证单一技术报告，给出是否进入步骤七及下一步建议
**Status:** complete

### Phase 12：补充 assignment-only 随机验证接口
- [x] 设计并实现高层 assignment 与低层 execution 独立确定性控制
- [x] 增加 validation-only 策略采样 seed、报告字段与参数校验
- [x] 补充串行/并行运行时和 validation-only 回归测试
- [x] 运行定向测试及完整测试，确认既有确定性验证语义不变
- [x] 将同固定集与匹配基线 seed 的两条验证命令写入 `TEST_CLI.md`
**Status:** complete

### Phase 13：分析 assignment-only 随机验证结果
- [x] 核对两个验证目录的完成状态、checkpoint、环境 seed、policy seed 与两层策略模式
- [x] 独立复算总体和24v4/5/6任务指标，确认汇总与分场景一致
- [x] 与同场景确定性 actor、capacity-aware 及步骤六训练 rollout 对齐比较
- [x] 量化随机—确定性差距及统计不确定性，判断 entropy 是否为当前主阻塞
- [x] 给出继续步骤七、补采样复验或调整 entropy 的明确下一步
**Status:** complete

### Phase 14：分析 it25 在 20263000 的确定性泛化结果
- [x] 核对validation-only完成状态、checkpoint SHA、环境seed与两层确定性模式
- [x] 独立复算总体及24v4/5/6指标并验证汇总一致性
- [x] 与20262000确定性actor、20263000 capacity-aware及随机actor同口径比较
- [x] 量化跨环境差异和统计不确定性，更新实际可用性判断
- [x] 在现有完整MCP技术报告上做依赖更新并完成验证/渲染
**Status:** complete

### Phase 15：固化 Stage 2 最终命令并清理过程测试产物
- [x] 盘点 `CLI_COMMAND.md`、评审文档、实际 run manifest 与 Stage 2 输出谱系
- [x] 确定最终保留清单和可恢复删除清单，保护最终 checkpoint、训练日志及正式验证结果
- [x] 将可直接执行的最终训练、续训和确定性验证命令更新到 `CLI_COMMAND.md`
- [x] 将过程 smoke、preflight、失败/临时分析产物移入系统回收站
- [x] 校验命令语法、文件引用、保留产物完整性及目录清理结果
**Status:** complete

## 错误记录
| 错误 | 尝试 | 处理 |
|---|---:|---|
| 当前目录无 Git 元数据 | 1 | 改用逐文件读取与显式变更记录，不执行任何回滚操作 |
| 24v4、180 s CPU 完整验证超过 120 s 工具超时 | 1 | 已确认无残留进程；改用真实 checkpoint 单更新 smoke test、5 s 有界轨迹和 CUDA 吞吐/显存基准覆盖训练前验证 |
| 搜索基准脚本时假设存在 `scripts/` 目录 | 1 | 仓库没有该目录；改为直接使用现有训练入口和 rollout 实现 |
| 候选 checkpoint 深检脚本假设存在 `assignment_critic_d_value_components` 序列化键 | 1 | 该维数为模型配置派生值；改为实例化 critic 并检查实际 value-components 输出维度 |
| 最终 `bash -n` 行范围多包含 Markdown 文本 | 1 | 按代码围栏精确缩小到 296-344、361-399、414-415 行重跑 |
| planning 技能 `check-complete.sh` 为 CRLF，直接 Bash 执行失败 | 1 | 不修改技能文件，通过 `tr -d '\r'` 后管道给 Bash 执行 |
| 当前目录无 Git 元数据（本轮再次确认） | 1 | 继续使用逐文件核对、测试和计划文件记录变更 |
| Stage 2 resume 补丁中的通用代码围栏上下文命中 Stage 1 首个命令块 | 1 | 立即读取命中位置，移除误加行并改用 `for each attempt` 唯一上下文写入 resume 命令 |
| 一次组合源码/测试搜索因最后一个 `rg` 无匹配返回退出码 1 | 1 | 前序源码内容已正常读取；后续对允许无匹配的探查单独加 `|| true`，不重复组合失败命令 |
| CLI help 冒烟遗漏 `PYTHONPATH=src`，conda 环境无法定位本地包 | 1 | checkpoint SHA 已成功核对；后续用与交付命令一致的 `PYTHONPATH=src conda run ...` 重新验证 CLI，不重复缺环境变量命令 |
| 新诊断脚本直接把旧汇总中的 `trial_count=96.0` 传给 `math.comb` | 1 | 在统计比较入口把成功数/试验数显式规范为整数；保留原始浮点率并重新执行完整断言 |
| 检查旧 report artifact 时假设 `manifest.sources` 必然存在，jq 对 null 迭代失败 | 1 | 旧artifact把canonical sources放在顶层；先读取实际顶层/manifest键，再按真实结构构建完整修订报告 |
| 格式/pytest 配置搜索同时传入不存在的配置文件导致 `rg` 退出码 2 | 1 | Python 编译已通过；改用 `rg --files` 先解析实际存在的配置文件，不重复该命令 |
| `rg --files` 未找到格式/pytest 配置文件并返回退出码 1 | 1 | 确认仓库没有这些配置文件；使用既有 `pytest` 命令和 Python 编译验证 |
| 汇总 5-update 分块时 jq 的管道/加法表达式优先级导致编译错误 | 1 | 改为先绑定分块索引变量，再计算 block start/end；不重复原表达式 |
| 首次构建MCP报告时读取的压缩前日志JSON超过单次shell输出边界，返回内容在约20k字符处拼接导致JSON解析失败 | 1 | 缩减为报告实际需要的字段，保证中间JSON小于输出边界后再验证；未调用可见渲染器 |
| MCP报告验证器拒绝JSONL路径作为卡片/图表来源，且612环境没有DuckDB | 1 | 不安装依赖；生成并实际运行SQLite兼容的复算SQL文件，作为所有报告数据集的可执行来源 |
| `612` 环境缺少 SciPy，首次统计复算脚本导入失败 | 1 | 不安装依赖；改用 Python 标准库实现 Wilson/Newcombe、Fisher 精确检验与二项检验，并保留公式口径 |
| 报告表格默认排序引用了未声明的辅助列 | 1 | 改为按已展示的 `dataset` / `blue_count` 列排序后重新验证 |
| MCP 报告要求图表/表格来源包含实际执行的 SQL，Python 复算路径不足以作为可展开数据源 | 1 | 为三份报告数据集补 SQLite 兼容的 `VALUES` CTE 查询并实际执行校验，再作为可审计来源 |
| 首次写入步骤四规划时错误假设 `progress.md` 标题为“进度日志” | 1 | 读取实际标题后改用“工作进度”上下文；第一次补丁原子失败，未修改文件 |
| 第二次步骤四规划补丁的既有错误行上下文少了一个空格 | 2 | 拆分为更小的精确补丁，不再复用整块多文件上下文 |
| 首次更新 Stage 2 v2 命令时多段补丁中的 resume 说明上下文不精确 | 1 | 原子补丁未修改文件；改为按标题、命令块和说明分别应用小补丁 |
| 步骤五首次批量更新 `train_env.py` 时 `_build_ppo_config` 上下文与实际换行不一致 | 1 | 原子补丁未修改文件；改为读取精确行段并拆成配置构建、恢复语义、argparse 三组小补丁 |
| 步骤五首次批量迁移 rollout 缓存时单环境 scenario metadata 的行格式与假设不一致 | 1 | 原子补丁未修改文件；按 helper、单环境 append、并行 append、batch 组装四块拆分 |
| 步骤五首轮完整测试 4 项仍断言 CUDA 常驻 batch，且测试构造值硬编码 CUDA device | 1 | 79/83 已通过；按新 CPU-pinned storage 契约更新设备断言与测试数据 device，并让私有 GAE 测试显式取得 training view |
| 步骤五定向测试引用了不存在的阶段转换测试函数名，pytest 在收集期退出 | 1 | 用 `rg` 解析实际函数名后重跑；本次未执行任何测试，不重复错误 node id |
| 阶段转换测试新增 CLI 参数的补丁命中了文件中更早的相同 `assignment-reward-learning-scale` 上下文 | 1 | 3/4 定向测试通过；按 `transitioned_args` 唯一上下文把三项步骤五参数移入正确命令块 |
| 文档/文件/进程联合盘点命令因 `rg '[r]ed_swarm_policy.train_env'` 未匹配进程而整体返回 1 | 1 | 前面的文档、文件时间与行数输出均有效；将“无训练残留进程”作为已核实事实，后续命令避免让预期空匹配影响整体退出码 |
| 首决策诊断对 5 checkpoints × 3 场景重复执行 post-boost reset，超过 120 s | 1 | 无部分输出、无残留 cell；改为只生成一个代表性初始观测并在五个 checkpoint 间复用，避免重复物理预热 |
| 新分析脚本把 v2 新增的 assignment 调度字段当成旧 v1 必有字段，读取旧日志时报 `KeyError` | 1 | 对旧日志使用语义正确的默认值 0；不改变 v2 字段的严格读取 |
| 分析脚本假设 latest checkpoint 已含终止后的 `stop_reason`，实际 latest 在 early-stop 事件前保存 | 1 | 改为显式记录字段是否存在；终止原因由最终 manifest/metrics 控制，不伪造 checkpoint 内值 |
| MCP 校验前通过 shell 读取 93KB artifact，桌面终端输出在约 37KB 截断并拼接，导致 JSON 解析失败 | 1 | 不重复大文件直读；把报告数据物化为本地 SQLite 表，source SQL 改为短 SELECT，并移除冗余顶层 sources，缩小 canonical payload 后再校验 |
| 精确清理失败 run 时直接删除命令被破坏性操作保护拒绝 | 1 | 未删除任何文件；改用系统回收站移动两个已解析并校验的目标，保证重训路径干净且可恢复 |
| `conda run -n 612 python -` 未执行 heredoc 中的 checkpoint 检查代码且无输出 | 1 | 不重复 stdin 方案；后续把检查写入分析脚本并以显式脚本路径运行 |
| 分析脚本输出路径保持相对路径，最终打印 `relative_to(ROOT)` 时抛 `ValueError` | 1 | 输出 JSON 已写出但最终回执失败；把 `OUT` 解析为绝对路径后重跑全脚本与断言 |
| 读取 artifact 的 shell 结果被命令回执头包裹，直接 `JSON.parse` 报 `Unexpected token E` | 1 | 未调用 validator；后续从首个 `{"surface"` 起截取 JSON payload 后再解析，不重复直接解析包装字符串 |
| 49.9KB artifact 单次 shell 回传在约34KB处截断，截取 payload 后仍无法解析 | 2 | 未调用 validator；改为按16KB读取并 base64 编码，各块在 JS 中按字节重组 UTF-8，避免回执截断和中文转码问题 |
| functions.exec V8 环境没有全局 `atob`，分块 base64 解码时报 `ReferenceError` | 1 | 未调用 validator；保留分块方案，改用内联纯 JS base64 解码器，不依赖 Node/浏览器全局对象 |
| functions.exec V8 环境同样没有 `TextDecoder`，已解码字节无法直接转 UTF-8 | 1 | 未调用 validator；增加最小 UTF-8 解码函数处理1–4字节序列，继续使用已验证的16KB分块边界 |

## 变更记录
| 文件 | 变更 |
|---|---|
| `src/red_swarm_policy/core/config.py` | 解耦高层 critic head mode |
| `src/red_swarm_policy/policy/critic.py` | 高层 critic 使用独立分量数 |
| `src/red_swarm_policy/train_env.py` | schema 14 与安全阶段转换 |
| `src/red_swarm_policy/env/observation.py` | 短时失锁预测目标槽 |
| `tests/test_smoke.py` | critic/失锁/schema 回归测试 |
| `tests/test_training_readiness.py` | stage1→stage2 转换门禁测试 |
| `CLI_COMMAND.md` | 新增 Stage 2 预检、三种子训练、续训和首轮监控命令 |
| `src/red_swarm_policy/train_env.py` | 本轮重置 Stage 2 局部计数并持久化 `stage_origin` |
| `tests/test_training_readiness.py` | 本轮覆盖 Stage 2 从 1 开始、周期 checkpoint 和同阶段恢复 |
| `CLI_COMMAND.md` | 明确 Stage 2 首次 iteration/计数与周期 checkpoint 编号 |
| `outputs/stage2_v1_high_batch16/analysis/iteration_1_28_report_source.sql` | 新增锁定日志前69行的只读报告数据复算脚本 |
| `STAGE2_V1_HIGH_BATCH16_REVIEW_20260811.md` | 新增阶段一/阶段二源码与结果独立合理性复核报告（快照 iteration 1-39） |
| `progress.md` | 更新当前状态并追加第二轮独立复核操作日志 |
| `STAGE2_V1_HIGH_BATCH16_REVIEW_20260811.md` | 补 §0「是否需要修改并重新训练」判定表、§5 执行计划（禁改边界+7 步+决策闸门+验收标准）；更正 §2.5/§3-P0-2 中「capacity_aware 基线零代码改动」的错误说法 |
| `src/red_swarm_policy/train_env.py` | 步骤 2 方案 A：新增 `ASSIGNMENT_MODES` / `VALIDATION_ASSIGNMENT_MODE_CHOICES` 常量 |
| `src/red_swarm_policy/train_env.py` | `_validation_plan` 增加第 4 个可选参数 `assignment_mode_override`，显式指定时覆盖返回的 `assignment_mode`，非法值抛 `ValueError` |
| `src/red_swarm_policy/train_env.py` | `_fixed_validation_metrics` 增加关键字参数 `assignment_mode_override` 并透传；训练循环内的验证调用同步传入 |
| `src/red_swarm_policy/train_env.py` | 新增 CLI `--validation-assignment-mode {auto,actor,capacity_aware}`（默认 auto 保持现状）与 `--validation-only`（要求 `--iterations 0` + `--resume-checkpoint`，一次固定验证后退出，不落盘） |
| `src/red_swarm_policy/train_env.py` | `--validation-only` 时不继承 checkpoint 的 `validation_config`，命令行为唯一权威；`validation_config` 新增 `assignment_mode_override` / `validation_only` 字段 |
| `tests/test_training_readiness.py` | 新增 3 个用例覆盖 plan override 语义、validation-only 端到端产物与参数校验（74 → 77 passed） |
| `TEST_CLI.md` | 写入步骤 2 的四组 capacity_aware/actor 基线验证命令、结果读取脚本与分支判据 |
| `outputs/stage2_baseline_20260811/analysis/analyze_stage2_baseline.py` | 新增只读标准库复算脚本：完整性、计数、区间、检验与分支判断 |
| `outputs/stage2_baseline_20260811/analysis/{run_comparison,scenario_comparison,training_validation}.sql` | 新增并实际执行的 SQLite 兼容报告数据源 |
| Data Analytics MCP 报告 | 已验证并渲染《Stage 2 步骤二基线验证分析》，含 1 图、3 表、技术结论、方法、限制与下一步 |
| `src/red_swarm_policy/env/types.py` | 步骤 4a：把全目标成功从 damage/waste 的低优先级跨度中剥离，失败/超时仍参与字典序校验 |
| `src/red_swarm_policy/core/config.py` | 步骤 4c：新增并校验 `assignment_reward_learning_scale` |
| `src/red_swarm_policy/training/mappo.py` | 步骤 4c：高层 GAE 使用缩放后的学习奖励，原始任务奖励与日志口径不变 |
| `src/red_swarm_policy/train_env.py` | 步骤 4b/4c：新增 assignment 调度、best restore、CLI/状态落盘与 stage1→stage2 受控配置覆盖 |
| `tests/test_smoke.py` | 新增成功奖励反例、高层 scale 等价性和 critic 梯度 512 倍缩放测试 |
| `tests/test_training_readiness.py` | 新增 assignment scheduler/restore、高层退化早停回滚及阶段转换/同阶段恢复测试 |
| `CLI_COMMAND.md` | Stage 2 新实验迁移到 `stage2_v2_p0_batch16`，显式加入步骤四奖励、scale、调度与早停参数 |
| `outputs/stage2_v2_p0_batch16/step4_preflight_20260811/{transition_metrics.json,run_manifest.json}` | 真实 Stage 1 best 的零更新 Stage 2 v2 过渡预检产物 |
| `src/red_swarm_policy/policy/{actor,critic}.py` | 步骤 5a/5e：输入容器支持 non-blocking/pin；删除高层 `_distributions()` 死前向并加入当前目标滞回 logits |
| `src/red_swarm_policy/training/{rollout,mappo}.py` | 步骤 5a：rollout 快照逐步回 CPU、CUDA 时整批锁页；PPO 按时间块异步搬 observation，high-only 跳过冻结低层前向 |
| `src/red_swarm_policy/core/config.py` | 步骤 5b/5c：assignment entropy 默认 0.001；新增并校验 `assignment_stickiness_logit_bonus` |
| `src/red_swarm_policy/env/types.py` | 步骤 5d：新训练 high potential 默认权重提高到 512 |
| `src/red_swarm_policy/train_env.py` | 步骤 5b-5d：新 CLI、默认值、Stage 1→2 受控覆盖与普通 Stage 2 resume 禁止漂移 |
| `tests/{test_smoke,test_training_readiness}.py` | CPU-pinned/minibatch、滞回、dead-forward、冻结低层零前向及 checkpoint 恢复回归测试 |
| `outputs/stage2_v2_p1_batch48/step5_smoke_streamed_20260811/**` | GPU 9 上 48-env × 3-iteration 最终步骤五 smoke 产物 |
| `TEST_CLI.md` | 步骤六单 seed、48-env、最多 50 iteration 正式训练、安全续训和首轮冻结检查命令 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/analyze_stage2_v2_p1.py` | 新增步骤六只读复算脚本：任务趋势、验收闸门、checkpoint 深检与首决策消融 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/analysis_summary.json` | 保存完整复算结果与可审计结论 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/build_report_artifact.py` | 生成并复核技术报告 canonical artifact、SQLite 数据源与图表映射 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/{report_artifact.json,report_data.sqlite,report_source.sql,chart_map.json}` | 报告 payload、已执行查询数据源及可视化 QA 记录 |
| Data Analytics MCP 技术报告 | 已验证并渲染《Stage 2 步骤六单种子重训分析：未通过，no-target 滞回导致确定性策略自锁》 |
| `src/red_swarm_policy/training/rollout.py` | validation runtime 与并行 episode evaluator 支持 assignment/execution 独立 deterministic 覆盖 |
| `src/red_swarm_policy/validate_checkpoint.py` | 串行 trial 透传两层独立确定性控制，保留旧全局 deterministic 默认语义 |
| `src/red_swarm_policy/train_env.py` | 新增 validation-only assignment 随机开关、restore 后策略 seed、参数门禁及审计字段 |
| `tests/test_training_readiness.py` | 新增独立控制、RNG 复现、进程池端到端与非法 CLI 组合回归；完整测试 89 passed |
| `TEST_CLI.md` | 新增固定集 20262000 与匹配基线集 20263000 的两条 assignment-only 随机验证命令 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/analyze_assignment_stochastic.py` | 新增两环境块随机验证复算脚本：谱系、指标、Wilson/Newcombe/Fisher及对照断言 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/assignment_stochastic_analysis_summary.json` | 保存37/192合并结果、分规模结果、统计比较与下一步判定 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/build_assignment_stochastic_report.py` | 在原20-block报告基础上构建包含随机验证证据的完整修订报告 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/{assignment_stochastic_report_artifact.json,assignment_stochastic_report_data.sqlite,assignment_stochastic_report_source.sql,assignment_stochastic_chart_map.json}` | 通过MCP validator并成功渲染的报告payload、可执行数据源与图表QA映射 |
| Data Analytics MCP 技术报告 | 已验证并渲染《Stage 2 步骤六补充诊断：随机分配仅19%，策略分布确认是主要阻塞》 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/analyze_deterministic_generalization.py` | 新增it25确定性独立环境块复算脚本：谱系、重聚合、区间与基线比较 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/deterministic_generalization_analysis_summary.json` | 保存95/96独立结果、187/192合并结果、分规模指标和可用性判定 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/build_deterministic_generalization_report.py` | 在完整随机诊断报告上新增确定性泛化证据并更新受控可用性结论 |
| `outputs/stage2_v2_p1_batch48/A2_stage1_seed_20260713/analysis/{deterministic_generalization_report_artifact.json,deterministic_generalization_report_data.sqlite,deterministic_generalization_report_source.sql,deterministic_generalization_chart_map.json}` | 通过MCP validator并渲染的30-block报告、可执行数据源与图表QA映射 |
| Data Analytics MCP 技术报告 | 已验证并渲染《Stage 2 步骤六补充诊断：确定性策略跨环境达97.40%，随机分布仍是主要阻塞》 |

## 本轮决策
| 决策 | 原因 |
|---|---|
| 只在 `low_only → high_only` 阶段转换时重置局部计数 | 普通同阶段 checkpoint 恢复必须继续原 Stage 2 编号 |
| Stage 1 原计数写入 `stage_transition.source_*` | 新阶段从 1 编号，同时保留可审计训练谱系 |
| Stage 2 场景 RNG 使用重置后的局部 iteration | 使新 Stage 2 seed 的采样序列从本阶段第 1 批自然开始 |
| 在 checkpoint `training_state` 中持久化 `stage_origin` | Stage 2 后续断点续训仍能追溯 Stage 1 来源计数 |
| 步骤四直接执行，不做步骤 3.5 | 步骤二 capacity-aware 合并全成功率 83.33%（<90%），满足文档的直接进入步骤四分支 |
| 步骤四仅修改 4a/4b/4c 与测试/命令 | 遵守复核文档禁改边界，不改网络结构、阶段一产物、环境物理或既有输出 |

## 本轮分析原则
| 原则 | 原因 |
|---|---|
| 固定验证指标优先于训练 rollout 指标 | 验证可跨 update 与 seed 比较，训练批次场景和策略采样均随机 |
| 严格按全目标毁伤→无效损失→时间→控制消耗排序 | 与项目的训练、验证及最佳 checkpoint 选择契约一致 |
| 单指标不下结论，使用任务结果+分配行为+PPO 稳定性联合判断 | 能区分探索期、分配退化、价值网络失真和实现异常 |
| 结论必须以本次目录的最新完整快照为准 | 训练可能在上一轮检查后继续写入，旧的 iteration 1–14 结论不能直接外推 |
| 报告面向技术读者并保留代码/产物证据路径 | 用户要求结合代码判断合理性，需要能复核口径和结论 |
