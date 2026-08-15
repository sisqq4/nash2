# 工作进度

## it25 确定性泛化验证结果分析（2026-08-14）

- 用户已完成当前it25 best在环境seed 20263000上的确定性actor验证，开始Phase 14。
- 本轮采用metric-diagnostics → validate-data → visualize-data → build-report路线；延续technical + MCP app单一报告面，在上一版完整artifact上做窄依赖更新，不创建并行HTML。
- 已重新读取planning-with-files及Data Analytics诊断、验证、可视化、technical report和MCP app合同，并执行session catch-up；本轮只读分析现有结果，不启动训练或追加验证。
- 新验证目录完成性与模式核对通过：it25 best、env 20263000、两层deterministic、actor assignment、24v4/5/6各32场；总体95/96=98.96%，分规模32/32、32/32、31/32。
- 独立环境块比20262000固定集多3场成功，且24v5/24v6计数完全一致；开始复算Wilson/Newcombe/Fisher并与同环境capacity-aware及随机actor比较。
- 已新增并执行`analyze_deterministic_generalization.py`，完成metrics/manifest/JSONL重复值、checkpoint SHA、全部源码hash、分场景重聚合及有限值断言。
- 复算结果：独立块95/96（95% CI 94.33%–99.82%），两确定性块合并187/192=97.40%；相对20262000差+3.13 pp且无退化证据，相对同seed capacity-aware显著高+10.42 pp。
- 当前判断更新为：支持在现有仿真包线内进行两层deterministic的受控试运行；不支持随机/unrestricted使用，且原步骤六闸门3仍阻止进入步骤七。
- 已基于上一版完整25-block artifact构建依赖更新：保留全部旧块/卡/图/表/source/dataset，新增确定性泛化3卡、1张分组柱图、1张统计表和5个叙事/证据块。
- 新完整报告为30 blocks、10 cards、6 charts、4 tables、14 sources、13 datasets；SQLite新增三张表实际物化为1/12/3行，Python编译和结构断言通过。
- MCP artifact validator返回`ok=true`、13 datasets、14 sources、snapshot=`ready`；验证成功后仅调用一次renderer，完整报告渲染成功。
- Phase 14完成。最终评级为Share with caveats：支持当前仿真包线内两层deterministic、带模式断言和capacity-aware回退的受控试运行；不支持随机/unrestricted使用，步骤六闸门3和步骤七仍保持阻塞。

## assignment-only 随机验证结果分析（2026-08-14）

- 用户已完成环境 seed 20262000/20263000、policy seed 20265001 的两组 assignment-only 随机验证，开始 Phase 13。
- 本轮按 metric diagnostics → validate-data 路线：先验证产物谱系与指标模式，再与确定性 actor/capacity-aware 同场景对齐，最后判断是否需要调 entropy。
- Data Analytics 路由要求诊断结果使用单一 durable report 面；本轮选择 technical + MCP app，并补入 visualize-data 与 build-report，不创建并行 HTML。
- 已完整读取 technical report、MCP app、analytics core 与 visualization 合同；报告将更新此前完整技术报告，而不是另建缩略副本，并至少加入一张原生对比图。
- 已执行 planning session catch-up；不会启动训练或追加验证，只读分析现有结果。
- 两个验证目录完整性和配置语义已核对：均正常 validation-only，best SHA/policy seed一致，高层随机、低层确定性。初步结果为19/96与18/96，下一步对齐基线并复算区间/检验。
- 已对齐capacity-aware：同环境分别75/96与85/96，随机actor分别低56场与67场；it25训练随机rollout为27.08%，与新结果同量级。当前it25缺少20263000确定性actor结果，作为泛化 caveat 保留。
- 已新增标准库复算脚本；首次运行发现旧baseline汇总把trial_count序列化为96.0，已在Fisher入口规范为整数后重跑，未改变任何源结果。
- 复算已通过：合并37/192=19.27%（Wilson 95% CI 14.32%–25.43%）；同固定集随机比确定性低76.04 pp，保守独立Fisher p=3.25e-16。开始基于旧完整artifact构建修订报告。
- 关键分解完成：24v4/5/6合并随机成功率42.19%/14.06%/1.56%；两个环境seed随机结果无差异证据。报告采用全量修订而非缩略副本，保留旧20 blocks并新增1图1表和对应解释。
- 已核对旧artifact实际结构、7个数据集和4图chart map；修订版将保留旧块/卡/图/表/source ID不变，新增独立source与3个快照数据集，避免新证据污染旧来源口径。
- 标准库复算脚本最终执行和Python编译均通过；输出固定为37/192=19.27%，并断言不进入步骤七。
- 已生成完整修订报告：25 blocks、7 cards、5 charts、3 tables、10 datasets、11 sources；新增SQLite三张表实际物化为1/20/5行。
- MCP artifact validator返回`ok=true`、snapshot=`ready`；按合同仅在验证成功后调用一次renderer，渲染成功。
- Phase 13完成。结论为步骤六闸门3被直接反证：当前主要阻塞是高层策略分布过宽；下一步先补assignment采样温度接口并对T=0.5/0.25做短验证，不启动步骤七或直接长时重训。

## assignment-only 随机验证接口（2026-08-14）

- 用户要求补齐只随机高层 assignment、保持低层 execution 确定性的验证接口，并交付两条验证命令。
- 已重新读取 planning-with-files 技能、执行 session catch-up，并建立 Phase 12。
- 设计约束：复用 validation-only 的进程池；不使用会同时随机化两层且串行的 `validate_checkpoint.py --stochastic`；增加独立策略采样 seed，避免 checkpoint RNG 恢复覆盖实验 seed。
- 已完成调用链盘点：需要同步修改单环境 runtime、并行 episode evaluator、fixed validation 汇总、validation-only CLI/报告及其端到端测试；既有调用通过可选覆盖参数保持兼容。
- 已确定安全边界：新随机开关只用于 validation-only actor；必须显式给策略采样 seed；训练期 checkpoint 选优仍保持两层 deterministic。
- 已修改 `HierarchicalPolicyRuntime`、`evaluate_parallel_episodes` 与串行 `_run_trial`：新增两层独立 deterministic 覆盖，默认行为保持向后兼容；下一步接入 train_env CLI、RNG 与报告字段。
- 已接通 train_env 新参数、restore 后策略 RNG、两层 policy mode 报告与 validation config；编译通过，4 项定向测试通过（含进程池 validation-only）。best checkpoint SHA 仍为 `ac210d75...85cbb`。
- 已在 `TEST_CLI.md` 新增环境 seed 20262000/20263000 两条 assignment-only 随机验证命令；两段 Bash 语法和 CLI help 均通过检查。
- 完整回归 `pytest -q tests` 通过：89 passed in 49.04 s。Phase 12 完成，未启动任何正式验证或训练任务。

## 步骤六修复后重训结果分析（2026-08-14）

- 已完整读取 planning-with-files、Data Analytics 路由、metric diagnostics、visualization、technical report 与 validate-data 工作流，并执行 session catch-up。
- 采用 technical 受众、单一 MCP app report 交付；先确认修复后运行谱系与完整性，再独立复算四项验收闸门。
- 已建立 Phase 11；本轮保留此前失败 run 的分析记录，仅把它作为对照，所有当前结论重新从现有输出产物计算。
- 已新增并运行 `analysis/analyze_rerun.py`：解析 45 个 iteration、9 个验证点、四类对照与 11 个 checkpoint，复算精确 Spearman、Wilson/Newcombe、Fisher、PPO/冻结/哈希契约并生成 `rerun_analysis_summary.json`。
- 当前结论：修复后任务效果显著恢复，best=95.83%、稳定性/基线/KL 三项通过；但随机训练 rollout 与确定性验证的全成功率差未收窄，故原步骤六四闸门严格总体仍为 fail。下一步诊断 entropy 口径与 gap 来源，并构建技术报告。
- 已核对 entropy 与采样源码：训练高层随机采样、自回归联合熵按高层时间步平均；固定验证为确定性 actor。报告面已确认可调用 artifact validator/renderer，将按 MCP app 单一交付面生成，不并行创建 HTML。
- 已生成技术报告 SQLite 快照、canonical artifact 与 chart map；最终报告为 20 blocks、4 charts、2 tables、7 datasets、8 sources，artifact validator 与 renderer 均成功。
- 最终 QA 通过：验证轨迹34行、分场景16行、训练/验证差18行、entropy 45行、闸门4行、checkpoint诊断5行；报告与 `rerun_analysis_summary.json` 的核心数值一致。
- Phase 11 完成。严格判定为步骤六未通过：闸门1、2、4通过，闸门3因随机 rollout—确定性验证差距未收窄而失败；当前建议先做随机 actor 配对验证，不进入步骤七。

## 步骤六缺陷修复与重训准备（2026-08-12）

- 用户要求修复已定位的 no-target 滞回自锁、清理无关产物并重新执行步骤六。
- 已重新读取 planning-with-files 技能并完成 session catch-up；Phase 10 开始。
- 当前目录无 Git 元数据；将通过精确补丁、哈希/测试和显式文件清单追踪变更，不触碰无关用户文件。
- 已盘点失败运行目录：原始训练产物位于 `seed_20260810/`，本次分析派生产物位于 `analysis/`；先完成代码与测试修复，再按重训路径需求精确清理。
- 已修改高层 actor：滞回向量与有效目标 mask 相交，并显式将 no-target 槽清零。
- 已补两个缺失回归：no-target 不受奖励、不可用的历史物理目标不受奖励；原有测试继续覆盖有效物理目标概率提升与无额外 head 前向。
- CPU 定向回归 3/3、完整 `pytest tests` 86/86 通过。
- 原始 Stage 1 best 的 24v4 固定种子真实首决策复核通过：初始化当前槽计数 `[24,0,0,0,0]`，修复后的确定性 actor 输出 `[8,4,4,4,4]`。
- 已将修复前失败 run `seed_20260810/`（约 298 MiB）及其 `analysis/` 派生产物（约 284 KiB）移入系统回收站；目标根目录复核为空，可恢复但未永久擦除。
- `TEST_CLI.md` 已补充必须从原始 Stage 1 checkpoint 重训的说明；新训练代码块通过 `bash -n`，Stage 1 checkpoint/quality gate 存在，输出防覆盖路径不存在。Phase 10 完成。

## 步骤六结果分析日志（2026-08-12）

- 已运行 planning session catch-up；仓库已有前序计划文件，采用追加 Phase 9 的方式继续，不覆盖历史结论。
- 已选择 metric diagnostics → visualization → technical report 路线；下一步完整读取相关技能说明和步骤六文档，再盘点输出目录。
- 已完整读取 Data Analytics 路由、metric diagnostics、visualization、report building 与 planning-with-files 说明；本轮按技术受众生成一个可复核报告，图表用于固定验证离散比较、训练趋势和 PPO/调度诊断。
- 已核实 run 于 iteration 25 正常触发 assignment early stop；调度/回滚链路本身有事件证据，但 5 个固定验证点全部为零任务效果，开始定位 actor 确定性 no-target、checkpoint 恢复或验证链路中的根因。
- 已定位到可疑代码：stickiness bonus 未排除 no-target 槽；下一步检查 `current_assignment` 的重置编码，并用 best/periodic checkpoint 做首决策 logits/action 对照（bonus=1 与 bonus=0）。
- 首轮首决策实验因重复 15 次 post-boost 预热超时；已改为单场景观测复用方案，不重复原实验结构。
- 单场景复用实验通过：bonus=1 令五个 checkpoint 首决策全部 24/24 no-target；bonus=0 令同权重首决策覆盖 4 个目标各 4 枚。根因已闭环，开始量化训练侧学习、PPO稳定性和冻结/回滚契约。
- 复算脚本已通过并生成 `analysis_summary.json`；开始按 validate-data 逐项核对口径、基线、checkpoint 与报告图表，当前最高严重级问题仍是 no-target 自锁。
- 报告 canonical artifact 已生成（19 blocks、4 charts、2 tables），但首次传给 MCP validator 时 93KB JSON 被终端输出边界截断；正在把重复的 VALUES SQL 改为已执行的 SQLite 表查询以缩小 payload，未产生可见破损报告。
- 已将报告数据物化为本地 SQLite 表并把 source SQL 改为短 SELECT；完整 artifact 再次验证通过并只渲染一次最终 MCP 技术报告。
- 最终 QA 通过：两份 Python 脚本编译成功；只读复算断言总体 fail、KL-stop=0、低层位级冻结；SQLite 7 表行数 1/7/75/25/10/4/6；报告 19 blocks、4 charts、2 tables、snapshot ready。Phase 9 完成。


## 步骤四执行日志（2026-08-11）

- 完整读取 planning-with-files 技能并执行 session catch-up。
- 读取 `STAGE2_V1_HIGH_BATCH16_REVIEW_20260811.md` 的步骤四及禁改边界。
- 确认步骤二 `<90%` 决策闸门已经满足，当前无残留训练进程且目录无 Git 元数据。
- 建立 Phase 7；下一步检查 4a/4b/4c 当前实现与既有测试口径。
- 已定位 4a/4c 缺口：成功奖励被错误纳入低优先级跨度；高层 GAE 没有 reward scale，低层已有可对称复用的实现。
- 已审计 4b 全链路：当前 scheduler、restore、CLI、checkpoint/manifest/metrics 全为 execution 命名，主循环在 `high_only` 下仍错误调用它；决定保留低层兼容路径并补 assignment 对称路径与模式路由。
- 已确认 checkpoint 向后兼容与测试模板：assignment restore/scale 可镜像已有 execution 实现，但需为 stage1→stage2 增加受控的高层 scale 覆盖语义。
- 已确定步骤四配置：成功奖励 512、高层 critic scale 1/512；高层使用独立 plateau/restore/early-stop 状态并将后续命令迁移到新的 Stage 2 v2 输出目录。
- 已完成 4a/4b/4c 第一版代码与单元/端到端测试编写，Python 编译通过；下一步运行定向 pytest 并修复回归。
- 定向测试 9/9 通过；新增的 critic 梯度回归确认高层 reward scale 令 preclip 梯度精确缩小 512 倍。
- 完整测试最终 83 passed；Stage 2 v2 新训/续训命令 Bash 语法通过。
- 使用真实 Stage 1 best 与质量门完成零更新 Stage 2 v2 过渡预检，配置、5 分量 critic 重建、计数清零和禁改哈希全部通过。Phase 7 完成。

## 当前状态
已完成复核文档步骤二的 `stage2_baseline_20260811` 独立分析。结论：capacity-aware 两个 seed 块为 78.13%/88.54%，触发预设 `<90%` 分支；it10 actor 的增量真实，但新 holdout 仅 +4/96、尚未单独显著，且不改变 it10 后退化与必须修复后重训的结论。技术报告已通过 Data Analytics artifact 校验并渲染。

## 操作日志

- 2026-08-11：开始分析 `outputs/stage2_baseline_20260811`（复核文档步骤二）；采用 metric diagnostics → visualization → technical report → validation 路线，先锁定步骤二判据与产物口径，再独立复算结果。
- 2026-08-11：完成步骤二文档口径与四组 `validation.json` 初检；actor 原 seed 精确复现 96.875%，capacity-aware 两集合分别为 78.125%/88.542%，初步落入预设 `<90%` 分支；待核验 manifest/console、独立计算置信区间与最终报告。
- 2026-08-11：完成指标口径源码核对、manifest/console/checkpoint 完整性检查与标准库统计复算。确认 `<90%` 分支；独立 holdout 的 actor 优势为 +4/96，方向一致但单批次未显著，需在报告中与原选优集合强优势分开表述。
- 2026-08-11：新增并实际运行 `outputs/stage2_baseline_20260811/analysis/analyze_stage2_baseline.py`，复算四组/分规模/合并指标、Wilson/Newcombe/Fisher/二项检验与 manifest/checkpoint/source 完整性；脚本编译和断言全部通过。
- 2026-08-11：为报告新增 3 份 SQLite 兼容查询并全部实际执行（4/6/7 行）；MCP 报告 artifact 校验通过（3 datasets、5 sources、ready），最终渲染成功。Phase 6 完成。
- 已读取 planning-with-files 技能说明并初始化检查计划。
- 已列出仓库源文件、测试与第一阶段输出；确认 3 个 A2 独立种子及完整验证产物。
- 已核对 AGENTS 需求、现有 CLI 文档和两份阶段一分析；锁定阶段二输入为 seed 20260713 的 best 检查点。
- 已验证 checkpoint 哈希和内部 schema，并定位高层转换、冻结及固定验证相关代码入口。
- 已阅读高层转换与主循环：确认低层确定性执行、高层独占更新、阶段 best 重置机制；正在核查 resume 配置/RNG 语义。
- 已对照 manifest 验证训练核心源码完整性，并加载 checkpoint 检查所有模型张量有限。
- 已审计高层自回归容量分配、联合 log-prob、PPO ratio 与高层 GAE 时间尺度；这些核心契约一致。
- 发现阻断问题：stage1 的 scalar critic 配置会错误传播给高层 critic；确定采用高低层 critic 头解耦并仅重建未训练高层 critic。
- 已核对高层奖励、字典序门禁、观测和 critic 输入；开始检查短时失锁与周期重决策的稳定性契约。
- 已完成代码修复和回归测试；定向测试 5 passed，全套测试 74 passed。
- 已使用真实 stage1 best checkpoint 完成 24v4 单更新 smoke test：仅高层网络更新，低层 actor/critic 位级不变，高层 critic 为 5 分量，输出参数全部有限。
- 24v4、180 s CPU 完整验证超过 120 s 工具时限，已确认无残留进程；5 s 有界策略轨迹正常。
- 正在评估 high_only 采样缓存的 CUDA 显存和吞吐，以确定第二阶段安全并行度。
- 已完成 GPU 9 上 4 环境完整 high_only 基准：1154 s、峰值 2360 MiB，更新成功；正在以 32 环境短轨迹实测并行扩展和峰值。
- 32 环境短基准成功：49 s、峰值 1340 MiB；32×40 完整更新成功但峰值达 18016 MiB，且有 2/32 轨迹因事件驱动重决策较多而未终止。
- 为覆盖完整 episode 并保留显存余量，正式候选改为 16 环境 × 64 步，正在执行精确配置验证。
- 精确 16×64 候选验证成功：1369.6 s、峰值 9212 MiB、16/16 episode 完整终止；checkpoint 结构、有限性、冻结与更新计数全部通过。
- 已在 `CLI_COMMAND.md` 新增 Stage 2 预检、三种子训练、续训和首轮监控命令；质量门、CLI 参数和 Bash 语法均已检查。
- 最终完整测试：74 passed in 33.03s；GPU 9 已释放，正式 Stage 2 输出目录尚不存在。
- 2026-08-10：确认当前 Stage 2 继承 Stage 1 `completed_iterations=25`，新阶段日志错误地从 26 开始；开始实现阶段局部计数清零和来源计数留档。
- 本轮仓库状态检查再次确认目录没有 Git 元数据；不执行任何回滚，通过显式文件记录追踪修改。
- 已完成计数与恢复契约审计：周期 checkpoint、随机场景采样和 iteration 日志均由 `start_iteration` 驱动；普通同阶段恢复已有连续编号测试。
- 已修改 `train_env.py`：阶段转换的 `start_iteration` 和累计更新数归零，新增可跨 Stage 2 checkpoint 续训保留的 `stage_origin` 元数据。
- 已扩展阶段转换回归测试：实际执行 1 个小型 high-only update，检查首条 iteration=1、`iteration_000001.pt`，并零步恢复该 Stage 2 checkpoint 检查编号连续与来源元数据保留。
- `train_env.py` 与定向测试文件均通过 `py_compile`；配置搜索因传入不存在文件返回退出码 2，已记录并改用先列文件的方式。
- 阶段转换定向回归测试通过：1 passed in 4.29s。
- 训练就绪测试通过：22 passed in 25.81s。
- 完整测试套件通过：74 passed in 34.70s。
- 使用真实 Stage 1 best checkpoint 完成零步转换检查：Stage 2 局部计数均为 0，来源计数 25/25/20/20 正确保留。
- 已确认真实 checkpoint 的高层 actor/critic optimizer 状态为空；转换后的高层训练没有 Stage 1 高层优化历史泄漏。
- 已更新 `CLI_COMMAND.md`，明确首条 Stage 2 update 必须为 iteration 1、第五次生成 `iteration_000005.pt`，同阶段恢复继续编号。
- 本轮实现、定向测试、完整回归、真实 checkpoint 转换检查和命令文档复核全部完成。
- 2026-08-11：开始基于实际日志字段和计算代码梳理 Stage 2 重点监控指标。
- 已发现 seed 20260810 的 14 次真实 update 和两次固定验证；开始核对每类指标计算口径。
- 已核对固定验证字典序、分配行为诊断和高层 PPO 的 KL/clip/更新/EV 口径。
- 已提取 iteration 1–14 趋势及 iteration 5/10 固定验证，开始形成分级监控与联合异常诊断规则。
- 已完成 Stage 2 日志指标优先级、字段含义、趋势阈值、联合异常诊断和当前训练快照分析。
- 2026-08-11：收到对 `stage2_v1_high_batch16` 当前结果合理性的代码结合分析请求；新增 Phase 5，准备重新盘点最新产物并独立复算关键指标。
- 已确认训练仍在运行；锁定 2026-08-11 08:47 CST 的前 69 行/iteration 1–28 作为本轮一致分析快照。
- 当前仅 seed 20260810 有 Stage 2 产物；已有周期 checkpoint 5/10/15/20/25，best 仍为 iteration 10。
- 已抽取 iteration 1–28 的任务、分配和 PPO 指标，并完整复核五次 96 场固定验证；确认 best 留在 iteration 10 符合五级字典序，后期固定验证回落主要集中在 24v6。
- 5-update 分块汇总首次 jq 表达式因运算优先级编译失败，已记录并改用预绑定分块索引的实现。
- 已完成训练分块、场景覆盖、冻结网络和 checkpoint 语义复核；确认单点 KL 超阈值后恢复，冻结低层位级不变。
- 发现设计层主要风险：训练 reward 线性优化平均毁伤而没有全成功终端项，不能在策略分布层面严格保证“全目标成功率第一”；此外反复复用同一固定验证集选 best，最终必须补独立 Stage 2 holdout。
- 已计算总体与24v6成功率的Wilson区间并完成三张报告图的可视化契约；准备生成技术型MCP报告。
- MCP报告首次组装未进入验证器：中间日志JSON超过shell返回边界后无法解析；已定位为读取字段过多，改用最小字段集重建。
- 报告验证器要求原生卡片/图表/表格具有SQL来源；未安装DuckDB，因此用SQLite复算脚本物化同一快照并在内存中验证全部视图行数和汇总值。
- 技术报告已通过数据/结构验证并成功渲染；报告固定使用日志前69行（iteration 1–28），不会与仍在继续写入的训练日志混用。
- 最终判定：iteration 10 是当前固定验证集上的合法 best；iteration 20/25 出现真实但统计区间重叠的回落。单种子、重复复用同一验证集选优以及 reward 与“全目标成功率第一”并非严格同构，是当前不能验收的三个主要限制。
- 2026-08-11（第二轮，独立复核）：收到"结合代码分析阶段一/阶段二结果是否合理"的请求；以 iteration 1-39（日志前 93 行）为快照重新盘点，`pytest tests` 74 passed。
- 已全量阅读 env/policy/training/core 与 `train_env.py`；确认物理、导引头、命中判定、自回归容量约束采样、联合 log-prob、GAE 时标缩放与 high_only 冻结契约均正确。
- 已用日志逐 iteration 验证高层回报解析式：`episode_high_reward_mean` 与 `512·D − 64·W − 2·T` 残差 ≤0.02，证明势函数塑形对回报贡献严格为 0、终端成功项缺失。
- 已实测 `validate_lexicographic_priority` 在 24 红下把 `terminal_success_reward` 上界锁死在 0.666，确认目标函数缺陷是结构性的而非参数遗漏。
- 已确认阶段二无早停（默认 0 且未传参），且 LR plateau/restore-best 只作用于冻结的 execution optimizer，高层无任何自适应控制。
- 已用趋势检验修订前一轮结论：平均毁伤率 it10 后 6 点严格单调递减（Spearman ρ=−1.000，p≈0.0014），全成功率 it(10,15) 186/192 vs it(30,35) 174/192（z=2.55，p≈0.011），退化统计显著。
- 已定位阶段二最大评估缺口：无基线对照，而 `capacity_aware` 启发式分配在 `training/rollout.py:416` 已现成可用。
- 已输出 `STAGE2_V1_HIGH_BATCH16_REVIEW_20260811.md`，含结论、证据路径、P0/P1/P2 建议与"已核实无问题"清单。
- 已补充执行计划：在报告中新增 §0 阶段一/阶段二「是否修改重训」判定表与 §5 供 agent 按序执行的 7 步计划（含禁改边界、阻塞性决策闸门、分支判据、逐步验收标准）。
- 2026-08-11（步骤 2 实现）：按 §5 步骤 2 方案 A 打通并行 `capacity_aware` 基线入口——`_validation_plan` 支持 assignment_mode 覆盖，新增 `--validation-assignment-mode` 与 `--validation-only`，复用既有 `--validation-parallel-envs` 池与 `_summarize_validation_values` 统计口径。
- `--validation-only` 明确不继承 checkpoint 的 `validation_config`，否则新 holdout seed 会被 checkpoint 里的 20262000 静默覆盖；口径字段写入 metrics/manifest 便于追溯。
- 新增 3 个回归用例（plan override、validation-only 端到端不落盘且命令行压过 checkpoint 配置、参数校验）；`pytest tests` 77 passed。
- 已用真实 `stage2_high_best.pt` 做 validation-only 冒烟（24v4-6 各 1 场、capacity_aware、3 worker），确认 resume/口径/进程池路径可用。
- 已把步骤 2 的四组验证命令、结果读取脚本与分支判据写入 `TEST_CLI.md`，交由用户执行。
- 更正前述错误结论：`capacity_aware` 基线并非零代码改动——`validate_checkpoint.py` 未把 `assignment_mode` 接到 CLI（`:484` 调用处恒为 actor）且 trial 串行，`validate_stage1_low_checkpoint.py` 写死 `blue_count=1`，`_validation_plan` 仅在 low_only 分支返回 capacity_aware；串行 96 场需 30 h 以上，需打通并行验证入口。
# 步骤五执行日志（2026-08-11）

- 已读取 `planning-with-files` 完整技能说明并运行 session catch-up。
- 已检查进程：没有本仓库 `train_env` 残留；未启动正式训练。
- 已在 `task_plan.md` 增加 Phase 8，开始核对步骤五/六与现有实现。
- 已锁定评审文档步骤五/六全文和当前默认参数：entropy=0.01、potential=1；开始检查 rollout 数据生命周期、奖励边界与 CLI/checkpoint 配置链路。
- 已完成 5a 代码路径设计：逐步卸载 CPU、最终锁页、训练按 slice 异步回 GPU，并消除 high-only 对冻结低层 critic 的无用求值。
- 已实现 5b（entropy 0.001）、5c（1.0 logit 滞回接口）、5d（potential 512）与 5e（删除死 logits 前向）；已接入 Stage 1→2 受控覆盖/同阶段恢复保护。
- 已实现 CPU rollout storage、CUDA pinned batch 和按训练层级 selective transfer；相关 Python 文件编译通过，下一步补回归测试并跑现有测试定位兼容问题。
- 完整回归最终 84/84 通过（41.52 s）；开始选择空闲 GPU 并运行 48-env、3-iteration high-only 有界 smoke，不启动步骤六正式训练。
- GPU 9 上 48-env × 3-iteration high-only smoke 已通过；execution actor/critic 位级冻结、更新/loss/grad 全零、日志全有限、无残留进程。步骤五已满足文档验收，开始写 `TEST_CLI.md` 的步骤六单种子正式训练与安全续训命令。
- 最终实现进一步改为 observation 全程 CPU pinned、PPO 按 sequence/time chunk non-blocking 迁移；完整测试再次 84/84 通过，最终 streamed 48-env smoke 与位级冻结复核通过。
- 已写入并完成语法校验 `TEST_CLI.md`；没有创建正式 `seed_20260810` 输出目录，也没有启动步骤六训练。Phase 8 完成。
# 2026-08-14：Stage 2 最终 CLI 与产物清理

- 已恢复既有规划记录并新增 Phase 15；当前目录无 Git 元数据，继续采用显式文件清单和校验。
- 已确认 `CLI_COMMAND.md` 的 Stage 2 主命令仍是废弃的 P0 batch16 三种子配置；真实最终 run 是 P1 batch48、seed 20260810、最多 50 updates、iteration 45 早停、iteration 25 best。
- 已核对最终 best SHA256 `ac210d751f1ae226ca6681a201eed09cc867f2f6822f7d604c5f8d9f9ac85cbb`，以及 env 20263000 全确定性验证 `95/96`。
- 已完成 Stage 2 输出盘点；将用 `gio trash` 进行可恢复清理，不直接永久删除。
- 已将 `CLI_COMMAND.md` Stage 2 第 7–11 节更新为最终 P1 batch48 训练、安全续训、首轮检查和全确定性独立验证命令；移除了旧 P0 batch16 路径。
- 已对文档内全部 Bash 代码块执行 `bash -n`，并用 `train_env --help` 核对 P1 与 validation-only 参数接口，均通过。
- 已将 32 个过程目标（旧 v1/baseline、P0 preflight、两个 P1 smoke、9 个周期 checkpoint、`TEST_CLI.md`、随机验证原始目录与中间分析构建物）移入系统回收站，约 768 MiB，可恢复。
- 清理后 `outputs/` 下只剩 Stage 2 最终根 `stage2_v2_p1_batch48`，总大小约 86 MiB；best/latest、训练 metrics/log/manifest、env 20263000 确定性验证和最终综合报告数据均保留。
- 三份保留 SQLite 均通过 `PRAGMA integrity_check=ok`，最终报告 artifact JSON 结构、best SHA256 及清理目标不存在性均已校验。
- 最终复核：`CLI_COMMAND.md` 全部 Bash 代码块语法通过、报告声明的全部 source 路径存在、schema 14 best checkpoint 可由 `612` 环境正常反序列化；planning 完成检查为 15/15。
