"""按第二版报告章节离线导出中文图表及逐图数据文档。

直接运行本文件，不导入项目包、模型或仿真环境。参见 BLUE_REPORT_CHARTS.md。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path


def _numeric_dependency_message(error: Exception) -> str:
    executable = f'& "{sys.executable}"' if sys.platform == "win32" else f'"{sys.executable}"'
    return (
        f"绘图依赖导入失败：{type(error).__name__}: {error}\n\n"
        "请使用运行本脚本的同一个 Python 解释器执行：\n"
        f"  {executable} -m pip install --upgrade "
        '"numpy>=1.24,<3" "matplotlib>=3.7,<4"\n'
        "建议在独立虚拟环境中安装，避免已有科学计算包发生版本冲突。"
    )


try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except Exception as error:
    raise SystemExit(_numeric_dependency_message(error)) from error

try:
    from .plot_localization import apply_chinese_plot_style
    from .report_chart_data import read_run, load_trace_summary
except ImportError:  # Support direct execution from this source directory.
    from plot_localization import apply_chinese_plot_style
    from report_chart_data import read_run, load_trace_summary


def write_chart_document(*args, **kwargs):
    """Delay the optional Word dependency until a real export is requested."""
    try:
        from .report_chart_documents import write_chart_document as implementation
    except ImportError:  # Support direct execution from this source directory.
        from report_chart_documents import write_chart_document as implementation
    return implementation(*args, **kwargs)


DEFAULT_RUN = Path(__file__).resolve().parents[2] / "outputs/blue_rl/reward_ablations_six/05_rl_all_mechanisms"
ORIENTATIONS = {
    "toward_missile_swarm": "迎头", "positive_90_deg": "正九十度侧向",
    "negative_90_deg": "负九十度侧向", "away_from_missile_swarm": "背向",
}
VERDICTS = {"near_vertical": "近垂直飞行", "small_radius_spiral": "小半径螺旋",
            "steep_low_horizontal_speed": "陡峭低水平速度", "near_stationary_reversal": "近静止反转",
            "altitude_boundary": "高度边界", "safety_reliance": "安全过滤依赖"}
TERMINATIONS = {"success": "蓝机被命中", "red_failure": "来弹全部失效", "timeout": "达到总时限"}
LOSSES = {"valid_hit": "命中", "post_closest_miss": "经过最近点后脱靶",
          "guidance_timeout": "制导超时", "low_speed": "低速失效", "ground_impact": "落地",
          "out_of_bounds": "越界", "unknown": "原因未记录"}
REWARDS = {"joint_potential_shaping": "联合势函数塑形", "terminal": "终止奖励",
           "timing_penalty": "时机惩罚", "direction_penalty": "方向惩罚", "overload_penalty": "过载惩罚"}
COLORS = ["#2674A6", "#D06A4B", "#399484", "#9670A3"]


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 缺失或不是有限数值")
    return float(value)


def wilson(k, n):
    if n == 0:
        return None, None
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0., c - radius), min(1., c + radius)


def table(title, headers, rows):
    return {"title": title, "headers": headers, "rows": rows}


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return [None] * 7
    return [float(x) for x in [a.mean(), a.std(), a.min(), *np.quantile(a, [.25, .5, .75]), a.max()]]


def validate_data(training, evaluation):
    windows = training.get("iterations", [])
    rows = evaluation.get("results", [])
    if not windows or not rows:
        raise ValueError("需要训练窗口 iterations 和测试逐回合 results")
    seen, completed = set(), 0
    for w in windows:
        ids = w.get("episode_ids", [])
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("训练窗口缺少回合编号或重复记录同一回合")
        seen.update(ids)
        completed += len(ids)
        if w.get("completed_episodes") != completed:
            raise ValueError("训练完成计数与窗口回合编号数量不一致")
        for key in ["reward_mean", "survival_rate", "simulation_time_mean_s", "optimizer_updates_in_window"]:
            number(w.get(key), key)
        if not 0 <= w["survival_rate"] <= 1:
            raise ValueError("训练逃逸率不在零到一区间")
        if abs(w["survival_rate"] * len(ids) - round(w["survival_rate"] * len(ids))) > 1e-6:
            raise ValueError("训练窗口逃逸率与回合计数不一致")
    ids = [r.get("episode") for r in rows]
    if any(type(x) is not int for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("测试回合编号缺失或重复")
    for r in rows:
        if type(r.get("blue_survived")) is not bool:
            raise ValueError("测试存活标记必须是真实布尔值，不能以缺失代替失败")
        if r.get("blue_orientation") not in ORIENTATIONS:
            raise ValueError(f"未识别的朝向分类：{r.get('blue_orientation')}，请补充中文映射")
        if type(r.get("missile_count")) is not int or r["missile_count"] < 1:
            raise ValueError("来弹数量无效")
        for key in ["reward", "miss_distance_m", "simulation_time_s", "decision_steps"]:
            number(r.get(key), key)
        if r["miss_distance_m"] < 0:
            raise ValueError("脱靶距离不能为负")
    if isinstance(evaluation.get("episodes"), int) and evaluation["episodes"] != len(rows):
        raise ValueError("测试声明回合数与逐回合记录数量不一致")
    return windows, sorted(rows, key=lambda r: r["episode"])


def smooth(values, windows):
    """后向日志点平均；当前点缺失时保持缺失，不跨越缺口绘线。"""
    result = []
    for i, v in enumerate(values):
        a = [x for x in values[max(0, i - windows + 1):i + 1] if x is not None and math.isfinite(x)]
        result.append(sum(a) / len(a) if v is not None and a else None)
    return result


def setup_font(font_path=None):
    name = apply_chinese_plot_style(font_path)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlepad": 12})
    return name


def axes(count=1, figsize=None):
    fig, ax = plt.subplots(1, count, figsize=figsize or (7.8 * count, 4.8), layout="constrained", squeeze=False)
    for a in ax[0]:
        a.grid(axis="y", alpha=.18)
        a.set_axisbelow(True)
    return fig, ax[0]


class Exporter:
    def __init__(self, output, sources, dpi):
        self.output, self.sources, self.dpi = output, sources, dpi
        self.items = []
        self.output.mkdir(parents=True, exist_ok=True)

    def emit(self, section, slug, title, caption, tables, fig=None, notes=(), extra_tables=()):
        folder = self.output / f"{section}_{slug}"
        folder.mkdir(parents=True, exist_ok=True)
        if fig is not None:
            fig.suptitle(title, fontsize=15, fontweight="bold")
            fig.savefig(folder / "图.png", dpi=self.dpi, facecolor="white")
            fig.savefig(folder / "图.svg", facecolor="white")
            plt.close(fig)
        # CSV/JSON保留所有有效数字，Word做适度显示舍入。
        for i, t in enumerate([*tables, *extra_tables], 1):
            with (folder / f"数据表{i:02d}.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(t["headers"])
                writer.writerows(t["rows"])
        content = {"章节": section, "标题": title, "图注": caption, "统计说明": list(notes),
                   "数据来源": self.sources, "文档表格": tables, "完整数据表格": list(extra_tables)}
        (folder / "数据.json").write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        write_chart_document(folder / "关键数据.docx", title, section, caption, list(notes), tables, self.sources)
        md = [f"# {title}", "", caption, "", *[f"- {n}" for n in notes], ""]
        for t in tables:
            md += [f"## {t['title']}", "", "| " + " | ".join(t["headers"]) + " |", "| " + " | ".join(["---"] * len(t["headers"])) + " |"]
            md += ["| " + " | ".join("未记录" if v is None else str(v) for v in row) + " |" for row in t["rows"]]
            md.append("")
        (folder / "关键数据.md").write_text("\n".join(md), encoding="utf-8")
        self.items.append({"章节": section, "标题": title, "类型": "图及数据文档" if fig is not None else "表格文档", "目录": folder.name})
        print(f"已导出 {section} {title}", flush=True)


def group_stats(label, rows):
    n = len(rows)
    k = sum(r["blue_survived"] for r in rows)
    lo, hi = wilson(k, n)
    return [label, n, k, 100 * k / n if n else None,
            100 * lo if lo is not None else None, 100 * hi if hi is not None else None,
            float(np.mean([r["miss_distance_m"] for r in rows])) if n else None,
            float(np.median([r["miss_distance_m"] for r in rows])) if n else None]


GROUP_HEADERS = ["类别", "回合数", "存活数", "逃逸率（百分比）", "区间下限（百分比）", "区间上限（百分比）", "脱靶均值（米）", "脱靶中位数（米）"]


def rate_plot(groups, threshold):
    fig, (ax,) = axes(figsize=(9, 5.2))
    for i, g in enumerate(groups):
        if g[3] is None:
            ax.text(i, 2, "无样本", ha="center")
            continue
        ax.bar(i, g[3], width=.55, color=COLORS[i % 4], alpha=.85)
        ax.errorbar(i, g[3], yerr=[[g[3] - g[4]], [g[5] - g[3]]], color="#333333", capsize=5)
        ax.text(i, g[5] + 3, f"{g[3]:.2f}%\n{g[2]}/{g[1]} 回合", ha="center", fontsize=10)
    ax.axhline(threshold * 100, color=COLORS[1], linestyle="--", label=f"指标门槛 {threshold:.0%}")
    ax.set(xticks=range(len(groups)), xticklabels=[g[0] for g in groups], ylabel="逃逸率", ylim=(0, 110))
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.legend()
    return fig


def training_charts(ex, windows, smoothing):
    counts = [len(w["episode_ids"]) for w in windows]
    spans = sorted(set(counts))
    x = [w["completed_episodes"] for w in windows]
    window_text = f"每窗口 {spans[0]} 回合" if len(spans) == 1 else "窗口回合数见数据表"
    note = f"后向平均采用最近 {smoothing} 个日志点的算术平均，起始阶段使用已有点；不等同于按回合数加权的滑动逃逸率。"
    if len(spans) == 1:
        note += f"完整平滑窗口对应 {smoothing * spans[0]} 个完成回合。"
    else:
        complete_spans = [sum(counts[i-smoothing+1:i+1]) for i in range(smoothing-1, len(counts))]
        note += f"原始窗口实际含 {'、'.join(map(str, spans))} 回合。"
        if complete_spans:
            note += f"完整平滑窗口实际覆盖 {min(complete_spans)} 至 {max(complete_spans)} 个完成回合。"
    specifications = [
        ("3.2.1", "训练损失", "loss_mean", "损失", False),
        ("3.2.2", "训练回报", "reward_mean", "回合累计回报", False),
        ("3.2.3", "训练逃逸率", "survival_rate", "逃逸率", True),
        ("3.2.4", "学习率与梯度范数", "gradient_norm_mean", "梯度范数", False),
        ("3.2.5", "训练仿真结束时刻", "simulation_time_mean_s", "仿真结束时刻（秒）", False),
    ]
    for section, title, key, ylabel, percent in specifications:
        raw = [number(w[key], key) if w.get(key) is not None else None for w in windows]
        avg = smooth(raw, smoothing)
        fig, aa = axes(2 if section == "3.2.4" else 1, figsize=(12, 4.8) if section == "3.2.4" else None)
        ax = aa[0]
        ax.plot(x, raw, color=COLORS[0], alpha=.4, linewidth=1, label=f"原始窗口（{window_text}）")
        ax.plot(x, avg, color=COLORS[0], linewidth=2, label=f"最近 {smoothing} 点后向平均")
        ax.set(xlabel="累计完成回合数", ylabel=ylabel)
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(0, 1)
        ax.legend()
        point_rows = [[x[i], counts[i], raw[i], avg[i], w["optimizer_updates_in_window"], sum(counts[max(0, i-smoothing+1):i+1])] for i, w in enumerate(windows)]
        ts = [table("全部绘图点", ["累计完成回合数", "窗口回合数", ylabel, "后向平均", "窗口优化器更新次数", "平滑实际覆盖回合数"], point_rows)]
        if percent:
            ts.append(table("每窗口存活计数", ["累计完成回合数", "存活数", "回合数", "逃逸率（百分比）"],
                            [[x[i], round(raw[i] * counts[i]), counts[i], raw[i] * 100] for i in range(len(x))]))
        weighted = key in ("loss_mean", "gradient_norm_mean")
        weight_key = "optimizer_updates_in_window" if weighted else "episode_ids"
        summary = []
        # 只按完整日志窗口分组，决不拆分聚合窗口假装拥有更新明细。
        stage_size = 100 if key == "reward_mean" else 200
        for start in range(0, x[-1], stage_size):
            indices = [i for i, w in enumerate(windows) if start < w["completed_episodes"] <= start + stage_size]
            if not indices:
                continue
            valid = [i for i in indices if raw[i] is not None]
            weights = [windows[i][weight_key] if weighted else counts[i] for i in valid]
            total = sum(weights)
            val = sum(raw[i] * a for i, a in zip(valid, weights)) / total if total else None
            summary.append([x[indices[0]] - counts[indices[0]] + 1, x[indices[-1]], sum(counts[i] for i in indices), total, val])
        ts.insert(0, table("训练阶段汇总", ["实际完成次序起点", "实际完成次序终点", "完成回合数", "汇总权重合计", "加权均值"], summary))
        if key == "reward_mean":
            distributions = []
            for start in range(0, x[-1], 100):
                group = [w for w in windows if start < w["completed_episodes"] <= start + 100]
                n = sum(len(w["episode_ids"]) for w in group)
                if not n:
                    continue
                mean = sum(len(w["episode_ids"]) * w["reward_mean"] for w in group) / n
                variance = sum(len(w["episode_ids"]) * (number(w.get("reward_std"), "窗口回报标准差")**2 + w["reward_mean"]**2) for w in group) / n - mean**2
                distributions.append([group[0]["completed_episodes"] - len(group[0]["episode_ids"]) + 1, group[-1]["completed_episodes"], mean, math.sqrt(max(0, variance)),
                                      sum(round(w["survival_rate"] * len(w["episode_ids"])) for w in group),
                                      sum(len(w["episode_ids"]) * w["simulation_time_mean_s"] for w in group) / n])
            ts.insert(1, table("百回合回报分布与结果", ["实际次序起点", "实际次序终点", "平均回报", "总体标准差", "存活回合数", "平均结束时刻（秒）"], distributions))
        notes = [note, "横轴与分组按日志完成次序；并行任务编号不能代替完成次序。",
                 "阶段汇总使用完整窗口，权重为" + ("优化器更新次数。" if weighted else "完成回合数。"),
                 "未通过曲线自动判定收敛轮次。缺失日志点保留缺口，不补零。"]
        if section == "3.2.4":
            lr = [number(w["learning_rate"], "学习率") for w in windows]
            aa[1].plot(x, lr, color=COLORS[2], linewidth=2, label="记录学习率")
            aa[1].set(xlabel="累计完成回合数", ylabel="学习率")
            aa[1].ticklabel_format(axis="y", style="plain", useOffset=False)
            aa[1].legend()
            ts.append(table("学习率全部记录", ["累计完成回合数", "学习率"], list(map(list, zip(x, lr)))))
        if section == "3.2.5":
            notes.append("该量是绝对仿真结束时刻，不是实际运行耗时，也不是策略激活后时长。")
        ex.emit(section, title, title, f"{title}的原始日志窗口与后向平均。", ts, fig, notes)


def termination_chart(ex, rows):
    terms = Counter(r["termination_reason"] for r in rows)
    final = Counter()
    for r in rows:
        alive, reasons = r.get("red_alive"), r.get("red_loss_reasons")
        if not isinstance(alive, list) or not isinstance(reasons, list) or len(alive) != r["missile_count"] or len(reasons) != len(alive):
            raise ValueError("来弹末态记录不完整，不能由原因缺失推断存活")
        for a, reason in zip(alive, reasons):
            if type(a) is not bool:
                raise ValueError("来弹存活标志缺失")
            label = "回合结束时仍存活" if a else LOSSES.get(reason, "其他已记录失效原因")
            final[label] += 1
    fig, aa = axes(2, figsize=(13, 5))
    ts = []
    for ax, data, denominator, title in [(aa[0], {TERMINATIONS.get(k, "其他终止原因"): v for k, v in terms.items()}, len(rows), "回合终止类别"),
                                         (aa[1], final, sum(r["missile_count"] for r in rows), "来弹最终状态")]:
        keys, values = list(data), list(data.values())
        ax.barh(keys, values, color=COLORS[0])
        ax.invert_yaxis()
        ax.set(xlabel="数量", title=f"{title}（分母 {denominator}）", xlim=(0, max(values) * 1.5))
        for i, v in enumerate(values):
            ax.text(v + max(values) * .02, i, f"{v}（{v / denominator:.2%}）", va="center")
        ts.append(table(title, ["类别", "数量", "分母", "比例（百分比）"], [[k, v, denominator, 100 * v / denominator] for k, v in data.items()]))
    ex.emit("2.2.2", "终止与末态", "回合终止类别与来弹最终状态", "回合类别和来弹末态分别统计，采用各自分母。", ts, fig,
            ["沿用程序记录的终止原因；达到时限时即使来弹已失效，也不改写终止类别。", "仍存活的来弹只表示回合结束时的实体状态。"])


def quality_chart(ex, rows):
    summary = []
    for key, label in VERDICTS.items():
        vals = [r.get("flight_quality", {}).get("verdicts", {}).get(key) for r in rows]
        valid = [v for v in vals if type(v) is bool]
        failed = sum(not v for v in valid)
        summary.append([label, len(valid), failed, 100 * failed / len(valid) if valid else None, len(rows) - len(valid)])
    fig, (ax,) = axes(figsize=(10, 5))
    ax.barh([s[0] for s in summary], [s[3] if s[3] is not None else 0 for s in summary], color=COLORS[0])
    for i, s in enumerate(summary):
        ax.text((s[3] or 0) + 1, i, "缺失" if s[3] is None else f"{s[3]:.2f}%", va="center")
    ax.set(xlim=(0, 118), xlabel="已有飞行质量检查项未通过比例")
    ax.xaxis.set_major_formatter(PercentFormatter(100))
    ex.emit("2.2.1", "飞行状态核对", "蓝机飞行状态核对", "依据保存的逐回合飞行质量判据统计未通过比例。",
            [table("已有轨迹判据统计", ["检查项", "有效回合数", "未通过数", "未通过比例（百分比）", "缺失数"], summary)], fig,
            ["采用原记录的判据布尔值，不更改验收阈值。", "该结果是已有轨迹状态核对；数据未包含单独的标志性机动专测，不能代替专测验收。"])


def result_charts(ex, rows, metadata, threshold):
    overall = group_stats("总体", rows)
    ex.emit("3.3.1", "总体逃逸率", "总体逃逸率", "误差线为百分之九十五威尔逊置信区间。",
            [table("总体统计", GROUP_HEADERS, [overall])], rate_plot([overall], threshold),
            ["逃逸按回合结束时蓝机仍存活计算。", "门槛判断与置信区间描述分别保留。"])
    lethal = metadata.get("environment_config", {}).get("missile", {}).get("lethal_radius_m")
    fig, aa = axes(2, figsize=(12, 5))
    ts = []
    raw = []
    for ok, label, color in [(False, "被命中", COLORS[1]), (True, "存活", COLORS[0])]:
        values = sorted(r["miss_distance_m"] for r in rows if r["blue_survived"] == ok)
        if values:
            yy = np.arange(1, len(values) + 1) / len(values)
            for a in aa:
                a.step(values, yy, where="post", label=f"{label}（{len(values)} 回合）", color=color)
            raw += [[label, x, float(y)] for x, y in zip(values, yy)]
        ts.append([label, len(values), *describe(values)])
    for a in aa:
        a.set(xlabel="逐回合脱靶距离（米）", ylabel="组内累计比例")
        a.yaxis.set_major_formatter(PercentFormatter(1))
        if lethal is not None:
            a.axvline(lethal, ls="--", color="#666666", label=f"记录命中半径 {lethal:g} 米")
        a.legend(fontsize=9)
    aa[0].set_xlim(0, max(10, 2 * (lethal or 0)))
    aa[0].set_title("命中边界附近")
    aa[1].set_xscale("symlog", linthresh=1)
    aa[1].set_title("完整范围（大于一米部分采用对数刻度）")
    ex.emit("3.3.1", "脱靶量分布", "逐回合脱靶量分布", "分别显示命中组与存活组的经验累计分布，保留全部长尾样本。",
            [table("脱靶量描述统计", ["结局", "回合数", "均值（米）", "总体标准差（米）", "最小值（米）", "下四分位（米）", "中位数（米）", "上四分位（米）", "最大值（米）"], ts)], fig,
            ["多弹回合仅计一个环境裁决最小距离。图内分母为各结局组有效回合数。", "完整绘图坐标存于额外数据表，未截断远距离样本。"],
            [table("累计分布全部坐标", ["结局", "脱靶距离（米）", "组内累计比例"], raw)])
    counts = sorted({r["missile_count"] for r in rows})
    for sec, slug, title, groups in [
        ("3.3.2", "来弹数量", "不同来弹数量下的逃逸率", [group_stats(f"{c} 枚来弹", [r for r in rows if r["missile_count"] == c]) for c in counts]),
        ("3.3.4", "初始朝向", "不同初始朝向下的逃逸率", [group_stats(v, [r for r in rows if r["blue_orientation"] == k]) for k, v in ORIENTATIONS.items()])]:
        tables = [table("分层统计", GROUP_HEADERS, groups)]
        if sec == "3.3.4":
            timing, contingency = [], []
            for key, label in ORIENTATIONS.items():
                rs = [r for r in rows if r["blue_orientation"] == key]
                active = [r["learning_activation_time_s"] for r in rs if r.get("learning_activation_time_s") is not None]
                timing.append([label, len(rs), len(active), float(np.mean(active)) if active else None,
                               float(np.mean([r["simulation_time_s"] for r in rs])) if rs else None,
                               float(np.mean([r["decision_steps"] for r in rs])) if rs else None])
                k = sum(r["blue_survived"] for r in rs)
                contingency.append([label, k, len(rs) - k])
            tables.append(table("朝向分组过程时间", ["朝向", "回合数", "有效激活记录数", "平均激活时刻（秒）", "平均结束时刻（秒）", "平均决策步数"], timing))
            tables.append(table("朝向与结局列联表", ["朝向", "存活", "被命中"], contingency))
            observed = np.array([r[1:] for r in contingency], dtype=float)
            observed = observed[observed.sum(axis=1) > 0]
            if len(observed) == 4 and np.all(observed.sum(axis=0) > 0):
                expected = observed.sum(axis=1)[:, None] * observed.sum(axis=0)[None, :] / observed.sum()
                chi2 = float(np.sum((observed - expected) ** 2 / expected))
                # 四组两种结局，自由度3；卡方生存函数的半整数闭式，无需SciPy。
                z = math.sqrt(chi2 / 2)
                p = math.erfc(z) + 2 / math.sqrt(math.pi) * z * math.exp(-chi2 / 2)
                tables.append(table("朝向与结局关联检验", ["统计量", "数值"], [["皮尔逊卡方统计量", chi2], ["自由度", 3], ["显著性概率", p], ["最小期望频数", float(expected.min())]]))
        ex.emit(sec, slug, title, "误差线为百分之九十五威尔逊置信区间；柱上同时标出存活数和总回合数。",
                tables, rate_plot(groups, threshold), ["朝向使用日志保存的助推后相对于来弹群分类。", "无样本组保留缺失标记，不以零逃逸率填充。", "关联统计不解释为因果关系；期望频数过小时不宜依赖卡方近似。"])
    joint, matrix = [], np.full((len(counts), len(ORIENTATIONS)), np.nan)
    fig, (ax,) = axes(figsize=(10, 5.5))
    for i, c in enumerate(counts):
        for j, (key, label) in enumerate(ORIENTATIONS.items()):
            g = group_stats(f"{c} 枚来弹 {label}", [r for r in rows if r["missile_count"] == c and r["blue_orientation"] == key])
            joint.append(g)
            matrix[i, j] = g[3] if g[3] is not None else np.nan
            ax.text(j, i, "无样本" if g[3] is None else f"{g[3]:.2f}%\n{g[2]}/{g[1]} 回合", ha="center", va="center", color="white" if (g[3] or 0) > 55 else "black")
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set(xticks=range(len(ORIENTATIONS)), xticklabels=list(ORIENTATIONS.values()), yticks=range(len(counts)), yticklabels=[f"{c} 枚来弹" for c in counts])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="逃逸率（百分比）")
    observed = [g[3] for g in joint if g[3] is not None]
    ex.emit("3.3.5", "联合分层", "来弹数量与初始朝向联合分层", "每格显示逃逸率、存活数与总回合数。",
            [table("全部联合场景", GROUP_HEADERS, joint), table("汇总核对", ["指标", "数值"], [["各组回合数合计", sum(g[1] for g in joint)], ["已观测联合组等权逃逸率（百分比）", float(np.mean(observed))]])], fig,
            ["等权均值只包含有样本组；与按回合计数的总体逃逸率不同。", "置信区间完整保存在数据文档。"])


def process_chart(ex, rows, evaluation):
    groups = [(f"{c} 枚来弹", [r for r in rows if r["missile_count"] == c]) for c in sorted({r["missile_count"] for r in rows})] + [("总体", rows)]
    values = []
    for label, rs in groups:
        active = [r["learning_activation_time_s"] for r in rs if r.get("learning_activation_time_s") is not None]
        values.append([label, len(rs), float(np.mean(active)) if active else None, len(active),
                       float(np.mean([r["simulation_time_s"] for r in rs])), float(np.mean([r["decision_steps"] for r in rs])), float(np.mean([r["reward"] for r in rs]))])
    fig, aa = axes(2, figsize=(12, 5))
    xx = np.arange(len(values))
    aa[0].bar(xx - .18, [v[2] if v[2] is not None else np.nan for v in values], width=.36, label="策略激活时刻", color=COLORS[0])
    aa[0].bar(xx + .18, [v[4] for v in values], width=.36, label="仿真结束时刻", color=COLORS[2])
    aa[0].set(ylabel="仿真时刻（秒）")
    aa[0].legend()
    aa[1].bar(xx, [v[5] for v in values], color=COLORS[0])
    aa[1].set(ylabel="平均有效决策步数")
    for a in aa:
        a.set_xticks(xx, [v[0] for v in values])
    runtime = [[label, evaluation.get(key)] for key, label in [("elapsed_s", "整批运行耗时（秒）"), ("episodes_per_hour", "记录吞吐量（回合每小时）"), ("parallel_envs", "并行环境数"), ("optimizer_updates_during_evaluation", "测试优化器更新次数"), ("target_updates_during_evaluation", "测试目标网络更新次数"), ("replay_transitions_added", "测试回放新增数"), ("learner_state_unchanged", "学习状态保持不变")]]
    ex.emit("3.3.6", "测试过程时长", "测试过程时长与决策步数", "按实际测试场景汇总过程指标，运行耗时单独列入文档。",
            [table("过程指标", ["类别", "回合数", "平均激活时刻（秒）", "有效激活记录数", "平均结束时刻（秒）", "平均决策步数", "平均奖励"], values), table("运行效率与学习状态", ["指标", "记录值"], runtime)], fig,
            ["激活时刻缺失不补零，其均值使用有效记录数。", "整批耗时包含仿真和记录处理，不等同于单次网络推理延迟。"])


def reward_charts(ex, rows, block_size):
    blocks = []
    for i in range(0, len(rows), block_size):
        rs = rows[i:i + block_size]
        g = group_stats(f"区块 {i // block_size + 1}", rs)
        blocks.append([g[0], rs[0]["episode"], rs[-1]["episode"], *g[1:6]])
    fig, aa = axes(2, figsize=(12, 5))
    xx = np.arange(1, len(blocks) + 1)
    aa[0].errorbar(xx, [b[5] for b in blocks], yerr=[[b[5] - b[6] for b in blocks], [b[7] - b[5] for b in blocks]], fmt="o-", capsize=4, color=COLORS[0])
    aa[0].set(xticks=xx, xlabel=f"按回合编号排序的连续区块（每组至多 {block_size} 回合）", ylabel="逃逸率", ylim=(0, 100))
    aa[0].yaxis.set_major_formatter(PercentFormatter(100))
    rewards = [r["reward"] for r in rows]
    edges = np.histogram_bin_edges(rewards, bins=24)
    hist_rows = []
    for ok, label, color in [(False, "被命中", COLORS[1]), (True, "存活", COLORS[0])]:
        values = [r["reward"] for r in rows if r["blue_survived"] == ok]
        hist, _ = np.histogram(values, edges)
        aa[1].stairs(hist, edges, fill=True, alpha=.6, label=label, color=color)
        hist_rows += [[label, float(edges[i]), float(edges[i+1]), int(n), i == len(hist)-1] for i, n in enumerate(hist)]
    aa[1].set(xlabel="回合累计奖励", ylabel="回合数")
    aa[1].legend()
    ex.emit("3.3.7", "区块与奖励分布", "连续评估区块的逃逸率与回合奖励分布", "固定模型的样本区块波动与按结局分组的奖励直方图。",
            [table("区块逃逸率", ["区块", "首回合编号", "末回合编号", "回合数", "存活数", "逃逸率（百分比）", "区间下限（百分比）", "区间上限（百分比）"], blocks),
             table("奖励直方图绘图值", ["结局", "箱下界", "箱上界", "回合数", "上界是否包含"], hist_rows)], fig,
            ["区块按回合编号排序，不是并行完成的墙钟顺序；最后一个区块使用实际样本量。", "测试无学习更新，区块变化不表示训练收敛。", "直方图左闭右开，最后一箱包含右端点。"],
            [table("逐回合奖励", ["回合编号", "结局", "奖励"], [[r["episode"], "存活" if r["blue_survived"] else "被命中", r["reward"]] for r in rows])])
    components, errors = [], []
    for r in rows:
        c = r.get("reward_component_sums", {})
        vv = [number(c.get(k), k) for k in REWARDS]
        components.append(vv)
        errors.append(abs(sum(vv) - r["reward"]))
    means = np.mean(components, axis=0)
    fig, (ax,) = axes(figsize=(9, 5))
    ax.barh(list(REWARDS.values()), means, color=COLORS[0])
    ax.set_xlabel("独立奖励项的每回合均值")
    ax.axvline(0, color="#555555", lw=.8)
    for i, v in enumerate(means):
        ax.text(v, i, f" {v:.6f}", va="center", ha="left" if v >= 0 else "right")
    ax.margins(x=.25)
    ex.emit("3.3.7", "奖励构成", "累计奖励的独立组成", "只统计五项独立加和项，子项不重复累加。",
            [table("独立组成", ["分项", "每回合均值"], [[v, float(means[i])] for i, v in enumerate(REWARDS.values())] + [["分项均值合计", float(sum(means))], ["实际奖励均值", float(np.mean(rewards))], ["逐回合最大加和误差", max(errors)]])], fig,
            ["战术塑形及远离、切向、俯冲等子项已包含在联合势函数中，不能再加一次。", "加和误差大于十亿分之一时须核查奖励字段版本。"],
            [table("逐回合独立奖励", ["回合编号", *REWARDS.values(), "实际奖励", "加和绝对误差"], [[r["episode"], *components[i], r["reward"], errors[i]] for i, r in enumerate(rows)])])
    if max(errors) > 1e-9:
        raise ValueError("奖励五项重建误差过大，已保存误差明细，请检查数据契约")


def write_conditions(ex, training, evaluation, metadata, windows, rows, threshold):
    env = metadata.get("environment_config", {})
    adapter = metadata.get("adapter_config", {})
    options = metadata.get("evaluation_options", {})
    train_config = training.get("experiment_config", {})
    aircraft = env.get("aircraft", {})
    missile = env.get("missile", {})
    runtime = [
        ["实际训练回合数", windows[-1]["completed_episodes"]], ["计划训练回合数", train_config.get("episodes")],
        ["实际独立测试回合数", len(rows)], ["训练基础随机种子", train_config.get("seed")],
        ["测试基础随机种子", options.get("seed")], ["训练日志点数", len(windows)],
        ["每窗口完成回合数", "、".join(str(x) for x in sorted({len(w['episode_ids']) for w in windows}))],
        ["测试实际来弹数量", "、".join(str(x) for x in sorted({r['missile_count'] for r in rows}))],
        ["物理积分步长（秒）", env.get("time_step_s")], ["决策周期（秒）", adapter.get("decision_interval_s")],
        ["助推时长（秒）", missile.get("boost_duration_s")],
        ["任务截止时刻（秒）", env.get("max_steps", 0) * env.get("time_step_s", 0) or None],
        ["策略激活距离（米）", adapter.get("threat_detection_range_m")],
        ["蓝机配置最大过载（重力加速度倍数）", aircraft.get("max_load_factor_g")],
        ["蓝机高度下界（米）", aircraft.get("min_altitude_m")], ["蓝机高度上界（米）", aircraft.get("max_altitude_m")],
        ["蓝机速度下界（米每秒）", aircraft.get("min_speed_mps")], ["蓝机速度上界（米每秒）", aircraft.get("max_speed_mps")],
        ["当前记录命中半径（米）", missile.get("lethal_radius_m")], ["测试并行环境数", evaluation.get("parallel_envs")],
        ["检查点累计优化器更新次数", evaluation.get("checkpoint_optimizer_updates")],
        ["测试期间优化器更新次数", evaluation.get("optimizer_updates_during_evaluation")],
    ]
    ex.emit("1", "测试条件", "测试条件与统计口径", "本表读取当前运行保存的配置和实际记录数量，供报告第一章及测试要求引用。",
            [table("实际运行条件", ["项目", "记录值"], runtime),
             table("统计口径", ["对象", "口径"], [["训练窗口", "依次读取完成日志；横轴是累计完成回合数"], ["训练阶段回报与逃逸率", "完成回合数加权"], ["训练阶段损失与梯度范数", "窗口优化器更新次数加权"], ["逃逸率", "逐回合存活数除以总回合数"], ["置信区间", "百分之九十五威尔逊区间"], ["脱靶量", "每回合一个环境裁决最小距离"], ["缺失数据", "不补零，注明有效样本量"]])],
            notes=["运行目录名称不能代替实际配置。特别是命中半径、初始高度、回合数和日志窗口可能不同于格式参考报告。", "本次仅导出已有记录，不调用训练、评估或仿真入口。"])
    requirements = [["机动最大过载", "不低于九倍重力加速度", "核对配置及名义指令；差分尖峰不参与能力验收"],
                    ["总体逃逸概率", f"不低于 {threshold:.0%}", "由当前独立测试的存活数和回合数计算"],
                    ["收敛轮次", "不超过一百万回合", "训练曲线只提供记录，缺少明确收敛判定时标为未验证"],
                    ["脱靶量", "不大于五米", "逐回合计数，并区分命中组、存活组和当前命中半径"]]
    for section, title in [("2.1", "功能测试指标要求"), ("3.1", "性能测试指标要求")]:
        ex.emit(section, "指标要求", title, "沿用第二版报告的验收要求，不以旧报告统计结果替代当前记录。",
                [table("指标要求与核对口径", ["指标", "报告要求", "本次处理方式"], requirements)],
                notes=["实际核对结果见第四章技术指标资料。基线比较和完整回合案例按要求排除。"])


def trace_chart(ex, trace, metadata):
    episodes = trace.get("episodes", [])
    notes = ["名义指令取保存的执行动作对应合成指令，并非逐物理步插值后的实际指令。",
             "速度差分估计过载受高度边界速度修正影响，不能当作实际机动能力。",
             "只分析峰值附近的短时间数值诊断片段，不导出完整回合案例。", *trace.get("warnings", [])]
    if not episodes or not trace.get("peak"):
        ex.emit("2.2.3", "过载与高度边界", "过载指令与高度边界核对", "缺少可用轨迹，仅保留数据可用性说明。",
                [table("数据可用性", ["项目", "状态"], [["轨迹", "缺少有效记录，无法验证"]])], notes=notes)
        return
    def extreme(key, method=max):
        vals = [r[key] for r in episodes if r.get(key) is not None]
        return method(vals) if vals else None
    samples = sum(r["samples"] for r in episodes)
    high = sum(r.get("above_20g_samples", 0) for r in episodes)
    summary = [["匹配评估回合数", len(episodes)], ["记录采样点总数", samples],
               ["最高名义指令过载（重力加速度倍数）", extreme("max_commanded_load_g")],
               ["最高速度差分估计过载（重力加速度倍数）", extreme("max_estimated_actual_load_g")],
               ["估计过载超过二十倍重力加速度的采样点数", high],
               ["上述采样点占全部记录点比例（百分比）", 100 * high / samples if samples else None],
               ["保存时刻接触高度下界的回合数", sum(bool(r.get("lower_boundary_contact")) for r in episodes)],
               ["保存时刻接触高度上界的回合数", sum(bool(r.get("upper_boundary_contact")) for r in episodes)],
               ["最低记录高度（米）", extreme("min_altitude_m", min)], ["最高记录高度（米）", extreme("max_altitude_m")],
               ["最低记录速度（米每秒）", extreme("min_speed_mps", min)], ["最高记录速度（米每秒）", extreme("max_speed_mps")]]
    peak = trace["peak"]
    labels = {"episode": "回合编号", "time_s": "峰值时刻（秒）", "previous_altitude_m": "上一采样高度（米）", "altitude_m": "峰值采样高度（米）",
              "previous_vertical_speed_mps": "上一采样垂直速度（米每秒）", "vertical_speed_mps": "峰值采样垂直速度（米每秒）",
              "interval_s": "采样间隔（秒）", "commanded_load_g": "名义指令过载（重力加速度倍数）", "estimated_actual_load_g": "差分估计过载（重力加速度倍数）"}
    event = [[label, peak.get(key)] for key, label in labels.items()]
    sample_rows = trace.get("peak_window", [])
    fig, aa = plt.subplots(3, 1, figsize=(9.5, 8), sharex=True, layout="constrained")
    times = [r["time_s"] for r in sample_rows]
    aa[0].plot(times, [r.get("altitude_m") for r in sample_rows], color=COLORS[0], marker=".")
    aa[0].set_ylabel("高度（米）")
    aircraft = metadata.get("environment_config", {}).get("aircraft", {})
    for key in ["min_altitude_m", "max_altitude_m"]:
        boundary = aircraft.get(key)
        if boundary is not None and sample_rows:
            height = peak.get("altitude_m")
            if height is not None and abs(boundary - height) < 500:
                aa[0].axhline(boundary, color=COLORS[1], ls="--", label="配置高度边界")
                aa[0].legend()
    aa[1].plot(times, [r.get("vertical_speed_mps") for r in sample_rows], color=COLORS[0], marker=".")
    aa[1].set_ylabel("垂直速度（米每秒）")
    aa[2].plot(times, [r.get("estimated_actual_load_g") for r in sample_rows], color=COLORS[1], label="速度差分估计过载", marker=".")
    aa[2].plot(times, [r.get("commanded_load_g") for r in sample_rows], color=COLORS[0], label="名义执行动作指令")
    aa[2].set(xlabel="仿真时刻（秒）", ylabel="过载（重力加速度倍数）")
    aa[2].legend()
    for ax in aa:
        ax.axvline(peak["time_s"], color="#777777", ls=":")
        ax.grid(alpha=.2)
    fields = ["episode", "samples", "min_altitude_m", "max_altitude_m", "min_speed_mps", "max_speed_mps", "max_commanded_load_g", "max_estimated_actual_load_g", "above_20g_samples", "lower_boundary_contact", "upper_boundary_contact", "start_s", "end_s"]
    headers = ["回合编号", "采样点数", "最低高度（米）", "最高高度（米）", "最低速度（米每秒）", "最高速度（米每秒）", "最高名义指令过载", "最高差分估计过载", "估计过载超过二十倍采样数", "接触高度下界", "接触高度上界", "记录起点（秒）", "记录终点（秒）"]
    short_fields = ["time_s", "altitude_m", "vertical_speed_mps", "commanded_load_g", "estimated_actual_load_g"]
    ex.emit("2.2.3", "过载与高度边界", "最大差分估计过载的边界数值诊断", f"回合 {peak['episode']} 的全样本最大差分估计过载及其邻域。",
            [table("全样本边界统计", ["统计量", "值"], summary), table("峰值事件", ["项目", "值"], event),
             table("峰值邻域全部绘图点", ["仿真时刻（秒）", "高度（米）", "垂直速度（米每秒）", "名义指令过载", "差分估计过载"], [[r.get(k) for k in short_fields] for r in sample_rows])], fig, notes,
            [table("逐回合状态统计", headers, [[r.get(k) for k in fields] for r in episodes])])


def conclusions(ex, windows, rows, trace, metadata, threshold):
    n, k = len(rows), sum(r["blue_survived"] for r in rows)
    lo, hi = wilson(k, n)
    aircraft = metadata.get("environment_config", {}).get("aircraft", {})
    configured_load = aircraft.get("max_load_factor_g")
    loads = [r["max_commanded_load_g"] for r in trace.get("episodes", []) if r.get("max_commanded_load_g") is not None]
    peak_command = max(loads) if loads else None
    under5 = sum(r["miss_distance_m"] <= 5 for r in rows)
    judgment = [
        ["机动过载不低于九倍重力加速度", f"配置值 {configured_load}；最高名义执行指令 {peak_command}", "名义指令层达到" if configured_load is not None and peak_command is not None and min(configured_load, peak_command) >= 9 else "未验证达到"],
        [f"总体逃逸率不低于 {threshold:.0%}", f"{k}/{n}；{k/n:.2%}；区间 {lo:.2%} 至 {hi:.2%}", "样本比例达到" if k / n >= threshold else "样本比例未达到"],
        ["收敛不超过一百万回合", f"已保存 {windows[-1]['completed_episodes']} 回合训练记录", "未记录收敛轮次判定，不能据此验收"],
        ["脱靶量不大于五米", f"{under5}/{n} 回合符合；命中组与存活组分别统计", "全体回合满足" if under5 == n else "全体回合未全部满足"],
    ]
    ex.emit("4.1", "技术指标核对", "技术指标与实际记录核对", "技术要求沿用格式参考报告，实测值来自当前运行。",
            [table("技术指标核对", ["报告指标", "当前记录", "判断范围"], judgment)],
            notes=["第二章与第三章的指标要求可引用本表。", "报告五米脱靶量指标与当前仿真的命中半径是两个不同数值，不能互相替换。", "差分估计过载尖峰不用于机动能力验收。"])
    training_rows = []
    for label, subset in [("前二百完成回合对应窗口", [w for w in windows if w["completed_episodes"] <= 200]),
                          ("后二百完成回合对应窗口", [w for w in windows if w["completed_episodes"] > windows[-1]["completed_episodes"] - 200]),
                          ("全部训练窗口", windows)]:
        total = sum(len(w["episode_ids"]) for w in subset)
        if total:
            training_rows.append([label, total, *[sum(len(w["episode_ids"]) * w[key] for w in subset) / total for key in ["reward_mean", "survival_rate", "simulation_time_mean_s"]]])
    ex.emit("4.2", "训练与测试结果分析", "训练与测试结果分析数据", "保留前后训练阶段与固定模型测试的独立统计，供撰写结果分析使用。",
            [table("训练阶段比较", ["阶段", "实际回合数", "平均回报", "逃逸率（零到一）", "平均仿真结束时刻（秒）"], training_rows),
             table("独立测试", ["回合数", "存活数", "逃逸率（百分比）", "平均奖励"], [[n, k, 100*k/n, float(np.mean([r['reward'] for r in rows]))]])],
            notes=["训练包含探索和变化的策略；测试加载固定模型，二者不能当作同一分布的重复试验。", "阶段变化仅描述已有记录，不自动判定收敛，也不替代排除的基线对比。"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="包含 train 和 evaluation 的运行目录")
    parser.add_argument("--train-dir", "--training-dir", dest="train_dir", type=Path,
                        help="训练数据目录；默认使用 RUN_DIR/train")
    parser.add_argument("--evaluation-dir", "--eval-dir", dest="evaluation_dir", type=Path,
                        help="测试数据目录；默认使用 RUN_DIR/evaluation")
    parser.add_argument("--output", type=Path, help="导出目录，默认使用运行目录下 report_charts_zh")
    parser.add_argument("--font", type=Path, help="中文字体文件")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--smooth-windows", type=int, default=10, help="后向平滑日志点数；每点实际回合数写入数据表")
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--escape-threshold", type=float, default=.30, help="报告逃逸率指标门槛")
    args = parser.parse_args(argv)
    if args.smooth_windows < 1 or args.block_size < 1 or args.dpi < 72 or not 0 <= args.escape_threshold <= 1:
        parser.error("平滑点数、区块大小、分辨率或逃逸率门槛无效")
    if (args.train_dir is None) != (args.evaluation_dir is None):
        parser.error("--train-dir 与 --evaluation-dir 必须同时提供")
    if args.train_dir is not None:
        training_dir = args.train_dir.resolve()
        evaluation_dir = args.evaluation_dir.resolve()
        if args.run_dir is not None:
            root = args.run_dir.resolve()
        else:
            try:
                root = Path(os.path.commonpath((training_dir, evaluation_dir))).resolve()
            except ValueError:
                root = training_dir.parent
    else:
        root = (args.run_dir or DEFAULT_RUN).resolve()
        training_dir = (root / "train").resolve()
        evaluation_dir = (root / "evaluation").resolve()
    default_output = (evaluation_dir / "report_charts_zh"
                      if args.evaluation_dir is not None else root / "report_charts_zh")
    output = (args.output or default_output).resolve()
    input_locations = {"运行目录": str(root), "训练目录": str(training_dir),
                       "测试目录": str(evaluation_dir)}
    try:
        font = setup_font(args.font)
        training, evaluation, metadata = read_run(
            root, training_dir=training_dir, evaluation_dir=evaluation_dir)
        windows, rows = validate_data(training, evaluation)
        sources = [str(training.get("source", training_dir / "training_metrics.json")),
                   str(evaluation.get("source", evaluation_dir / "evaluation.json")),
                   str(metadata.get("source", evaluation_dir / "flight_quality/evaluation_metadata.json")),
                   str(evaluation_dir / "flight_quality/flight_quality_episodes.jsonl")]
        ex = Exporter(output, sources, args.dpi)
        (output / "导出清单.json").write_text(
            json.dumps({"状态": "进行中", **input_locations}, ensure_ascii=False), encoding="utf-8")
        write_conditions(ex, training, evaluation, metadata, windows, rows, args.escape_threshold)
        quality_chart(ex, rows)
        termination_chart(ex, rows)
        trace = load_trace_summary(root, evaluation, evaluation_dir=evaluation_dir)
        trace_chart(ex, trace, metadata)
        training_charts(ex, windows, args.smooth_windows)
        result_charts(ex, rows, metadata, args.escape_threshold)
        process_chart(ex, rows, evaluation)
        reward_charts(ex, rows, args.block_size)
        conclusions(ex, windows, rows, trace, metadata, args.escape_threshold)
        hashes = {}
        for p in sources:
            if Path(p).is_file():
                digest = hashlib.sha256()
                with Path(p).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                hashes[p] = digest.hexdigest()
        manifest = {"状态": "完成", **input_locations, "中文字体": font, "训练回合数": windows[-1]["completed_episodes"], "测试回合数": len(rows),
                    "排除章节": ["3.3.3 与基线模型结果对比测试", "3.4 完整回合案例分析"], "产物": ex.items,
                    "输入摘要": hashes}
        (output / "导出清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# 第二版报告中文图表", "", f"训练 {windows[-1]['completed_episodes']} 回合，测试 {len(rows)} 回合。", "", "按第二版报告章节组织；统计值全部重新读取当前运行。", ""]
        lines += [f"- {i['章节']} [{i['标题']}]({i['目录']}/关键数据.md)（{i['类型']}）" for i in ex.items]
        lines += ["", "排除 3.3.3 基线对比和 3.4 完整回合分析。每个图目录包含中文图片、Word 文档、Markdown 文档和精确数值表。", "数值表中空值表示未记录，布尔值表示原记录判据。不同运行请使用新的输出目录，避免混入旧结果。"]
        (output / "图表索引.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"完成：{len(ex.items)} 个图表资料目录，输出至 {output}")
    except (OSError, ValueError, KeyError) as error:
        print(f"导出失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
