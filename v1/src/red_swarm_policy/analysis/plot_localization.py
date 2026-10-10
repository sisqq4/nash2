"""Chinese text and font configuration for Matplotlib figures.

The plotting modules use :func:`chinese_plot_context` when they must leave the
caller's global Matplotlib settings unchanged.  Command line exporters may use
:func:`apply_chinese_plot_style` when a process-wide style is intentional.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator


_FONT_ENVIRONMENT_VARIABLE = "RED_SWARM_PLOT_FONT"
_CHINESE_FONT_FAMILIES = (
    "Microsoft YaHei",
    "DengXian",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "SimHei",
    "SimSun",
    "WenQuanYi Zen Hei",
)


def _font_name_from_path(path: Path, *, source: str) -> str:
    """Register one user or system font file and return its family name."""
    from matplotlib import font_manager

    candidate = path.expanduser()
    try:
        candidate = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{source}指定的中文字体文件不存在：{candidate}") from error
    if not candidate.is_file():
        raise RuntimeError(f"{source}指定的中文字体路径不是文件：{candidate}")
    try:
        font_manager.fontManager.addfont(str(candidate))
        name = font_manager.FontProperties(fname=str(candidate)).get_name()
    except Exception as error:
        raise RuntimeError(f"无法加载{source}指定的中文字体文件 {candidate}：{error}") from error
    if not name:
        raise RuntimeError(f"无法识别{source}指定字体文件的字体族名称：{candidate}")
    return str(name)


def _common_font_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    windows_root = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
    if windows_root:
        font_dir = Path(windows_root) / "Fonts"
        paths.extend(font_dir / name for name in (
            "msyh.ttc", "msyh.ttf", "Deng.ttf", "simhei.ttf", "simsun.ttc"
        ))
    paths.extend(Path(value) for value in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansSC-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/local/share/fonts/NotoSansCJK-Regular.ttc",
        "/usr/local/share/fonts/SourceHanSansSC-Regular.otf",
    ))
    # Preserve priority while avoiding duplicate paths when WINDIR and
    # SystemRoot resolve to the same directory.
    return tuple(dict.fromkeys(paths))


def resolve_chinese_font(font_path: str | Path | None = None) -> str:
    """Return an available Chinese font family, registering a file if needed.

    Resolution priority is an explicit path, ``RED_SWARM_PLOT_FONT``, an
    already registered family, then common Windows and Linux font locations.
    Explicitly configured paths fail loudly instead of silently selecting a
    different font and producing unexpected glyphs.
    """
    if font_path is not None:
        return _font_name_from_path(Path(font_path), source="参数")

    environment_path = os.environ.get(_FONT_ENVIRONMENT_VARIABLE, "").strip()
    if environment_path:
        return _font_name_from_path(
            Path(environment_path), source=f"环境变量 {_FONT_ENVIRONMENT_VARIABLE} "
        )

    from matplotlib import font_manager

    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for family in _CHINESE_FONT_FAMILIES:
        if family in installed:
            return family

    path_errors: list[str] = []
    for candidate in _common_font_paths():
        if not candidate.is_file():
            continue
        try:
            return _font_name_from_path(candidate, source="系统")
        except RuntimeError as error:
            path_errors.append(str(error))

    detail = f"；已找到但加载失败：{'；'.join(path_errors)}" if path_errors else ""
    raise RuntimeError(
        "未找到可显示中文的 Matplotlib 字体。请安装 Microsoft YaHei、等线、黑体、"
        "Noto Sans CJK SC 或思源黑体，或者通过 font_path 参数或 "
        f"{_FONT_ENVIRONMENT_VARIABLE} 环境变量指定字体文件{detail}"
    )


def _plot_rc(font_name: str) -> dict[str, Any]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": list(dict.fromkeys((font_name, "DejaVu Sans"))),
        "axes.unicode_minus": False,
    }


def chinese_plot_rc(font_path: str | Path | None = None) -> dict[str, Any]:
    """Build Matplotlib rc settings that display Chinese and negative signs."""
    return _plot_rc(resolve_chinese_font(font_path))


@contextmanager
def chinese_plot_context(font_path: str | Path | None = None) -> Iterator[str]:
    """Temporarily apply the Chinese plot style and yield the selected family."""
    import matplotlib

    font_name = resolve_chinese_font(font_path)
    with matplotlib.rc_context(rc=_plot_rc(font_name)):
        yield font_name


def apply_chinese_plot_style(font_path: str | Path | None = None) -> str:
    """Apply the Chinese plot style process-wide and return the selected family."""
    import matplotlib

    font_name = resolve_chinese_font(font_path)
    matplotlib.rcParams.update(_plot_rc(font_name))
    return font_name
