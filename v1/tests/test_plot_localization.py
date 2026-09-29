from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import matplotlib
from matplotlib import font_manager
import pytest

MODULE_PATH = (Path(__file__).resolve().parents[1] / "src" / "red_swarm_policy" /
               "plot_localization.py")
SPEC = importlib.util.spec_from_file_location("plot_localization", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
plot_localization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_localization)


def _bundled_font() -> Path:
    return Path(font_manager.findfont("DejaVu Sans", fallback_to_default=False))


def test_explicit_font_path_has_priority_over_invalid_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RED_SWARM_PLOT_FONT", "missing-environment-font.ttf")

    name = plot_localization.resolve_chinese_font(_bundled_font())

    assert name == "DejaVu Sans"


def test_environment_font_path_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RED_SWARM_PLOT_FONT", str(_bundled_font()))

    assert plot_localization.resolve_chinese_font() == "DejaVu Sans"


def test_registered_chinese_family_is_selected_in_priority_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RED_SWARM_PLOT_FONT", raising=False)
    monkeypatch.setattr(
        font_manager.fontManager,
        "ttflist",
        [SimpleNamespace(name="SimHei"), SimpleNamespace(name="Microsoft YaHei")],
    )

    assert plot_localization.resolve_chinese_font() == "Microsoft YaHei"


def test_common_system_font_path_is_registered_as_last_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    font_path = _bundled_font()
    monkeypatch.delenv("RED_SWARM_PLOT_FONT", raising=False)
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [])
    monkeypatch.setattr(plot_localization, "_common_font_paths", lambda: (font_path,))

    assert plot_localization.resolve_chinese_font() == "DejaVu Sans"


def test_missing_configured_font_has_clear_chinese_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RED_SWARM_PLOT_FONT", str(MODULE_PATH.with_name("不存在.ttf")))

    with pytest.raises(RuntimeError, match="环境变量.*中文字体文件不存在"):
        plot_localization.resolve_chinese_font()


def test_no_available_chinese_font_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RED_SWARM_PLOT_FONT", raising=False)
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [])
    monkeypatch.setattr(plot_localization, "_common_font_paths", lambda: ())

    with pytest.raises(RuntimeError, match="未找到可显示中文.*RED_SWARM_PLOT_FONT"):
        plot_localization.resolve_chinese_font()


def test_plot_rc_contains_chinese_family_and_unicode_minus_setting() -> None:
    settings = plot_localization.chinese_plot_rc(_bundled_font())

    assert settings["font.family"] == "sans-serif"
    assert settings["font.sans-serif"] == ["DejaVu Sans"]
    assert settings["axes.unicode_minus"] is False


def test_context_restores_global_rc_settings() -> None:
    keys = ("font.family", "font.sans-serif", "axes.unicode_minus")
    before = {key: matplotlib.rcParams[key].copy() if isinstance(matplotlib.rcParams[key], list)
              else matplotlib.rcParams[key] for key in keys}

    with plot_localization.chinese_plot_context(_bundled_font()) as name:
        assert name == "DejaVu Sans"
        assert matplotlib.rcParams["font.family"] == ["sans-serif"]
        assert matplotlib.rcParams["font.sans-serif"] == ["DejaVu Sans"]
        assert matplotlib.rcParams["axes.unicode_minus"] is False

    after = {key: matplotlib.rcParams[key].copy() if isinstance(matplotlib.rcParams[key], list)
             else matplotlib.rcParams[key] for key in keys}
    assert after == before


def test_apply_style_updates_current_rc_context() -> None:
    with matplotlib.rc_context():
        name = plot_localization.apply_chinese_plot_style(_bundled_font())
        assert name == "DejaVu Sans"
        assert matplotlib.rcParams["font.family"] == ["sans-serif"]
        assert matplotlib.rcParams["font.sans-serif"] == ["DejaVu Sans"]
        assert matplotlib.rcParams["axes.unicode_minus"] is False
