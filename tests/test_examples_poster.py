"""Structural tests for examples/scripts/poster.py path handling."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "scripts" / "poster.py"
PLOT_DEPS = ("numpy", "pandas", "matplotlib.pyplot", "nibabel", "cmasher")


def available(name: str) -> bool:
    # find_spec raises ModuleNotFoundError when the top-level package is missing.
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


MISSING_PLOT_DEPS = tuple(name for name in PLOT_DEPS if not available(name))
requires_plot_deps = pytest.mark.skipif(
    bool(MISSING_PLOT_DEPS),
    reason="optional plotting dependencies not installed: "
    + ", ".join(MISSING_PLOT_DEPS),
)


def read_source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def parse_source() -> ast.Module:
    return ast.parse(read_source(), filename=str(SCRIPT))


def is_main_guard(node: ast.If) -> bool:
    return "__main__" in ast.unparse(node.test)


def module_scope_mkdir_lines() -> list[int]:
    """Line numbers of mkdir calls that run when the module is imported.

    Calls inside functions, classes and the ``__main__`` guard are skipped.
    """
    lines: list[int] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.If) and is_main_guard(child):
                continue
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in ("mkdir", "makedirs"):
                    lines.append(child.lineno)
            visit(child)

    visit(parse_source())
    return lines


def load_poster_module():
    spec = importlib.util.spec_from_file_location("poster_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main_block(module) -> None:
    tree = parse_source()
    guard = next(
        node for node in tree.body if isinstance(node, ast.If) and is_main_guard(node)
    )
    body = ast.Module(body=guard.body, type_ignores=[])
    ast.fix_missing_locations(body)
    exec(compile(body, str(SCRIPT), "exec"), module.__dict__)


def test_no_absolute_personal_home_defaults():
    assert "/home/" not in read_source()


def test_no_directory_creation_at_import_time():
    assert module_scope_mkdir_lines() == []


@requires_plot_deps
def test_import_creates_no_directories(monkeypatch):
    # Import the third-party dependencies first: they may create their own
    # config directories, which is out of scope here.
    for name in PLOT_DEPS:
        importlib.import_module(name)

    created = []

    def record_mkdir(self, *args, **kwargs):
        created.append(self)

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    module = load_poster_module()

    assert created == []
    assert module.OUTPUT_DIR == REPO_ROOT / "poster_figures"


@requires_plot_deps
def test_main_block_creates_output_dir_before_generating(monkeypatch, tmp_path):
    module = load_poster_module()
    out_dir = tmp_path / "poster_figures"
    monkeypatch.setattr(module, "OUTPUT_DIR", out_dir)

    events = []
    real_mkdir = Path.mkdir

    def recording_mkdir(self, *args, **kwargs):
        events.append(("mkdir", self, kwargs))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", recording_mkdir)

    def make_stub(name):
        def stub():
            events.append(("generated", name, out_dir.exists()))

        return stub

    for name in sorted(vars(module)):
        if name.startswith("generate_"):
            monkeypatch.setattr(module, name, make_stub(name))

    run_main_block(module)

    mkdirs = [event for event in events if event[0] == "mkdir"]
    assert mkdirs == [("mkdir", out_dir, {"parents": True, "exist_ok": True})]
    assert events[0] == mkdirs[0]
    generated = [event for event in events if event[0] == "generated"]
    assert generated
    assert all(exists for _, _, exists in generated)
    assert out_dir.is_dir()
