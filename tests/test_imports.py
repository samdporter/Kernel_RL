"""Tests for the package's public import surface."""

import json
import os
import subprocess
import sys


def test_public_surface_imports():
    import krl
    from krl.algorithms import MAPRL, LBFGSBOptimizer, LBFGSBOptions, RichardsonLucy

    assert krl.MAPRL is MAPRL
    assert krl.RichardsonLucy is RichardsonLucy
    assert krl.LBFGSBOptimizer is LBFGSBOptimizer
    assert krl.LBFGSBOptions is LBFGSBOptions


_IMPORT_LOGGING_SCRIPT = r"""
import json
import logging


def snapshot(logger):
    return {
        "handlers": [
            {
                "type": type(handler).__name__,
                "level": logging.getLevelName(handler.level),
                "formatter": type(handler.formatter).__name__ if handler.formatter else None,
            }
            for handler in logger.handlers
        ],
        "level": logging.getLevelName(logger.level),
        "propagate": logger.propagate,
    }


maprl_logger = logging.getLogger("krl.algorithms.maprl")
before = {"root": snapshot(logging.getLogger()), "maprl": snapshot(maprl_logger)}

import krl  # noqa: F401,E402
import krl.algorithms.maprl  # noqa: F401,E402

after = {"root": snapshot(logging.getLogger()), "maprl": snapshot(maprl_logger)}
print(json.dumps({"before": before, "after": after}))
"""


def run_import_snapshot(env_extra=None):
    env = os.environ.copy()
    env.pop("MAPRL_LOG_LEVEL", None)
    if env_extra:
        env.update(env_extra)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_LOGGING_SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_import_does_not_configure_logging():
    snapshot = run_import_snapshot()

    assert snapshot["after"] == snapshot["before"]
    assert snapshot["after"]["maprl"]["level"] == "NOTSET"
    assert snapshot["after"]["maprl"]["propagate"] is True
    assert snapshot["after"]["maprl"]["handlers"] == []


def test_maprl_log_level_env_has_no_effect():
    default = run_import_snapshot()
    with_env = run_import_snapshot({"MAPRL_LOG_LEVEL": "DEBUG"})

    assert with_env["after"] == with_env["before"]
    assert with_env["after"]["maprl"] == default["after"]["maprl"]
