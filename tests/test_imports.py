"""Tests for the package's public import surface."""


def test_public_surface_imports():
    import krl
    from krl.algorithms import MAPRL, LBFGSBOptimizer, LBFGSBOptions, RichardsonLucy

    assert krl.MAPRL is MAPRL
    assert krl.RichardsonLucy is RichardsonLucy
    assert krl.LBFGSBOptimizer is LBFGSBOptimizer
    assert krl.LBFGSBOptions is LBFGSBOptions
