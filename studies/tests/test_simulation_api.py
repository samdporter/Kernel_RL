import numpy as np

from krl_studies.simulation import _api


class _FakeImage:
    def as_array(self):
        return np.zeros((4, 5, 6), dtype=np.float32)

    def voxel_sizes(self):
        return (2.0, 1.5, 1.25)


class _FakeAcquisition:
    def create_uniform_image(self, value):
        assert value == 1.0
        return _FakeImage()


def test_scanner_grid_returns_array_shape_and_voxel_order(monkeypatch):
    monkeypatch.setattr(_api, "_require_sirf", lambda: None)
    assert _api.scanner_grid(_FakeAcquisition()) == ((4, 5, 6), (2.0, 1.5, 1.25))
