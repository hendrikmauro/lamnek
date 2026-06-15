"""Tests für CUDA-Setup."""

from __future__ import annotations

from lamnek.cuda_setup import setup_cuda


def test_setup_cuda_returns_list() -> None:
    paths = setup_cuda()
    assert isinstance(paths, list)
