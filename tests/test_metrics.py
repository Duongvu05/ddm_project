"""Tests for MAP@K metric."""

import pytest
from ddm_project.baselines import _ap_at_k, map_at_k


def test_ap_perfect():
    predicted = ["a", "b", "c"]
    actual = {"a", "b", "c"}
    assert _ap_at_k(predicted, actual, k=3) == pytest.approx(1.0)


def test_ap_all_wrong():
    predicted = ["x", "y", "z"]
    actual = {"a", "b"}
    assert _ap_at_k(predicted, actual, k=3) == pytest.approx(0.0)


def test_ap_empty_actual():
    assert _ap_at_k(["a", "b"], set(), k=2) == 0.0


def test_ap_partial():
    # hits at position 1 and 3 → (1/1 + 2/3) / 2
    predicted = ["a", "x", "b"]
    actual = {"a", "b"}
    expected = (1.0 + 2 / 3) / 2
    assert _ap_at_k(predicted, actual, k=3) == pytest.approx(expected)


def test_ap_k_truncation():
    predicted = ["x", "x", "x", "a"]
    actual = {"a"}
    assert _ap_at_k(predicted, actual, k=3) == pytest.approx(0.0)


def test_map_single_user():
    preds = {"u1": ["a", "b"]}
    gt = {"u1": {"a", "b"}}
    assert map_at_k(preds, gt, k=2) == pytest.approx(1.0)


def test_map_missing_user_gets_zero():
    preds: dict = {}
    gt = {"u1": {"a"}}
    assert map_at_k(preds, gt, k=12) == pytest.approx(0.0)


def test_map_multiple_users():
    preds = {"u1": ["a", "b"], "u2": ["x", "y"]}
    gt = {"u1": {"a"}, "u2": {"a"}}
    score = map_at_k(preds, gt, k=2)
    assert 0.0 < score < 1.0
