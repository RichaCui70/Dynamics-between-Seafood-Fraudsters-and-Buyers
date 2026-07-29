"""Tests for summary metrics and per-timestep Shapley attributions of Pᵐ."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.constants import DEFAULT_PARAMS
from core.metrics import (
    market_price_from_fp_and_harvest,
    shapley_fp_and_harvest,
    compute_summary_metrics,
    heatmap_display_percents,
    build_heatmap_display_rows,
)


GAMMA_M = DEFAULT_PARAMS['gamma_m']
E_D = DEFAULT_PARAMS['e_d']
E_SM = DEFAULT_PARAMS['e_sm']


def test_market_price_matches_system_formula():
    fp, h = 0.2, 0.05
    expected = GAMMA_M * np.sqrt(((1 - fp) ** E_D) / (h ** E_SM))
    assert market_price_from_fp_and_harvest(fp, h, GAMMA_M, E_D, E_SM) == pytest.approx(expected)


def test_shapley_efficiency():
    fp_ref, h_ref = 0.0, 0.04
    fp_act, h_act = 0.3, 0.08
    phi_fp, phi_h = shapley_fp_and_harvest(
        fp_act, h_act, fp_ref, h_ref, GAMMA_M, E_D, E_SM,
    )
    pm_act = market_price_from_fp_and_harvest(fp_act, h_act, GAMMA_M, E_D, E_SM)
    pm_ref = market_price_from_fp_and_harvest(fp_ref, h_ref, GAMMA_M, E_D, E_SM)
    assert phi_fp + phi_h == pytest.approx(pm_act - pm_ref)


def test_shapley_zero_when_at_reference():
    phi_fp, phi_h = shapley_fp_and_harvest(
        0.0, 0.05, 0.0, 0.05, GAMMA_M, E_D, E_SM,
    )
    assert phi_fp == pytest.approx(0.0)
    assert phi_h == pytest.approx(0.0)


def test_shapley_vectorized_efficiency():
    fp = np.array([0.0, 0.2, 0.5])
    h = np.array([0.04, 0.06, 0.02])
    fp_ref, h_ref = 0.0, 0.04
    phi_fp, phi_h = shapley_fp_and_harvest(
        fp, h, fp_ref, h_ref, GAMMA_M, E_D, E_SM,
    )
    pm_act = market_price_from_fp_and_harvest(fp, h, GAMMA_M, E_D, E_SM)
    pm_ref = market_price_from_fp_and_harvest(fp_ref, h_ref, GAMMA_M, E_D, E_SM)
    np.testing.assert_allclose(phi_fp + phi_h, pm_act - pm_ref)


def test_compute_summary_metrics_per_timestep_tracks_mean_price_gap():
    """Averaged per-timestep φ sums to mean_t Pᵐ(FP_t,H_t) − Pᵐ(ref)."""
    n, burn = 100, 0.6
    burn_steps = int(n * burn)
    seafood = np.concatenate([np.full(burn_steps, 0.1), np.full(n - burn_steps, 0.5)])
    harvest_post = np.linspace(0.02, 0.08, n - burn_steps)
    harvest = np.concatenate([np.full(burn_steps, 0.9), harvest_post])
    fp_post = np.linspace(0.1, 0.4, n - burn_steps)
    fp = np.concatenate([np.full(burn_steps, 0.9), fp_post])
    market_post = market_price_from_fp_and_harvest(fp_post, harvest_post, GAMMA_M, E_D, E_SM)
    market = np.concatenate([np.full(burn_steps, 99.0), market_post])

    harvest_ref = 0.04
    metrics = compute_summary_metrics(
        seafood, harvest, market, fp,
        gamma_m=GAMMA_M, e_d=E_D, e_sm=E_SM,
        harvest_ref=harvest_ref, fraud_perception_ref=0.0,
        burn_in_fraction=burn,
    )
    assert metrics['S̄'] == pytest.approx(0.5)
    assert metrics['H̄'] == pytest.approx(float(np.mean(harvest_post)))
    assert metrics['P̄ᵐ'] == pytest.approx(float(np.mean(market_post)))

    pm_ref = market_price_from_fp_and_harvest(0.0, harvest_ref, GAMMA_M, E_D, E_SM)
    assert metrics['φ_FP'] + metrics['φ_H'] == pytest.approx(
        float(np.mean(market_post)) - pm_ref
    )

    # Differs from Shapley-of-means when the orbit varies
    means_phi_fp, means_phi_h = shapley_fp_and_harvest(
        float(np.mean(fp_post)), float(np.mean(harvest_post)),
        0.0, harvest_ref, GAMMA_M, E_D, E_SM,
    )
    assert metrics['φ_FP'] + metrics['φ_H'] != pytest.approx(
        means_phi_fp + means_phi_h, abs=1e-6
    )


def test_heatmap_display_percents_option_two():
    baseline = {'S̄': 1.0, 'H̄': 0.05, 'P̄ᵐ': 10.0, 'φ_FP': 0.0, 'φ_H': 0.0}
    metrics_by_param = {
        'a': {'S̄': 1.2, 'H̄': 0.04, 'P̄ᵐ': 9.0, 'φ_FP': -1.5, 'φ_H': -0.5},
        'b': {'S̄': 1.0, 'H̄': 0.05, 'P̄ᵐ': 10.0, 'φ_FP': 0.0, 'φ_H': 0.0},
    }
    percents = heatmap_display_percents(metrics_by_param, ['a', 'b'], baseline)
    assert percents['S̄'][0] == pytest.approx(20.0)
    assert percents['H̄'][0] == pytest.approx(-20.0)
    assert percents['P̄ᵐ'][0] == pytest.approx(-10.0)
    assert percents['φ_FP'][0] == pytest.approx(-15.0)
    assert percents['φ_H'][0] == pytest.approx(-5.0)
    assert percents['φ_FP'][1] == pytest.approx(0.0)
    assert percents['φ_H'][1] == pytest.approx(0.0)


def test_build_heatmap_display_rows_baseline_column_near_zero_phi():
    n = 200
    baseline_series = {
        'Seafood': np.full(n, 0.6),
        'Harvest': np.full(n, 0.05),
        'Market Price': np.full(n, 10.0),
        'Perception of Fraud': np.full(n, 0.0),
    }
    active_series = {
        'Seafood': np.full(n, 0.5),
        'Harvest': np.full(n, 0.08),
        'Market Price': np.full(n, 7.0),
        'Perception of Fraud': np.full(n, 0.4),
    }
    baseline_means = {
        'Seafood': 0.6,
        'Harvest': 0.05,
        'Market Price': 10.0,
    }
    percents = build_heatmap_display_rows(
        {'Baseline': baseline_series, 1.0: active_series},
        ['Baseline', 1.0],
        gamma_m=GAMMA_M, e_d=E_D, e_sm=E_SM,
        baseline_means=baseline_means,
    )
    assert percents['S̄'][0] == pytest.approx(0.0)
    assert percents['H̄'][0] == pytest.approx(0.0)
    assert percents['P̄ᵐ'][0] == pytest.approx(0.0)
    assert percents['φ_FP'][0] == pytest.approx(0.0)
    assert percents['φ_H'][0] == pytest.approx(0.0)
    assert percents['φ_FP'][1] != 0.0 or percents['φ_H'][1] != 0.0
