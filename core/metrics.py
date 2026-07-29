"""Trajectory summary metrics for heatmaps and analysis.

Computes post-burn means and 2-player Shapley attributions of market price
to fraud perception (FP) and harvest (H).
"""

from __future__ import annotations

import numpy as np

SUMMARY_METRIC_KEYS = ('S̄', 'H̄', 'P̄ᵐ', 'φ_FP', 'φ_H')


def _post_burn_slice(series, burn_in_fraction: float) -> np.ndarray:
    """Return the post-burn-in portion of a 1D series as float64."""
    values = np.asarray(series, dtype=np.float64)
    burn_in_steps = int(len(values) * burn_in_fraction)
    return values[burn_in_steps:]


def _post_burn_mean(series, burn_in_fraction: float) -> float:
    """Return the mean of the post-burn-in portion of a series."""
    return float(np.mean(_post_burn_slice(series, burn_in_fraction)))


def market_price_from_fp_and_harvest(fraud_perception, harvest, gamma_m, e_d, e_sm):
    """Dimensionalized market price: γ_M · √((1−FP)^{ε_d} / H^{ε_sm}).

    Args:
        fraud_perception: Scalar or array of fraud perception values in [0, 1].
        harvest: Scalar or array of harvest values (strictly positive after flooring).
        gamma_m: Market price scaling γ_M.
        e_d: Demand elasticity ε_d.
        e_sm: Market supply elasticity ε_sm.

    Returns:
        Market price with the same shape as the broadcast inputs.
    """
    fp = np.clip(np.asarray(fraud_perception, dtype=np.float64), 0.0, 1.0 - 1e-15)
    h = np.maximum(np.asarray(harvest, dtype=np.float64), np.finfo(np.float64).eps)
    return gamma_m * np.sqrt(((1.0 - fp) ** e_d) / (h ** e_sm))


def shapley_fp_and_harvest(
    fraud_perception_actual,
    harvest_actual,
    fraud_perception_ref,
    harvest_ref,
    gamma_m,
    e_d,
    e_sm,
):
    """2-player Shapley values of (FP, H) for the market-price gap vs a reference.

    Supports scalars or equal-length arrays. Efficiency holds pointwise:
    φ_FP + φ_H = Pᵐ(act) − Pᵐ(ref).

    Args:
        fraud_perception_actual: Actual FP (scalar or array).
        harvest_actual: Actual harvest (scalar or array).
        fraud_perception_ref: Reference FP (typically 0).
        harvest_ref: Reference harvest (typically no-fraud baseline mean).
        gamma_m: Market price scaling γ_M.
        e_d: Demand elasticity ε_d.
        e_sm: Market supply elasticity ε_sm.

    Returns:
        Tuple (φ_FP, φ_H) with the same shape as the actual inputs.
    """
    value_empty = market_price_from_fp_and_harvest(
        fraud_perception_ref, harvest_ref, gamma_m, e_d, e_sm,
    )
    value_fp_only = market_price_from_fp_and_harvest(
        fraud_perception_actual, harvest_ref, gamma_m, e_d, e_sm,
    )
    value_h_only = market_price_from_fp_and_harvest(
        fraud_perception_ref, harvest_actual, gamma_m, e_d, e_sm,
    )
    value_both = market_price_from_fp_and_harvest(
        fraud_perception_actual, harvest_actual, gamma_m, e_d, e_sm,
    )
    phi_fp = 0.5 * (value_fp_only - value_empty) + 0.5 * (value_both - value_h_only)
    phi_h = 0.5 * (value_h_only - value_empty) + 0.5 * (value_both - value_fp_only)
    if np.ndim(phi_fp) == 0:
        return float(phi_fp), float(phi_h)
    return phi_fp, phi_h


def compute_summary_metrics(
    seafood_time_series,
    harvest_time_series,
    market_price_time_series,
    fraud_perception_time_series,
    *,
    gamma_m,
    e_d,
    e_sm,
    harvest_ref,
    fraud_perception_ref=0.0,
    burn_in_fraction=0.6,
):
    """Calculate summary metrics for a simulated trajectory.

    S̄ / H̄ / P̄ᵐ are post-burn means of the corresponding series. φ_FP and φ_H
    are per-timestep Shapley attributions of Pᵐ(FP, H) vs
    (fraud_perception_ref, harvest_ref), then averaged over the post-burn window.
    That yields φ_FP + φ_H = mean_t Pᵐ(FP_t, H_t) − Pᵐ(ref), which tracks the
    observed average price gap much more closely than Shapley-of-means.

    Args:
        seafood_time_series: Seafood biomass series.
        harvest_time_series: Harvest series.
        market_price_time_series: Observed market price series (for P̄ᵐ).
        fraud_perception_time_series: Fraud perception series.
        gamma_m: Market price scaling γ_M.
        e_d: Demand elasticity ε_d.
        e_sm: Market supply elasticity ε_sm.
        harvest_ref: Reference harvest for Shapley counterfactuals.
        fraud_perception_ref: Reference FP for Shapley counterfactuals.
        burn_in_fraction: Fraction of the series discarded as burn-in.

    Returns:
        Dict with keys S̄, H̄, P̄ᵐ, φ_FP, φ_H (φ in price units).
    """
    seafood_mean = _post_burn_mean(seafood_time_series, burn_in_fraction)
    harvest_mean = _post_burn_mean(harvest_time_series, burn_in_fraction)
    market_price_mean = _post_burn_mean(market_price_time_series, burn_in_fraction)

    harvest_post_burn = _post_burn_slice(harvest_time_series, burn_in_fraction)
    fraud_perception_post_burn = _post_burn_slice(
        fraud_perception_time_series, burn_in_fraction,
    )
    phi_fp_series, phi_h_series = shapley_fp_and_harvest(
        fraud_perception_post_burn,
        harvest_post_burn,
        fraud_perception_ref,
        harvest_ref,
        gamma_m,
        e_d,
        e_sm,
    )
    return {
        'S̄': seafood_mean,
        'H̄': harvest_mean,
        'P̄ᵐ': market_price_mean,
        'φ_FP': float(np.mean(phi_fp_series)),
        'φ_H': float(np.mean(phi_h_series)),
    }


def _percent_of_baseline(value, baseline_value) -> float:
    """Return percent change of value relative to baseline_value."""
    if baseline_value == 0:
        return 0.0
    return (value - baseline_value) / abs(baseline_value) * 100.0


def _percent_of_reference(value, reference_value) -> float:
    """Return value as a percent of reference_value."""
    if reference_value == 0:
        return 0.0
    return value / abs(reference_value) * 100.0


def heatmap_display_percents(metrics_by_param, param_values, baseline_metrics):
    """Convert raw summary metrics into heatmap cell percents.

    S̄ / H̄ / P̄ᵐ → % change vs baseline means.
    φ_FP / φ_H → φ as % of baseline P̄ᵐ.

    Args:
        metrics_by_param: Map from param value → summary-metric dict.
        param_values: Column order for the heatmap.
        baseline_metrics: Summary metrics for the no-fraud baseline.

    Returns:
        Dict mapping each metric name to a list of percents aligned with
        param_values.
    """
    baseline_market_price = baseline_metrics['P̄ᵐ']
    percent_by_metric = {metric: [] for metric in SUMMARY_METRIC_KEYS}

    for param_value in param_values:
        metrics = metrics_by_param[param_value]
        percent_by_metric['S̄'].append(
            _percent_of_baseline(metrics['S̄'], baseline_metrics['S̄'])
        )
        percent_by_metric['H̄'].append(
            _percent_of_baseline(metrics['H̄'], baseline_metrics['H̄'])
        )
        percent_by_metric['P̄ᵐ'].append(
            _percent_of_baseline(metrics['P̄ᵐ'], baseline_metrics['P̄ᵐ'])
        )
        percent_by_metric['φ_FP'].append(
            _percent_of_reference(metrics['φ_FP'], baseline_market_price)
        )
        percent_by_metric['φ_H'].append(
            _percent_of_reference(metrics['φ_H'], baseline_market_price)
        )

    return percent_by_metric


def build_heatmap_display_rows(
    time_series_by_param,
    param_values,
    *,
    gamma_m,
    e_d,
    e_sm,
    baseline_means,
    burn_in_fraction=0.6,
    fraud_perception_ref=0.0,
):
    """Build display-ready heatmap rows from time series + no-fraud baseline means.

    ``baseline_means`` uses series keys from ``*_baseline()`` helpers:
    ``Seafood``, ``Harvest``, ``Market Price``.

    Args:
        time_series_by_param: Map from param value → generate_time_series dict.
        param_values: Column order for the heatmap.
        gamma_m: Market price scaling γ_M.
        e_d: Demand elasticity ε_d.
        e_sm: Market supply elasticity ε_sm.
        baseline_means: Post-burn means from the no-fraud baseline run.
        burn_in_fraction: Fraction of each series discarded as burn-in.
        fraud_perception_ref: Reference FP for Shapley counterfactuals.

    Returns:
        Dict mapping each heatmap metric to a list of display percents.
    """
    harvest_ref = baseline_means['Harvest']
    baseline_metrics = {
        'S̄': baseline_means['Seafood'],
        'H̄': baseline_means['Harvest'],
        'P̄ᵐ': baseline_means['Market Price'],
        'φ_FP': 0.0,
        'φ_H': 0.0,
    }

    metrics_by_param = {}
    for param_value in param_values:
        series = time_series_by_param[param_value]
        metrics_by_param[param_value] = compute_summary_metrics(
            series['Seafood'],
            series['Harvest'],
            series['Market Price'],
            series['Perception of Fraud'],
            gamma_m=gamma_m,
            e_d=e_d,
            e_sm=e_sm,
            harvest_ref=harvest_ref,
            fraud_perception_ref=fraud_perception_ref,
            burn_in_fraction=burn_in_fraction,
        )

    return heatmap_display_percents(metrics_by_param, param_values, baseline_metrics)


def market_price_params_from_overrides(system_param_overrides=()):
    """Resolve γ_M, ε_d, ε_sm from DEFAULT_PARAMS + scenario UI overrides.

    Args:
        system_param_overrides: Optional iterable of (key, value) pairs from the
            Streamlit system-parameter UI.

    Returns:
        Tuple (gamma_m, e_d, e_sm).
    """
    from .constants import DEFAULT_PARAMS

    params = DEFAULT_PARAMS.copy()
    if system_param_overrides:
        params.update(dict(system_param_overrides))
    return params['gamma_m'], params['e_d'], params['e_sm']
